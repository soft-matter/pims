###############################################################################

# Reader for CINE files produced by Vision Research Phantom Software
# Author: Dustin Kleckner
# dkleckner@uchicago.edu

# Modified by Thomas A Caswell (tcaswell@uchicago.edu)
# Added to PIMS by Thomas A Caswell (tcaswell@gmail.com)

# Modified by B. Neel
###############################################################################

from pims.frame import Frame
from pims.base_frames import FramesSequence, index_attr
from pims.utils.misc import FileLocker
import time
import struct
import numpy as np
from numpy import array, frombuffer, where
from threading import Lock
import datetime
import hashlib
import warnings
from collections.abc import Iterable

__all__ = ('Cine', )


# '<' for little endian (cine documentation)
def _build_struct(dtype):
    return struct.Struct(str("<" + dtype))


FRACTION_MASK = (2**32-1)
MAX_INT = 2**32

# Harmonized/simplified cine file data types with Python struct doc
UINT8 = 'B'
CHAR = 'b'
UINT16 = 'H'
INT16 = 'h'
BOOL = 'i'
UINT32 = 'I'
INT32 = 'i'
INT64 = 'q'
FLOAT = 'f'
DOUBLE = 'd'
TIME64 = 'Q'
RECT = '4i'
WBGAIN = '2f'
IMFILTER = '28i'
# TODO: get correct format for TrigTC
TC = '8s'

CFA_NONE = 0    # gray sensor
CFA_VRI = 1     # gbrg/rggb sensor
CFA_VRIV6 = 2   # bggr/grbg sensor
CFA_BAYER = 3   # gb/rg sensor
CFA_BAYERFLIP = 4   #rg/gb sensor

TAGGED_FIELDS = {
    1000: ('ang_dig_sigs', ''),
    1001: ('image_time_total', TIME64),
    1002: ('image_time_only', TIME64),
    1003: ('exposure_only', UINT32),
    1004: ('range_data', ''),
    1005: ('binsig', ''),
    1006: ('anasig', ''),
    1007: ('time_code', '')}

HEADER_FIELDS = [
    ('type', '2s'),
    ('header_size', UINT16),
    ('compression', UINT16),
    ('version', UINT16),
    ('first_movie_image', INT32),
    ('total_image_count', UINT32),
    ('first_image_no', INT32),
    ('image_count', UINT32),
    # Offsets of following sections
    ('off_image_header', UINT32),
    ('off_setup', UINT32),
    ('off_image_offsets', UINT32),
    ('trigger_time', TIME64),
]

BITMAP_INFO_FIELDS = [
    ('bi_size', UINT32),
    ('bi_width', INT32),
    ('bi_height', INT32),
    ('bi_planes', UINT16),
    ('bi_bit_count', UINT16),
    ('bi_compression', UINT32),
    ('bi_image_size', UINT32),
    ('bi_x_pels_per_meter', INT32),
    ('bi_y_pels_per_meter', INT32),
    ('bi_clr_used', UINT32),
    ('bi_clr_important', UINT32),
]


SETUP_FIELDS = [
    ('frame_rate_16', UINT16),
    ('shutter_16', UINT16),
    ('post_trigger_16', UINT16),
    ('frame_delay_16', UINT16),
    ('aspect_ratio', UINT16),
    ('contrast_16', UINT16),
    ('bright_16', UINT16),
    ('rotate_16', UINT8),
    ('time_annotation', UINT8),
    ('trig_cine', UINT8),
    ('trig_frame', UINT8),
    ('shutter_on', UINT8),
    ('description_old', '121s'),
    ('mark', '2s'),
    ('length', UINT16),
    ('binning', UINT16),
    ('sig_option', UINT16),
    ('bin_channels', INT16),
    ('samples_per_image', UINT8),
    ] + [('bin_name{:d}'.format(i), '11s') for i in range(8)] + [
    ('ana_option', UINT16),
    ('ana_channels', INT16),
    ('res_6', UINT8),
    ('ana_board', UINT8),
    ] + [('ch_option{:d}'.format(i), INT16) for i in range(8)] + [
    ] + [('ana_gain{:d}'.format(i), FLOAT) for i in range(8)] + [
    ] + [('ana_unit{:d}'.format(i), '6s') for i in range(8)] + [
    ] + [('ana_name{:d}'.format(i), '11s') for i in range(8)] + [
    ('i_first_image', INT32),
    ('dw_image_count', UINT32),
    ('n_q_factor', INT16),
    ('w_cine_file_type', UINT16),
    ] + [('sz_cine_path{:d}'.format(i), '65s') for i in range(4)] + [
    ('b_mains_freq', UINT16),
    ('b_time_code', UINT8),
    ('b_priority', UINT8),
    ('w_leap_sec_dy', UINT16),
    ('d_delay_tc', DOUBLE),
    ('d_delay_pps', DOUBLE),
    ('gen_bits', UINT16),
    ('res_1', INT32),  
    ('res_2', INT32),
    ('res_3', INT32),
    ('im_width', UINT16),
    ('im_height', UINT16),
    ('edr_shutter_16', UINT16),
    ('serial', UINT32),
    ('saturation', INT32),
    ('res_5', UINT8),
    ('auto_exposure', UINT32),
    ('b_flip_h', BOOL),
    ('b_flip_v', BOOL),
    ('grid', UINT32),
    ('frame_rate', UINT32),
    ('shutter', UINT32),
    ('edr_shutter', UINT32),
    ('post_trigger', UINT32),
    ('frame_delay', UINT32),
    ('b_enable_color', BOOL),
    ('camera_version', UINT32),
    ('firmware_version', UINT32),
    ('software_version', UINT32),
    ('recording_time_zone', INT32),
    ('cfa', UINT32),
    ('bright', INT32),
    ('contrast', INT32),
    ('gamma', INT32),
    ('res_21', UINT32),
    ('auto_exp_level', UINT32),
    ('auto_exp_speed', UINT32),
    ('auto_exp_rect', RECT),
    ('wb_gain', '8f'),
    ('rotate', INT32),
    ('wb_view', WBGAIN),
    ('real_bpp', UINT32),
    ('conv_8_min', UINT32),
    ('conv_8_max', UINT32),
    ('filter_code', INT32),
    ('filter_param', INT32),
    ('uf', IMFILTER),
    ('black_cal_sver', UINT32),
    ('white_cal_sver', UINT32),
    ('gray_cal_sver', UINT32),
    ('b_stamp_time', BOOL),
    ('sound_dest', UINT32),
    ('frp_steps', UINT32),
    ] + [('frp_img_nr{:d}'.format(i), INT32) for i in range(16)] + [
    ] + [('frp_rate{:d}'.format(i), UINT32) for i in range(16)] + [
    ] + [('frp_exp{:d}'.format(i), UINT32) for i in range(16)] + [
    ('mc_cnt', INT32),
    ] + [('mc_percent{:d}'.format(i), FLOAT) for i in range(64)] + [
    ('ci_calib', UINT32),
    ('calib_width', UINT32),
    ('calib_height', UINT32),
    ('calib_rate', UINT32),
    ('calib_exp', UINT32),
    ('calib_edr', UINT32),
    ('calib_temp', UINT32),
    ] + [('header_serial{:d}'.format(i), UINT32) for i in range(4)] + [
    ('range_code', UINT32),
    ('range_size', UINT32),
    ('decimation', UINT32),
    ('master_serial', UINT32),
    ('sensor', UINT32),
    ('shutter_ns', UINT32),
    ('edr_shutter_ns', UINT32),
    ('frame_delay_ns', UINT32),
    ('im_pos_xacq', UINT32),
    ('im_pos_yacq', UINT32),
    ('im_width_acq', UINT32),
    ('im_height_acq', UINT32),
    ('description', '4096s'),
    ('rising_edge', BOOL),
    ('filter_time', UINT32),
    ('long_ready', BOOL),
    ('shutter_off', BOOL),
    ('res_4', '16s'),
    ('b_meta_WB', BOOL),
    ('hue', INT32),
    ('black_level', INT32),
    ('white_level', INT32),
    ('lens_description', '256s'),
    ('lens_aperture', FLOAT),
    ('lens_focus_distance', FLOAT),
    ('lens_focal_length', FLOAT),
    ('f_offset', FLOAT),
    ('f_gain', FLOAT),
    ('f_saturation', FLOAT),
    ('f_hue', FLOAT),
    ('f_gamma', FLOAT),
    ('f_gamma_R', FLOAT),
    ('f_gamma_B', FLOAT),
    ('f_flare', FLOAT),
    ('f_pedestal_R', FLOAT),
    ('f_pedestal_G', FLOAT),
    ('f_pedestal_B', FLOAT),
    ('f_chroma', FLOAT),
    ('tone_label', '256s'),
    ('tone_points', INT32),
    ('f_tone', ''.join(32*['2f'])),
    ('user_matrix_label', '256s'),
    ('enable_matrices', BOOL),
    ('f_user_matrix', '9'+FLOAT),
    ('enable_crop', BOOL),
    ('crop_left_top_right_bottom', '4i'),
    ('enable_resample', BOOL),
    ('resample_width', UINT32),
    ('resample_height', UINT32),
    ('f_gain16_8', FLOAT),
    ('frp_shape', '16'+UINT32),
    ('trig_TC', TC),
    ('f_pb_rate', FLOAT),
    ('f_tc_rate', FLOAT),
    ('cine_name', '256s')
]

#from VR doc: This field is maintained for compatibility with old versions but
#a new field was added for that information. The new field can be larger or may
#have a different measurement unit.
UPDATED_FIELDS = {
        'frame_rate_16': 'frame_rate',
        'shutter_16': 'shutter_ns',
        'post_trigger_16': 'post_trigger',
        'frame_delay_16': 'frame_delay_ns',
        'edr_shutter_16': 'edr_shutter_ns',
        'saturation': 'f_saturation',
        'shutter': 'shutter_ns',
        'edr_shutter': 'edr_shutter_ns',
        'frame_delay': 'frame_delay_ns',
        'bright': 'f_offset',
        'contrast': 'f_gain',
        'gamma': 'f_gamma',
        'conv_8_max': 'f_gain16_8',
        'hue': 'f_hue',
        }

#from VR doc: to be ignored, not used anymore
TO_BE_IGNORED_FIELDS = {
        'contrast_16': 'res_7',
        'bright_16': 'res_8',
        'rotate_16': 'res_9',
        'time_annotation': 'res_10',
        'trig_cine': 'res_11',
        'shutter_on': 'res_12',
        'binning': 'res_13',
        'b_mains_freq': 'res_14', 
        'b_time_code': 'res_15',
        'b_priority': 'res_16',
        'w_leap_sec_dy': 'res_17',
        'd_delay_tc': 'res_18',
        'd_delay_pps': 'res_19',
        'gen_bits': 'res_20',
        'conv_8_min': '',
        }

# from VR doc: last setup field appearing in software version
# TODO: keep up-to-date with newer and more precise doc, if available
END_OF_SETUP = {
        551: 'software_version',
        552: 'recording_time_zone',
        578: 'rotate',
        605: 'b_stamp_time',
        606: 'mc_percent63',
        607: 'head_serial3',
        614: 'decimation',
        624: 'master_serial',
        625: 'sensor',
        631: 'frame_delay_ns',
        637: 'description',
        671: 'hue',
        691: 'lens_focal_length',
        693: 'f_gain16_8',
        701: 'f_tc_rate',
        702: 'cine_name',
        }


class Cine(FramesSequence):
    """Read cine files

    Read cine files, the out put from Vision Research high-speed phantom
    cameras.  Support uncompressed monochrome and color files.

    Nominally thread-safe, but this assertion is not tested.


    Parameters
    ----------
    filename : string
        Path to cine (or chd) file.
    
    Notes
    -----
    For a .chd file, this class only reads the header, not the images.
    
    """
    @classmethod
    def class_exts(cls):
        return {'cine'} | super(Cine, cls).class_exts()

    propagate_attrs = ['frame_shape', 'pixel_type', 'filename', 'frame_rate',
                       'get_fps', 'compression', 'cfa', 'off_set']

    def __init__(self, filename):
        super(Cine, self).__init__()
        self.f = open(filename, 'rb')
        self._filename = filename

        ### HEADER
        self.header_dict = self._read_header(HEADER_FIELDS)
        self.bitmapinfo_dict = self._read_header(BITMAP_INFO_FIELDS,
                                                self.off_image_header)
        self.setup_fields_dict = self._read_header(SETUP_FIELDS, self.off_setup)
        self.setup_fields_dict = self.clean_setup_dict()

        self._width = self.bitmapinfo_dict['bi_width']
        self._height = self.bitmapinfo_dict['bi_height']
        self._pixel_count = self._width * self._height

        # Allows Cine object to be accessed from multiple threads!
        self.file_lock = Lock()

        self._hash = None

        self._im_sz = (self._width, self._height)

        # sort out the data type by reading the meta-data
        if self.bitmapinfo_dict['bi_bit_count'] in (8, 24):
            self._data_type = 'u1'
        else:
            self._data_type = 'u2'
        self.tagged_blocks = self._read_tagged_blocks()
        self.frame_time_stamps = self.tagged_blocks['image_time_only']
        self.all_exposures = self.tagged_blocks['exposure_only']
        self.stack_meta_data = dict()
        self.stack_meta_data.update(self.bitmapinfo_dict)
        self.stack_meta_data.update({k: self.setup_fields_dict[k]
                                     for k in set(('trig_frame',
                                                   'gamma',
                                                   'frame_rate',
                                                   'shutter_ns'
                                                   )
                                                   )
                                                   })
        self.stack_meta_data.update({k: self.header_dict[k]
                                     for k in set(('first_image_no',
                                                   'image_count',
                                                   'total_image_count',
                                                   'first_movie_image'
                                                   )
                                                   )
                                                   })
        self.stack_meta_data['trigger_time'] = self.trigger_time

        ### IMAGES
        # Move to images offset to test EOF...
        self.f.seek(self.off_image_offsets)
        if self.f.read(1) != b'':
            # ... If no, read images
            self.image_locations = self._unpack('%dQ' % self.image_count,
                                               self.off_image_offsets)
            if type(self.image_locations) not in (list, tuple):
                self.image_locations = [self.image_locations]
        # TODO: add support for reading sequence within the same framework, when data
        # has been saved in another format (.tif, image sequence, etc)

    def clean_setup_dict(self):
        r"""Clean setup dictionary by removing newer fields, when compared to the
        software version, and trailing null character b'\x00' in entries.

        Notes
        -----
        The method is called after building the setup from the raw cine header.
        It can be overridden to match more specific purposes (e.g. filtering
        out TO_BE_IGNORED_ and UPDATED_FIELDS).

        See also
        --------
        `Vision Research Phantom documentation <http://phantomhighspeed-knowledge.force.com/servlet/fileField?id=0BE1N000000kD2i>`_
        """
        setup = self.setup_fields_dict.copy()
        # End setup at correct field (according to doc)
        versions = sorted(END_OF_SETUP.keys())
        fields = [v[0] for v in SETUP_FIELDS]
        v = setup['software_version']
        # Get next field where setup is known to have ended, according to VR
        try:
            v_up = versions[sorted(where(array(versions) >= v)[0])[0]]
            last_field = END_OF_SETUP[v_up]
            for k in fields[fields.index(last_field)+1:]:
                del setup[k]
        except IndexError:
            # Or go to the end (waiting for updated documentation)
            pass

        # Remove blank characters
        setup = _convert_null_byte(setup)

        # Filter out 'res_' (reserved/obsolete) fields
        #k_res = [k for k in setup.keys() if k.startswith('res_')]
        #for k in k_res:
        #    del setup[k]

        # Format f_tone properly
        if 'f_tone' in setup.keys():
            tone = setup['f_tone']
            setup['f_tone'] = tuple((tone[2*k], tone[2*k+1])\
                                    for k in range(setup['tone_points']))
        return setup


    @property
    def filename(self):
        return self._filename
    
    @property
    def frame_rate(self):
        """Frame rate (setting in Phantom PCC software) (Hz).
        May differ from computed average one.
        """
        return self.setup_fields_dict['frame_rate']

    @property
    def frame_rate_avg(self):
        """Actual frame rate, averaged on frame timestamps (Hz)."""
        return self.get_frame_rate_avg()

    # use properties for things that should not be changeable
    @property
    def cfa(self):
        return self.setup_fields_dict['cfa']

    @property
    def compression(self):
        return self.header_dict['compression']

    @property
    def pixel_type(self):
        return np.dtype(self._data_type)

    # TODO: what is this field??? (baneel)
    @property
    def off_set(self):
        return self.header_dict['offset']

    @property
    def setup_length(self):
        return self.setup_fields_dict['length']

    @property
    def off_image_offsets(self):
        return self.header_dict['off_image_offsets']

    @property
    def off_image_header(self):
        return self.header_dict['off_image_header']

    @property
    def off_setup(self):
        return self.header_dict['off_setup']

    @property
    def image_count(self):
        return self.header_dict['image_count']

    @property
    def frame_shape(self):
        return self._im_sz

    @property
    def shape(self):
        """Shape of virtual np.array containing images."""
        W, H = self.frame_shape
        return self.len(), H, W

    def get_frame(self, j):
        md = dict()
        md['exposure'] = self.all_exposures[j]
        ts, sec_frac = self.frame_time_stamps[j]
        md['frame_time'] = {'datetime': ts,
                            'second_fraction': sec_frac,
                            'time_to_trigger': self.get_time_to_trigger(j),
                            }
        return Frame(self._get_frame(j), frame_no=j, metadata=md)

    def _unpack(self, fs, offset=None):
        if offset is not None:
            self.f.seek(offset)
        s = _build_struct(fs)
        vals = s.unpack(self.f.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals

    def _read_tagged_blocks(self):
        """Reads the tagged block meta-data from the header."""
        tmp_dict = dict()
        if not self.off_setup + self.setup_length < self.off_image_offsets:
            return
        next_tag_exists = True
        next_tag_offset = 0
        while next_tag_exists:
            block_size, next_tag_exists = self._read_tag_block(next_tag_offset,
                                                               tmp_dict)
            next_tag_offset += block_size
        return tmp_dict

    def _read_tag_block(self, off_set, accum_dict):
        '''
        Internal helper-function for reading the tagged blocks.
        '''
        with FileLocker(self.file_lock):
            self.f.seek(self.off_setup + self.setup_length + off_set)
            block_size = self._unpack(UINT32)
            b_type = self._unpack(UINT16)
            more_tags = self._unpack(UINT16)

            if b_type == 1004:
                # docs say to ignore range data it seems to be a poison flag,
                # if see this, give up tag parsing
                return block_size, 0

            try:
                d_name, d_type = TAGGED_FIELDS[b_type]

            except KeyError:
                return block_size, more_tags

            if d_type == '':
                # print "can't deal with  <" + d_name + "> tagged data"
                return block_size, more_tags

            s_tmp = _build_struct(d_type)
            if (block_size-8) % s_tmp.size != 0:
                #            print 'something is wrong with your data types'
                return block_size, more_tags

            d_count = (block_size-8)//(s_tmp.size)

            data = self._unpack('%d' % d_count + d_type)
            if not isinstance(data, tuple):
                # fix up data due to design choice in self.unpack
                data = (data, )

            # parse time
            if b_type == 1002 or b_type == 1001:
                data = [(datetime.datetime.fromtimestamp(d >> 32),
                         (FRACTION_MASK & d)/MAX_INT) for d in data]
            # convert exposure to seconds
            if b_type == 1003:
                data = [d/(MAX_INT) for d in data]

            accum_dict[d_name] = data

        return block_size, more_tags

    def _read_header(self, fields, offset=0):
        self.f.seek(offset)
        tmp = dict()
        for name, format in fields:
            val = self._unpack(format)
            tmp[name] = val

        return tmp

    def _get_frame(self, number):
        with FileLocker(self.file_lock):
            # get basic information about the frame we want
            image_start = self.image_locations[number]
            annotation_size = self._unpack(UINT32, image_start)
            # this is not used, but is needed to advance the point in the file
            annotation = self._unpack('%db' % (annotation_size - 8))
            image_size = self._unpack(UINT32)

            cfa = self.cfa
            compression = self.compression

            # sort out data type looking at the cached version
            data_type = self._data_type

            # actual bit per pixel
            actual_bits = image_size * 8 // (self._pixel_count)

            # so this seem wrong as 10 or 12 bits won't fit in 'u1'
            # but I (TAC) may not understand and don't have a packed file
            # (which the docs seem to imply don't exist) to test on so
            # I am leaving it.  good luck.
            if actual_bits in (10, 12):
                data_type = 'u1'

            # move the file to the right point in the file
            self.f.seek(image_start + annotation_size)

            # suck the data out of the file and shove into linear
            # numpy array
            frame = frombuffer(self.f.read(image_size), data_type)

            # if mono-camera
            if cfa == CFA_NONE:
                if compression != 0:
                    raise ValueError("Can not deal with compressed files\n" +
                                     "compression level: " +
                                     "{}".format(compression))
                # we are working with a monochrome camera
                # un-pack packed data
                if (actual_bits == 10):
                    frame = _ten2sixteen(frame)
                elif (actual_bits == 12):
                    frame = _twelve2sixteen(frame)
                elif (actual_bits % 8):
                    raise ValueError('Data should be byte aligned, ' +
                         'or 10 or 12 bit packed (appears to be' +
                        ' %dbits/pixel?!)' % actual_bits)

                # re-shape to an array
                # flip the rows
                frame = frame.reshape(self._height, self._width)[::-1]

                if actual_bits in (10, 12):
                    frame = frame[::-1, :]
                    # Don't know why it works this way, but it does...
            # else, some sort of color layout
            else:
                if compression == 0:
                    # and re-order so color is RGB (naively saves as BGR)
                    frame = frame.reshape(self._height, self._width,
                                          3)[::-1, :, ::-1]
                elif compression == 2:
                    raise ValueError("Can not process un-interpolated movies")
                else:
                    raise ValueError("Should never hit this, " +
                                     "you have an un-documented file\n" +
                                     "compression level: " +
                                     "{}".format(compression))

        return frame

    def __len__(self):
        return self.image_count

    len = __len__

    @index_attr
    def get_time(self, i):
        """Return the time of frame i in seconds, relative to first frame."""
        warnings.warn("This is not guaranteed to be the actual time. "\
                      +"See self.get_time_to_trigger(i) method.",
                      category=PendingDeprecationWarning)
        return float(i) / self.frame_rate

    @index_attr
    def get_time_to_trigger(self, i):
        """Get actual time (s) of frame i, relative to trigger."""
        ti = self.frame_time_stamps[i]
        ti = ti[0].timestamp() + ti[1]
        tt= self.trigger_time
        tt = tt['datetime'].timestamp() + tt['second_fraction']
        return ti - tt

    def get_frame_rate_avg(self, error_tol=1e-3):
        """Compute mean frame rate (Hz), on the basis of frame time stamps.

        Parameters
        ----------
        error_tol : float, optional.
            Tolerance on relative error (standard deviation/mean),
            above which a warning is raised.

        Returns
        -------
        fps : float.
            Actual mean frame rate, based on the frames time stamps.
        """
        times = np.r_[[self.get_time_to_trigger(i) for i in range(self.len())]]
        freqs = 1 / np.diff(times)
        fps, std = freqs.mean(), freqs.std()
        error = std / fps
        if error > error_tol:
            warnings.warn('Relative precision on the average frame rate is '\
                          +'{:.2f}%.'.format(1e2*error))
        return fps

    def get_fps(self):
        """Get frame rate (setting in Phantom PCC software) (Hz).
        May differ from computed average one.

        See also
        --------
        PCC setting (all fields refer to the same value)
            self.frame_rate
            self.setup_fields_dict['frame_rate']
        
        Computed average
            self.frame_rate_avg
            self.get_frame_rate_avg()
        """
        return self.frame_rate

    def close(self):
        self.f.close()

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()

    def __unicode__(self):
        return self.filename

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {frame_shape!r}
Pixel Datatype: {dtype}""".format(frame_shape=self.frame_shape,
                                  count=len(self),
                                  filename=self.filename,
                                  dtype=self.pixel_type)

    @property
    def trigger_time(self):
        '''Returns the time of the trigger, tuple of (datatime_object,
        fraction_in_s)'''
        trigger_time = self.header_dict['trigger_time']
        ts, sf = (datetime.datetime.fromtimestamp(trigger_time >> 32),
                   float(FRACTION_MASK & trigger_time)/(MAX_INT))

        return {'datetime': ts, 'second_fraction': sf}

    @property
    def hash(self):
        if self._hash is None:
            self._hash_fun()
        return self._hash

    def __hash__(self):
        return int(self.hash, base=16)

    def _hash_fun(self):
        """Generates the md5 hash of the header of the file.  Here the
        header is defined as everything before the first image starts.

        This includes all of the meta-data (including the plethora of
        time stamps) so this will be unique.
        """
        # get the file lock (so we don't screw up any other reads)
        with FileLocker(self.file_lock):

            self.f.seek(0)
            max_loc = self.image_locations[0]
            md5 = hashlib.md5()

            chunk_size = 128*md5.block_size
            chunk_count = (max_loc//chunk_size) + 1

            for j in range(chunk_count):
                md5.update(self.f.read(128*md5.block_size))

            self._hash = md5.hexdigest()

    def __eq__(self, other):
        return self.hash == other.hash

    def __ne__(self, other):
        return not self == other


# Should be divisible by 3, 4 and 5!  This seems to be near-optimal.
CHUNK_SIZE = 6 * 10 ** 5


def _ten2sixteen(a):
    """Convert array of 10bit uints to array of 16bit uints."""
    b = np.zeros(a.size//5*4, dtype='u2')

    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2, a3, a4) = [a[j+i:j+CHUNK_SIZE:5].astype('u2')
                                for i in range(5)]

        k = j//5 * 4
        k2 = k + CHUNK_SIZE//5 * 4

        b[k+0:k2:4] = ((a0 & 0b11111111) << 2) + ((a1 & 0b11000000) >> 6)
        b[k+1:k2:4] = ((a1 & 0b00111111) << 4) + ((a2 & 0b11110000) >> 4)
        b[k+2:k2:4] = ((a2 & 0b00001111) << 6) + ((a3 & 0b11111100) >> 2)
        b[k+3:k2:4] = ((a3 & 0b00000011) << 8) + ((a4 & 0b11111111) >> 0)

    return b


def _sixteen2ten(b):
    """Convert array of 16bit uints to array of 10bit uints."""
    a = np.zeros(b.size//4*5, dtype='u1')

    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1, b2, b3) = [b[j+i:j+CHUNK_SIZE:4] for i in range(4)]

        k = j//4 * 5
        k2 = k + CHUNK_SIZE//4 * 5

        a[k+0:k2:5] =                              ((b0 & 0b1111111100) >> 2)
        a[k+1:k2:5] = ((b0 & 0b0000000011) << 6) + ((b1 & 0b1111110000) >> 4)
        a[k+2:k2:5] = ((b1 & 0b0000001111) << 4) + ((b2 & 0b1111000000) >> 6)
        a[k+3:k2:5] = ((b2 & 0b0000111111) << 2) + ((b3 & 0b1100000000) >> 8)
        a[k+4:k2:5] = ((b3 & 0b0011111111) << 0)

    return a


def _twelve2sixteen(a):
    """Convert array of 12bit uints to array of 16bit uints."""
    b = np.zeros(a.size//3*2, dtype='u2')

    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2) = [a[j+i:j+CHUNK_SIZE:3].astype('u2') for i in range(3)]

        k = j//3 * 2
        k2 = k + CHUNK_SIZE//3 * 2

        b[k+0:k2:2] = ((a0 & 0xFF) << 4) + ((a1 & 0xF0) >> 4)
        b[k+1:k2:2] = ((a1 & 0x0F) << 8) + ((a2 & 0xFF) >> 0)

    return b


def _sixteen2twelve(b):
    """Convert array of 16bit uints to array of 12bit uints."""
    a = np.zeros(b.size//2*3, dtype='u1')

    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1) = [b[j+i:j+CHUNK_SIZE:2] for i in range(2)]

        k = j//2 * 3
        k2 = k + CHUNK_SIZE//2 * 3

        a[k+0:k2:3] =                       ((b0 & 0xFF0) >> 4)
        a[k+1:k2:3] = ((b0 & 0x00F) << 4) + ((b1 & 0xF00) >> 8)
        a[k+2:k2:3] = ((b1 & 0x0FF) << 0)

    return a


def _convert_null_byte(dic):
    """
    Convert binary null character b'\x00' to empty string in dictionary entries.

    Parameters
    ----------
    dic : dict
        Dictionary to clean. Function loops over the key-value pairs and
        converts the null byte `b'\x00'` to empty string `''`.

    Returns
    -------
    clean_dic : dict
        Cleaned dictionary.

    Notes
    -----
    The routine is intended to work on string-like bytes array (resp. Iterable
    of such arrays), and return a string (resp. a list of strings).
    """
    for k, v in dic.items():
        if isinstance(v, bytes):
            try:
                dic[k] = v.decode('utf8').replace('\x00', '')
            except (UnicodeDecodeError):
                pass
        elif isinstance(v, Iterable):
            try:
                dic[k] = [el.decode('utf8').replace('\x00', '')\
                          for el in v]
            except (AttributeError, UnicodeDecodeError):
                pass
    return dic
