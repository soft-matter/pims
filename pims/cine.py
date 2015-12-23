###############################################################################

# Reader for CINE files produced by Vision Research Phantom Software
# Author: Dustin Kleckner
# dkleckner@uchicago.edu

# Modified by Thomas A Caswell (tcaswell@uchicago.edu)
# Added to PIMS by Thomas A Caswell (tcaswell@gmail.com)
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from pims.frame import Frame
from pims.base_frames import FramesSequence, index_attr
from pims.utils.misc import FileLocker
import time
import struct
import numpy as np
from numpy import array, frombuffer
from threading import Lock
import datetime
import hashlib

__all__ = ('Cine', )


def _build_struct(dtype):
    return struct.Struct(str("<" + dtype))


FRACTION_MASK = (2**32-1)
MAX_INT = 2**32

BYTE = 'B'
WORD = 'H'
INT16 = 'h'
SHORT = 'h'
BOOL = 'i'
DWORD = 'I'
UINT = 'I'
LONG = 'l'
INT = 'l'
FLOAT = 'f'
DOUBLE = 'd'
TIME64 = 'Q'
RECT = '4i'
WBGAIN = '2f'
IMFILTER = '28i'

CFA_NONE = 0
CFA_VRI = 1
CFA_VRIV6 = 2
CFA_BAYER = 3
CFA_BAYERFLIP = 4

TAGGED_FIELDS = {
    1000: ('ang_dig_sigs', ''),
    1001: ('image_time_total', TIME64),
    1002: ('image_time_only', TIME64),
    1003: ('exposure_only', DWORD),
    1004: ('range_data', ''),
    1005: ('binsig', ''),
    1006: ('anasig', ''),
    # 1007 exists in my files, but is not in documentation I can find
    1007: ('undocumented', '')}

HEADER_FIELDS = [
    ('type', '2s'),
    ('header_size', WORD),
    ('compression', WORD),
    ('version', WORD),
    ('first_movie_image', LONG),
    ('total_image_count', DWORD),
    ('first_image_no', LONG),
    ('image_count', DWORD),
    ('off_image_header', DWORD),
    ('off_setup', DWORD),
    ('off_image_offsets', DWORD),
    ('trigger_time', TIME64),
]

BITMAP_INFO_FIELDS = [
    ('bi_size', DWORD),
    ('bi_width', LONG),
    ('bi_height', LONG),
    ('bi_planes', WORD),
    ('bi_bit_count', WORD),
    ('bi_compression', DWORD),
    ('bi_image_size', DWORD),
    ('bi_x_pels_per_meter', LONG),
    ('bi_y_pels_per_meter', LONG),
    ('bi_clr_used', DWORD),
    ('bi_clr_important', DWORD),
]

SETUP_FIELDS = [
    ('frame_rate_16', WORD),
    ('shutter_16', WORD),
    ('post_trigger_16', WORD),
    ('frame_delay_16', WORD),
    ('aspect_ratio', WORD),
    ('contrast_16', WORD),
    ('bright_16', WORD),
    ('rotate_16', BYTE),
    ('time_annotation', BYTE),
    ('trig_cine', BYTE),
    ('trig_frame', BYTE),
    ('shutter_on', BYTE),
    # Guessed at length... because it isn't documented!  This seems to work.
    ('description_old', '121s'),
    ('mark', '2s'),
    ('length', WORD),
    ('binning', WORD),
    ('sig_option', WORD),
    ('bin_channels', SHORT),
    ('samples_per_image', BYTE)] + \
    [('bin_name%d' % i, '11s') for i in range(8)] + [
        ('ana_option', WORD),
        ('ana_channels', SHORT),
        ('res_6', BYTE),
        ('ana_board', BYTE)] + \
    [('ch_option%d' % i, SHORT) for i in range(8)] + \
    [('ana_gain%d' % i, FLOAT) for i in range(8)] + \
    [('ana_unit%d' % i, '6s') for i in range(8)] + \
    [('ana_name%d' % i, '11s') for i in range(8)] + [
    ('i_first_image', LONG),
    ('dw_image_count', DWORD),
    ('n_q_factor', SHORT),
    ('w_cine_file_type', WORD)] + \
    [('sz_cine_path%d' % i, '65s') for i in range(4)] + [
    ('b_mains_freq', WORD),
    ('b_time_code', BYTE),
    ('b_priority', BYTE),
    ('w_leap_sec_dy', DOUBLE),
    ('d_delay_tc', DOUBLE),
    ('d_delay_pps', DOUBLE),
    ('gen_bits', WORD),
    ('res_1', INT16),  # Manual says INT, but this is clearly wrong!
    ('res_2', INT16),
    ('res_3', INT16),
    ('im_width', WORD),
    ('im_height', WORD),
    ('edr_shutter_16', WORD),
    ('serial', UINT),
    ('saturation', INT),
    ('res_5', BYTE),
    ('auto_exposure', UINT),
    ('b_flip_h', BOOL),
    ('b_flip_v', BOOL),
    ('grid', UINT),
    ('frame_rate', UINT),
    ('shutter', UINT),
    ('edr_shutter', UINT),
    ('post_trigger', UINT),
    ('frame_delay', UINT),
    ('b_enable_color', BOOL),
    ('camera_version', UINT),
    ('firmware_version', UINT),
    ('software_version', UINT),
    ('recording_time_zone', INT),
    ('cfa', UINT),
    ('bright', INT),
    ('contrast', INT),
    ('gamma', INT),
    ('reserved1', UINT),
    ('auto_exp_level', UINT),
    ('auto_exp_speed', UINT),
    ('auto_exp_rect', RECT),
    ('wb_gain', '8f'),
    ('rotate', INT),
    ('wb_view', WBGAIN),
    ('real_bpp', UINT),
    ('conv_8_min', UINT),
    ('conv_8_max', UINT),
    ('filter_code', INT),
    ('filter_param', INT),
    ('uf', IMFILTER),
    ('black_cal_sver', UINT),
    ('white_cal_sver', UINT),
    ('gray_cal_sver', UINT),
    ('b_stamp_time', BOOL),
    ('sound_dest', UINT),
    ('frp_steps', UINT),
    ] + [('frp_img_nr%d' % i, INT) for i in range(16)] + \
        [('frp_rate%d' % i, UINT) for i in range(16)] + \
        [('frp_exp%d' % i, UINT) for i in range(16)] + [
    ('mc_cnt', INT),
    ] + [('mc_percent%d' % i, FLOAT) for i in range(64)] + [
    ('ci_calib', UINT),
    ('calib_width', UINT),
    ('calib_height', UINT),
    ('calib_rate', UINT),
    ('calib_exp', UINT),
    ('calib_edr', UINT),
    ('calib_temp', UINT),
    ] + [('header_serial%d' % i, UINT) for i in range(4)] + [
    ('range_code', UINT),
    ('range_size', UINT),
    ('decimation', UINT),
    ('master_serial', UINT),
    ('sensor', UINT),
    ('shutter_ns', UINT),
    ('edr_shutter_ns', UINT),
    ('frame_delay_ns', UINT),
    ('im_pos_xacq', UINT),
    ('im_pos_yacq', UINT),
    ('im_width_acq', UINT),
    ('im_height_acq', UINT),
    ('description', '4096s')
]


class Cine(FramesSequence):
    """Read cine files

    Read cine files, the out put from Vision Research high-speed phantom
    cameras.  Support uncompressed monochrome and color files.

    Nominally thread-safe, but this assertion is not tested.


    Parameters
    ----------
    filename : string
        Path to cine file.
    process_func : function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    dtype : numpy datatype, optional
        Image arrays will be converted to this datatype.
    as_grey : boolean, optional
        Convert color images to greyscale. False by default.
        May not be used in conjunction with process_func.
    """
    # TODO: Unit tests using a small sample cine file.
    @classmethod
    def class_exts(cls):
        return {'cine'} | super(Cine, cls).class_exts()

    propagate_attrs = ['frame_shape', 'pixel_type', 'filename', 'frame_rate',
                       'get_fps', 'compression', 'cfa', 'off_set']

    def __init__(self, filename, process_func=None,
                 dtype=None, as_grey=False):
        super(Cine, self).__init__()
        self.f = open(filename, 'rb')
        self._filename = filename

        self.header_dict = self.read_header(HEADER_FIELDS)
        self.bitmapinfo_dict = self.read_header(BITMAP_INFO_FIELDS,
                                                self.off_image_header)
        self.setup_fields_dict = self.read_header(SETUP_FIELDS, self.off_setup)
        self.image_locations = self.unpack('%dQ' % self.image_count,
                                           self.off_image_offsets)
        if type(self.image_locations) not in (list, tuple):
            self.image_locations = [self.image_locations]

        self._width = self.bitmapinfo_dict['bi_width']
        self._height = self.bitmapinfo_dict['bi_height']
        self._pixel_count = self._width * self._height

        # Allows Cine object to be accessed from multiple threads!
        self.file_lock = Lock()

        self._hash = None

        # validate gray/process func
        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

        self._im_sz = (self._width, self._height)

        # sort out the data type by reading the meta-data
        if self.bitmapinfo_dict['bi_bit_count'] in (8, 24):
            self._data_type = 'u1'
        else:
            self._data_type = 'u2'

        # sort out what type to return data as
        if dtype is None:
            self._dtype = np.dtype(self._data_type)
        else:
            self._dtype = dtype
        self.tagged_blocks = self.read_tagged_blocks()
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

    @property
    def filename(self):
        return self._filename

    @property
    def frame_rate(self):
        return self.setup_fields_dict['frame_rate']

    # use properties for things that should not be changeable
    @property
    def cfa(self):
        return self.setup_fields_dict['cfa']

    @property
    def compression(self):
        return self.header_dict['compression']

    @property
    def pixel_type(self):
        return self._dtype

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

    def get_frame(self, j):
        md = dict()
        md['exposure'] = self.all_exposures[j]
        ts, sec_frac = self.frame_time_stamps[j]
        md['frame_time'] = {'datetime': ts,
                            'second_fraction': sec_frac}
        return Frame(self.process_func(self._get_frame(j)),
                     frame_no=j, metadata=md)

    def unpack(self, fs, offset=None):
        if offset is not None:
            self.f.seek(offset)
        s = _build_struct(fs)
        vals = s.unpack(self.f.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals

    def read_tagged_blocks(self):
        '''
        Reads the tagged block meta-data from the header
        '''
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
            block_size = self.unpack(DWORD)
            b_type = self.unpack(WORD)
            more_tags = self.unpack(WORD)

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

            data = self.unpack('%d' % d_count + d_type)
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

    def read_header(self, fields, offset=0):
        self.f.seek(offset)
        tmp = dict()
        for name, format in fields:
            val = self.unpack(format)
            tmp[name] = val

        return tmp

    def _get_frame(self, number):
        with FileLocker(self.file_lock):
            # get basic information about the frame we want
            image_start = self.image_locations[number]
            annotation_size = self.unpack(DWORD, image_start)
            # this is not used, but is needed to advance the point in the file
            annotation = self.unpack('%db' % (annotation_size - 8))
            image_size = self.unpack(DWORD)

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
                # and the cast to proper type
                frame = frame.reshape(self._height,
                                      self._width)[::-1].astype(self._dtype)

                if actual_bits in (10, 12):
                    frame = frame[::-1, :]
                    # Don't know why it works this way, but it does...
            # else, some sort of color layout
            else:
                if compression == 0:
                    # and re-order so color is RGB (naively saves as BGR)
                    frame = frame.reshape(self._height,
                                          self._width,
                                          3)[::-1, :, ::-1].astype(self._dtype)
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
        '''Returm the time of frame i in seconds.'''
        # TODO: This is not guaranteed to be the actual time.
        # Frames may be unevenly spaced due to e.g. external sync.
        # The actual time is available from the timestamp tagged block,
        # which is read above.
        # See NorpixSeq for a timestamp API that solves this problem.
        return float(i) / self.frame_rate

    def get_fps(self):
        return self.frame_rate

    def close(self):
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
        fraction_in_ns)'''
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
        """
        generates the md5 hash of the header of the file.  Here the
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
    """
    Convert array of 10bit uints to array of 16bit uints
    """
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
    """
    Convert array of 16bit uints to array of 10bit uints
    """
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
    """
    Convert array of 12bit uints to array of 16bit uints
    """
    b = np.zeros(a.size//3*2, dtype='u2')

    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2) = [a[j+i:j+CHUNK_SIZE:3].astype('u2') for i in range(3)]

        k = j//3 * 2
        k2 = k + CHUNK_SIZE//3 * 2

        b[k+0:k2:2] = ((a0 & 0xFF) << 4) + ((a1 & 0xF0) >> 4)
        b[k+1:k2:2] = ((a1 & 0x0F) << 8) + ((a2 & 0xFF) >> 0)

    return b


def _sixteen2twelve(b):
    """
    Convert array of 16bit uints to array of 12bit uints
    """
    a = np.zeros(b.size//2*3, dtype='u1')

    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1) = [b[j+i:j+CHUNK_SIZE:2] for i in range(2)]

        k = j//2 * 3
        k2 = k + CHUNK_SIZE//2 * 3

        a[k+0:k2:3] =                       ((b0 & 0xFF0) >> 4)
        a[k+1:k2:3] = ((b0 & 0x00F) << 4) + ((b1 & 0xF00) >> 8)
        a[k+2:k2:3] = ((b1 & 0x0FF) << 0)

    return a
