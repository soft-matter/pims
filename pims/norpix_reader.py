"""Reader for Norpix .seq files

Author: Nathan C. Keim
Based heavily on cine.py by Kleckner and Caswell
"""

from pims.frame import Frame
from pims.base_frames import FramesSequence, index_attr
from pims.utils.misc import FileLocker
import os, struct, itertools
from warnings import warn
import datetime
import numpy as np
from threading import Lock

__all__ = ['NorpixSeq',]

DWORD = 'L'
LONG = 'l'
DOUBLE = 'd'
USHORT = 'H'

HEADER_FIELDS = [
    ('magic', DWORD),
    ('name', '24s'),
    ('version', LONG),
    ('header_size', LONG),
    ('description', '512s'),
    ('width', DWORD),
    ('height', DWORD),
    ('bit_depth', DWORD),
    ('bit_depth_real', DWORD),
    ('image_size_bytes', DWORD),
    ('image_format', DWORD),
    ('allocated_frames', DWORD),
    ('origin', DWORD),
    ('true_image_size', DWORD),
    ('suggested_frame_rate', DOUBLE),
    ('description_format', LONG),
    ('reference_frame', DWORD),
    ('fixed_size', DWORD),
    ('flags', DWORD),
    ('bayer_pattern', LONG),
    ('time_offset_us', LONG),
    ('extended_header_size', LONG),
    ('compression_format', DWORD),
    ('reference_time_s', LONG),
    ('reference_time_ms', USHORT),
    ('reference_time_us', USHORT)
    # More header values not implemented
]


class NorpixSeq(FramesSequence):
    """Read Norpix sequence (.seq) files

    This is the native format of StreamPix software, owned by NorPix Inc.
    The format is described in the StreamPix documentation.

    Currently only supports uncompressed files (with the same number
    of bytes for each frame).

    Only unsigned 8, 16 or 32 bit monochrome files are directly supported.
    Other color and monochrome pixel formats can be handled by setting
    the as_raw flag which results in frames with unsigned 8-bit data type.
    and the correct number of rows.  (If a raw frame size in bytes is not
    evenly divisible by the number of rows, a 1-D array is returned).

    Nominally thread-safe.

    Parameters
    ----------
    filename : string
        Path to the .seq file
    as_raw : boolean, optional
        Required for non-monochrome frames.
        Images will be returned as an ndarray of bytes.
        2-dimensional if the image height evenly divides the bytes per image,
        1-dimensional otherwise.
    """
    @classmethod
    def class_exts(cls):
        return {'seq'} | super(NorpixSeq, cls).class_exts()

    propagate_attrs = ['frame_shape', 'pixel_type', 'get_time',
                       'get_time_float', 'filename', 'width', 'height',
                       'frame_rate']

    def __init__(self, filename, as_raw=False):
        super(NorpixSeq, self).__init__()
        self._file = open(filename, 'rb')
        self._filename = filename

        self.header_dict = self._read_header(HEADER_FIELDS)

        if self.header_dict['magic'] != 0xFEED:
            raise IOError('The format of this .seq file is unrecognized')
        if self.header_dict['compression_format'] != 0:
            raise IOError('Only uncompressed images are supported in .seq files')
        if self.header_dict['image_format'] != 100 and not as_raw:
            raise IOError('Non-monochrome images are only supported as_raw in .seq files')

        # File-level metadata
        if self.header_dict['version'] >= 5:  # StreamPix version 6
            self._image_offset = 8192
            # Timestamp = 4-byte unsigned long + 2-byte unsigned short (ms)
            #   + 2-byte unsigned short (us)
            self._timestamp_struct = struct.Struct('<LHH')
            self._timestamp_micro = True
        else:  # Older versions
            self._image_offset = 1024
            self._timestamp_struct = struct.Struct('<LH')
            self._timestamp_micro = False
        self._image_block_size = self.header_dict['true_image_size']
        self._filesize = os.stat(self._filename).st_size
        self._image_count = int((self._filesize - self._image_offset) /
                                self._image_block_size)

        # Image metadata
        self._width = self.header_dict['width']
        self._height = self.header_dict['height']
        self._image_bytes = self.header_dict['image_size_bytes']
        if as_raw:
            self._pixel_count = self._image_bytes
            if self._pixel_count % self._height == 0:
                self._shape = (self._height, int(self._pixel_count / self._height))
            else:
                self._shape = (self._image_bytes,)
            self._dtype = 'uint8'
        else:
            try:
                self._pixel_count = self._width * self._height
                dtype_native = 'uint%i' % self.header_dict['bit_depth']
                self._dtype = np.dtype(dtype_native)
                self._shape = (self._height, self._width)
            except TypeError as e:
                raise IOError(self._dtype + " pixels not supported; use as_raw and convert")

        # Public metadata
        self.metadata = {k: self.header_dict[k] for k in
                         ('description', 'bit_depth_real', 'origin',
                          'suggested_frame_rate', 'width', 'height')}
        self.metadata['gamut'] = 2**self.metadata['bit_depth_real'] - 1

        self._file_lock = Lock()

    def _read_header(self, fields, offset=0):
        self._file.seek(offset)
        tmp = dict()
        for name, format in fields:
            val = self._unpack(format)
            tmp[name] = val

        return tmp

    def _unpack(self, fs, offset=None):
        if offset is not None:
            self._file.seek(offset)
        s = struct.Struct('<' + fs)
        vals = s.unpack(self._file.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals

    def _verify_frame_no(self, i):
        if int(i) != i:
            raise ValueError("Frame numbers can only be integers")
        if i >= self._image_count or i < 0:
            raise ValueError("Frame number is out of range: " + str(i))

    def set_process_func(self, process_func):
        # Expose the _validate_process_func for use after the header is
        # available
        self._validate_process_func(process_func)

    def get_frame(self, i):
        self._verify_frame_no(i)
        with FileLocker(self._file_lock):
            self._file.seek(self._image_offset + self._image_block_size * i)
            imdata = np.fromfile(self._file, self.pixel_type, self._pixel_count
                                 ).reshape(self._shape)
            # Timestamp immediately follows
            tfloat, ts = self._read_timestamp()
            md = {'time': ts, 'time_float': tfloat,
                  'gamut': self.metadata['gamut']}
            return Frame(imdata, frame_no=i, metadata=md)

    def _read_timestamp(self):
        """Read a timestamp at the current position in the file.

        Returns a floating-point representation in seconds, and a datetime instance.
        """
        if self._timestamp_micro:
            tsecs, tms, tus = self._timestamp_struct.unpack(self._file.read(8))
            tfloat = tsecs + float(tms) / 1000. + float(tus) / 1.0e6
        else:
            tsecs, tms = self._timestamp_struct.unpack(self._file.read(6))
            tfloat = tsecs + float(tms) / 1000.
        return tfloat, datetime.datetime.fromtimestamp(tfloat)

    def _get_time(self, i):
        """Call _read_timestamp() for a given frame."""
        self._verify_frame_no(i)
        with FileLocker(self._file_lock):
            self._file.seek(self._image_offset + self._image_block_size * i
                            + self._image_bytes)
            return self._read_timestamp()

    @index_attr
    def get_time(self, i):
        """Return the time of frame i as a datetime instance.

        Calling this function in a different timezone than where the movie
        was recorded will result in an offset. The .seq format does not
        store UTC or timezone information.
        """
        return self._get_time(i)[1]

    @index_attr
    def get_time_float(self, i):
        """Return the time of frame i as a floating-point number of seconds."""
        return self._get_time(i)[0]

    def dump_times_float(self):
        """Return all frame times in file, as an array of floating-point numbers."""
        return np.array([self._get_time(i)[0] for i in range(len(self))])

    @property
    def filename(self):
        return self._filename

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def width(self):
        return self.metadata['width']

    @property
    def height(self):
        return self.metadata['height']

    @property
    def frame_shape(self):
        return (self.metadata['height'], self.metadata['width'])

    @property
    def frame_rate(self):
        return self.metadata['suggested_frame_rate']

    def __len__(self):
        return self._image_count

    def close(self):
        self._file.close()

    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()

    def __repr__(self):
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w}w x {h}h
Pixel Datatype: {dtype}""".format(filename=self.filename,
                                  count=len(self),
                                  h=self.frame_shape[0],
                                  w=self.frame_shape[1],
                                  dtype=self.pixel_type)
