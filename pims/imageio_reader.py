from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from pims.base_frames import FramesSequence
from pims.frame import Frame

try:
    import imageio
except ImportError:
    imageio = None


def available():
    return imageio is not None


class ImageIOReader(FramesSequence):
    class_priority = 6
    @classmethod
    def class_exts(cls):
        return {'tiff', 'bmp', 'cut', 'dds', 'exr', 'g3', 'hdr', 'iff', 'j2k',
                'jng', 'jp2', 'jpeg', 'jpg', 'koala', 'pbm', 'pbmraw', 'pcd',
                'pcx', 'pfm', 'pgm', 'pgmraw', 'pict', 'png', 'ppm', 'ppmraw',
                'psd', 'ras', 'raw', 'sgi', 'targa', 'fi_tiff', 'wbmp', 'webp',
                'xbm', 'xpm', 'ico', 'gif', 'dicom', 'npz', 'fits', 'itk',
                'gdal', 'dummy', 'gif', 'ffmpeg', 'avbin', 'swf', 'fits',
                'gdal', 'mov', 'mp4', 'avi', 'mpeg', 'wmv', 'mkv', 'ts', 'tif'}

    def __init__(self, filename, **kwargs):
        if imageio is None:
            raise ImportError('The ImageIOReader requires imageio to work.')
        self.reader = imageio.get_reader(filename, **kwargs)
        self.filename = filename
        self._len = self.reader.get_length()

        first_frame = self.get_frame(0)
        self._shape = first_frame.shape
        self._dtype = first_frame.dtype

    def get_frame(self, i):
        frame = self.reader.get_data(i)
        return Frame(frame, frame_no=i, metadata=frame.meta)

    def get_metadata(self):
        return self.reader.get_meta_data(None)

    def __len__(self):
        return self._len

    def __iter__(self):
        iterable = self.reader.iter_data()
        for i in range(len(self)):
            frame = next(iterable)
            yield Frame(frame, frame_no=i, metadata=frame.meta)

    @property
    def frame_rate(self):
        return self.get_metadata()['fps']

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype

    def close(self):
        self.reader.close()
