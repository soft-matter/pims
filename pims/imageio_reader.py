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
    def __init__(self, filename):
        if imageio is None:
            raise ImportError('The ImageIOReader requires imageio to work.')
        self.reader = imageio.get_reader(filename)
        self.filename = filename
        self._len = len(self.reader)

        first_frame = self.get_frame(0)
        self._shape = first_frame.shape
        self._dtype = first_frame.dtype

    def get_frame(self, i):
        return Frame(self.reader.get_data(i), frame_no=i,
                     metadata=self.reader.get_meta_data(i))

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype  

    def close(self):
        self.reader.close()
