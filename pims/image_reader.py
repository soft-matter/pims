import numpy as np

from pims.base_frames import FramesSequence, FramesSequenceND
from pims.frame import Frame

try:
    from skimage.io import imread
except ImportError:
    try:
        from imageio import v2 as iio
    except ImportError:
        import imageio as iio

    def imread(*args, **kwargs):  # Strip metadata for consistency.
        return np.asarray(iio.imread(*args, **kwargs))


class ImageReader(FramesSequence):
    """Reads a single image into a length-1 reader.

    Simple wrapper around skimage.io.imread or matplotlib.pyplot.imread,
    in that priority order."""
    @classmethod
    def class_exts(cls):
        return {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'ico'}

    class_priority = 12

    def __init__(self, filename, **kwargs):
        self._data = imread(filename, **kwargs)

    def get_frame(self, i):
        return Frame(self._data, frame_no=0)

    def __len__(self):
        return 1

    @property
    def pixel_type(self):
        return self._data.dtype

    @property
    def frame_shape(self):
        return self._data.shape


class ImageReaderND(FramesSequenceND):
    """Reads a single image into a dimension-aware reader.

    Simple wrapper around skimage.io.imread or matplotlib.pyplot.imread,
    in that priority order."""
    @classmethod
    def class_exts(cls):
        return {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'ico'}

    class_priority = 11

    def __init__(self, filename, **kwargs):
        super(ImageReaderND, self).__init__()

        self._data = Frame(imread(filename, **kwargs), frame_no=0)
        shape = self._data.shape
        if len(shape) == 2:   # greyscale
            self._init_axis('y', shape[0])
            self._init_axis('x', shape[1])
            self._register_get_frame(self.get_frame_2D, 'yx')
            self.bundle_axes = 'yx'
        elif (len(shape) == 3) and (shape[2] in (3, 4)):   # grayscale
            self._init_axis('y', shape[0])
            self._init_axis('x', shape[1])
            self._init_axis('c', shape[2])
            self._register_get_frame(self.get_frame_2D, 'yxc')
            self.bundle_axes = 'yxc'
        else:
            raise IOError('The image has a shape that is not grayscale nor RGB:'
                          ' {}'.format(shape))

    def get_frame_2D(self, **ind):
        return Frame(self._data, frame_no=0)

    @property
    def pixel_type(self):
        return self._data.dtype

    @property
    def frame_shape(self):
        return self._data.shape
