import numpy as np
from slicerator import pipeline, Pipeline


@pipeline
def as_grey(frame):
    """Convert a 2D image or PIMS reader to greyscale.

    This weights the color channels according to their typical
    response to white light.

    It does nothing if the input is already greyscale.
    """
    if len(frame.shape) == 2:
        return frame
    else:
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        return 0.2125 * red + 0.7154 * green + 0.0721 * blue

# "Gray" is the more common spelling
as_gray = as_grey

# Source of this patch: https://github.com/scikit-image/scikit-image/pull/3556
# See also: https://github.com/numpy/numpy/pull/11966

from numpy.lib.arraypad import _as_pairs

def validate_lengths(ar, crop_width):
    return _as_pairs(crop_width, ar.ndim, as_index=True)

def _crop(frame, bbox):
    return frame[bbox[0]:bbox[2], bbox[1]:bbox[3]]


@pipeline
class crop(Pipeline):
    """Crop image or image-reader`reader` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
       Input array.
    crop_width : {sequence, int}
       Number of values to remove from the edges of each axis.
       ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
       unique crop widths at the start and end of each axis.
       ``((before, after),)`` specifies a fixed start and end crop
       for every axis.
       ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
       before = after = ``n`` for all axes.
    order : {'C', 'F', 'A', 'K'}, optional
       control the memory layout of the copy. See ``np.copy``.
    Returns
    -------
    cropped : array
       The cropped array.

    See Also
    --------
    Source: ``skimage.util.crop`` (v0.12.3)
    """
    def __init__(self, reader, crop_width, order='K'):
        # We have to know the frame shape that is returned by the reader.
        try:  # In case the reader is a FramesSequence, there is an attribute
            shape = reader.frame_shape
            first_frame = np.empty(shape, dtype=bool)
        except AttributeError:
            first_frame = reader[0]
            shape = first_frame.shape
        # Validate the crop widths on the first frame
        crops = validate_lengths(first_frame, crop_width)
        self._crop_slices = tuple([slice(a, shape[i] - b)
                             for i, (a, b) in enumerate(crops)])
        self._crop_shape = tuple([shape[i] - b - a
                                  for i, (a, b) in enumerate(crops)])
        self._crop_order = order
        # We could pass _crop to proc_func. However this adds an extra copy
        # operation. Therefore we define our own here.
        super(self.__class__, self).__init__(None, reader)

    def _get(self, key):
        ar = self._ancestors[0][key]
        return np.array(ar[self._crop_slices], order=self._crop_order,
                        copy=True)

    @property
    def frame_shape(self):
        return self._crop_shape
