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


# Vendored from numpy @7649fe2e
# This function is
#
#  Copyright (c) 2005-2024, NumPy Developers.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#      * Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above
#         copyright notice, this list of conditions and the following
#         disclaimer in the documentation and/or other materials provided
#         with the distribution.
#
#      * Neither the name of the NumPy Developers nor the names of any
#         contributors may be used to endorse or promote products derived
#         from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def _as_pairs(x, ndim, as_index=False):
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.

    Parameters
    ----------
    x : {None, scalar, array-like}
        The object to broadcast to the shape (`ndim`, 2).
    ndim : int
        Number of pairs the broadcasted `x` will have.
    as_index : bool, optional
        If `x` is not None, try to round each element of `x` to an integer
        (dtype `np.intp`) and ensure every element is positive.

    Returns
    -------
    pairs : nested iterables, shape (`ndim`, 2)
        The broadcasted version of `x`.

    Raises
    ------
    ValueError
        If `as_index` is True and `x` contains negative elements.
        Or if `x` is not broadcastable to the shape (`ndim`, 2).
    """
    if x is None:
        # Pass through None as a special case, otherwise np.round(x) fails
        # with an AttributeError
        return ((None, None),) * ndim

    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)

    if x.ndim < 3:
        # Optimization: Possibly use faster paths for cases where `x` has
        # only 1 or 2 elements. `np.broadcast_to` could handle these as well
        # but is currently slower

        if x.size == 1:
            # x was supplied as a single value
            x = x.ravel()  # Ensure x[0] works for x.ndim == 0, 1, 2
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim

        if x.size == 2 and x.shape != (2, 1):
            # x was supplied with a single value for each side
            # but except case when each dimension has a single value
            # which should be broadcasted to a pair,
            # e.g. [[1], [2]] -> [[1, 1], [2, 2]] not [[1, 2], [1, 2]]
            x = x.ravel()  # Ensure x[0], x[1] works
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim

    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")

    # Converting the array with `tolist` seems to improve performance
    # when iterating and indexing the result (see usage in `pad`)
    return np.broadcast_to(x, (ndim, 2)).tolist()


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
