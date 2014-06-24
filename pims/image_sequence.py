from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import map
import os
import glob
from warnings import warn
from scipy.ndimage import imread as scipy_imread
from matplotlib.pyplot import imread as mpl_imread
from pims.base_frames import FramesSequence
from pims.frame import Frame


class ImageSequence(FramesSequence):
    """Iterable object that returns frames of video as numpy arrays.

    Parameters
    ----------
    pathname : string
       a directory or, safer, a pattern like path/to/images/*.png
       which will ignore extraneous files
    gray : Convert color image to grayscale. True by default.
    invert : Invert black and white. True by default.

    Examples
    --------
    >>> video = ImageSequence('path/to/images/*.png')  # or *.tif, or *.jpg
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[1][0:10][0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """

    def __init__(self, pathname, process_func=None, dtype=None):
        self.pathname = os.path.abspath(pathname)  # used by __repr__
        if os.path.isdir(pathname):
            warn("Loading ALL files in this directory. To ignore extraneous "
                 "files, use a pattern like 'path/to/images/*.png'",
                 UserWarning)
            directory = pathname
            filenames = os.listdir(directory)
            make_full_path = lambda filename: (
                os.path.abspath(os.path.join(directory, filename)))
            filepaths = list(map(make_full_path, filenames))
        else:
            filepaths = glob.glob(pathname)
        filepaths.sort()  # listdir returns arbitrary order
        self._filepaths = filepaths
        self._count = len(self._filepaths)

        if process_func is None:
            process_func = lambda x: x
        if not callable(process_func):
            raise ValueError("process_func must be a function, or None")
        self.process_func = process_func

        tmp = scipy_imread(self._filepaths[0])

        # hacky solution to PIL problem
        if tmp.ndim == 0:  # obviously bad
            tmp = mpl_imread(self._filepaths[0])
            self.imread = mpl_imread
        else:
            self.imread = scipy_imread

        self._first_frame_shape = tmp.shape

        if dtype is None:
            self._dtype = tmp.dtype
        else:
            self._dtype = dtype

    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        res = self.imread(self._filepaths[j])
        if res.dtype != self._dtype:
            res = res.astype(self._dtype)
        res = Frame(self.process_func(res), frame_no=j)
        return res

    def __len__(self):
        return self._count

    @property
    def frame_shape(self):
        return self._first_frame_shape

    @property
    def pixel_type(self):
        return self._dtype

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {pathname}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  pathname=self.pathname,
                                  dtype=self.pixel_type)
