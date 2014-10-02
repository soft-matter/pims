from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import map
import os
import glob
from warnings import warn
from pims.base_frames import FramesSequence
from pims.frame import Frame

# skimage.io.plugin_order() gives a nice hierarchy of implementations of imread.
# If skimage is not available, go down our own hard-coded hierarchy.
try:
    from skimage.io import imread
except ImportError:
    try:
        from matplotlib.pyplot import imread
    except ImportError:
        from scipy.ndimage import imread


class ImageSequence(FramesSequence):
    """Read a directory of sequentially numbered image files into an
    iterable that returns images as numpy arrays.

    Parameters
    ----------
    path_spec : string or iterable of strings
       a directory or, safer, a pattern like path/to/images/*.png
       which will ignore extraneous files or a list of files to open
       in the order they should be loaded.
    process_func : function, optional
        callable with signalture `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    dtype : numpy datatype, optional
        Image arrays will be converted to this datatype.
    as_grey : boolean, optional
        Convert color images to greyscale. False by default.
        May not be used in conjection with process_func.
    plugin : string
        Passed on to skimage.io.imread if scikit-image is available.
        If scikit-image is not available, this will be ignored and a warning
        will be issued.

    Examples
    --------
    >>> video = ImageSequence('path/to/images/*.png')  # or *.tif, or *.jpg
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """
    def __init__(self, path_spec, process_func=None, dtype=None,
                 as_grey=False, plugin=None):
        try:
            import skimage
        except ImportError:
            if plugin is not None:
                warn("A plugin was specified but ignored. Plugins can only "
                     "be specified if scikit-image is available. Instead, "
                     "ImageSequence will try using matplotlib and scipy "
                     "in that order.")
            self.kwargs = dict()
        else:
            self.kwargs = dict(plugin=plugin)

        self._get_files(path_spec)

        tmp = imread(self._filepaths[0], **self.kwargs)
        self._first_frame_shape = tmp.shape

        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

        if dtype is None:
            self._dtype = tmp.dtype
        else:
            self._dtype = dtype

    def _get_files(self, path_spec):
        # deal with if input is _not_ a string
        if not isinstance(path_spec, six.string_types):
            # assume it is iterable and off we go!
            self._filepaths = list(path_spec)
            self._count = len(path_spec)
            return

        self.pathname = os.path.abspath(path_spec)  # used by __repr__
        if os.path.isdir(path_spec):
            warn("Loading ALL files in this directory. To ignore extraneous "
                 "files, use a pattern like 'path/to/images/*.png'",
                 UserWarning)
            directory = path_spec
            filenames = os.listdir(directory)
            make_full_path = lambda filename: (
                os.path.abspath(os.path.join(directory, filename)))
            filepaths = list(map(make_full_path, filenames))
        else:
            filepaths = glob.glob(path_spec)
        filepaths.sort()  # listdir returns arbitrary order
        self._filepaths = filepaths
        self._count = len(self._filepaths)

        # If there were no matches, this was probably a user typo.
        if self._count == 0:
            raise IOError("No files were found matching that path.")


    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        res = imread(self._filepaths[j], **self.kwargs)
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
        try:
            source = self.pathname
        except AttributeError:
            source = '(list of images)'
        return """<Frames>
Source: {pathname}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  pathname=source,
                                  dtype=self.pixel_type)
