import os
import glob
import fnmatch
from warnings import warn
import re
import zipfile
from io import BytesIO
from functools import partial

import numpy as np

import pims
from pims.base_frames import FramesSequence, FramesSequenceND
from pims.frame import Frame
from pims.image_reader import imread
from pims.utils.sort import natural_keys


class ImageSequence(FramesSequence):
    """Read a directory of sequentially numbered image files into an
    iterable that returns images as numpy arrays.

    Parameters
    ----------
    path_spec : string or iterable of strings
        a directory or, safer, a pattern like path/to/images/*.png
        which will ignore extraneous files or a list of files to open
        in the order they should be loaded. When a path to a zipfile is
        specified, all files in the zipfile will be loaded.
    plugin : string
        Passed on to skimage.io.imread if scikit-image is available.
        If scikit-image is not available, this will be ignored and a warning
        will be issued. Not available in combination with zipfiles.

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
    def __init__(self, path_spec, plugin=None):
        if not imread.__module__.startswith("skimage"):
            if plugin is not None:
                warn("A plugin was specified but ignored. Plugins can only "
                     "be specified if scikit-image is available. Instead, "
                     "ImageSequence will use imageio")
            self.kwargs = dict()
        else:
            self.kwargs = dict(plugin=plugin)

        self._is_zipfile = False
        self._zipfile = None
        self._get_files(path_spec)

        tmp = self.imread(self._filepaths[0], **self.kwargs)
        self._first_frame_shape = tmp.shape
        self._dtype = tmp.dtype

    def close(self):
        if self._is_zipfile:
            self._zipfile.close()
        super(ImageSequence, self).close()

    def __del__(self):
        self.close()

    def imread(self, filename, **kwargs):
        if self._is_zipfile:
            file_handle = BytesIO(self._zipfile.read(filename))
            return imread(file_handle, **kwargs)
        else:
            return imread(filename, **kwargs)

    def _get_files(self, path_spec):
        # deal with if input is _not_ a string
        if not isinstance(path_spec, str):
            # assume it is iterable and off we go!
            self._filepaths = list(path_spec)
            self._count = len(self._filepaths)
            return

        if zipfile.is_zipfile(path_spec):
            self._is_zipfile = True
            self.pathname = os.path.abspath(path_spec)
            self._zipfile = zipfile.ZipFile(path_spec, 'r')
            filepaths = [fn for fn in self._zipfile.namelist()
                         if fnmatch.fnmatch(fn, '*.*')]
            self._filepaths = sorted(filepaths, key=natural_keys)
            self._count = len(self._filepaths)
            if 'plugin' in self.kwargs and self.kwargs['plugin'] is not None:
                warn("A plugin cannot be combined with reading from an "
                     "archive. Extract it if you want to use the plugin.")
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
        self._filepaths = sorted(filepaths, key=natural_keys)
        self._count = len(self._filepaths)

        # If there were no matches, this was probably a user typo.
        if self._count == 0:
            raise IOError("No files were found matching that path.")

    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        res = self.imread(self._filepaths[j], **self.kwargs)
        return Frame(res, frame_no=j)

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
Frame Shape: {frame_shape!r}
Pixel Datatype: {dtype}""".format(frame_shape=self.frame_shape,
                                  count=len(self),
                                  pathname=source,
                                  dtype=self.pixel_type)


def filename_to_indices(filename, identifiers='tzc'):
    """ Find ocurrences of axis indices (e.g. t001, z06, c2)
    in a filename and returns a list of indices.

    Parameters
    ----------
    filename : string
        filename to be searched for indices
    identifiers : string or list of strings, optional
        iterable of N strings preceding axis indices, in that order

    Returns
    ---------
    list of int
        axis indices. Elements default to 0 when index was not found.

    """
    escaped = [re.escape(a) for a in identifiers]
    axes = re.findall('(' + '|'.join(escaped) + r')(\d+)',
                            filename)
    if len(axes) > len(identifiers):
        axes = axes[-3:]
    order = [a[0] for a in axes]
    result = [0] * len(identifiers)
    for (i, col) in enumerate(identifiers):
        try:
            result[i] = int(axes[order.index(col)][1])
        except ValueError:
            result[i] = 0
    return result


class ReaderSequence(FramesSequenceND):
    """Construct a reader from a directory of ND image files.

    Parameters
    ----------
    path_spec : string or iterable of strings
        a directory or, safer, a pattern like path/to/images/*.png
        which will ignore extraneous files or a list of files to open
        in the order they should be loaded. When a path to a zipfile is
        specified, all files in the zipfile will be loaded. The filenames
        should contain the indices of T, Z and C, preceded by a axis
        identifier such as: 'file_t001c05z32'.
    axis_name : string, optional
        The name of the added axis. Default 't'.
    """
    def __init__(self, path_spec, reader_cls=None, axis_name='t', **kwargs):
        FramesSequenceND.__init__(self)

        self.kwargs = kwargs
        if reader_cls is None:
            self.reader_cls = pims.open
        else:
            self.reader_cls = reader_cls
        self._get_files(path_spec)

        with self.reader_cls(self._filepaths[0], **self.kwargs) as reader:
            if not isinstance(reader, FramesSequenceND):
                raise ValueError("Reader is not subclass of FramesSequenceND")
            for ax in reader.axes:
                self._init_axis(ax, reader.sizes[ax])
            self._pixel_type = reader.pixel_type
        self._imseq_axis = axis_name
        self._init_axis(axis_name, self._count)
        self.iter_axes = [axis_name]

    @property
    def bundle_axes(self):
        return self._bundle_axes[:]

    @bundle_axes.setter
    def bundle_axes(self, value):
        """Overrides the baseclass 'smart' bundle_axes, as _get_frame_wrapped
        uses the child reader bundle axes logic."""
        value = list(value)
        invalid = [k for k in value if k not in self._sizes]
        if invalid:
            raise ValueError("axes %r do not exist" % invalid)

        if self._imseq_axis in self.bundle_axes:
            raise ValueError('The sequence axis cannot be bundled.')
        for k in value:
            if k in self._iter_axes:
                del self._iter_axes[self._iter_axes.index(k)]
        self._bundle_axes = value
        self._get_frame_wrapped = self._get_seq_frame

    def _get_seq_frame(self, **coords):
        i = coords.pop(self._imseq_axis)
        with self.reader_cls(self._filepaths[i], **self.kwargs) as reader:
            # check whether the reader has the expected shape
            for ax in self.sizes:
                if ax == self._imseq_axis:
                    continue
                if ax not in reader.sizes:
                    raise RuntimeError('{} does not have '
                                       'axis {}'.format(self._filepaths[i], ax))
                if reader.sizes[ax] != self.sizes[ax]:
                    raise RuntimeError('In {}, the size of axis {} was unexpect'
                                       'ed'.format(self._filepaths[i], ax))
            reader.bundle_axes = self.bundle_axes
            result = reader._get_frame_wrapped(**coords)
        return result

    @property
    def pixel_type(self):
        return self._pixel_type

    def __repr__(self):
        try:
            source = self.pathname
        except AttributeError:
            source = '(list of images)'
        s = "<ReaderSequence>\nSource: {0}\n".format(source)
        s += "Axes: {0}\n".format(self.ndim)
        for dim in self._sizes:
            s += "Axis '{0}' size: {1}\n".format(dim, self._sizes[dim])
        s += """Pixel Datatype: {dtype}""".format(dtype=self.pixel_type)
        return s

    def _get_files(self, path_spec):
        # deal with if input is _not_ a string
        if not isinstance(path_spec, str):
            # assume it is iterable and off we go!
            self._filepaths = sorted(list(path_spec), key=natural_keys)
            self._count = len(path_spec)
            return

        if zipfile.is_zipfile(path_spec):
            self._is_zipfile = True
            self.pathname = os.path.abspath(path_spec)
            self._zipfile = zipfile.ZipFile(path_spec, 'r')
            filepaths = [fn for fn in self._zipfile.namelist()
                         if fnmatch.fnmatch(fn, '*.*')]
            self._filepaths = sorted(filepaths, key=natural_keys)
            self._count = len(self._filepaths)
            if 'plugin' in self.kwargs and self.kwargs['plugin'] is not None:
                warn("A plugin cannot be combined with reading from an "
                     "archive. Extract it if you want to use the plugin.")
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
        self._filepaths = sorted(filepaths, key=natural_keys)
        self._count = len(self._filepaths)

        # If there were no matches, this was probably a user typo.
        if self._count == 0:
            raise IOError("No files were found matching that path.")


class ImageSequenceND(FramesSequenceND, ImageSequence):
    """Read a directory of multi-indexed image files into an iterable that
    returns images as numpy arrays. By default, the extra axes are
    denoted with t, z, c.

    Parameters
    ----------
    path_spec : string or iterable of strings
        a directory or, safer, a pattern like path/to/images/*.png
        which will ignore extraneous files or a list of files to open
        in the order they should be loaded. When a path to a zipfile is
        specified, all files in the zipfile will be loaded. The filenames
        should contain the indices of T, Z and C, preceded by a axis
        identifier such as: 'file_t001c05z32'.
    plugin : string, optional
        Passed on to skimage.io.imread if scikit-image is available.
        If scikit-image is not available, this will be ignored and a warning
        will be issued. Not available in combination with zipfiles.
    axes_identifiers : iterable of strings, optional
        N strings preceding axes indices. Default 'tzc'. x and y are not
        allowed. c is not allowed when images are RGB.

    Attributes
    ----------
    axes : list of strings
        List of all available axes
    ndim : int
        Number of image axes
    sizes : dict of int
        Dictionary with all axis sizes
    frame_shape : tuple of int
        Shape of frames that will be returned by get_frame
    iter_axes : iterable of strings
        This determines which axes will be iterated over by the FramesSequence.
        The last element in will iterate fastest. x and y are not allowed.
    bundle_axes : iterable of strings
        This determines which axes will be bundled into one Frame. The axes in
        the ndarray that is returned by get_frame have the same order as the
        order in this list. The last two elements have to be ['y', 'x'].
        If the 'z' axis exists then it defaults to ['z', 'y', 'x']
    default_coords : dict of int
        When an axis is not present in both iter_axes and bundle_axes, the
        coordinate contained in this dictionary will be used.
    is_rgb : boolean
        True when the input image is an RGB image.
    is_interleaved : boolean
        Applicable to RGB images. Signifies the position of the rgb axis in
        the input image. True when color data is stored in the last dimension.
    """
    def __init__(self, path_spec, plugin=None, axes_identifiers='tzc'):
        FramesSequenceND.__init__(self)
        if 'x' in axes_identifiers:
            raise ValueError("Axis 'x' is reserved")
        if 'y' in axes_identifiers:
            raise ValueError("Axis 'y' is reserved")
        self.axes_identifiers = axes_identifiers
        ImageSequence.__init__(self, path_spec, plugin)
        shape = self._first_frame_shape
        if len(shape) == 2:
            self._init_axis('y', shape[0])
            self._init_axis('x', shape[1])
            self._register_get_frame(self.get_frame_2D, 'yx')
            self.is_rgb = False
        elif len(shape) == 3 and shape[2] in [3, 4]:
            self._init_axis('y', shape[0])
            self._init_axis('x', shape[1])
            self._init_axis('c', shape[2])
            self._register_get_frame(self.get_frame_2D, 'yxc')
            self.is_rgb = True
            self.is_interleaved = True
        elif len(shape) == 3:
            self._init_axis('c', shape[0])
            self._init_axis('y', shape[1])
            self._init_axis('x', shape[2])
            self._register_get_frame(self.get_frame_2D, 'cyx')
            self.is_rgb = True
            self.is_interleaved = False
        else:
            raise IOError("Could not interpret image shape.")

        if self.is_rgb and 'c' in self.axes_identifiers:
            raise ValueError("Axis identifier 'c' is reserved when "
                             "images are rgb.")

        if 't' in self.axes:
            self.iter_axes = ['t']  # iterate over t
        if 'z' in self.axes:
            self.bundle_axes = ['z', 'y', 'x']  # return z-stacks

    def _get_files(self, path_spec):
        super(ImageSequenceND, self)._get_files(path_spec)
        self._toc = np.array([filename_to_indices(f, self.axes_identifiers)
                              for f in self._filepaths])
        for n, name in enumerate(self.axes_identifiers):
            if np.all(self._toc[:, n] == 0):
                self._toc = np.delete(self._toc, n, axis=1)
            else:
                self._toc[:, n] = self._toc[:, n] - min(self._toc[:, n])
                self._init_axis(name, max(self._toc[:, n]) + 1)
        self._filepaths = np.array(self._filepaths)

    def get_frame(self, i):
        frame = super(ImageSequenceND, self).get_frame(i)
        return Frame(frame, frame_no=i)

    def get_frame_2D(self, **ind):
        if self.is_rgb:
            c = ind['c']
            row = [ind[name] for name in self.axes_identifiers if name != 'c']
        else:
            row = [ind[name] for name in self.axes_identifiers]
        i = np.argwhere(np.all(self._toc == row, 1))[0, 0]
        return self.imread(self._filepaths[i], **self.kwargs)

    def __repr__(self):
        try:
            source = self.pathname
        except AttributeError:
            source = '(list of images)'
        s = "<ImageSequenceND>\nSource: {0}\n".format(source)
        s += "Axes: {0}\n".format(self.ndim)
        for dim in self._sizes:
            s += "Axis '{0}' size: {1}\n".format(dim, self._sizes[dim])
        s += """Pixel Datatype: {dtype}""".format(dtype=self.pixel_type)
        return s


def customize_image_sequence(imread_func, name=None):
    """Class factory for ImageSequence with customized image reader.

    Parameters
    ----------
    imread_func : callable
        image reader
    name : str or None
        name of class returned; if None, 'CustomImageSequence' is used.

    Returns
    -------
    type : a subclass of ImageSequence
        This subclass has its image-opening method, imread, overriden
        by the passed function.

    Example
    -------
    >>> # my_func accepts a filename and returns a numpy array
    >>> MyImageSequence = customize_image_sequence(my_func)
    >>> frames = MyImageSequence('path/to/my_weird_files*')
    """
    class CustomImageSequence(ImageSequence):
        def imread(self, filename, **kwargs):
            return imread_func(filename, **kwargs)
    if name is not None:
        CustomImageSequence.__name__ = name
    return CustomImageSequence
