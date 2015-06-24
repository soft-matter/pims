from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six import with_metaclass
from six.moves import range
import os
import numpy as np
import collections
import itertools
from .frame import Frame
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import wraps
from warnings import warn


class FramesStream(with_metaclass(ABCMeta, object)):
    """
    A base class for wrapping input data which knows how to
    advance to the next frame, but does not have random access.

    The length does not need to be finite.

    Does not support slicing.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __iter__(self):
        pass

    @abstractproperty
    def pixel_type(self):
        """Returns a numpy.dtype for the data type of the pixel values"""
        pass

    @abstractproperty
    def frame_shape(self):
        """Returns the shape of a single frame as a tuple ex (10, 12)"""
        pass

    @classmethod
    def class_exts(cls):
        """
        Return a set of the file extensions that this reader can deal with.

        Sub-classes should over-ride this function to list what extensions
        they deal with.

        The default interpretation of the returned set is 'file
        extensions including but not exclusively'.
        """
        return set()

    @property
    def exts(self):
        """
        Property to get the extensions of a FramesStream class.

        Calls relevant classmethod.
        """
        return type(self).class_ext()

    def close(self):
        """
        A method to clean up anything that need to be cleaned up.

        Sub-classes should use super to call up the MRO stack and then
        do any class-specific clean up
        """
        pass

    def _validate_process_func(self, process_func):
        if process_func is None:
            process_func = lambda x: x
        if not callable(process_func):
            raise ValueError("process_func must be a function, or None")
        self.process_func = process_func

    def _as_grey(self, as_grey, process_func):
        # See skimage.color.colorconv in the scikit-image project.
        # As noted there, the weights used in this conversion are calibrated
        # for contemporary CRT phosphors. Any alpha channel is ignored."""

        if as_grey:
            if process_func is not None:
                raise ValueError("The as_grey option cannot be used when "
                                 "process_func is specified. Incorpate "
                                 "greyscale conversion in the function "
                                 "passed to process_func.")
            shape = self.frame_shape
            ndim = len(shape)
            # Look for dimensions that look like color channels.
            rgb_like = shape.count(3) == 1
            rgba_like = shape.count(4) == 1
            if ndim == 2:
                # The image is already greyscale.
                process_func = None
            elif ndim == 3 and (rgb_like or rgba_like):
                reduced_shape = list(shape)
                if rgb_like:
                    color_axis_size = 3
                    calibration = [0.2125, 0.7154, 0.0721]
                else:
                    color_axis_size = 4
                    calibration = [0.2125, 0.7154, 0.0721, 0]
                reduced_shape.remove(color_axis_size)
                self._im_sz = tuple(reduced_shape)
                def convert_to_grey(img):
                    color_axis = img.shape.index(color_axis_size)
                    img = np.rollaxis(img, color_axis, 3)
                    grey = (img * calibration).sum(2)
                    return grey.astype(img.dtype)  # coerce to original dtype
                self.process_func = convert_to_grey
            else:
                raise NotImplementedError("I don't know how to convert an "
                                          "image of shaped {0} to greyscale. "
                                          "Write you own function and pass "
                                          "it using the process_func "
                                          "keyword argument.".format(shape))

    # magic functions to make all sub-classes usable as context managers
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  dtype=self.pixel_type)


class SliceableIterable(object):

    def __init__(self, ancestor, indices, length=None):
        """A generator that supports fancy indexing

        When sliced using any iterable with a known length, it return another
        object like itself, a SliceableIterable. When sliced with an integer,
        it returns the data payload.

        Also, this retains the attributes of the ultimate ancestor that
        created it (or its parent, or its parent's parent, ...).

        Parameters
        ----------
        ancestor : object
            must support __getitem__ with an integer argument
        indices : iterable
            giving indices into `ancestor`
        length : integer, optional
            length of indicies
            This is required if `indices` is a generator,
            that is, if `len(indices)` is invalid

        Examples
        --------
        # Slicing on a SliceableIterable returns another SliceableIterable...
        >>> v = SliceableIterable([0, 1, 2, 3], range(4), 4)
        >>> v1 = v[:2]
        >>> type(v[:2])
        SliceableIterable
        >>> v2 = v[::2]
        >>> type(v2)
        SliceableIterable
        >>> v2[0]
        0
        # ...unless the slice itself has an unknown length, which makes
        # slicing impossible.
        >>> v3 = v2((i for i in [0]))  # argument is a generator
        >>> type(v3)
        generator
        """
        if length is None:
            try:
                length = len(indices)
            except TypeError:
                raise ValueError("The length parameter is required in this "
                                 "case because len(indices) is not valid.")
        self._len = length
        self._ancestor = ancestor
        self._indices = indices
        self._counter = 0
        self._proc_func = lambda image: image

    @property
    def indices(self):
        # Advancing indices won't affect this new copy of self._indices.
        indices, self._indices = itertools.tee(iter(self._indices))
        return indices

    def _get(self, key):
        "Wrap ancestor's get_frame method in a processing function."
        return self._proc_func(self._ancestor[key])

    def __repr__(self):
        msg = "Sliced and/or processed {0}. Original repr:\n".format(
                type(self._ancestor).__name__)
        old = '\n'.join("    " + ln for ln in repr(self._ancestor).split('\n'))
        return msg + old

    def __iter__(self):
        return (self._get(i) for i in self.indices)

    def __len__(self):
        return self._len

    def __getattr__(self, key):
        # Remember this only gets called if __getattribute__ raises an
        # AttributeError. Try the ancestor object.
        return getattr(self._ancestor, key)

    def __getitem__(self, key):
        """for data access"""
        _len = len(self)
        abs_indices = self.indices

        if isinstance(key, slice):
            # if input is a slice, return another SliceableIterable
            start, stop, step = key.indices(_len)
            rel_indices = range(start, stop, step)
            new_length = len(rel_indices)
            indices = _index_generator(rel_indices, abs_indices)
            return SliceableIterable(self._ancestor, indices, new_length)
        elif isinstance(key, collections.Iterable):
            # if the input is an iterable, doing 'fancy' indexing
            if isinstance(key, np.ndarray) and key.dtype == np.bool:
                # if we have a bool array, set up masking but defer
                # the actual computation, returning another SliceableIterable
                rel_indices = np.arange(len(self))[key]
                indices = _index_generator(rel_indices, abs_indices)
                new_length = key.sum()
                return SliceableIterable(self._ancestor, indices, new_length)
            if any(_k < -_len or _k >= _len for _k in key):
                raise IndexError("Keys out of range")
            try:
                new_length = len(key)
            except TypeError:
                # The key is a generator; return a plain old generator.
                # Without knowing the length of the *key*,
                # we can't give a SliceableIterable
                gen = (self[_k if _k >= 0 else _len + _k] for _k in key)
                return gen
            else:
                # The key is a list of in-range values. Build another
                # SliceableIterable, again deferring computation.
                rel_indices = ((_k if _k >= 0 else _len + _k) for _k in key)
                indices = _index_generator(rel_indices, abs_indices)
                return SliceableIterable(self._ancestor, indices, new_length)
        else:
            if key < -_len or key >= _len:
                raise IndexError("Key out of range")
            try:
                abs_key = self._indices[key]
            except TypeError:
                key = key if key >= 0 else _len + key
                rel_indices, self._indices = itertools.tee(self._indices)
                for _, i in zip(range(key + 1), rel_indices):
                    abs_key = i
            return self._get(abs_key)

    def close(self):
        "Closing this child slice of the original reader does nothing."
        pass


class FramesSequence(FramesStream):
    """Baseclass for wrapping data buckets that have random access.

    Support random access.

    Supports standard slicing and fancy slicing, but returns a
    generator.

    Must be finite length.

    """
    def __getitem__(self, key):
        """If getting a scalar, a specific frame, call get_frame. Otherwise,
        be 'lazy' and defer to the slicing logic of SliceableIterable."""
        if isinstance(key, int):
            i = key if key >= 0 else len(self) + key
            return self.get_frame(i)
        else:
            return SliceableIterable(self, range(len(self)), len(self))[key]

    def __iter__(self):
        return iter(self[:])

    @abstractmethod
    def __len__(self):
        """
        It is obligatory that sub-classes define a length.
        """
        pass

    @abstractmethod
    def get_frame(self, ind):
        """
        Sub classes must over-ride this function for how to get a given
        frame out of the file.  Any data-type specific internal-state
        nonsense should be dealt with in this function.
        """
        pass

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count = len(self),
                                  dtype=self.pixel_type)


class FrameRewindableStream(FramesStream):
    """
    A base class for holding the common code for
    wrapping data sources that do not rewind easily.
    """
    @abstractmethod
    def rewind(self, j=0):
        """
        Resets the stream to frame j

        j : int
            Frame to rewind the stream to
        """
        pass

    @abstractmethod
    def skip_forward(self, j):
        """
        Skip the stream forward by j frames.

        j : int
           Number of frames to skip
        """
        pass

    @abstractmethod
    def next(self):
        """
        return the next frame in the stream
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractproperty
    def current(self):
        """
        The current location in the stream.

        Can be an int if in stream or None if out the end.

        """
        pass

    def __iter__(self):
        self.rewind(0)
        return self

    def __getitem__(self, arg):
        """
        Returns a generator which yields frames
        """
        if isinstance(arg, slice):
            # get value from slice
            start, stop, step = arg.start, arg.stop, arg.step
            # sanitize step
            if step is None:
                step = 1
            if step < 1:
                raise ValueError("step must be positive")
            # make sure the stream is in the right place to start
            if start is None:
                start = 0
            if start < self.current:
                self.rewind(start)
            if start > self.current:
                self.skip_forward(start - self.current)

            # sanity check
            if stop is not None and stop < start:
                raise ValueError("start must be less than stop")
            # special case, we can't just return self, because __iter__ rewinds
            if step == 1 and stop is None:
                # keep going until exhausted
                return (self.next() for _ in itertools.repeat(True))

            return self._step_gen(step, stop)

        elif isinstance(arg, int):
            self.rewind(arg)
            return self.next()
        else:
            raise ValueError("Invalid arguement, use either a `slice` or " +
                             "or an `int`. not {t}".format(t=str(type(arg))))

    def _step_gen(self, step, stop):
        """
        Wraps up the logic of stepping forward by step > 1
        """
        while stop is None or self.current < stop:
            yield self.next()
            self.skip_forward(step - 1)
        else:
            raise StopIteration

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count = len(self),
                                  dtype=self.pixel_type)


def _index_generator(new_indices, old_indices):
    """Find locations of new_indicies in the ref. frame of the old_indices.
    
    Example: (1, 3), (1, 3, 5, 10) -> (3, 10)

    The point of all this trouble is that this is done lazily, returning
    a generator without actually looping through the inputs."""
    # Use iter() to be safe. On a generator, this returns an identical ref.
    new_indices = iter(new_indices)
    n = next(new_indices)
    last_n = None
    done = False
    while True:
        old_indices_, old_indices = itertools.tee(iter(old_indices))
        for i, o in enumerate(old_indices_):
            # If new_indices is not strictly monotonically increasing, break
            # and start again from the beginning of old_indices.
            if last_n is not None and n <= last_n:
                last_n = None
                break
            if done:
                raise StopIteration
            if i == n:
                last_n = n
                try:
                    n = next(new_indices)
                except StopIteration:
                    done = True
                    # Don't stop yet; we still have one last thing to yield.
                yield o
            else:
                continue


def pipeline(func):
    """Decorator to make function aware of pims objects.

    When the function is applied to a pims reader or a slice of one, it
    returns another lazily-evaluated, sliceable object.

    When the function is applied to any other object, it falls back on its
    normal behavhior.

    Parameters
    ----------
    func : callable
        function that accepts an image as its first argument

    Returns
    -------
    processed_images : pims.SliceableIterator

    Example
    -------
    Apply the pipeline decorator to your image processing function.
    >>> @pipeline
    ...  def color_channel(image, channel):
    ...      return image[channel, :, :]
    ...

    Load images with PIMS.
    >>> images = pims.ImageSequence(...)

    Passing the PIMS class to the function return another PIMS object
    that "lazily" applies the function when the images come out. Different
    functions can be applied to the same underlying images, creating
    independent objects.
    >>> red_images = color_channel(images, 0)
    >>> green_images = color_channel(images, 1)

    Pipeline functions can also be composed.
    >>> @pipeline
    ... def rescale(image):
    ... return (image - image.min())/image.ptp()
    ...
    >>> rescale(color_channel(images, 0))

    The function can still be applied to ordinary images. The decorator
    only takes affect when a PIMS object is passed.
    >>> single_img = images[0]
    >>> red_img = red_channel(single_img)  # normal behavior
    """
    @wraps(func)
    def process(img_or_iterable, *args, **kwargs):
        if isinstance(img_or_iterable, (SliceableIterable, FramesSequence)):
            _len = len(img_or_iterable)
            s = SliceableIterable(img_or_iterable, range(_len), _len)
            s._proc_func = lambda image: func(image, *args, **kwargs)
            return s
        else:
            # Fall back on normal behavior of func, interpreting input
            # as a single image.
            return func(img_or_iterable)

    if process.__doc__ is None:
        process.__doc__ = ''
    process.__doc__ = ("This function has been made pims-aware. When passed\n"
                       "a pims reader or SliceableIterable, it will return a \n"
                       "new SliceableIterable of the results. When passed \n"
                       "other objects, its behavior is "
                       "unchanged.\n\n") + process.__doc__
    return process


class FramesSequenceND(FramesSequence):
    """ A base class defining a FramesSequence with an arbitrary number of
    axes. In the context of this reader base class, dimensions like 'x', 'y',
    't' and 'z' will be called axes. Indices along these axes will be called
    coordinates.

    The properties `bundle_axes`, `iter_axes`, and `default_coords` define
    to which coordinates each index points. See below for a description of
    each attribute.

    Subclassed readers only need to define `get_frame_2D`, `pixel_type` and
    `__init__`. In the `__init__`, at least axes y and x need to be
    initialized using `_init_axis(name, size)`.

    The attributes `__len__`, `frame_shape`, and `get_frame` are defined by
    this base_class; these are not meant to be changed.

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
    default_coords: dict of int
        When a dimension is not present in both iter_axes and bundle_axes, the
        coordinate contained in this dictionary will be used.

    Examples
    --------
    >>> class MDummy(FramesSequenceND):
    ...    @property
    ...    def pixel_type(self):
    ...        return 'uint8'
    ...    def __init__(self, shape, **axes):
    ...        self._init_axis('y', shape[0])
    ...        self._init_axis('x', shape[1])
    ...        for name in axes:
    ...            self._init_axis(name, axes[name])
    ...    def get_frame_2D(self, **ind):
    ...        return np.zeros((self.sizes['y'], self.sizes['x']),
    ...                        dtype=self.pixel_type)

    >>> frames = MDummy((64, 64), t=80, c=2, z=10, m=5)
    >>> frames.bundle_axes = 'czyx'
    >>> frames.iter_axes = 't'
    >>> frames.default_coords['m'] = 3
    >>> frames[5]  # returns Frame at T=5, M=3 with shape (2, 10, 64, 64)
    """
    def _init_axis(self, name, size, default=0):
        # check if the axes have been initialized, if not, do it here
        if not hasattr(self, '_sizes'):
            self._sizes = {}
            self._default_coords = {}
            self._iter_axes = []
            self._bundle_axes = ['y', 'x']
        elif name in self._sizes:
            raise ValueError("dimension '{}' already exists".format(name))
        self._sizes[name] = int(size)
        if not (name == 'x' or name == 'y'):
            self.default_coords[name] = int(default)

    def __len__(self):
        return int(np.prod([self._sizes[d] for d in self._iter_axes]))

    @property
    def frame_shape(self):
        """ Returns the shape of the frame as returned by get_frame. """
        return tuple([self._sizes[d] for d in self._bundle_axes])

    @property
    def axes(self):
        """ Returns a list of all axes. """
        return [k for k in self._sizes]

    @property
    def ndim(self):
        """ Returns the number of axes. """
        return len(self._sizes)

    @property
    def sizes(self):
        """ Returns a dict of all axis sizes. """
        return self._sizes

    @property
    def bundle_axes(self):
        """ This determines which dimensions will be bundled into one Frame.
        The ndarray that is returned by get_frame has the same dimension order
        as the order of `bundle_axes`.
        The last two elements have to be ['y', 'x'].
        """
        return self._bundle_axes

    @bundle_axes.setter
    def bundle_axes(self, value):
        invalid = [k for k in value if k not in self._sizes]
        if invalid:
            raise ValueError("axes %r do not exist" % invalid)

        if len(value) < 2 or not (value[-1] == 'x' and value[-2] == 'y'):
            raise ValueError("bundle_axes should end with ['y', 'x']")

        for k in value:
            if k in self._iter_axes:
                del self._iter_axes[self._iter_axes.index(k)]

        self._bundle_axes = list(value)

    @property
    def iter_axes(self):
        """ This determines which axes will be iterated over by the
        FramesSequence. The last element will iterate fastest.
        x and y are not allowed. """
        return self._iter_axes

    @iter_axes.setter
    def iter_axes(self, value):
        invalid = [k for k in value if k not in self._sizes]
        if invalid:
            raise ValueError("axes %r do not exist" % invalid)

        if 'x' in value or 'y' in value:
            raise ValueError("axes 'y' and 'x' cannot be iterated")

        for k in value:
            if k in self._bundle_axes:
                del self._bundle_axes[self._bundle_axes.index(k)]

        self._iter_axes = list(value)

    @property
    def default_coords(self):
        """ When a axis is not present in both iter_axes and bundle_axes, the
        coordinate contained in this dictionary will be used. """
        return self._default_coords

    @default_coords.setter
    def default_coords(self, value):
        invalid = [k for k in value if k not in self._sizes]
        if invalid:
            raise ValueError("axes %r do not exist" % invalid)
        self._default_coords.update(**value)

    @abstractmethod
    def get_frame_2D(self, **ind):
        """ The actual frame reader, defined by the subclassed reader.

        This method should take exactly one keyword argument per axis,
        reflecting the coordinate along each axis. It returns a two dimensional
        ndarray with shape (sizes['y'], sizes['x']) and dtype `pixel_type`. It
        may also return a Frame object, so that metadata will be propagated. It
        will only propagate metadata if every bundled frame gives the same
        fields.
        """
        pass

    def get_frame(self, i):
        """ Returns a Frame of shape deterimend by bundle_axes. The index value
        is interpreted according to the iter_axes property. Coordinates not
        present in both iter_axes and bundle_axes will be set to their default
        value (see default_coords). """

        # start with the default coordinates
        coords = self._default_coords.copy()

        # list sizes of iterate dimensions
        iter_sizes = [self._sizes[k] for k in self._iter_axes]
        # list how much i has to increase to get an increase of coordinate n
        iter_cumsizes = np.append(np.cumprod(iter_sizes[::-1])[-2::-1], 1)
        # calculate the coordinates and update the coords dictionary
        iter_coords = (i // iter_cumsizes) % iter_sizes
        coords.update(**{k: v for k, v in zip(self._iter_axes, iter_coords)})

        shape = self.frame_shape
        if len(shape) == 2:  # simple case of only one frame
            result = self.get_frame_2D(**coords)
            if hasattr(result, 'metadata'):
                metadata = result.metadata
            else:
                metadata = None
        else:  # general case of N dimensional frame
            Nframes = int(np.prod(shape[:-2]))
            result = np.empty([Nframes] + list(shape[-2:]),
                              dtype=self.pixel_type)

            # read all 2D frames and properly iterate through the coordinates
            mdlist = [{}] * Nframes
            for n in range(Nframes):
                frame = self.get_frame_2D(**coords)
                result[n] = frame
                if hasattr(frame, 'metadata'):
                    mdlist[n] = frame.metadata
                for dim in self._bundle_axes[-3::-1]:
                    coords[dim] += 1
                    if coords[dim] >= self._sizes[dim]:
                        coords[dim] = 0
                    else:
                        break
            # reshape the array into the desired shape
            result.shape = shape

            # propagate metadata
            metadata = {}
            if not np.all([md == {} for md in mdlist]):
                keys = mdlist[0].keys()
                for k in keys:
                    try:
                        metadata[k] = [row[k] for row in mdlist]
                    except KeyError:
                        # if a field is not present in every frame, ignore it
                        warn('metadata field {} is not propagated')
                    else:
                        # if all values are equal, only return one value
                        if metadata[k][1:] == metadata[k][:-1]:
                            metadata[k] = metadata[k][0]
                        else:  # cast into ndarray
                            metadata[k] = np.array(metadata[k])
                            metadata[k].shape = shape[:-2]

        return Frame(result, frame_no=i, metadata=metadata)

    def __repr__(self):
        s = "<FramesSequenceND>\nDimensions: {0}\n".format(self.ndim)
        for dim in self._sizes:
            s += "Dimension '{0}' size: {1}\n".format(dim, self._sizes[dim])
        s += """Pixel Datatype: {dtype}""".format(dtype=self.pixel_type)
        return s
