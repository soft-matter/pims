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


class FramesDimension(object):
    """
    This class defines a dimension for use in the Multidimensional base class.

    Parameters
    ----------
    name : string, dimension name
    size : int, size of dimension
    aggregate : boolean, default False
        when True, this sets the dimension to be aggregated inside the Frame
    iterate : boolean, default True
        when True, this sets the dimension to be iterable
    default : int
        when aggregate and iterate are both False, this is the fallback value
    """
    def __init__(self, name, size, aggregate=False, iterate=True, default=0):
        if name.lower() in ['x', 'y']:
            raise ValueError('The names x and y are reserved and cannot be '
                             'used.')
        self.name = name
        self._default = default
        self.size = size
        if aggregate and iterate:
            raise ValueError('Dimensions cannot aggregate and be iterable '
                             'simultaneously.')
        self._aggregate = aggregate
        self._iterate = iterate

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if value <= 0:
            raise ValueError('Dimension size should be greater than zero.')
        self._size = int(value)
        self.default = self._default  # recheck default value validity

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        if value >= self.size:
            raise ValueError('Default value out of bounds')
        self._default = value

    @property
    def aggregate(self):
        return self._aggregate

    @aggregate.setter
    def aggregate(self, value):
        self._aggregate = value
        if value:  # iterate and aggregate cannot both be True
            self._iterate = False

    @property
    def iterate(self):
        return self._iterate

    @iterate.setter
    def iterate(self, value):
        self._iterate = value
        if value:  # iterate and aggregate cannot both be True
            self._aggregate = False

    def __repr__(self):
        s = "<FramesDimension>; Name: {0}; Size: {1}; Status: "
        s = s.format(self.name, self.size)
        if self.iterate:
            s += "iterate"
        elif self.aggregate:
            s += "aggregate"
        else:
            s += "defaults to {}".format(self.default)
        return s


class Multidimensional(FramesSequence):
    """ A base class defining a FramesSequence with an arbitrary number of
    dimensions. The properties `aggregate` and `iterate` define the functions
    of each dimension.

    Subclassed readers need to define `frame_shape_2D` and `get_frame_2D`.
    Dimensions need to be instanciated using the available method `add_dim`.

    Examples
    --------
    >>> class MDummy(Multidimensional):
    ...    @property
    ...    def pixel_type(self):
    ...        return 'uint8'
    ...    @property
    ...    def frame_shape_2D(self):
    ...        return self._frame_shape_2D
    ...    def __init__(self, shape, **dims):
    ...        for name in dims:
    ...            self.add_dim(name, dims[name])
    ...        self._frame_shape_2D = shape
    ...    def get_frame_2D(self, **ind):
    ...        return np.zeros(self.frame_shape_2D, dtype=self.pixel_type)

    >>> frames = MDummy((64, 64), t=80, c=2, z=10, m=5)
    >>> frames.aggregate = 'cz'
    >>> frames.iterate = 't'
    >>> frames.dims['m'].default = 3
    >>> frames[5]  # returns Frame at T=5, M=3 with shape (2, 10, 64, 64)
    """
    def clear_dims(self):
        self._dims = []

    def add_dim(self, name, size, aggregate=False, iterate=True,
                default=0):
        new_dim = FramesDimension(name, size, aggregate, iterate, default)
        self._dims += [new_dim]

    def __len__(self):
        return np.prod([d.size for d in self._dims if d.iterate])

    @abstractproperty
    def frame_shape_2D(self, **ind):
        """ This property should return the shape of a single 2D image. """
        pass

    @property
    def frame_shape(self):
        """ Returns the shape of the frame as returned by get_frame. """
        shape = [d.size for d in self._dims if d.aggregate]
        return tuple(shape + list(self.frame_shape_2D))

    @property
    def dims(self):
        """ Returns a dictionary of all FramesDimension objects. """
        return {d.name: d for d in self._dims}

    @property
    def ndim(self):
        """ Returns the number of dimensions, including x and y. """
        return len(self._dims) + 2

    @property
    def sizes(self):
        """ Returns a dict of all dimension sizes, including x and y. """
        result = {d.name: d.size for d in self._dims}
        result['y'], result['x'] = self.frame_shape_2D
        return result

    @property
    def aggregate(self):
        """ This determines which dimensions will be aggregated into one Frame.
        The ndarray that is returned by get_frame has the same dimension order
        as the order of aggregate. """
        return [d.name for d in self._dims if d.aggregate]

    @aggregate.setter
    def aggregate(self, value):
        for dim_aggregate in value:
            if dim_aggregate.lower() in ['x', 'y']:
                raise ValueError('Aggregate does not take dimensions x or y.')
            if dim_aggregate not in self.dims:
                raise ValueError(('Dimension named ''{}'' does not exist ' +
                                  'in this image.').format(dim_aggregate))

        new_dims_aggr = [0] * len(value)
        new_dims_rest = []
        for dim in self._dims:
            for n_aggr, dim_aggregate in enumerate(value):
                if dim.name == dim_aggregate:
                    new_dims_aggr[n_aggr] = dim
                    dim.aggregate = True
                    break
            else:
                new_dims_rest.append(dim)
                dim.aggregate = False
        self._dims = new_dims_rest + new_dims_aggr

    @property
    def iterate(self):
        """ This determines which dimensions will be iterated over by the
        FramesSequence. The last element will iterate fastest. """
        return [d.name for d in self._dims if d.iterate]

    @iterate.setter
    def iterate(self, value):
        for dim_iterate in value:
            if dim_iterate.lower() in ['x', 'y']:
                raise ValueError('Iterate does not take dimensions x or y.')
            if dim_iterate not in self.dims:
                raise ValueError(("Dimension named '{}' does not exist " +
                                  "in this image.").format(dim_iterate))

        new_dims_iter = [0] * len(value)
        new_dims_rest = []
        for dim in self._dims:
            for n_iter, dim_iterate in enumerate(value):
                if dim.name == dim_iterate:
                    new_dims_iter[n_iter] = dim
                    dim.iterate = True
                    break
            else:
                new_dims_rest.append(dim)
                dim.iterate = False
        self._dims = new_dims_iter + new_dims_rest

    @abstractmethod
    def get_frame_2D(self, **ind):
        """ The actual frame reader, defined by the subclassed reader.

        This method should take exactly one keyword argument per dimension,
        reflecting the index along each dimension. It returns a two dimensional
        ndarray with shape `frame_shape_2D` and dtype `pixel_type`.
        """
        pass

    def get_frame(self, i):
        """ Returns a Frame of shape deterimend by aggregate. The property
        iterate (together with the dimension default index) determines which.
        multidimensional index is returned. """
        # identify the indices to take along each dimension
        i_prev = 0
        for n, dim in enumerate(self._dims):
            dim.i = dim.default
            if dim.iterate:
                size = int(np.prod([d.size for d in self._dims[n+1:]
                                    if d.iterate]))
                dim.i = (i - i_prev) // size
                i_prev += dim.i * size

        # initialize a stack of 2D arrays to collect the Frame
        Nframes = int(np.prod(self.frame_shape[:-2]))
        result = np.empty([Nframes] + list(self.frame_shape_2D),
                          dtype=self.pixel_type)

        # read all 2D frames
        for n in range(Nframes):
            result[n] = self.get_frame_2D(**{d.name: d.i for d in self._dims})
            for dim in self._dims[::-1]:
                if dim.aggregate:
                    dim.i += 1
                    if dim.i >= dim.size:
                        dim.i = 0
                    else:
                        break

        # reshape the array into the desired shape
        result.shape = self.frame_shape
        return Frame(result, frame_no=i)

    def __getattr__(self, key):
        """ Sets an empty value for self._dims without using __init__.

        Enables dimensions to be accessed by for instance frames.c.
        Existing methods always precede over this: a dimension named 'ndim'
        would not be accessible by frames.ndim. """
        if key == '_dims':
            return []
        for dim in self._dims:
            if key == dim.name:
                return dim
        raise AttributeError
