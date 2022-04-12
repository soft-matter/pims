import numpy as np
import itertools
from slicerator import Slicerator, propagate_attr, index_attr
from .frame import Frame
from abc import ABC, abstractmethod, abstractproperty
from warnings import warn


class FramesStream(ABC):
    """
    A base class for wrapping input data which knows how to
    advance to the next frame, but does not have random access.

    The length does not need to be finite.

    Does not support slicing.
    """

    @abstractmethod
    def __iter__(self):
        pass

    @abstractproperty
    def pixel_type(self):
        """Returns a numpy.dtype for the data type of the pixel values"""
        pass

    @property
    def dtype(self):
        # The choice of using the separate name `pixel_type` was historical
        # and needlessly made PIMS objects look less like numpy arrays.
        return np.dtype(self.pixel_type)

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
        return type(self).class_exts()

    def close(self):
        """
        A method to clean up anything that need to be cleaned up.

        Sub-classes should use super to call up the MRO stack and then
        do any class-specific clean up
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Frame Shape: {frame_shape!r}
Pixel Datatype: {dtype}""".format(frame_shape=self.frame_shape,
                                  dtype=self.pixel_type)

@Slicerator.from_class
class FramesSequence(FramesStream):
    """Baseclass for wrapping data buckets that have random access.

    Support random access.

    Supports standard slicing and fancy slicing and returns a resliceable
    Slicerator object.

    Must be finite length.

    """
    propagate_attrs = ['frame_shape', 'pixel_type']

    def __getitem__(self, key):
        """__getitem__ is handled by Slicerator. In all pims readers, the data
        returning function is get_frame."""
        return self.get_frame(key)

    def __iter__(self):
        return iter(self[:])

    @property
    def shape(self):
        return (len(self), *self.frame_shape)

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
                                  count=len(self),
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
            raise ValueError("Invalid argument, use either a `slice` or " +
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
                                  count=len(self),
                                  dtype=self.pixel_type)

def _iter_attr(obj):
    try:
        for ns in [obj] + obj.__class__.mro():
            for attr in ns.__dict__:
                yield ns.__dict__[attr]
    except AttributeError:
        raise StopIteration  # obj has no __dict__


def _transpose(get_frame, expected_axes, desired_axes):
    if list(expected_axes) == list(desired_axes):
        return get_frame
    else:
        transposition = [expected_axes.index(a) for a in desired_axes]
        def get_frame_T(**ind):
            return get_frame(**ind).transpose(transposition)
        return get_frame_T


def _bundle(get_frame, expected_axes, to_iter, sizes, dtype):
    bundled_axes = to_iter + expected_axes
    shape = [sizes[a] for a in bundled_axes]
    iter_shape = shape[:len(to_iter)]
    def get_frame_bundled(**ind):
        result = np.empty(shape, dtype=dtype)
        md_list = []
        for indices in itertools.product(*[range(s) for s in iter_shape]):
            ind.update({n: i for n, i in zip(to_iter, indices)})
            frame = get_frame(**ind)
            result[indices] = frame

            if hasattr(frame, 'metadata'):
                if frame.metadata is not None:
                    md_list.append(frame.metadata)
        # propagate metadata
        if len(md_list) == np.prod(iter_shape):
            metadata = dict()
            keys = md_list[0].keys()
            for k in keys:
                try:
                    metadata[k] = [row[k] for row in md_list]
                except KeyError:
                    # if a field is not present in every frame, ignore it
                    warn('metadata field {} is not propagated')
                else:
                    # if all values are equal, only return one value
                    if metadata[k][1:] == metadata[k][:-1]:
                        metadata[k] = metadata[k][0]
                    else:  # cast into ndarray
                        metadata[k] = np.array(metadata[k])
                        metadata[k].shape = iter_shape
        else:
            metadata = None
        return Frame(result, metadata=metadata)
    return get_frame_bundled, bundled_axes


def _drop(get_frame, expected_axes, to_drop):
    # sort axes in descending order for correct function of np.take
    to_drop_inds = [list(expected_axes).index(a) for a in to_drop]
    indices = np.argsort(to_drop_inds)
    axes = [to_drop_inds[i] for i in reversed(indices)]
    to_drop = [to_drop[i] for i in reversed(indices)]
    result_axes = [a for a in expected_axes if a not in to_drop]

    def get_frame_dropped(**ind):
        result = get_frame(**ind)
        for (ax, name) in zip(axes, to_drop):
            result = np.take(result, ind[name], axis=ax)
        return result
    return get_frame_dropped, result_axes


def _make_get_frame(result_axes, get_frame_dict, sizes, dtype):
    methods = list(get_frame_dict.keys())
    result_axes = [a for a in result_axes]
    result_axes_set = set(result_axes)

    # search for get_frame methods that return the right axes
    for axes in methods:
        if len(set(axes) ^ result_axes_set) == 0:
            # _transpose does nothing when axes == result_axes
            return _transpose(get_frame_dict[axes], axes, result_axes)

    # we need either to drop axes or to iterate over axes:
    # collect some numbers to decide what to do
    arr = [None] * len(methods)
    for i, method in enumerate(methods):
        axes_set = set(method)
        to_iter_set = result_axes_set - axes_set
        to_iter = [x for x in result_axes if x in to_iter_set]  # fix the order
        n_iter = int(np.prod([sizes[ax] for ax in to_iter]))
        to_drop = list(axes_set - result_axes_set)
        n_drop = int(np.prod([sizes[ax] for ax in to_drop]))
        arr[i] = [method, axes_set, to_iter, n_iter, to_drop, n_drop]

    # try to read as less data as possible: try n_drop == 0
    # sort in increasing number of iterations
    arr.sort(key=lambda x: x[3])
    for method, axes_set, to_iter, n_iter, to_drop, n_drop in arr:
        if n_drop > 0:
            continue
        bundled_axes = to_iter + list(method)
        get_frame, after_bundle = _bundle(get_frame_dict[method], method,
                                          to_iter, sizes, dtype)
        return _transpose(get_frame, bundled_axes, result_axes)

    # try to iterate without dropping axes
    # sort in increasing number of dropped frames
    # TODO: sometimes dropping some data is better than having many iterations
    arr.sort(key=lambda x: x[5])
    for method, axes_set, to_iter, n_iter, to_drop, n_drop in arr:
        if n_iter > 0:
            continue
        get_frame, after_drop = _drop(get_frame_dict[method], method, to_drop)
        return _transpose(get_frame, after_drop, result_axes)

    # worst case: all methods have both too many axes and require iteration
    # take lowest number of dropped frames
    # if indecisive, take lowest number of iterations
    arr.sort(key=lambda x: (x[3], x[5]))
    method, axes_set, to_iter, n_iter, to_drop, n_drop = arr[0]

    get_frame, after_drop = _drop(get_frame_dict[method], method, to_drop)
    get_frame, after_bundle = _bundle(get_frame, after_drop, to_iter,
                                      sizes, dtype)
    return _transpose(get_frame, after_bundle, result_axes)


class DefaultCoordsDict(dict):
    """Dictionary that checks whether all keys are in `axes`"""
    def __init__(self, default_coords=None):
        """There is no check done here"""
        super(DefaultCoordsDict, self).__init__()
        self.axes = []

    def __setitem__(self, attr, value):
        if attr not in self.axes:
            raise ValueError("axes %r does not exist" % attr)
        super(DefaultCoordsDict, self).__setitem__(attr, value)

    def update(self, *args, **kwargs):
        # So that update does the check too
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class FramesSequenceND(FramesSequence):
    """ A base class defining a FramesSequence with an arbitrary number of
    axes. In the context of this reader base class, dimensions like 'x', 'y',
    't' and 'z' will be called axes. Indices along these axes will be called
    coordinates.

    The properties `bundle_axes`, `iter_axes`, and `default_coords` define
    to which coordinates each index points. See below for a description of
    each attribute.

    Subclassed readers only need to define `pixel_type` and `__init__`. At least
    one reader method needs to be registered as such using
    `self._register_get_frame(method, <list of axes>)`.
    In the `__init__`, axes need to be initialized using `_init_axis(name, size)`.
    It is recommended to set default values to `bundle_axes` and `iter_axes`.

    The attributes `__len__`, `get_frame`, and the attributes below are defined
    by this base_class; these should not be changed by derived classes.

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
        The last element in will iterate fastest. Default [].
    bundle_axes : iterable of strings
        This determines which axes will be bundled into one Frame. The axes in
        the ndarray that is returned by get_frame have the same order as the
        order in this list. Default ['y', 'x'].
    default_coords: dict of int
        When an axis is not present in both iter_axes and bundle_axes, the
        coordinate contained in this dictionary will be used. Default 0 for each.

    Examples
    --------
    >>> class DummyReaderND(FramesSequenceND):
    ...    @property
    ...    def pixel_type(self):
    ...        return 'uint8'
    ...    def __init__(self, shape, **axes):
    ...        super(DummyReaderND, self).__init__()  # properly initialize
    ...        self._init_axis('y', shape[0])
    ...        self._init_axis('x', shape[1])
    ...        for name in axes:
    ...            self._init_axis(name, axes[name])
    ...        self._register_get_frame(self.get_frame_2D, 'yx')
    ...        self.bundle_axes = 'yx'  # set default value
    ...        if 't' in axes:
    ...            self.iter_axes = 't'  # set default value
    ...    def get_frame_2D(self, **ind):
    ...        return np.zeros((self.sizes['y'], self.sizes['x']),
    ...                        dtype=self.pixel_type)

    >>> frames = MDummy((64, 64), t=80, c=2, z=10, m=5)
    >>> frames.bundle_axes = 'czyx'
    >>> frames.iter_axes = 't'
    >>> frames.default_coords['m'] = 3
    >>> frames[5]  # returns Frame at T=5, M=3 with shape (2, 10, 64, 64)
    """
    def __init__(self):
        self._clear_axes()
        self._get_frame_dict = dict()

    def _register_get_frame(self, method, axes):
        axes = tuple([a for a in axes])
        if not hasattr(self, '_get_frame_dict'):
            warn("Please call FramesSequenceND.__init__() at the start of the"
                 "the reader initialization.")
            self._get_frame_dict = dict()
        self._get_frame_dict[axes] = method

    def _clear_axes(self):
        self._sizes = {}
        self._default_coords = DefaultCoordsDict()
        self._iter_axes = []
        self._bundle_axes = ['y', 'x']
        self._get_frame_wrapped = None

    def _init_axis(self, name, size, default=0):
        # check if the axes have been initialized, if not, do it here
        if not hasattr(self, '_sizes'):
            warn("Please call FramesSequenceND.__init__() at the start of the"
                 "the reader initialization.")
            self._clear_axes()
            self._get_frame_dict = dict()
        if name in self._sizes:
            raise ValueError("axis '{}' already exists".format(name))
        self._sizes[name] = int(size)
        self.default_coords.axes = self.axes
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
        """ This determines which axes will be bundled into one Frame.
        The ndarray that is returned by get_frame has the same axis order
        as the order of `bundle_axes`.
        """
        return self._bundle_axes[:]  # return a copy

    @bundle_axes.setter
    def bundle_axes(self, value):
        value = list(value)
        invalid = [k for k in value if k not in self._sizes]
        if invalid:
            raise ValueError("axes %r do not exist" % invalid)

        for k in value:
            if k in self._iter_axes:
                del self._iter_axes[self._iter_axes.index(k)]

        self._bundle_axes = value
        if not hasattr(self, '_get_frame_dict'):
            warn("Please call FramesSequenceND.__init__() at the start of the"
                 "the reader initialization.")
            self._get_frame_dict = dict()
        if len(self._get_frame_dict) == 0:
            if hasattr(self, 'get_frame_2D'):
                # include get_frame_2D for backwards compatibility
                self._register_get_frame(self.get_frame_2D, 'yx')
            else:
                raise RuntimeError('No reader methods found. Register a reader '
                                   'method with _register_get_frame')

        # update the get_frame method
        get_frame = _make_get_frame(self._bundle_axes, self._get_frame_dict,
                                    self.sizes, self.pixel_type)
        self._get_frame_wrapped = get_frame

    @property
    def iter_axes(self):
        """ This determines which axes will be iterated over by the
        FramesSequence. The last element will iterate fastest. """
        return self._iter_axes[:]  # return a copy

    @iter_axes.setter
    def iter_axes(self, value):
        value = list(value)
        invalid = [k for k in value if k not in self._sizes]
        if invalid:
            raise ValueError("axes %r do not exist" % invalid)

        for k in value:
            if k in self._bundle_axes:
                del self._bundle_axes[self._bundle_axes.index(k)]

        self._iter_axes = value

    @property
    def default_coords(self):
        """ When a axis is not present in both iter_axes and bundle_axes, the
        coordinate contained in this dictionary will be used. """
        return self._default_coords  # this is a custom dict (DefaultCoordsDict)

    @default_coords.setter
    def default_coords(self, value):
        self._default_coords.update(**value)

    def get_frame(self, i):
        """ Returns a Frame of shape determined by bundle_axes. The index value
        is interpreted according to the iter_axes property. Coordinates not
        present in both iter_axes and bundle_axes will be set to their default
        value (see default_coords). """
        if i > len(self):
            raise IndexError('index out of range')
        if self._get_frame_wrapped is None:
            self.bundle_axes = tuple(self.bundle_axes)  # kick bundle_axes

        # start with the default coordinates
        coords = self.default_coords.copy()

        # list sizes of iteration axes
        iter_sizes = [self._sizes[k] for k in self.iter_axes]
        # list how much i has to increase to get an increase of coordinate n
        iter_cumsizes = np.append(np.cumprod(iter_sizes[::-1])[-2::-1], 1)
        # calculate the coordinates and update the coords dictionary
        iter_coords = (i // iter_cumsizes) % iter_sizes
        coords.update(**{k: v for k, v in zip(self.iter_axes, iter_coords)})

        result = self._get_frame_wrapped(**coords)
        if hasattr(result, 'metadata'):
            metadata = result.metadata
        else:
            metadata = dict()

        metadata_axes = set(self.axes) - set(self.bundle_axes)
        metadata_coords = {ax: coords[ax] for ax in metadata_axes}
        metadata.update(dict(axes=self.bundle_axes, coords=metadata_coords))
        return Frame(result, frame_no=i, metadata=metadata)

    def __repr__(self):
        s = "<FramesSequenceND>\nAxes: {0}\n".format(self.ndim)
        for dim in self._sizes:
            s += "Axis '{0}' size: {1}\n".format(dim, self._sizes[dim])
        s += """Pixel Datatype: {dtype}""".format(dtype=self.pixel_type)
        return s

