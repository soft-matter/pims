from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six import with_metaclass
from six.moves import xrange
import os
import numpy as np
import collections
import itertools
from .frame import Frame
from abc import ABCMeta, abstractmethod, abstractproperty


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
            if ndim == 2:
                # The image is already greyscale.
                process_func = None
            elif ndim == 3 and (shape.count(3) == 1):
                reduced_shape = list(shape)
                reduced_shape.remove(3)
                self._im_sz = tuple(reduced_shape)
                calibration = [0.2125, 0.7154, 0.0721]
                def convert_to_grey(img):
                    color_axis = img.shape.index(3)
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


class FramesSequence(FramesStream):
    """Baseclass for wrapping data buckets that have random access.

    Support random access.

    Supports standard slicing and fancy slicing, but returns a
    generator.

    Must be finite length.

    """
    def __getitem__(self, key):
        """for data access"""
        _len = len(self)

        if isinstance(key, slice):
            # if input is a slice, return a generator
            return (self.get_frame(_k) for _k
                    in xrange(*key.indices(_len)))
        elif isinstance(key, collections.Iterable):
            # if the input is an iterable, doing 'fancy' indexing

            if isinstance(key, np.ndarray) and key.dtype == np.bool:
                # if we have a bool array, do the right thing
                return (self.get_frame(_k) for _k in np.arange(len(self))[key])
            if any(_k < -_len or _k >= _len for _k in key):
                raise IndexError("Keys out of range")
            # else, return a generator looping over the keys
            return (self.get_frame(_k if _k >= 0 else _len + _k)
                    for _k in key)
        else:
            if key < -_len or key >= _len:
                raise IndexError("Key out of range")

            # else, fall back to `get_frame`
            return self.get_frame(key if key >= 0 else _len + key)

    def __iter__(self):
        return self[:]

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
