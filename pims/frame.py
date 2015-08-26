from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from base64 import b64encode
import six

from numpy import ndarray, asarray
from pims.display import _scrollable_stack, _as_png, to_rgb


WIDTH = 512  # width of rich display, in pixels
MAX_HEIGHT = 512  # maximum height of rich display, in pixels
MAX_STACK_DEPTH = 128  # max stack count of scrollable stack (for 3D images)


class Frame(ndarray):
    "Extends a numpy array with meta information"
    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, input_array, frame_no=None, metadata=None):
        # get a view of the input data as a Frame object
        obj = asarray(input_array).view(cls)

        # if no frame number is passed in, see if the input array has one
        if frame_no is None and hasattr(input_array, 'frame_no'):
            frame_no = getattr(input_array, 'frame_no')

        obj.frame_no = frame_no
        # check if the input object _has_ a metadata attribute
        if hasattr(input_array, 'metadata'):
            # and get a local (shallow) copy
            arr_metadata = dict(getattr(input_array, 'metadata'))
        else:
            # else, empty dict
            arr_metadata = dict()

        # validation on input
        if metadata is None:
            metadata = {}

        # override meta-data on input object with explicitly passed in metadata
        arr_metadata.update(metadata)

        # assign to the output
        obj.metadata = arr_metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.frame_no = getattr(obj, 'frame_no', None)
        self.metadata = getattr(obj, 'metadata', None)

    def __array_wrap__(self, out_arr, context=None):
        # Handle scalars so as not to break ndimage.
        # See http://stackoverflow.com/a/794812/1221924
        if out_arr.ndim == 0:
            return out_arr[()]
        return ndarray.__array_wrap__(self, out_arr, context)

    def __reduce__(self):
        """Necessary for making this object picklable"""
        object_state = list(ndarray.__reduce__(self))
        saved_attr = ['frame_no', 'metadata']
        subclass_state = {a: getattr(self, a) for a in saved_attr}
        object_state[2] = (object_state[2], subclass_state)
        return tuple(object_state)

    def __setstate__(self, state):
        """Necessary for making this object picklable"""
        nd_state, own_state = state
        ndarray.__setstate__(self, nd_state)

        for attr, val in own_state.items():
            setattr(self, attr, val)

    def _repr_html_(self):
        from jinja2 import Template
        # Identify whether image is multichannel, convert to rgb if necessary
        try:
            # if colors field exists, check if size agrees with image shape
            colors = self.metadata['colors']
            is_multichannel = len(colors) == 1 or len(colors) == self.shape[0]         
        except KeyError or AttributeError:
            # if colors field does not exist, guess from image shape
            colors = None
            is_multichannel = self.ndim > 2 and self.shape[0] < 5
        if is_multichannel:
            image = to_rgb(self, colors, False)
            has_color_channels = True
        else:
            image = self
            has_color_channels = image.shape[-1] == 3 or image.shape[-1] == 4
        # If Frame is 2D, display as a plain image.
        # We have to build the image tag ourselves; _repr_html_ expects HTML.
        if image.ndim == 2 or (image.ndim == 3 and has_color_channels):
            width = WIDTH
            if ((image.shape[0] * width) // image.shape[1]) > MAX_HEIGHT:
                width = (image.shape[1] * MAX_HEIGHT) // image.shape[0]
            tag = Template('<img src="data:image/png;base64,{{data}}" '
                           'style="width: {{width}}" />')
            return tag.render(data=b64encode(_as_png(image,
                                                     width)).decode('utf-8'),
                              width=width)
        # If Frame is 3D, display as a scrollable stack.
        elif image.ndim == 3 or (image.ndim == 4 and has_color_channels):
            if image.shape[0] > MAX_STACK_DEPTH:          
                raise ValueError("For 3D images, pims is limited to a stack "
                                 "depth of {0}.".format(MAX_STACK_DEPTH))
            width = WIDTH
            if ((image.shape[1] * width) // image.shape[2]) > MAX_HEIGHT:
                width = (image.shape[2] * MAX_HEIGHT) // image.shape[1]
            return _scrollable_stack(image, width=width)
        else:
            # This exception will be caught by IPython and displayed
            # as a FormatterWarning.
            raise ValueError("No rich representation is available for "
                             "{0}-dimensional Frames".format(self.ndim))
