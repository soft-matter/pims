from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from numpy import ndarray, asarray
from pims.export import _scrollable_stack, _as_png


WIDTH = 500  # width of rich display, in pixels


class Frame(ndarray):
    "Extends a numpy array with meta information"
    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, input_array, frame_no=None, metadata=None):
        obj = asarray(input_array).view(cls)
        obj.frame_no = frame_no
        if metadata is None:
            metadata = {}
        obj.metadata = metadata
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

    def _repr_png_(self):
        try:
            from PIL import Image
        except ImportError:
            # IPython will show this exception as a warning unless
            # _repr_png_() is explicitly called.
            raise ImportError("Install PIL or Pillow to enable "
                              "rich display of Frames.")
        return _as_png(self, WIDTH)

    def _repr_html_(self):
        from jinja2 import Template
        # If Frame is 2D, display as a plain image.
        # We have to build the image tag ourselves; _repr_html_ expects HTML.
        has_color_channels = (3 in self.shape) or (4 in self.shape)
        if self.ndim == 2 or (self.ndim == 3 and has_color_channels):
            tag = Template('<img src="data:image/png;base64,{{data}}" '
                           'style="width: {{width}}" />')
            return tag.render(data=_as_png(self, WIDTH).encode('base64'),
                              width=WIDTH)
        # If Frame is 3D, display as a scrollable stack.
        elif self.ndim == 3 or (self.ndim == 4 and has_color_channels):
            return _scrollable_stack(self, width=WIDTH)
        else:
            # This exception will be caught by IPython and displayed
            # as a FormatterWarning.
            raise ValueError("No rich representation is available for "
                             "{0}-dimensional Frames".format(self.ndim))


