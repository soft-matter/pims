from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from io import BytesIO

from numpy import ndarray, asarray


class Frame(ndarray):
    "Extends a numpy array with meta information"
    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, input_array, frame_no=None, metadata=None):
        obj = asarray(input_array).view(cls)
        obj.frame_no = frame_no
        obj.metadata = {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
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
        from PIL import Image
        w = 500
        h = self.shape[0] * w // self.shape[1] 
        x = asarray(Image.fromarray(self).resize((w, h)))
        x = (x - x.min()) / (x.max() - x.min())
        img = Image.fromarray((x*256).astype('uint8'))
        img_buffer = BytesIO()
        img.save(img_buffer, format='png')
        return img_buffer.getvalue()
