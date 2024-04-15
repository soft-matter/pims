from base64 import b64encode

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

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        # Handle scalars so as not to break ndimage.
        # See http://stackoverflow.com/a/794812/1221924
        if out_arr.ndim == 0:
            return out_arr[()]
        return ndarray.__array_wrap__(self, out_arr, context, return_scalar)

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
        try:
            from PIL import Image
        except ImportError:
            raise ImportError('Rich display in IPython requires PIL/Pillow.')
        ndim = self.ndim
        shape = self.shape
        image = self
        try:
            colors = self.metadata['colors']
            if len(colors) != shape[0]:
                colors = None
        except KeyError or AttributeError:
            colors = None

        # 2D, grayscale
        if ndim == 2:
            stack = False
        # 2D, has colors attribute
        elif ndim == 3 and colors is not None:
            stack = False
            image = to_rgb(image, colors, False)
        # 2D, RGB
        elif ndim == 3 and shape[2] in [3, 4]:
            stack = False
        # 2D, is multichannel
        elif ndim == 3 and shape[0] < 5:  # guessing; could be small z-stack
            stack = False
            image = to_rgb(image, None, False)
        # 3D, grayscale
        elif ndim == 3:
            stack = True
        # 3D, has colors attribute
        elif ndim == 4 and colors is not None:
            stack = True
            image = to_rgb(image, colors, False)
        # 3D, RGB
        elif ndim == 4 and shape[3] in [3, 4]:
            stack = True
        # 3D, is multichannel
        elif ndim == 4 and shape[0] < 5:
            stack = True
            image = to_rgb(image, None, False)
        else:
            # This exception will be caught by IPython and displayed
            # as a FormatterWarning.
            raise ValueError("No rich representation is available for "
                             "frames of shape {0}".format(shape))

        # Calculate display width
        if stack: # z, y, x[, c]
            frame_shape = shape[1:3]
        else: # y, x[, c]
            frame_shape = shape[:2]
        width = WIDTH
        if ((frame_shape[0] * width) // frame_shape[1]) > MAX_HEIGHT:
            width = (frame_shape[1] * MAX_HEIGHT) // frame_shape[0]

        # If Frame is 2D, display as a plain image.
        # We have to build the image tag ourselves; _repr_html_ expects HTML.
        if not stack:
            tag = Template('<img src="data:image/png;base64,{{data}}" '
                           'style="width: {{width}}" />')
            return tag.render(data=b64encode(_as_png(image,
                                                     width)).decode('utf-8'),
                              width=width)
        # If Frame is 3D, display as a scrollable stack.
        else:
            return _scrollable_stack(image, width=width)
