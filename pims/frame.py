from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from io import BytesIO

from numpy import ndarray, asarray


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
        return _as_png(self)

    def _repr_html_(self):
        from IPython.display import Javascript, HTML, display_png
        from jinja2 import Template

        # Only IPython will call _repr_html_ and we can count on it
        # to have the required deps or fail gracefully.
        # Therefore, we define the templates here.
        SCROLL_STACK_JS = Template("""
require(['jquery'], function() {
  if (!(window.PIMS)) {
    var stack_cursors = {};
    window.PIMS = {stack_cursors: {}};
  }
  $('#stack-{{stack_id}}-slice-0').css('display', 'block');
  window.PIMS.stack_cursors['{{stack_id}}'] = 0;
});

require(['jquery'],
$('#image-stack-{{stack_id}}').bind('mousewheel DOMMouseScroll', function(e) {
  var direction;
  var cursor = window.PIMS.stack_cursors['{{stack_id}}'];
  e.preventDefault();
  if (e.type == 'mousewheel') {
    direction = e.originalEvent.wheelDelta < 0;
  }
  else if (e.type == 'DOMMouseScroll') {
    direction = e.originalEvent.detail < 0;
  }
  var delta = direction * 2 - 1;
  if (cursor + delta < 0) {
    return;
  }
  else if (cursor + delta > {{length}} - 1) {
    return;
  }
  $('#stack-{{stack_id}}-slice-' + cursor).css('display', 'none');
  $('#stack-{{stack_id}}-slice-' + (cursor + delta)).css('display', 'block');
  window.PIMS.stack_cursors['{{stack_id}}'] = cursor + delta;
}));""")
        TAG = Template('<img src="data:image/png;base64,{{data}}" '
                       'style="display: none;" '
                       'id="stack-{{stack_id}}-slice-{{i}}" />')
        WRAPPER = Template('<div id="image-stack-{{stack_id}}", style="'
                           '"width: {{width}}; float: left; display: inline;">')

        # If Frame is 2D, display as a plain image.
        # We have to build the image tag ourselves: _repr_html_ expects HTML.
        has_color_channels = (3 in self.shape) or (4 in self.shape)
        if self.ndim == 2 or (self.ndim == 3 and has_color_channels):
            tag = Template('<img src="data:image/png;base64,{{data}}" />')
            return tag.render(data=_as_png(self).encode('base64'))
        # If Frame is 3D, display as a scrollable stack.
        elif self.ndim == 3 or (self.ndim == 4 and has_color_channels):
            stack_id = str(id(self))  # TODO Be more specific.
            js = SCROLL_STACK_JS.render(length=len(self), stack_id=stack_id)
            output = '<script>{0}</script>'.format(js)
            output += WRAPPER.render(width=WIDTH, stack_id=stack_id)
            for i, s in enumerate(self):
                output += TAG.render(data=_as_png(s).encode('base64'),
                                     stack_id=stack_id, i=i)
            output += "</div>"
            return output
        else:
            # This exception will be caught by IPython and displayed
            # as a FormatterWarning.
            raise ValueError("No rich representation is available for "
                             "{0}-dimensional Frames".format(self.ndim))


def _as_png(arr):
    from PIL import Image
    w = WIDTH  # for brevity
    h = arr.shape[0] * w // arr.shape[1]
    ptp = arr.max() - arr.min()
    # Handle edge case of a flat image.
    if ptp == 0:
        ptp = 1
    scaled_arr = (arr - arr.min()) / ptp
    img = Image.fromarray((scaled_arr * 256).astype('uint8')).resize((w, h))
    img_buffer = BytesIO()
    img.save(img_buffer, format='png')
    return img_buffer.getvalue()
