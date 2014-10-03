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
        if self.ndim == 2:
            w = 500
            h = self.shape[0] * w // self.shape[1]
            ptp = self.max() - self.min()
            # Handle edge case of a flat image.
            if ptp == 0:
                ptp = 1
            x = (self - self.min()) / ptp
            img = Image.fromarray((x * 256).astype('uint8')).resize((w, h))
            img_buffer = BytesIO()
            img.save(img_buffer, format='png')
            return img_buffer.getvalue()
        elif self.ndim == 3 and self.shape[0] > 4:
            return self.gen3DIpythonObject()
        else:
            return None
            
    def gen3DIpythonObject(self):
        from PIL import Image
        from os.path import join
        from IPython.display import HTML, Javascript, display
        from random import random #to force browser refresh of images
        
        
        savepath = 'IPython Notebooks\zstacks'
         #point this to ipython notebook folder
        
        d = self.shape[0]
        w = min(512,self.shape[2])
        h = self.shape[1] * w // self.shape[2]
        x = (self - self.min()) / (self.max() - self.min())
        
        for n in xrange(self.shape[0]):
            img = Image.fromarray((x[n] * 256).astype('uint8')).resize((w,h))
            img.save(join(savepath,'image'+str(n)+'.png'), format='png')
            
            
        # The piece of HTML/JS was adapted from http://codepen.io/will-moore/pen/Beuyc    
        mainimg = ''
        for n in xrange(d):
            mainimg = mainimg + '<img src="zstacks/image'+str(n)+'.png?' + str(random()) + '" class="large_image"/>'
        
        html = """<div id="large_img_container">
            """ + mainimg + """    
        </div>
        <p id="indicator">0</p>
        """
        
        css = """<style media="screen" type="text/css">
        body {
            font-family: arial;
        }
        
        #large_img_container {
            position:relative;
            float: left;
            width:""" + str(w) + """px;
            height:""" + str(h) + """px;
            overflow:hidden;
            Cache-Control: no-store
        }
        
        .large_image {
            position:absolute;
            top:0px; left:0px;
            display: none;
        }
        </style>
        """
        
        js = """
        (function($){
            $(function() {
        
            var SRC_ROOT = "zstacks/",
                SIZE_X = """ + str(w) + """,
                SIZE_Y = """ + str(h) + """,
                SIZE_Z = """ + str(d) + """,
                large_image_styles = [],
                z_index = 0;
                
            $(".large_image").each(function() {
                large_image_styles.push(this.style);
            });
            
            // show the specified plane in the main viewer
            var show_plane = function(theZ) {
                if (theZ < 0 || theZ >= SIZE_Z) return;
                z_index = theZ
                // hide all planes...
                for (var i=0; i<large_image_styles.length; i++){
                    large_image_styles[i].display = 'none';
                }
                large_image_styles[z_index].display = 'block';
                $("#indicator").text(z_index);
            }
            
            var onMouseWheel = function(e) {
                e = e.originalEvent;
                var incr = e.wheelDelta>0||e.detail<0?1:-1;
                show_plane(z_index + incr);
                return false;
            }
            
            $("#large_img_container").bind("mousewheel DOMMouseScroll", onMouseWheel);
        
            // finally, start by showing the first plane
            show_plane(""" + str(d//2) + """);
        });
        })(jQuery);
        """
        return display(HTML(css + html),Javascript(js))
