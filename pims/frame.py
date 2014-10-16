from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from io import BytesIO
from os import getcwd
from numpy import ndarray, asarray


class Frame(ndarray):
    "Extends a numpy array with meta information"
    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    iPythonWorkingFolder = getcwd()
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
        if self.ndim == 2 or (self.ndim == 3 and self.shape[0] <= 4):
            try:
                from PIL import Image
            except ImportError:
                # IPython will show this exception as a warning unless
                # _repr_png_() is explicitly called.
                raise ImportError("Install PIL or Pillow to enable "
                                  "rich display of Frames.")
            w = 500
            h = self.shape[0] * w // self.shape[1]
            ptp = self.max() - self.min()
            # Handle edge case of a flat image.
            if ptp == 0:
                ptp = 1
            x = (self - self.min()) / ptp
            img = Image.fromarray((x * 255).astype('uint8')).resize((w, h))
            img_buffer = BytesIO()
            img.save(img_buffer, format='png')
            return img_buffer.getvalue()
            
    def _repr_html_(self):
        if self.ndim == 3 and self.shape[0] > 4:
            self.image3D()
            return '<p>IndexT: ' + str(self.frame_no) + '</p>'
                   
    def image3D(self, annotate_func = None, annotate_args = {}):
        """
        Displays a scrollable z-stack, based on a combined HTML and JavaScript 
        iPython object. Working folder is saved at package load, to make sure
        this is the iPython root. In root\_temp_zstacks, temporary png images
        will be saved and displayed by the HTML/JavaScript object.
        
        If annotate_func is passed, this function is applied to each stack.
        Function has to accept:
            - image: 2D ndarray of 8-bit integers
            - z: index of z plane, to be used for annotation
            - frame_no: frame number, to be used for annotation
            - ax: axis object that will be used, in order to allow image3D to
                  suppress direct display
            - imshow_style: kwargs to be passed to pyplot.imshow
            
        The full stack is normalized, instead of all pictures separately.
        """
        
        from PIL import Image
        from os.path import join, exists
        from IPython.display import HTML, Javascript, display
        from random import random #to force browser refresh of images
        
        savepath = self.iPythonWorkingFolder + '\\_temp_zstacks'

        if not exists(savepath):
            print("Temp folder: " + savepath)
            raise IOError("Trying to use subfolder '_temp_zstacks' in iPython notebook " 
                          "folder. If it already exists: make sure pims is "
                          "loaded before changing the working directory.")

        annotate = not (annotate_func == None)

        d = self.shape[0]
        w = min(512, self.shape[2])
        h = self.shape[1] * w // self.shape[2]
        x = (((self - self.min()) / (self.max() - self.min())) * 255).astype('uint8')
        annotate_args.update({'imshow_style': {'vmin': 0, 'vmax': 255}})
        
        if 'ax' in annotate_args: del annotate_args['ax']
            
        for n in range(self.shape[0]):
            if annotate: 
                import matplotlib.pyplot as plt
                fig = plt.figure(0)
                ax = fig.add_subplot(111)
                ax = annotate_func(image=x[n], z=n, frame_no=self.frame_no, 
                                   ax=ax, **annotate_args)
                fig.savefig(join(savepath,'image'+str(n)+'.png'))
                if n == 0:
                    w, h = fig.get_dpi()*fig.get_size_inches()   
                plt.close(0)
            else:
                img = Image.fromarray(x[n]).resize((w, h))
                img.save(join(savepath,'image'+str(n)+'.png'), format='png')
        
        # The piece of HTML/JS was adapted from http://codepen.io/will-moore/pen/Beuyc    
        mainimg = ''
        for n in range(d):
            mainimg = mainimg + '<img src="_temp_zstacks/image'+str(n)+'.png?'+\
                                    str(random()) + '" class="XY_image"/>'
        
        html = """<div id="XY_container">
            """ + mainimg + """    
        </div>
        <div><p id="indicator">IndexZ: 0</p></div>
        """
        
        css = """<style media="screen" type="text/css">        
        #XY_container {
            position: relative;
            float: left;
            width:""" + str(w) + """px;
            height:""" + str(h) + """px;
            overflow: hidden;
            Cache-Control: no-store
        }
        
        .XY_image {
            position: absolute;
            top: 0px; left: 0px;
            display: none;
        }
        </style>
        """
        
        js = """
        (function($){
            $(function() {
        
            var Zcount = """ + str(d) + """,
                XY_image_styles = [],
                Zactive = """ + str(d//2) + """;
                
            $(".XY_image").each(function() {
                XY_image_styles.push(this.style);
            });
            
            // show the specified plane in the main viewer
            var show_plane = function(Z) {
                if (Z < 0 || Z >= Zcount) return;
                Zactive = Z
                // hide all planes...
                for (var i = 0; i < XY_image_styles.length; i++){
                    XY_image_styles[i].display = 'none';
                }
                XY_image_styles[Zactive].display = 'block';
                $("#indicator").text('IndexZ: ' + Zactive);
            }
            
            var onMouseWheel = function(e) {
                e = e.originalEvent;
                var incr = e.wheelDelta>0||e.detail<0?1:-1;
                show_plane(Zactive + incr);
                return false;
            }
            
            $("#XY_container").bind("mousewheel DOMMouseScroll", onMouseWheel);
        
            // finally, start by showing the first plane
            show_plane(Zactive);
        });
        })(jQuery);
        """
        display(HTML(css + html),Javascript(js))
        return None
