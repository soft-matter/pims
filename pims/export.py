from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import tempfile

def export(sequence, filename, rate=30, bitrate=None,
           width=None, height=None, codec='mpeg4', format='yuv420p',
           autoscale=True):
    """Export a sequence of images as a standard video file.

    N.B. If the quality and detail are insufficient, increase the
    bitrate.

    Parameters
    ----------
    sequence : any iterator or array of array-like images
        The images should have two dimensions plus an
        optional third dimensions representing color.
    filename : string
        name of output file
    rate : integer
        frame rate of output file, 30 by default
    bitrate : integer
        Video bitrate is crudely guessed if None is given.
    width : integer
        By default, set the width of the images.
    height : integer
        By default, set the  height of the images. If width is specified
        and height is not, the height is autoscaled to maintain the aspect
        ratio.
    codec : string
        a valid video encoding, 'mpeg4' by default
    format: string
        Video stream format, 'yuv420p' by default.
    autoscale : boolean
        Linearly rescale the brightness to use the full gamut of black to
        white values. If the datatype of the images is not 'uint8', this must
        be set to True, as it is by default.
        
    """
    try:
        import av
    except ImportError:
        raise("This feature requires PyAV with FFmpeg or libav installed.")
    output = av.open(filename, 'w')
    stream = output.add_stream(bytes(codec), rate)
    if bitrate is None:
        # Estimate a workabout bitrate.
        bitrate = int(128000 * rate / 30)
    stream.bit_rate = int(bitrate)
    stream.pix_fmt = bytes(format)

    ndim = None
    for frame_no, img in enumerate(sequence):
        if not frame_no:
            # Inspect first frame to set up stream.
            if width is None:
                stream.height = img.shape[0]
                stream.width = img.shape[1]
            else:
                stream.width = width
                stream.height = (height or
                                width * img.shape[0] // img.shape[1])
            ndim = img.ndim

        if ndim == 3:
            if img.shape.count(3) != 1:
                raise ValueError("Images have the wrong shape.")
            # This is a color image. Ensure that the color axis is axis 2.
            color_axis = img.shape.index(3)
            img = np.rollaxis(img, color_axis, 3)
        elif ndim == 2:
            # Expand into color to satisfy PyAV's expectation that images
            # be in color. (Without this, an assert is tripped.)
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        else:
            raise ValueError("Images have the wrong shape.")

        # PyAV requires uint8.
        if img.dtype is not np.uint8 and (not autoscale):
            raise ValueError("Autoscaling must be turned on if the image "
                             "data type is not uint8. Convert the datatype "
                             "manually if you want to turn off autoscale.")
        if autoscale:
            normed = (img - img.min()) / (img.max() - img.min())
            img = (255 * normed).astype('uint8')

        frame = av.VideoFrame.from_ndarray(np.asarray(img), format=b'bgr24')
        packet = stream.encode(frame)
        output.mux(packet)

    output.close()


def play(sequence, rate=30, bitrate=None,
         width=None, height=None, autoscale=True):
    """In an IPython notebook, display a sequence of images as
    an embedded video.

    N.B. If the quality and detail are insufficient, increase the
    bit rate.

    Parameters
    ----------
    sequence : any iterator or array of array-like images
        The images should have two dimensions plus an
        optional third dimensions representing color.
    rate : integer
        frame rate of output file, 30 by default
    bitrate : integer
        Video bitrate is crudely guessed if None is given.
    width : integer
        By default, set the width of the images.
    height : integer
        By default, set the  height of the images. If width is specified
        and height is not, the height is autoscaled to maintain the aspect
        ratio.
    autoscale : boolean
        Linearly rescale the brightness to use the full gamut of black to
        white values. If the datatype of the images is not 'uint8', this must
        be set to True, as it is by default.
        
    """
    try:
        from IPython.display import display
    except ImportError:
        raise ImportError("This feature requires IPython.")
    with tempfile.NamedTemporaryFile(suffix='.webm') as temp:
        export(sequence, bytes(temp.name), codec='libvpx', rate=rate, 
               width=width, height=height, bitrate=bitrate, format='yuv420p',
               autoscale=True)
        temp.flush()
        display(repr_video(temp.name, 'x-webm'))


def repr_video(fname, mimetype):
    """Load the video in the file `fname`, with given mimetype,
    and display as HTML5 video.
    """
    try:
       from IPython.display import HTML
    except ImportError:
        raise ImportError("This feature requires IPython.")
    video_encoded = open(fname, "rb").read().encode("base64")

    video_tag = """<video controls>
<source alt="test" src="data:video/{0};base64,{1}" type="video/webm">
Use Google Chrome browser.</video>""".format(mimetype, video_encoded)
    return HTML(data=video_tag)
