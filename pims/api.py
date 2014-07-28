from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import glob
import os
from warnings import warn

# has to be here for API stuff
from pims.image_sequence import ImageSequence  # noqa


def not_available(requirement):
    def raiser(*args, **kwargs):
        raise ImportError(
            "This reader requires {0}.".format(requirement))
    return raiser

try:
    import pims.ffmpeg_reader
    if pims.ffmpeg_reader.available():
        Video = pims.ffmpeg_reader.FFmpegVideoReader
    else:
        raise ImportError()
except (ImportError, IOError):
    Video = not_available("ffmpeg")

try:
    import pims.tiff_stack
    from pims.tiff_stack import TiffStack_pil, TiffStack_libtiff
    if pims.tiff_stack.libtiff_available():
        TiffStack = TiffStack_libtiff
    elif pims.tiff_stack.PIL_available():
        TiffStack = TiffStack_pil
    else:
        raise ImportError()
except ImportError:
    TiffStack = not_available("libtiff or PIL/PILLOW")

def open(sequence, process_func=None, dtype=None, as_gray=False, plugin=None):
    """Read a directory of sequentially numbered image files into an
    iterable that returns images as numpy arrays.

    Parameters
    ----------
    sequence : string, list of strings, or glob
       The sequence you want to load. This can be a directory containing images,
       a glob of images, a video file, or a tiff stack
    process_func : function, optional
        callable with signalture `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    dtype : numpy datatype, optional
        Image arrays will be converted to this datatype.
    as_grey : boolean, optional
        Convert color images to greyscale. False by default.
        May not be used in conjection with process_func.
    plugin : string
        Passed on to skimage.io.imread if scikit-image is available.
        If scikit-image is not available, this will be ignored and a warning
        will be issued.

    Examples
    --------
    >>> video = open('path/to/images/*.png')  # or *.tif, or *.jpg
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """
    file = glob.glob(sequence)
    if len(sequence) > 1:
        return ImageSequence(files, process_func, dtype, as_gray, plugin)
    if plugin is not None:
        warn("scikit-image plugin specification ignored because such plugins"
             "only apply when loading a sequence of image files. ")
    _, ext = os.path.splitext(sequence)
    ext = ext[1:].lower()
    if ext in TiffStack.class_exts()
        return TiffStack(sequence, process_func, dtype, as_gray)
    if ext in Video.class_exts():
        return Video(sequence, process_func, dtype, as_array)

    raise UnknownFormatError("Could not autodetect how to load a file of type {0}. Try manually specifying a loader class, ie Video({1})".format(ext, sequence))

class UnknownFormatError(Error):
    def __init__(self, message = ""):
        self.msg = message
    def __str__(self):
        return self.message
