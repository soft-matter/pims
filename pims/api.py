from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from slicerator import pipeline
from pims.base_frames import FramesSequence, FramesSequenceND
from pims.frame import Frame
from pims.display import (export, play, scrollable_stack, to_rgb, normalize,
                          plot_to_frame, plots_to_frame)

import six
import glob
import os
from warnings import warn

# has to be here for API stuff
from pims.image_sequence import ImageSequence, ImageSequenceND  # noqa
from .cine import Cine  # noqa
from .norpix_reader import NorpixSeq  # noqa
from pims.tiff_stack import TiffStack_tifffile  # noqa
from .spe_stack import SpeStack


def not_available(requirement):
    def raiser(*args, **kwargs):
        raise ImportError(
            "This reader requires {0}.".format(requirement))
    return raiser

try:
    import pims.pyav_reader
    if pims.pyav_reader.available():
        Video = pims.pyav_reader.PyAVVideoReader
    else:
        raise ImportError()
except (ImportError, IOError):
    Video = not_available("PyAV")

import pims.tiff_stack
from pims.tiff_stack import (TiffStack_pil, TiffStack_libtiff,
                                TiffStack_tifffile)
# First, check if each individual class is available
# and drop in placeholders as needed.
if not pims.tiff_stack.tifffile_available():
    TiffStack_tiffile = not_available("tifffile")
if not pims.tiff_stack.libtiff_available():
    TiffStack_libtiff = not_available("libtiff")
if not pims.tiff_stack.PIL_available():
    TiffStack_pil = not_available("PIL or Pillow")
# Second, decide which class to assign to the
# TiffStack alias.
if pims.tiff_stack.tifffile_available():
    TiffStack = TiffStack_tifffile
elif pims.tiff_stack.libtiff_available():
    TiffStack = TiffStack_libtiff
elif pims.tiff_stack.PIL_available():
    TiffStack = TiffStack_pil
else:
    TiffStack = not_available("tifffile, libtiff, or PIL/Pillow")


try:
    import pims.bioformats
    if pims.bioformats.available():
        Bioformats = pims.bioformats.BioformatsReader
    else:
        raise ImportError()
except (ImportError, IOError):
    BioformatsRaw = not_available("JPype")
    Bioformats = not_available("JPype")


def open(sequence, process_func=None, dtype=None, as_grey=False, plugin=None):
    """Read a filename, list of filenames, or directory of image files into an
    iterable that returns images as numpy arrays.

    Parameters
    ----------
    sequence : string, list of strings, or glob
       The sequence you want to load. This can be a directory containing
       images, a glob ('/path/foo*.png') pattern of images,
       a video file, or a tiff stack
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
    files = glob.glob(sequence)
    if len(files) > 1:
        # todo: test if ImageSequence can read the image type,
        #       delegate to subclasses as needed
        return ImageSequence(sequence, process_func, dtype, as_grey, plugin)

    # We are now not in an image sequence, so warn if plugin is specified,
    # since we will not be able to use it
    if plugin is not None:
        warn("scikit-image plugin specification ignored because such plugins "
             "only apply when loading a sequence of image files. ")
    _, ext = os.path.splitext(sequence)
    if ext is None or len(ext) < 2:
        raise UnknownFormatError(
            "Could not detect your file type because it did not have an "
            "extension. Try specifying a loader class, e.g. "
            "Video({0})".format(sequence))
    ext = ext.lower()[1:]

    all_handlers = _recursive_subclasses(FramesSequence)
    eligible_handlers = [h for h in all_handlers
                         if ext and ext in h.class_exts()]
    if len(eligible_handlers) < 1:
        raise UnknownFormatError(
            "Could not autodetect how to load a file of type {0}. "
            "Try manually "
            "specifying a loader class, e.g. Video({1})".format(ext, sequence))

    def sort_on_priority(handlers):
        # This uses optional priority information from subclasses
        # > 10 means that it will be used instead of than built-in subclasses
        def priority(cls):
            try:
                return cls.class_priority
            except AttributeError:
                return 10
        return sorted(handlers, key=priority, reverse=True)

    handler = sort_on_priority(eligible_handlers)[0]

    # TODO maybe we should wrap this in a try and loop to try all the
    # handlers if early ones throw exceptions
    return handler(sequence, process_func=process_func,
                   dtype=dtype, as_grey=as_grey)


class UnknownFormatError(Exception):
    pass


def _recursive_subclasses(cls):
    "Return all subclasses (and their subclasses, etc.)."
    # Source: http://stackoverflow.com/a/3862957/1221924
    return (cls.__subclasses__() +
        [g for s in cls.__subclasses__() for g in _recursive_subclasses(s)])
