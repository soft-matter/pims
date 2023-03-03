from slicerator import pipeline
from pims.base_frames import FramesSequence, FramesSequenceND
from pims.frame import Frame
from pims.display import (export, play, scrollable_stack, to_rgb, normalize,
                          plot_to_frame, plots_to_frame)
from itertools import chain

import glob
import os
from warnings import warn

# has to be here for API stuff
from pims.image_sequence import ImageSequence, ImageSequenceND, ReaderSequence  # noqa
from pims.image_reader import ImageReader, ImageReaderND  # noqa
from .cine import Cine  # noqa
from .norpix_reader import NorpixSeq  # noqa
from pims.tiff_stack import TiffStack_tifffile  # noqa
from .spe_stack import SpeStack
from pims.process import as_grey, as_gray


def not_available(requirement):
    def raiser(*args, **kwargs):
        raise ImportError(
            "This reader requires {0}.".format(requirement))
    return raiser

if export is None:
    export = not_available("PyAV or MoviePy")

try:
    import pims.pyav_reader
    if pims.pyav_reader.available():
        PyAVReaderTimed = pims.pyav_reader.PyAVReaderTimed
        PyAVReaderIndexed = pims.pyav_reader.PyAVReaderIndexed
        Video = PyAVReaderTimed
    else:
        raise ImportError()
except (ImportError, IOError):
    PyAVVideoReader = not_available("PyAV")
    PyAVReaderTimed = not_available("PyAV")
    PyAVReaderIndexed = not_available("PyAV")
    Video = None

PyAVVideoReader = PyAVReaderTimed


try:
    import pims.imageio_reader
    if pims.imageio_reader.available():
        ImageIOReader = pims.imageio_reader.ImageIOReader
        if Video is None:
            if pims.imageio_reader.ffmpeg_available():
                Video = ImageIOReader
    else:
        raise ImportError()
except (ImportError, IOError):
    ImageIOReader = not_available("ImageIO")


try:
    import pims.moviepy_reader
    if pims.moviepy_reader.available():
        MoviePyReader = pims.moviepy_reader.MoviePyReader
        if Video is None:
            Video = MoviePyReader
    else:
        raise ImportError()
except (ImportError, IOError):
    MoviePyReader = not_available("MoviePy")

if Video is None:
    Video = not_available("PyAV, MoviePy, or ImageIO")

import pims.tiff_stack
from pims.tiff_stack import (TiffStack_pil, TiffStack_tifffile)
# First, check if each individual class is available
# and drop in placeholders as needed.
if not pims.tiff_stack.tifffile_available():
    TiffStack_tiffile = not_available("tifffile")
if not pims.tiff_stack.PIL_available():
    TiffStack_pil = not_available("PIL or Pillow")
# Second, decide which class to assign to the
# TiffStack alias.
if pims.tiff_stack.tifffile_available():
    TiffStack = TiffStack_tifffile
elif pims.tiff_stack.PIL_available():
    TiffStack = TiffStack_pil
else:
    TiffStack = not_available("tifffile or PIL/Pillow")


try:
    import pims.bioformats
    if pims.bioformats.available():
        Bioformats = pims.bioformats.BioformatsReader
    else:
        raise ImportError()
except (ImportError, IOError):
    BioformatsRaw = not_available("JPype")
    Bioformats = not_available("JPype")


try:
    from pims_nd2 import ND2_Reader as ND2Reader_SDK

    class ND2_Reader(ND2Reader_SDK):
        class_priority = 0

        def __init__(self, *args, **kwargs):
            warn("'ND2_Reader' has been renamed to 'ND2Reader_SDK' and will be"
                 "removed in future pims versions. "
                 "Please use the new name, or try out the pure-Python one named "
                 "`ND2Reader`.")
            super(ND2_Reader, self).__init__(*args, **kwargs)
except ImportError:
    ND2Reader_SDK = not_available("pims_nd2")
    ND2_Reader = not_available("pims_nd2")

try:
    from nd2reader import ND2Reader
except ImportError:
    ND2Reader = not_available("nd2reader")

def open(sequence, **kwargs):
    """Read a filename, list of filenames, or directory of image files into an
    iterable that returns images as numpy arrays.

    Parameters
    ----------
    sequence : string, list of strings, or glob
        The sequence you want to load. This can be a directory containing
        images, a glob ('/path/foo*.png') pattern of images,
        a video file, or a tiff stack
    kwargs :
        All keyword arguments will be passed to the reader.

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
        return ImageSequence(sequence, **kwargs)

    _, ext = os.path.splitext(sequence)
    if ext is None or len(ext) < 2:
        raise UnknownFormatError(
            "Could not detect your file type because it did not have an "
            "extension. Try specifying a loader class, e.g. "
            "Video({0})".format(sequence))
    ext = ext.lower()[1:]

    # list all readers derived from the pims baseclasses
    all_handlers = chain(_recursive_subclasses(FramesSequence),
                         _recursive_subclasses(FramesSequenceND))
    # keep handlers that support the file ext. use set to avoid duplicates.
    eligible_handlers = set(h for h in all_handlers
                            if ext and ext in map(_drop_dot, h.class_exts()))
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

    messages = []
    for handler in sort_on_priority(eligible_handlers):
        try:
            return handler(sequence, **kwargs)
        except Exception as e:
            messages.append('{} errored: {}'.format(str(handler), str(e)))
    raise UnknownFormatError(
        "All handlers returned exceptions:\n" + "\n".join(messages))


class UnknownFormatError(Exception):
    pass


def _recursive_subclasses(cls):
    "Return all subclasses (and their subclasses, etc.)."
    # Source: http://stackoverflow.com/a/3862957/1221924
    return (cls.__subclasses__() +
        [g for s in cls.__subclasses__() for g in _recursive_subclasses(s)])

def _drop_dot(s):
    if s.startswith('.'):
        return s[1:]
    else:
        return s
