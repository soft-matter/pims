from pims.image_sequence import ImageSequence


def require_cv2_Video(*args, **kwargs):
    raise ImportError("To import frames from video files, you must install "
                      "OpenCV and the Python module cv2.")


def require_cv2_tools(*args, **kwargs):
    raise ImportError("To use video tools, you must install "
                      "OpenCV and the Python module cv2.")


def require_libtiff(*args, **kwargs):
    raise ImportError("To use TiffStack_libtiff, you must install libtiff. "
                      "Or, if you have PIL or Pillow, use TiffStack_pil")


def require_PIL_or_PILLOW(*args, **kwargs):
    raise ImportError("To use TiffStack_PIL, you must install PIL or Pillow. "
                      "Or, if you have libtiff, use TiffStack_libtiff.")


try:
    import cv2
except ImportError:
    Video = require_cv2_Video
    play = require_cv2_tools
else:
    from pims.video import Video
    from pims.playback import play

try:
    import libtiff
except ImportError:
    TiffStack = require_libtiff
else:
    from pims.tiff_stack import TiffStack_libtiff


try:
    from PIL import Image  # should work with PIL or PILLOW
except ImportError:
    TiffStack_pil = require_PIL_or_PILLOW
else:
    from pims.tiff_stack import TiffStack_pil
