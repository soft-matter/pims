from pims.base_frames import FramesSequence
from pims.frame import Frame
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None
except RuntimeError:
    # there is an incompatibility between moviepy 2.3.5 and imageio >= 2.5.0
    VideoFileClip = None


def available():
    return VideoFileClip is not None


class MoviePyReader(FramesSequence):
    """
    Read images from a video file via a MoviePy interface.
    """
    class_priority = 4
    @classmethod
    def class_exts(cls):
        return {'mov', 'mp4', 'avi', 'mpeg', 'wmv', 'mkv'}
    def __init__(self, filename):
        if VideoFileClip is None:
            raise ImportError('The MoviePyReader requires moviepy to work.')
        self.clip = VideoFileClip(filename)
        self.filename = filename
        self._fps = self.clip.fps
        self._len = int(self.clip.fps * self.clip.end)

        first_frame = self.clip.get_frame(0)
        self._shape = first_frame.shape
        self._dtype = first_frame.dtype

    def get_frame(self, i):
        return Frame(self.clip.get_frame(i / self._fps), frame_no=i)

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._shape

    @property
    def frame_rate(self):
        return self._fps

    @property
    def pixel_type(self):
        return self._dtype  
