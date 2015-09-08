from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from pims.base_frames import FramesSequence
from pims.frame import Frame

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None


class MoviePyReader(FramesSequence):
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
        return Frame(self.clip.get_frame(i / self._fps))

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype  
