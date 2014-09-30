from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import re

import numpy as np

from pims.base_frames import FramesSequence
from pims.frame import Frame


try:
    import av
except ImportError:
    av = None


def available():
    try:
        import av
    except ImportError:
        return False
    else:
        return True


class PyAVVideoReader(FramesSequence):
    """Read images from the frames of a standard video file into an
    iterable object that returns images as numpy arrays.

    Parameters
    ----------
    filename : string
    process_func : function, optional
        callable with signalture `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    dtype : numpy datatype, optional
        Image arrays will be converted to this datatype.
    as_grey : boolean, optional
        Convert color images to greyscale. False by default.
        May not be used in conjection with process_func.

    Examples
    --------
    >>> video = Video('video.avi')  # or .mov, etc.
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
    @classmethod
    def class_exts(cls):
        return {'mov', 'avi',
                'mp4'} | super(PyAVVideoReader, cls).class_exts()

    def __init__(self, filename, process_func=None, dtype=None,
                 as_grey=False):

        if dtype is not None:
            self._dtype = dtype
        else:
            # No need to detect dtype: PyAV always returns uint8.
            self._dtype = np.uint8

        self.filename = str(filename)
        self._initialize()

        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

    def _initialize(self):
        "Scan through and tabulate contents to enable random access."
        container = av.open(self.filename)
 
        # Build a toc 
        self._toc = np.cumsum([len(packet.decode())
                               for packet in container.demux()])
        self._len = self._toc[-1]

        video_stream = [s for s in container.streams
                        if isinstance(s, av.video.VideoStream)][0]
        # PyAV always returns frames in color, and we make that
        # assumption in get_frame() later below, so 3 is hardcoded here:
        self._im_sz = video_stream.width, video_stream.height, 3

        del container  # The generator is empty. Reload the file.
        self._load_fresh_file()

    def _load_fresh_file(self):
        self._demuxed_container = av.open(self.filename).demux()
        self._current_packet = next(self._demuxed_container).decode()
        self._packet_cursor = 0
        self._frame_cursor = 0

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._im_sz

    def get_frame(self, j):
        # Find the packet this frame is in.
        packet_no = self._toc.searchsorted(j, side='right')
        self._seek_packet(packet_no)
        # Find the location of the frame within the packet.
        if packet_no == 0:
            loc = j
        else:
            loc = j - self._toc[packet_no - 1]
        frame = self._current_packet[loc]  # av.VideoFrame
        if frame.index != j:
            raise AssertionError("Seeking failed to obtain the correct frame.")
        result = np.asarray(frame.to_rgb().to_image())
        return Frame(self.process_func(result).astype(self._dtype), frame_no=j)

    def _seek_packet(self, packet_no):
        """Advance through the container generator until we get the packet
        we want. Store that packet in self._current_packet."""
        if packet_no == self._packet_cursor:
            # We have the right packet and it is already decoded.
            return 
        if packet_no < self._packet_cursor:
            # "Rewind." This is not really possible, so we load a fresh
            # instance of the file object and then fast-forward.
            self._load_fresh_file()
        # Fast-forward if needed.
        while self._packet_cursor < packet_no:
            self._current_packet = next(self._demuxed_container).decode()
            self._packet_cursor += 1

    @property
    def pixel_type(self):
        raise NotImplemented()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  filename=self.filename)
