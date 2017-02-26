from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import re

import numpy as np

from pims.base_frames import FramesSequence
from pims.frame import Frame

from warnings import warn


try:
    import av
except ImportError:
    av = None


def available():
    return av is not None


def _gen_frames(demuxer, first_pts=0, index_base=1.):
    for packet in demuxer:
        for frame in packet.decode():
            # learn timestamp from the frame timestamp
            timestamp = frame.pts

            # if not available, take the packet timestamp
            if timestamp is None:
                timestamp = packet.pts

            # if not available, raise an exception
            if timestamp is None:
                raise IOError("Unable to read video: the frames do not contain"
                              " timestamps. Please use PyAVReaderIndexed.")

            i = int((timestamp - first_pts) * index_base)
            yield i, _to_nd_array(frame)


def _to_nd_array(frame):
    plane_rgb = frame.reformat(format="rgb24").planes[0]
    frame_arr = np.frombuffer(plane_rgb, np.uint8)
    frame_arr.shape = (frame.height, frame.width, -1)
    return frame_arr


class PyAVReaderTimed(FramesSequence):
    """Read images from a video file via a direct FFmpeg/AVbin interface.

    The frames are indexed according to their 'timestamp', starting at 0 at the
    timestamp of the first non-empty frame. Missing frames are filled in with
    empty frames. The number of frames in the video is estimated from the
    movie duration and the average frame rate.

    Parameters
    ----------
    filename : string
    cache_size : integer, optional
        the number of frames that are kept in memory. Default 16.

    Examples
    --------
    >>> video = PyAVVideoReader('video.avi')  # or .mov, etc.
    >>> video[0] # Show the first frame.
    >>> video[-1] # Show the last frame.
    >>> video[1][0:10, 0:10] # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """
    class_priority = 8
    @classmethod
    def class_exts(cls):
        return {'mov', 'avi', 'mp4', 'mpg', 'mkv', 'vob', 'webm', 'm4v',
                'flv', 'h264'} | super(PyAVReaderTimed, cls).class_exts()

    def __init__(self, filename, cache_size=16):
        self.filename = str(filename)
        self._container = av.open(self.filename)

        for s in self._container.streams:
            if isinstance(s, av.video.VideoStream):
                self._stream = s
                break
        else:
            raise IOError("No video stream found")

        self._cache = [(-1, None)] * cache_size
        self._fast_forward_thresh = cache_size * 2

        demuxer = self._container.demux(streams=self._stream)

        # obtain first frame to get first time point
        self._first_pts, frame = next(_gen_frames(demuxer))
        self._cache[0] = (0, frame)
        self._frame_shape = frame.shape
        self._last_frame = 0

        index_base = float(self._stream.time_base * self.frame_rate)
        self._frame_generator = _gen_frames(demuxer, self._first_pts,
                                            index_base)

    def __len__(self):
        return int(self._stream.duration * self._stream.time_base *
                   self._stream.average_rate)

    @property
    def duration(self):
        """The video duration in seconds."""
        return float(self._stream.duration * self._stream.time_base)

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def frame_rate(self):
        return float(self._stream.average_rate)

    def get_frame(self, i):
        cache_i = i % len(self._cache)
        did_seek = False
        if self._cache[cache_i][0] > i or \
           self._last_frame < i - self._fast_forward_thresh:
            self.seek(i)
            did_seek = True

        if self._cache[cache_i][0] == i:
            return Frame(self._cache[cache_i][1], frame_no=i)

        for index, frame in self._frame_generator:
            self._cache[index % len(self._cache)] = (index, frame)
            self._last_frame = index

            if index == i:
                break

            if index > i:
                if did_seek:
                    break
                else:
                    self.seek(i)
                    did_seek = True
        else:
            # restart the frame generator
            demuxer = self._container.demux(streams=self._stream)
            index_base = float(self._stream.time_base * self.frame_rate)
            self._frame_generator = _gen_frames(demuxer, self._first_pts,
                                                index_base)

        if self._cache[cache_i][0] != i:
            # the requested frame actually does not exist. Can occur
            # due to inaccuracy of __len__. Yield an empty frame.
            warn("Frame {} could not be found. Returning an"
                 "empty frame.".format(i))
            frame = np.zeros(self.frame_shape, dtype=self.pixel_type)
            self._cache[cache_i] = (i, frame)

        return Frame(self._cache[cache_i][1], frame_no=i)

    def seek(self, i):
        # flush the cache
        self._cache = [[-1, None]] * len(self._cache)
        # the ffmpeg decode cache is (apparently) flushed automatically

        timestamp = int(i / self.frame_rate * av.time_base)
        self._container.seek(timestamp)

    @property
    def pixel_type(self):
        return np.uint8

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Format: {format}
Source: {filename}
Duration: {duration:.3f} seconds
Frame rate: {frame_rate:.3f} fps
Length: {count} frames
Frame Shape: {frame_shape!r}
""".format(frame_shape=self.frame_shape,
           format=self._stream.long_name,
           duration=self.duration,
           frame_rate=self.frame_rate,
           count=len(self),
           filename=self.filename)


class PyAVReaderIndexed(FramesSequence):
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
    class_priority = 8

    @classmethod
    def class_exts(cls):
        return {'mov', 'avi',
                'mp4'} | super(PyAVReaderIndexed, cls).class_exts()

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
        self._im_sz = video_stream.height, video_stream.width, 3

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
        result = _to_nd_array(frame)
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
Frame Shape: {frame_shape!r}
""".format(frame_shape=self.frame_shape,
           count=len(self),
           filename=self.filename)
