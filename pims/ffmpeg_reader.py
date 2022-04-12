# The MIT License (MIT)
#
# Copyright (c) 2014 Zulko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# The MIT License (MIT)
# [OSI Approved License]
#
# https://github.com/Zulko/moviepy
#
# source files:
# moviepy/conf.py
# moviepy/video/io/ffmpeg_reader.py
# moviepy/tools.py
#
# Files heavily edited by PIMS contributors
# January 2014

import re
import subprocess as sp
import sys
import os

import numpy as np

from pims.base_frames import FramesSequence
from pims.frame import Frame


try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def try_ffmpeg(FFMPEG_BINARY):
    try:
        proc = sp.Popen([FFMPEG_BINARY],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
    except:
        return False
    else:
        return True


# Name (and location if needed) of the FFMPEG binary. It will be
# "ffmpeg" on linux, certainly "ffmpeg.exe" on windows, else any path.
FFMPEG_BINARY_SUGGESTIONS = ['ffmpeg', 'ffmpeg.exe']

FFMPEG_BINARY = None
for name in FFMPEG_BINARY_SUGGESTIONS:
    if try_ffmpeg(name):
        FFMPEG_BINARY = name
        break


def available():
    return FFMPEG_BINARY is not None

_pix_fmt_dict = {'rgb24': 3,
                 'rgba': 4}


class FFmpegVideoReader(FramesSequence):
    """Read images from the frames of a standard video file into an
    iterable object that returns images as numpy arrays.

    This reader, based on ffmpeg, should be able to read most video
    files

    Parameters
    ----------
    filename : string
    pix_fmt : string, optional
        Expected number of color channels. Either 'rgb24' or 'rgba'.
        Defaults to 'rgba'.
    use_cache : boolean, optional
        Saves time if the file was previously opened. Defaults to True.
        Set to False if the file may have changed since it was
        last read.

    Examples
    --------
    >>> video = FFmpegVideoReader('video.avi')  # or .mov, etc.
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
    def __init__(self, filename, pix_fmt="rgb24", use_cache=True):

        self.filename = filename
        self.pix_fmt = pix_fmt
        self._initialize(use_cache)
        try:
            self.depth = _pix_fmt_dict[pix_fmt]
        except KeyError:
            raise ValueError("invalid pixel format")
        w, h = self._size
        self._stride = self.depth*w*h

    def _initialize(self, use_cache):
        """ Opens the file, creates the pipe. """

        buffer_filename = '{0}.pims_buffer'.format(self.filename)
        meta_filename = '{0}.pims_meta'.format(self.filename)

        cmd = [FFMPEG_BINARY, '-i', self.filename,
                '-f', 'image2pipe',
                "-pix_fmt", self.pix_fmt,
                '-vcodec', 'rawvideo', '-']
        proc = sp.Popen(cmd, stdin=sp.PIPE,
                             stdout=sp.PIPE,
                             stderr=sp.PIPE)

        print("Decoding video file...")

        if (os.path.isfile(buffer_filename) and os.path.isfile(meta_filename)
            and use_cache):
            print("Reusing buffer from previous opening of this video.")
            self.data_buffer = open(buffer_filename, 'rb')
            self.metafile = open(meta_filename, 'r')
            self._len = int(self.metafile.readline())
            w = int(self.metafile.readline())
            h = int(self.metafile.readline())
            self._size = [w, h]
            return

        self.data_buffer = open(buffer_filename, 'wb')
        self.metafile = open(meta_filename, 'w')
        print ("Decoding video file. This is slow, but only the first time.")
        sys.stdout.flush()
        CHUNKSIZE = 2**14  # utterly arbitrary
        while True:
            try:
                chunk = proc.stdout.read(CHUNKSIZE)
                if len(chunk) == 0:
                    break
                self.data_buffer.write(chunk)
            except EOFError:
                break
        self.data_buffer.close()
        self.data_buffer = open(buffer_filename, 'rb')

        self._process_ffmpeg_stderr(proc.stderr.read())

        proc.terminate()
        for std in proc.stdin, proc.stdout, proc.stderr:
            std.close()

    def _process_ffmpeg_stderr(self, stderr, verbose=False):
        if verbose:
            print(stderr)

        lines = stderr.splitlines()
        if "No such file or directory" in lines[-1]:
            raise IOError("%s not found ! Wrong path ?" % self.filename)

        # get the output lines that describe the video
        line = [l for l in lines if ' Video: ' in l][0]
        # logic to parse all of the MD goes here

        # get the size, of the form 460x320 (w x h)
        match = re.search(" [0-9]*x[0-9]*(,| )", line)
        self._size = map(int, line[match.start():match.end()-1].split('x'))
        # this needs to be more robust
        self._len = int(lines[-2].split()[1])
        self.metafile.write('{0}\n'.format(self._len))
        self.metafile.write('{0}\n'.format(self._size[0]))
        self.metafile.write('{0}\n'.format(self._size[1]))
        self.metafile.close()

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._size

    def get_frame(self, j):
        self.data_buffer.seek(self._stride*j)
        s = self.data_buffer.read(self._stride)
        w, h = self._size
        result = np.frombuffer(s,
            dtype='uint8').reshape((h, w, self.depth))
        return Frame(result, frame_no=j)

    @property
    def pixel_type(self):
        raise np.uint8

    @classmethod
    def class_exts(cls):
        return set(['mov', 'avi', 'webm'])

    def close(self):
        self.metafile.close()
        self.data_buffer.close()
        super().close()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {frame_shape!r}
Pixel Format: {pix_fmt}""".format(frame_shape=self.frame_shape,
                                  count=len(self),
                                  filename=self.filename,
                                  pix_fmt=self.pix_fmt)
