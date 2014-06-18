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

## Name (and locatio if needed) of the FFMPEG binary. It will be
## "ffmpeg" on linux, certainly "ffmpeg.exe" on windows, else any path.
## If not provided (None), the system will look for the right version
## automatically each time you launch moviepy.
## If you run this script file it will check that the
## path to the ffmpeg binary (FFMPEG_BINARY)
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import re
import subprocess as sp
import sys

import numpy as np

from pims.base_frames import FramesSequence
from pims.frame import Frame


try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    import os
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

    def __init__(self, filename, pix_fmt="rgb24", process_func=None,
                 use_cache=True):

        self.filename = filename
        self.pix_fmt = pix_fmt
        self._initialize(use_cache)
        try:
            self.depth = _pix_fmt_dict[pix_fmt]
        except KeyError:
            raise ValueError("invalid pixel format")
        w, h = self._size
        self._stride = self.depth*w*h

        if process_func is None:
            process_func = lambda x: x
        if not callable(process_func):
            raise ValueError("process_func must be a function, or None")
        self.process_func = process_func

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
        result = np.fromstring(s,
            dtype='uint8').reshape((h, w, self.depth))
        return Frame(self.process_func(result), frame_no=j)

    @property
    def pixel_type(self):
        raise NotImplemented()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Format: {pix_fmt}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  filename=self.filename,
                                  pix_fmt=self.pix_fmt)
