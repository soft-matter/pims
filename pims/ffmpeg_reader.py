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

import re
import subprocess as sp
import tempfile

import numpy as np

from pims.base_frames import FrameRewindableStream


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


class FFmpegVideoReader(FrameRewindableStream):

    def __init__(self, filename, pix_fmt="rgb24", process_func=None):

        self.filename = filename
        self.pix_fmt = pix_fmt
        self._initialize()
        try:
            self.depth = _pix_fmt_dict[pix_fmt]
        except KeyError:
            raise ValueError("invalid pixel format")
        self._load_infos(print_infos=False)

        if process_func is None:
            process_func = lambda x: x
        if not callable(process_func):
            raise ValueError("process_func must be a function, or None")
        self.process_func = process_func


    def _initialize(self):
        """ Opens the file, creates the pipe. """

        self.data_buffer = tempfile.TemporaryFile()
        cmd = [FFMPEG_BINARY, '-i', self.filename,
                '-f', 'image2pipe',
                "-pix_fmt", self.pix_fmt,
                '-vcodec', 'rawvideo', '-']
        proc = sp.Popen(cmd, stdin=sp.PIPE,
                             stdout=sp.PIPE,
                             stderr=sp.PIPE)

        print "Decoding video file..."
        CHUNKSIZE = 2**14  # utterly arbitrary
        while True:
            try:
                chunk = proc.stdout.read(CHUNKSIZE)
                if len(chunk) == 0:
                    break
                self.data_buffer.write(chunk)
            except EOFError:
                break
        self.data_buffer.seek(0)

        proc.terminate()
        for std in proc.stdin, proc.stdout, proc.stderr:
            std.close()
        #del self.proc
        self.pos = 0

    def _load_infos(self, print_infos=False):
        """ reads the FFMPEG info on the file and sets self.size
            and self.fps """
        # open the file in a pipe, provoke an error, read output
        proc = sp.Popen([FFMPEG_BINARY, "-i", self.filename,
                         "-f", "null", "-"],
                stdin=sp.PIPE,
                stdout=DEVNULL,
                stderr=sp.PIPE)
        # let it fully play the movie to null so we can get a frame count
        proc.wait()
        infos = proc.stderr.read()
        if print_infos:
            # print the whole info text returned by FFMPEG
            print infos

        lines = infos.splitlines()
        if "No such file or directory" in lines[-1]:
            raise IOError("%s not found ! Wrong path ?" % self.filename)

        # get the output line that speaks about video
        line = [l for l in lines if ' Video: ' in l][0]
        # logic to parse all of the MD goes here

        # get the size, of the form 460x320 (w x h)
        match = re.search(" [0-9]*x[0-9]*(,| )", line)
        self._size = map(int, line[match.start():match.end()-1].split('x'))
        # this needs to be more robust
        self._len = int(lines[-2].split()[1])
        w, h = self._size
        self._stride = self.depth*w*h

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._size

    def close(self):
        del self.data_buffer

    def skip_forward(self, n=1):
        """ Reads and throws away n frames """
        w, h = self._size
        for i in range(n):
            self.data_buffer.read(self._stride)
            self.pos += 1

    def next(self):
        w, h = self._size
        # Normally, the readr should not read after the last frame...
        # if it does, raise an error.
        s = self.data_buffer.read(self._stride)
        result = np.fromstring(s,
            dtype='uint8').reshape((h, w, self.depth))

        self.pos += 1

        return self.process_func(result)

    def rewind(self, start_frame=0):
        """ Restarts the reading, starts at an arbitrary
            location (!! SLOW !!) """
        self.close()
        self._initialize()
        if start_frame != 0:
            self.skip_forward(start_frame)

    @property
    def current(self):
        return self.pos

    @property
    def pixel_type(self):
        raise NotImplemented()

    def __del__(self):
        self.close()
