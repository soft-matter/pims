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

import numpy as np
import subprocess as sp


FFMPEG_BINARY = None


def cvsecs(*args):
    """ converts a time to second. Either cvsecs(min,secs) or
    cvsecs(hours,mins,secs).
    >>> cvsecs(5.5) # -> 5.5 seconds
    >>> cvsecs(10, 4.5) # -> 604.5 seconds
    >>> cvsecs(1, 0, 5) # -> 3605 seconds
    """
    if len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return 60*args[0]+args[1]
    elif len(args) == 3:
        return 3600*args[0]+60*args[1]+args[2]


def tryffmpeg(FFMPEG_BINARY):
    try:
        proc = sp.Popen([FFMPEG_BINARY],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
        except:
            return False
        else:
            return True


if FFMPEG_BINARY is None:
    if tryffmpeg('ffmpeg'):
        FFMPEG_BINARY = 'ffmpeg'
    elif tryffmpeg('ffmpeg.exe'):
        FFMPEG_BINARY = 'ffmpeg.exe'
    else:
        raise IOError("FFMPEG binary not found.")


class FFMPEG_VideoReader:

    def __init__(self, filename, print_infos=False, pix_fmt="rgb24"):

        self.filename = filename
        self.pix_fmt = pix_fmt
        self.initialize()
        self.depth = 4 if pix_fmt == "rgba" else 3
        self.load_infos(print_infos)
        self.pos = 1
        self.lastread = self.read_frame()

    def initialize(self):
        """ Opens the file, creates the pipe. """

        cmd = [FFMPEG_BINARY, '-i', self.filename,
                '-f', 'image2pipe',
                "-pix_fmt", self.pix_fmt,
                '-vcodec', 'rawvideo', '-']
        self.proc = sp.Popen(cmd, stdin=sp.PIPE,
                                   stdout=sp.PIPE,
                                   stderr=sp.PIPE)

    def load_infos(self, print_infos=False):
        """ reads the FFMPEG info on the file and sets self.size
            and self.fps """
        # open the file in a pipe, provoke an error, read output
        proc = sp.Popen([FFMPEG_BINARY, "-i", self.filename, "-"],
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE)
        proc.stdout.readline()
        proc.terminate()
        infos = proc.stderr.read()
        if print_infos:
            # print the whole info text returned by FFMPEG
            print infos

        lines = infos.splitlines()
        if "No such file or directory" in lines[-1]:
            raise IOError("%s not found ! Wrong path ?" % self.filename)

        # get the output line that speaks about video
        line = [l for l in lines if ' Video: ' in l][0]

        # get the size, of the form 460x320 (w x h)
        match = re.search(" [0-9]*x[0-9]*(,| )", line)
        self.size = map(int, line[match.start():match.end()-1].split('x'))

        # get the frame rate
        match = re.search("( [0-9]*.| )[0-9]* (tbr|fps)", line)
        self.fps = float(line[match.start():match.end()].split(' ')[1])

        # get duration (in seconds)
        line = [l for l in lines if 'Duration: ' in l][0]
        match = re.search(" [0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9]", line)
        hms = map(float, line[match.start()+1:match.end()].split(':'))
        self.duration = cvsecs(*hms)
        self.nframes = int(self.duration*self.fps)

    def close(self):
        self.proc.terminate()
        for std in self.proc.stdin, self.proc.stdout, self.proc.stderr:
            std.close()
        del self.proc

    def skip_frames(self, n=1):
        """ Reads and throws away n frames """
        w, h = self.size
        for i in range(n):
            self.proc.stdout.read(self.depth*w*h)
            self.proc.stdout.flush()
        self.pos += n

    def read_frame(self):
        w, h = self.size
        try:
            # Normally, the readr should not read after the last frame...
            # if it does, raise an error.
            s = self.proc.stdout.read(self.depth*w*h)
            result = np.fromstring(s,
                             dtype='uint8').reshape((h, w, len(s)/(w*h)))
            self.proc.stdout.flush()
        except:
            self.proc.terminate()
            serr = self.proc.stderr.read()
            print "error: string: %s, stderr: %s"%(s, serr)
            raise

        self.lastread = result

        return result

    def reinitialize(self, starttime=0):
        """ Restarts the reading, starts at an arbitrary
            location (!! SLOW !!) """
        self.close()
        if starttime == 0:
            self.initialize()
        else:
            offset = min(1, starttime)
            cmd = [FFMPEG_BINARY, '-ss', "%.03f" % (starttime - offset),
                    '-i', self.filename,
                    '-ss', "%.03f" % offset,
                    '-f', 'image2pipe',
                    "-pix_fmt", self.pix_fmt,
                    '-vcodec', 'rawvideo', '-']
            self.proc = sp.Popen(cmd, stdin=sp.PIPE,
                                       stdout=sp.PIPE,
                                      stderr=sp.PIPE)

    def get_frame(self, t):
        """ Reads a frame at time t. Note for coders:
            getting an arbitrary frame in the video with ffmpeg can be
            painfully slow if some decoding has to be done. This
            function tries to avoid fectching arbitrary frames whenever
            possible, by moving between adjacent frames.
            """
        if t < 0:
            t = 0
        elif t > self.duration:
            t = self.duration

        pos = int(self.fps*t)+1
        if pos == self.pos:
            return self.lastread
        else:
            if (pos < self.pos) or (pos> self.pos + 100):
                self.reinitialize(t)
            else:
                self.skip_frames(pos-self.pos-1)
            result = self.read_frame()
            self.pos = pos
            return result


def read_image(filename, with_mask=True):
    pix_fmt = 'rgba' if with_mask else "rgb24"
    vf = FFMPEG_VideoReader(filename, pix_fmt=pix_fmt)
    vf.close()
    return vf.lastread
