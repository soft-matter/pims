pims: Python Image Sequence 
=========================

[![build status](https://travis-ci.org/soft-matter/pims.png?branch=master)](https://travis-ci.org/soft-matter/pims)

What Problem Does PIMS Solve?
-----------------------------

Scientific video can be packaged in various ways: familiar video formats like .AVI and .MOV, folders full of numbered images, or "stacks" of TIFF images. Each of these requires a separate Python module. And, once loaded, they have different methods for **accessing individual images, looping through the images in bulk, or access a specific range**. PIMS can do all of these using a consistent interface, handling the differences between different inputs invisibly.

Examples & Documentation
------------------------

Everything is demonstrated in [this IPython notebook](http://nbviewer.ipython.org/github/soft-matter/pims/blob/master/examples/loading%20video%20frames.ipynb).

Dependencies
------------


Basic Installation
------------------

Installation is simple on Windows, OSX, and Linux, even for Python novices.

To get started with Python on any platform, download and install
[Anaconda](https://store.continuum.io/cshop/anaconda/). It comes with the
common scientific Python packages built in.

If you are using Windows, I recommend 32-bit Anaconda even if your system is 64-bit. 
(One of the optional dependencies is not yet compatible with 64-bit Python.)

Open a command prompt. That's "Terminal" on a Mac, and 
"Start > Applications > Command Prompt" on Windows. Type these
lines:

    pip install http://github.com/soft-matter/pims/zipball/master

In the command prompt, type

    ipython notebook

Optional Dependencies
---------------------

### Reading Multi-Frame TIFF Stacks

You will need libtiff, which you can obtain by running the following command
in a command prompt:

    pip install -e svn+http://pylibtiff.googlecode.com/svn/trunk/

### Reading Video Files (AVI, MOV, etc.)

To load video files directly, you need OpenCV. You can work around this
requirement by converting any video files to folders full of images
using a utility like [ImageJ](http://rsb.info.nih.gov/ij/). Reading folders
of images is supported out of the box, without OpenCV.

* Linux: OpenCV is included with Anaconda
* OSX: OpenCV is easy to install on OSX using [homebrew](http://brew.sh/).
* Windows: OpenCV can be installed on Windows in a few steps, outlined below.
It is not as simple as the steps above, so beginners are encouraged
to experiment with a folder full of images first.

#### Installing OpenCV on Windows

1. Install the video software FFmepg using this [Windows installer](http://www.arachneweb.co.uk/software/windows/avchdview/FFmpegSetup.exe)
Make note of the directory where it is installed. It can be anywhere but, whatever it is,
you will need to know that location in the next step.
2. Right click on Computer (or perhaps "My Computer"), and click Properties. 
Click "Advanced System Settings", then "Properties". With "Path" highlighted,
click "Edit." This is a list of file paths separated by semicolons, you must 
type in an additional entry. ";C:\Program Files (x86)\ffmpeg" or wherever
FFmpeg was installed in Step 1.
3. Install the Windows 32 (Python 2.7) version of OpenCV available on [this page](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv).
4. Download [OpenCV for Windows](http://opencv.org/).
5. You will now have a folder called ``opencv``. We just need one file 
from this to make everything work.
6. Copy the file ``opencv\3rdparty\ffmpeg\opencv_ffmpeg.dll``.
7. Navigate to the directory where ffmpeg was installed, which you noted 
in Step 1. From this directory, navigate into ``win32-static\bin``.
Paste ``opencv_ffmpeg.dll`` here.

Now run ``ipython``. If you can execute ``import cv`` without any errors, the
installation is probably successful. If you can read video files using
``mr.Video('path/to/video_file.avi')`` then the installation is definitely working
as expected.


### Updating Your Instllation

The code is under active development. To update to the current development
version, run this in the command prompt:

    pip install --upgrade http://github.com/soft-matter/mr/zipball/master

Contributors
------------
* Daniel B. Allan
* Thomas A. Caswell (major refacotring, additional formats)

Supporting Grant
----------------
This package was originally developed and maintained by Daniel Allan, as part 
of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD. The work was supported by the National Science Foundation under grant number CBET-1033985.

Dan can be reached at dallan@pha.jhu.edu.
