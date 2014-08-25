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

One of the following is required:

* [scikit-image]
* [matplotlib]
* [scipy]

Depending on what file formats you want to read, you will also need:

* [ffmpeg](https://www.ffmpeg.org/) (video formats such as AVI, MOV)
* [Pillow](http://pillow.readthedocs.org/en/latest/) (improved TIFF support)
* [libtiff](https://code.google.com/p/pylibtiff/) (alternative TIFF support)
* Tifffile, which is included in PIMS

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

To load video files directly, you need FFmpeg. You can work around this
requirement by converting any video files to folders full of images
using a utility like [ImageJ](http://rsb.info.nih.gov/ij/). Reading folders
of images is supported out of the box, without OpenCV.

### Updating Your Instllation

The code is under active development. To update to the current development
version, run this in the command prompt:

    pip install --upgrade http://github.com/soft-matter/pims/zipball/master

Contributors
------------
* Daniel B. Allan
* Thomas A. Caswell (major refactoring, additional formats)

Supporting Grant
----------------
This package was originally developed and maintained by Daniel Allan, as part 
of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD. The work was supported by the National Science Foundation under grant number CBET-1033985.

Dan can be reached at daniel.b.allan@jhu.edu.
