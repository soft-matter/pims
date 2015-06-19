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

* [ffmpeg](https://www.ffmpeg.org/) and [PyAV](http://mikeboers.github.io/PyAV/) (video formats such as AVI, MOV)
* [jpype](http://jpype.readthedocs.org/en/latest/) (interface with bioformats to support [many](https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html) microscopy formats)
* [Pillow](http://pillow.readthedocs.org/en/latest/) (improved TIFF support)
* [libtiff](https://code.google.com/p/pylibtiff/) (alternative TIFF support)
* [tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) (alterative TIFF support)

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

    conda update conda
    conda install numpy matplotlib scikit-image pillow
    conda install pip

Then, to install pims:

    pip install pims

Finally, to try it out, type:

    ipython notebook

Optional Dependencies
---------------------

### Reading Multi-Frame TIFF Stacks

PIMS can read most TIFF files out of the box, so you should try reading
you files `(open('my_tiff_file.tif')` and revisit this section if you
encounter an error. Many camera and software manufacturers have their
own special variants of the TIFF format. Our default reader, built around
[Christoph Gohlke's tifffile.py](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html), handles all the formats we have personally enountered. But we have
alternative TIFF readers built around
Pillow (see above) and libtiff, which can be installed like so:

    pip install libtiff

### Reading Video Files (AVI, MOV, etc.)

To load video files directly, you need FFmpeg or libav. These can be tricky to
install, especially on Windows, so we advise less sophisticated users to simply
work around this requirement by converting their video files to folders full of
images using a utility like [ImageJ](http://rsb.info.nih.gov/ij/).

But if either FFmpeg or libav is available, PIMS enables fast random access to
video files. It relies on PyAV, which can be installed like so:

    pip install av

### Reading microscopy files via Bio-Formats

([List of supported formats](https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html))

Bio-Formats is an open-source java library for reading and writing
multidimensional image data, especially from microscopy files. To interface
with the java library, we use [JPype](https://github.com/originell/jpype) which
allows fast and easy access to all java functions. JRE or JDK are not required.
Install JPype as follows:

    pip install jpype1

On first use of `pims.Bioformats(filename)`, the required java library
`loci_tools.jar` will be automatically downloaded from
[openmicroscopy.org](http://downloads.openmicroscopy.org/bio-formats/).

#### Troubleshooting

If you use conda / Anaconda, watch out for an error like:

    version `GLIBC_2.15' not found

This seems to be because [conda includes an old version of a library needed by
PyAV](github.com/ContinuumIO/anaconda-issues/issues/182). To work around this,
simply delete anaconda's version of the library:

    rm ~/anaconda/lib/libm.so.6

and/or

    rm ~/anaconda/envs/name_of_your_environment/lib/libm.so.6

which will cause PyAV to use the your operating system's version of the
library.

### Updating Your Installation

The code is under active development. To update to the current development
version, run this in the command prompt:

    pip install --upgrade http://github.com/soft-matter/pims/zipball/master

Primary Contributors
--------------------
* Daniel B. Allan
* Thomas A. Caswell
* Casper van der Wel

Supporting Grant
----------------

This package was originally developed and maintained by Daniel Allan,
as part of his PhD thesis work on microrheology in Robert L. Leheny's
group at Johns Hopkins University in Baltimore, MD. The work was
supported by the National Science Foundation under grant number
CBET-1033985.


Dan can be reached at daniel.b.allan@jhu.edu.
