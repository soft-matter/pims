pims: Python Image Sequence
=========================

[![build status](https://travis-ci.org/soft-matter/pims.png?branch=master)](https://travis-ci.org/soft-matter/pims)

What Problem Does PIMS Solve?
-----------------------------

Scientific video can be packaged in various ways: familiar video formats like
.AVI and .MOV, folders full of numbered images, or "stacks" of TIFF images. Each
of these requires a separate Python module. And, once loaded, they have
different methods for **accessing individual images, looping through the images
in bulk, or access a specific range**. PIMS can do all of these using a
consistent interface, handling the differences between different inputs invisibly.

PIMS is based on readers by:
* [scikit-image](http://scikit-image.org/)
* [matplotlib](http://matplotlib.org/)
* [scipy](http://www.scipy.org/)
* [ffmpeg](https://www.ffmpeg.org/) and [PyAV](http://mikeboers.github.io/PyAV/) (video formats such as AVI, MOV)
* [jpype](http://jpype.readthedocs.org/en/latest/) (interface with bioformats to support [many](https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html) microscopy formats)
* [Pillow](http://pillow.readthedocs.org/en/latest/) (improved TIFF support)
* [libtiff](https://code.google.com/p/pylibtiff/) (alternative TIFF support)
* [tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) (alterative TIFF support)
* [pims_nd2](https://github.com/soft-matter/pims_nd2) (improved Nikon .nd2 support)

Examples & Documentation
------------------------

Everything is demonstrated in [this IPython notebook](http://nbviewer.ipython.org/github/soft-matter/pims/blob/master/examples/loading%20video%20frames.ipynb).

[**Read the documentation**](http://soft-matter.github.io/pims/) for
installation instructions, examples, and further reference.
