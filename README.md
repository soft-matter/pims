pims: Python Image Sequence
=========================

[![build status](https://travis-ci.org/soft-matter/pims.png?branch=master)](https://travis-ci.org/soft-matter/pims)

What Problem Does PIMS Solve?
-----------------------------

Scientific video can be packaged in various ways: familiar video formats like
.AVI and .MOV, folders full of numbered images, or "stacks" of TIFF images. Each
of these requires a separate Python module. And, once loaded, they have
different methods for **accessing individual images, looping through the images
in bulk, accessing a specific range, or dealing with multidimensional files**.
PIMS can do all of these using a consistent interface, handling the differences
between different inputs invisibly.

Formats readable by pims include:
* Directories or zipfiles of still images (most formats, including TIFF, JPEG, PNG, BMP), and TIFF stacks
* Microscope images supported by the [Bio-formats project](https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html), including Leica, Nikon, Olympus, and Zeiss formats. Requires separate installation; see below.
* Movie formats and codecs supported by ffmpeg, including AVI, QuickTime MOV, and H.264 (MP4). May require separate installation; see below.
* CINE files from Vision Research cameras
* SEQ files from NorPix StreamPix software

PIMS is based on readers by:
* [scikit-image](http://scikit-image.org/)
* [matplotlib](http://matplotlib.org/)
* [scipy](http://www.scipy.org/)
* [ffmpeg](https://www.ffmpeg.org/) and [PyAV](http://mikeboers.github.io/PyAV/) (video formats such as AVI, MOV)
* [jpype](http://jpype.readthedocs.org/en/latest/) (interface with Bio-formats)
* [Pillow](http://pillow.readthedocs.org/en/latest/) (improved TIFF support)
* [libtiff](https://code.google.com/p/pylibtiff/) (alternative TIFF support)
* [tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) (alterative TIFF support)
* [pims_nd2](https://github.com/soft-matter/pims_nd2) (improved Nikon .nd2 support)
* [imageio](https://imageio.github.io) (a multi-purpose reader package that
   reads and writes many formats)
* [moviepy](http://zulko.github.io/moviepy) (a Python module that supports
   video editing)

Examples & Documentation
------------------------

Everything is demonstrated in [this IPython notebook](http://nbviewer.ipython.org/github/soft-matter/pims/blob/master/examples/loading%20video%20frames.ipynb).

[**Read the documentation**](http://soft-matter.github.io/pims/) for
installation instructions, examples, and further reference.

Core Contributors
-----------------

  * **Daniel Allan** founding contributor, slicing and iteration logic,
    basic readers, display tools
  * **Thomas Caswell** major refactor, abstract base class
  * **Casper van der Wel** bioformats readers, display tools
  * **Thomas Dimiduk** filetype-detecting dispatch logic

Support
-------

This package was developed in part by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD. The work was supported by the National Science Foundation
under grant number CBET-1033985. Later work was supported by Brookhaven
National Lab. Dan can be reached at dallan@bnl.gov.

This package was developed in part by Thomas A Caswell as part of his
PhD thesis work in Sidney R Nagel's and Margaret L Gardel's groups at
the University of Chicago, Chicago IL.  This work was supported in
part by NSF Grant DMR-1105145 and NSF-MRSEC DMR-0820054. Later work was
supported by Brookhaven National Lab. Tom can be
reached at tcaswell@gmail.com.

This package was developed in part by Casper van der Wel, as part of his
PhD thesis work in Daniela Kraft's group at the Huygens-Kamerlingh-Onnes laboratory,
Institute of Physics, Leiden University, The Netherlands. This work was
supported by the Netherlands Organisation for Scientific Research (NWO/OCW).
