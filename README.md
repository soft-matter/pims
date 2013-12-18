pims: Python Image Sequence 
=========================

What Problem Does PIMS Solve?
-----------------------------

Scientific video can be packaged in various ways: familiar video formats like .AVI and .MOV, folders full of numbered images, or "stacks" of TIFF images. Each of these requires a separate Python module. And, once loaded, they have different methods for **accessing individual images, looping through the images in bulk, or access a specific range**. PIMS can do all of these using a consistent interface, handling the differences between different inputs invisibly.

Examples & Documentation
------------------------

Everything is demonstrated in [this IPython notebook](http://nbviewer.ipython.org/github/soft-matter/pims/blob/master/examples/loading%20video%20frames.ipynb).

Dependencies
------------

Essential:

  * ``numpy``
  * ``scipy``

Optional:

  * [``cv2``](http://opencv.org/downloads.html) for reading video files (such .MOV, .AVI)
  * ``libtiff`` for reading multi-frame tiff images

Background
----------

This package was developed and is maintained by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD. The work was supported by the National Science Foundation under grant number CBET-1033985.

Dan can be reached at dallan@pha.jhu.edu.
