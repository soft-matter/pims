TiffStack
=========

PIMS can read most TIFF files out of the box, so you should try reading
you files ``pims.open('my_tiff_file.tif')`` and revisit this section if you
encounter an error.

A tiff stack is a single file (.tif or .tiff) containing several images,
often a time series or "Z stack."

``TiffStack`` expects a single filename. To load a collection of many
single-image tiff files (e.g., :file:`img-1.tif`, :file:`img-2.tif`) see
:doc:`image_sequence`.

Dependencies
------------

There are several Python packages for reading TIFFs. Our default reader, built
around `Christoph Gohlke's tifffile.py <http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`__,
handles all the formats we have personally encountered. But we have
an alternative TIFF reader built around Pillow (see above).
To use a specific reader, use
``TiffStack_tifffile`` or ``TiffStack_pil``, which
depend, respectively, on the packages below. The "default" reader,
``TiffStack`` is an alias. At import time, it is pointed to the first
reader for which the required package is installed.

* `tifffile <https://pypi.python.org/pypi/tifffile>`_
* `Pillow <https://pillow.readthedocs.org/>`_ or `PIL <http://www.pythonware.com/products/pil/>`_

Tifffile is installed with the PIMS conda package.
