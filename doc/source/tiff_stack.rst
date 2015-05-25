TiffStack
=========

A tiff stack is a single file (.tif or .tiff) containing several images,
often a time series or "Z stack."

``TiffStack`` expects a single filename. To load a collection of many
single-image tiff files (e.g., :file:`img-1.tif`, :file:`img-2.tif`) see
:doc:`image_sequence`.

Dependences
-----------

There are several Python packages for reading TIFFs. One may work better than
others for your application. To use a specific reader, use
``TiffStack_tifffile``, ``TiffStack_libtiff``, or ``TiffStack_pil``, which
depend, respectively, on the packages below. The "default" reader,
``TiffStack`` is an alias. At import time, it is pointed to the first
reader for which the required package is installed.

* `tifffile <https://pypi.python.org/pypi/tifffile>`_
* `pylibtiff <https://pypi.python.org/pypi/libtiff>`_
* `Pillow <https://pillow.readthedocs.org/>`_ or `PIL <http://www.pythonware.com/products/pil/>`_

Tifffile is installed with the PIMS conda package.
