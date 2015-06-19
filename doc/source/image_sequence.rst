.. currentmodule:: pims

ImageSequence
=============

An image sequence is a collection of image files with sequential
filenames.

``ImageSequence`` can be instatiated using:

* a directory name, such as a ``'my_directory'``
* a "glob" string, such as ``'my_directory/*.png'``, which is safer than
  using a directory because directories sometimes contain stray files
* the filepath of a zipped archive, such as ``'my_directory/all-images.zip'``
* a list of filepaths, such as ``['image1.png', 'image2.png']``

Dependencies
------------

Several widely-used Python packages have slightly different implementations
of ``imread``, a general purpose image-reading function that understands
popular formats like PNG, JPG, TIFF, and others. PIMS requires **one of
the following** packages, in order of decreasing preference.

* `scikit-image <http://scikit-image.org/>`_
* `matplotlib <http://scikit-image.org/>`_
* `scipy <http://scikit-image.org/>`_

Scikit-image is installed with the PIMS conda package.
