Opening Files
=============

QuickStart
----------

Where possible, pims detects the type of file(s) automatically. Here are some
examples.

.. code-block:: python

   import pims
   images = pims.open('my_directory/*.png')  # many PNGs with sequential names
   images = pims.open('my_directory/*.tif')  # many TIFs with sequential names
   images = pims.open('tiff_stack.tif')  # one TIF file containing many frames

Using Specific Readers
----------------------

PIMS has several built-in readers. If you don't want to use ``open`` to
dispatch to them automatically, you can use them directly. For example:

.. code-block:: python

   import pims
   images = pims.ImageSequence('my_directory/*.png')

The main readers are:

* :doc:`image_sequence`
* :doc:`tiff_stack`
* :doc:`video`
* :doc:`bioformats`

See their individual pages for details.

If you have a file format not yet supported by PIMS, it is easy to define your
own reader and get PIMS lazy-loading and slicing behavhior "for free."
See :doc:`custom_readers`.
