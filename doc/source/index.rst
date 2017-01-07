PIMS: Python Image Sequence
===========================

PIMS is a lazy-loading interface to sequential data with numpy-like slicing.

Key features:

* One consistent interface to many file formats
* Numpy-like slicing returns lazy-loading, iterable, sliceable objects

Contents
--------

.. toctree::
   :maxdepth: 1

   install
   release_notes
   opening_files
   slicing
   pipelines
   frame
   multidimensional
   custom_readers

Built-in Readers
----------------

.. toctree::
   :maxdepth: 1

   image_sequence
   tiff_stack
   bioformats
   video

Example
-------

Everything is demonstrated in `this IPython
notebook <http://nbviewer.ipython.org/github/soft-matter/pims/blob/master/examples/loading%20video%20frames.ipynb>`__.

Load a sequence of images from a directory, where the images are named
:file:`img-0.png`, :file:`img-1.png`, etc.

.. ipython:: python
   :suppress:

   from pims.tests.test_common import save_dummy_png, clean_dummy_png
   filenames = ['img-{0}.png'.format(i) for i in range(9)]
   save_dummy_png('.', filenames, (256, 256))

.. ipython:: python

   from pims import ImageSequence
   images = ImageSequence('img-*.png')
   images

Images can be randomly accessed with standard Python slicing syntax.

.. ipython:: python

   images[0]  # first image
   images[-5]  # fifth from the end

The images are iterable. Data is loaded one image at a time, conserving
memory.

.. ipython:: python

   import numpy as np
   for image in images:
       np.sum(image)  # do something

Slicing ``images`` returns another lazy-loading object that is also iterable
and sliceable.

.. ipython:: python

   subsection = images[:5]  # the first five images
   len(images)
   len(subsection)
   for image in subsection:
       np.sum(image)  # do something

   subsubsection = subsection[::2]  # every other of the first five images

Fancy numpy-like slicing is supported.

.. ipython:: python

   subsection2 = images[[0, 3, 7]]
   mask = [True, False, False, False, False, True, False, False, False, False]
   subsection3 = images[mask]


.. ipython:: python
   :suppress:

   clean_dummy_png('.', filenames)

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
