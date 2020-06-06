Slicing and Iteration
=====================

.. ipython:: python
   :suppress:

   from pims.tests.test_common import save_dummy_png, clean_dummy_png
   filenames = ['img-{0}.png'.format(i) for i in range(9)]
   save_dummy_png('.', filenames, (256, 256))
   from pims import ImageSequence
   images = ImageSequence('img-*.png')

PIMS objects are like lists or numpy arrays that lazily load and "stream" data
like Python generators. Unlike Python
generators, they can be sliced and indexed, and they have a length. But unlike
lists or arrays, the underlying data is only accessed one element at a time.

Accessing Individual Images
---------------------------

Images can be randomly accessed with standard Python slicing syntax.
The built-in PIMS readers all enable random access, even if the underlying
file format does not support it natively, as with video files.

.. ipython:: python

   images[0]  # first image
   images[-5]  # fifth from the end


Iteration
---------

The images are iterable. Data is loaded one image at a time, conserving
memory.

.. ipython:: python

   import numpy as np
   for image in images:
       np.sum(image)  # do something

Slices -- and Slices of Slices
------------------------------

Slicing ``images`` returns another lazy-loading object that is also iterable
and sliceable.

.. ipython:: python

   subsection = images[:5]  # the first five images
   len(images)
   len(subsection)
   for image in subsection:
       np.sum(image)  # do something

   subsubsection = subsection[::2]  # every other of the first five images

You can creates slices of slices of slices to any depth.

Fancy Indexing
--------------

Numpy-like "fancy indexing" is supported. As above, the examples below
create more sliceable, iterable objects.

.. ipython:: python

   subsection2 = images[[0, 3, 7]]
   mask = [True, False, False, False, False, True, False, False, False, False]
   subsection3 = images[mask]


.. ipython:: python
   :suppress:

   clean_dummy_png('.', filenames)

Slicing in Space
----------------

It's easy to make a sequence with all images cropped using the
:py:class:`pims.process.crop` pipeline, or to make a sequence with any sort of
slicing by creating a new pipeline. See the `doc:pipelines` documentation
for details.