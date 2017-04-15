Multidimensional Readers
========================

.. ipython:: python
   :suppress:

   from pims.tests.test_common import save_dummy_png, clean_dummy_png
   from itertools import product
   filenames = ['img-t{0}z{1}c{2}.png'.format(i, j, z) for i, j, z in product(range(4), range(10), range(2))]
   save_dummy_png('.', filenames, (64, 128))
   from pims import ImageSequenceND
   images = ImageSequenceND('img-*.png', axes_identifiers='tzc')

Multidimensional files are files that have for instance an extra z-axis (making
the image 3D, an extra channel axis, or a multipoint axis. PIMS provides a
flexible and uniform interface for working with these kind of files through the
base class ``FramesSequenceND``. The readers ``ImageSequenceND`` and ``Bioformats``
(see :doc:`image_sequence` or :doc:`bioformats`).
are based on this baseclass and provide the axis-aware methods describe below.

To avoid ambiguity, we call a point along one axis a *coordinate*. The word
*index* refers to the working of ``frames[index]``. The index is equal to the
coordinate along the iterating axis (normally, `t`), except when the iterating
axis is nested (see below).

The names and sizes of each axis is provided with the ``sizes`` property:

.. ipython:: python

   images.sizes


Axes bundling
-------------

The ``bundle_axes`` property defines which axes will be present in a single frame.
The ``frame_shape`` property is changed accordingly:

.. ipython:: python

   images.bundle_axes = 'czyx'
   images.frame_shape

   images.bundle_axes = 'yx'
   images.frame_shape

Currently, the last two axes have to be ``'yx'``. For multi-symbol axis names,
provide a list like this: ``images.bundle_axes = ['one', 'two']``.


Axes iteration
--------------

The ``iter_axes`` property defines which axis will be used as the index axis. The
reader length is updated accordingly:

.. ipython:: python

   images.iter_axes = 't'
   len(images)

When multiple axes are provided to ``iter_axes``, a nested iteration will be
performed in which the last element will iterate fastest:

.. ipython:: python

   images.iter_axes = 'tz'
   len(images)  # 4 * 10
   images[12];  # returns the image at t == 1 and z = 2

Default coordinates
-------------------

What if an axis is not present in ``bundle_axes`` and ``iter_axes``? Then the
*default coordinate* is returned, as defined in the dictionary ``default_coords``:

.. ipython:: python

   images.bundle_axes = 'zyx'
   images.iter_axes = 't'
   images.default_coords['c'] = 1

   images[2];  # returns the 3D image at t == 2 and c = 1

.. ipython:: python
   :suppress:

   clean_dummy_png('.', filenames)

Make your own multidimensional reader
-------------------------------------

Making a multidimensional reader class yourself is simple. The following
example is already a fully-functioning multidimensional reader. The crucial
method here is ``_register_get_frame``, that registers a ``get_frame`` method
and tells the reader which axes to expect from that method. You can also define
multiple ``get_frame`` methods to increase the reader performance.

The reader then figures out how to efficiently use this function, to present
the image in the shape that corresponds with the ``bundle_axes`` settings.

.. code-block:: python

   from pims import FramesSequenceND
   import numpy as np

   class IndexReturningReader(FramesSequenceND):
      @property
      def pixel_type(self):
          return np.uint8  # the pixel datatype

      def __init__(self, size_c, size_t, size_z):
          # first call the baseclass initialization
          super(IndexReturningReader, self).__init__()
          self._init_axis('x', 3)
          self._init_axis('y', 1)
          self._init_axis('c', size_c)
          self._init_axis('t', size_t)
          self._init_axis('z', size_z)
          # register the get_frame function
          self._register_get_frame(self.get_frame_func, 'yx')

      def get_frame_func(self, c, t, z):
          return np.array([[c, t, z]], dtype=np.uint8)
