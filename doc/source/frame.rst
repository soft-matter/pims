Frame
=====

PIMS returns images as ``Frame`` objects. Frames can be treated precisely
the same as numpy arrays.

They are a subclass of ``numpy.ndarray``, adding two new attributes:

* ``frame_no``, an integer
* ``metadata``, a dictionary

These can be used by the PIMS readers to provide any metadata stored in the
image files. Setting these attributes is optional.


IPython Rich Display
--------------------

Frame objects hook into IPython's rich display framework. In IPython notebooks, 2D Frames are displayed as actual images. 3D Frames ("Z stacks") are displayed as a stack of images. The user can scroll through the images with the scroll wheel.

Caveats
-------

As with a numpy array, if some mathematical operation reduces a Frame to
a scalar, the output is a standard Python scalar. Any extra attributes are
discarded.

.. ipython:: python

   from pims import Frame
   frame = Frame(np.ones((5, 5)))
   frame
   frame.sum()

.. warning::

   Propagating metadata is a hard problem, and it is not one that PIMS has not
   yet solved. If you combine two Frames in, for example, addition, the
   metadata of the left object is propagated. This will be addressed in a
   future release of PIMS.
