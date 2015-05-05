Frame
=====

PIMS returns images as ``Frame`` objects. Frames can be treated precisely
the same as numpy arrays.

They are a subclass of ``numpy.ndarray``, adding two new attributes:

* ``frame_no``, an integer
* ``metadata``, a dictionary

These can be used by the PIMS readers to provide any metadata stored in the
image files. Setting these attriburtes is optional.

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
