Pipelines
=========

.. ipython:: python
   :suppress:

   from pims.tests.test_common import save_dummy_png, clean_dummy_png
   filenames = ['img-{0}.png'.format(i) for i in range(9)]
   save_dummy_png('.', filenames, (256, 256, 3))
   from pims import ImageSequence
   video = ImageSequence('img-*.png')

   def _as_grey(frame):
       red = frame[:, :, 0]
       green = frame[:, :, 1]
       blue = frame[:, :, 2]
       return 0.2125 * red + 0.7154 * green + 0.0721 * blue

   from slicerator import pipeline
   as_grey = pipeline(_as_grey)

Videos loaded by pims (``FramesSequence`` objects) are like lists of numpy
arrays. Unlike Python lists of arrays they are "lazy", they only load the data
from the harddrive when it is necessary.

In order to modify a ``FramesSequence``, for instance to convert RGB color
videos to grayscale, one could load all the video frames in memory and do the
conversion. This however costs a lot of time and memory, and for very large
videos this is just not feasible. To solve this problem, PIMS uses
so-called pipeline decorators from a sister project called ``slicerator``.
A pipeline-decorated function is only evaluated when needed, so that the
underlying video data is only accessed one element at a time.

Conversion to greyscale
-----------------------

Say we want to convert an RGB video to greyscale. A pipeline to do this is
already provided as ``pims.as_grey``, but it is also easy to make our own.
We define a function as follows and decorate it with ``@pipeline`` to turn
it into a pipeline:

.. code-block:: python

   from slicerator import pipeline  # or: from pims import pipeline

   @pipeline
   def as_grey(frame):
       red = frame[:, :, 0]
       green = frame[:, :, 1]
       blue = frame[:, :, 2]
       return 0.2125 * red + 0.7154 * green + 0.0721 * blue


The behavior of ``as_grey`` is unchanged if it is used on a single frame:

.. ipython:: python

   frame = video[0]
   print(frame.shape)   # the shape of the example video is RGB
   processed_frame = as_grey(frame)
   print(processed_frame.shape)  # the converted frame is indeed greyscale


However, the ``@pipeline`` decorator enables lazy evaluation of full videos:

.. ipython:: python

   processed_video = as_grey(video)  # this would not be possible without @pipeline
   # nothing has been converted yet!

   processed_frame = processed_video[0]  # now the conversion takes place
   print(processed_frame.shape)

This means that the modified video can be used exactly as you would use the
original one. In most cases, it will look as though you are accessing
a grayscale video file, even though the file on disk is still in color.
Please keep in mind that these simple pipelines do not change the reader
properties, such as ``video.frame_shape``. Propagating metadata properly through
pipelines is partly implemented, but currently still experimental.
For a detailed description of this tricky point, please consult
`this <https://github.com/soft-matter/slicerator/pull/5#issuecomment-143560978>`_
discussion on GitHub.


Converting existing functions to a pipeline
-------------------------------------------

.. note:: This supersedes the ``process_func`` and ``as_grey`` reader keyword
 arguments starting from PIMS v0.4

We are now going to do exactly the same greyscale conversion, but using an
existing function from ``skimage``:

.. ipython:: python

   from skimage.color import rgb2gray
   rgb2gray_pipeline = pipeline(rgb2gray)
   processed_video = rgb2gray_pipeline(video)
   processed_frame = processed_video[0]
   print(processed_frame.shape)


Any function that takes a single frame and returns a single frame can be converted
into a pipeline in this way.


Dtype conversion using lambda functions
---------------------------------------

.. note:: This supersedes the ``dtype`` reader keyword argument starting from PIMS v0.4

We are now going to convert the data type of a video to float using an
unnamed lambda function in a single line:

.. ipython:: python

   processed_video = pipeline(lambda x: x.astype(np.float))(video)
   processed_frame = processed_video[0]
   print(processed_frame.shape)

.. ipython:: python
   :suppress:

   clean_dummy_png('.', filenames)