Video
=====

PIMS provides reading of video through :ref:`video-pyav`,
:ref:`ImageIO <video-imageio-moviepy>`
or :ref:`MoviePy <video-imageio-moviepy>`.

.. _video-pyav:

PyAV (fastest)
--------------


`PyAV <https://github.com/PyAV-Org/PyAV>`_ can be installed via Anaconda, as follows:

.. code-block:: bash

    conda install av -c conda-forge

Non-anaconda users will have to compile PyAV themselves, which is complicated,
especially on Windows. For this we refer the
users to the `PyAV documentation <https://pyav.org/docs/>`_.

There are two ways PIMS provides random access to video files, which is not
something that video formats natively support:

* :class:`PyAVReaderTimed <pims.PyAVReaderTimed>` bases the indices of the video frames on the
  ``frame_rate`` that is reported by the video file, along with the timestamps
  that are imprinted on the separate video frames. The readers ``PyAVVideoReader``
  and ``Video`` are different names for this reader.
* :class:`PyAVReaderIndexed <pims.PyAVReaderIndexed>` scans through the entire video to build a table
  of contents. This means that opening the file can take some time, but
  once it is open, random access is fast. In the case timestamps or `frame_rate``
  are not available, this reader is the preferred option.

.. autoclass:: pims.PyAVReaderTimed
   :members:

.. autoclass:: pims.PyAVReaderIndexed
   :members:

.. _video-imageio-moviepy:

ImageIO and MoviePy
-------------------
`imageio-ffmpeg <https://github.com/imageio/imageio-ffmpeg>`_ and `moviepy <https://github.com/Zulko/moviepy>`_ 
can be installed via Anaconda, as follows:

.. code-block:: bash

    conda install imageio-ffmpeg -c conda-forge
    conda install moviepy -c conda-forge

Both `ImageIO <https://imageio.github.io>`_ and `MoviePy <http://zulko.github.io/moviepy>`_
implement interfaces with ffmpeg through a Pipe. These are implemented through
:class:`ImageIOReader <pims.ImageIOReader>` and :class:`MoviePyReader <pims.MoviePyReader>`, respectively.

.. autoclass:: pims.ImageIOReader
   :members:

.. The import of pims.moviepy_reader fails. MoviePy seems to use imageio in turn.
    .. autoclass:: pims.MoviePyReader
    :members:
