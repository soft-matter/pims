Video
=====

.. note::

   There are many types of video formats and encodings, and many of them can
   be read by FFmpeg or libav. However, some file types can be tricky to read.
   For instance, frame indices can be inaccurate and different between
   the several FFmpeg/libav interfaces that PIMS implements. Therefore we
   advise less sophisticated users to simply work around this requirement by
   converting their video files to folders full of images using a utility like
   `ImageJ <http://rsb.info.nih.gov/ij/>`_.

   But if either FFmpeg or libav is available, PIMS enables fast random access
   to video files.

PyAV (fastest)
--------------

`PyAV <http://mikeboers.github.io/PyAV/>`_ can be installed via Anaconda, as follows:

.. code-block:: bash

    conda install av -c conda-forge

Non-anaconda users will have to compile PyAV themselves, which is complicated,
especially on Windows. For this we refer the
users to the `PyAV documentation <https://mikeboers.github.io/PyAV/>`_.

There are two ways PIMS provides random access to video files, which is not
something that video formats natively support:

* ``PyAVReaderTimed`` bases the indices of the video frames on the
``frame_rate`` that is reported by the video file, along with the timestamps
that are imprinted on the separate video frames. The readers ``PyAVVideoReader``
and ``Video`` are different names for this reader.
* ``PyAVReaderIndexed`` scans through the entire video to build a table
of contents. This means that opening the file can take some time, but
once it is open, random access is fast. In the case timestamps or `frame_rate``
are not available, this reader is the preferred option.


Troubleshooting
---------------

If you use conda / Anaconda, watch out for an error like:

.. code-block:: bash

    version `GLIBC_2.15' not found

This seems to be because `conda includes an old version of a library
needed by PyAV <github.com/ContinuumIO/anaconda-issues/issues/182>`__.
To work around this, simply delete anaconda's version of the library:

.. code-block:: bash

    rm ~/anaconda/lib/libm.so.6

and/or

.. code-block:: bash

    rm ~/anaconda/envs/name_of_your_environment/lib/libm.so.6

which will cause PyAV to use the your operating system's version of the
library.


ImageIO and MoviePy
-------------------
Both `ImageIO <https://imageio.github.io>`_ and `MoviePy <http://zulko.github.io/moviepy>`_
implement interfaces with ffmpeg through a Pipe. These are implemented through
``ImageIOReader`` and ``MoviePyReader``, respectively.
