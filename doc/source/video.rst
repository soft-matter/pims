Video
=====

.. note::

   To load video files directly, you need FFmpeg or libav. These can be tricky
   to install, especially on Windows, so we advise less sophisticated users to
   simply work around this requirement by converting their video files to
   folders full of images using a utility like `ImageJ <http://rsb.info.nih.gov/ij/>`_.

   But if either FFmpeg or libav is available, PIMS enables fast random access to video files.

PyAV is installed with the PIMS conda package. You can install it via pip like so:

.. code-block:: bash

    pip install av

The ``Video`` reader can open any file format supported by FFmepg and libav.

In order to provide random access by frame number, which is not something that
video formats natively support, PIMS scans through the entire video to build
a table of contents. This means that opening the file can take some time, but
once it is open, random access is fast.

Dependencies
------------

* `PyAV <http://mikeboers.github.io/PyAV/>`_

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
