Video
=====

.. note::

   To load video files directly, you need FFmpeg or libav. These can be tricky to install, especially on Windows, so we advise less sophisticated users to simply work around this requirement by converting their video files to folders full of images using a utility like `ImageJ <http://rsb.info.nih.gov/ij/>`_.

   But if either FFmpeg or libav is available, PIMS enables fast random access to video files. 

The ``Video`` reader can open any file format supported by FFmepg and libav.

In order to provide random access by frame number, which is not something that
video formats natively support, PIMS scans through the entire video to build
a table of contents. This means that opening the file can take some time, but
once it is open, random access is fast.

Dependencies
------------

* `FFmpeg <http://ffmpeg.org/>`_ or `libav <http://libav.org/>`_
* `PyAV <http://mikeboers.github.io/PyAV/>`_

FFmpeg and libav can be installed using a package manager like aptitude or
homebrew. PyAV can be installed using pip.
