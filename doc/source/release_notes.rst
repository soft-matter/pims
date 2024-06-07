Release notes
=============

v0.7
------
This is a major release that includes compatibility with numpy 2.0, the
end of libtiff support, and some bug fixes.

-  MNT: numpy2 compatibility by @tacaswell in
   https://github.com/soft-matter/pims/pull/457
-  MNT: Drop support for libtiff by @nkeim in
   https://github.com/soft-matter/pims/pull/441
-  FIX: Be more lenient when parsing the TIFF DateTime metadata tag. by
   @anntzer in https://github.com/soft-matter/pims/pull/424
-  FIX: Don’t emit warnings that duplicate exceptions on opening
   failure. by @anntzer in https://github.com/soft-matter/pims/pull/434
-  FIX: Close files in case of reader failure by @nkeim in
   https://github.com/soft-matter/pims/pull/445
-  MNT: Update/fix CI setup. by @anntzer in
   https://github.com/soft-matter/pims/pull/437
-  DOC: Update imageio link by @dstansby in
   https://github.com/soft-matter/pims/pull/450
-  DOC: branch name in link to example nb by @dmarx in
   https://github.com/soft-matter/pims/pull/428
-  DOC: Clarify sorting rule in ImageSequence docs by @anntzer in
   https://github.com/soft-matter/pims/pull/423

New Contributors:

-  @dmarx made their first contribution in
   https://github.com/soft-matter/pims/pull/428
-  @dstansby made their first contribution in
   https://github.com/soft-matter/pims/pull/450


v0.6.1
------
This is a bugfix release recommended for all users.

- Fixed imageio "infinite reader" bug (PR 415)
- Added installation instructions for imageio-ffmpeg and moviepy (PR 417)


v0.6
----
This is a major release recommended for all users.

- Added direct dependence on imageio (PR 392); imageio-ffmpeg is optional (PR 358)
- Added compatibility with Python 3.10 and 3.11 (PR 383, PR 399)
- Ended support for Python 2 (PR 408)
- PIMS reader objects now support the ``dtype`` and ``shape`` attributes from the
  standard numpy array interface (PR 254)
- Fixed frame numbering bug for PyAV (PR 370)
- Fixed handling of nonstandard datetime tags in TIFFs (PR 375)
- Updated documentation (PR 376, PR 390)
- Fixed and improved automated testing and development (PR 380, PR 383,
  PR 399, PR 391, PR 407, PR 410)
- Removed use of deprecated skimage.external.tifffile (PR 362)
- Fixed imageio video API support (PR 406)


v0.5
----
This is a major release recommended for all users. Your existing code to
open files may need to be edited slightly -- see the first change listed below.

- API: all readers do not support the keyword arguments ``process_func``,
  ``dtype`` and ``as_grey`` anymore. Please consult the documentation on
  Pipelines on how to convert videos. (see :doc:`pipelines`) (PR 250)
- New built-in ``as_grey`` pipeline for convenient conversion to greyscale
  (PR 305)
- New built-in ``crop`` pipeline for making a cropped sequence (PR 247)
- Major speedup for random access in PyAVReaderIndexed (PR 340)
- Bumped Bioformats version to 6.5.x (PR 301)
- Added instructions for installing/updating Bioformats (PR 346)
- Enhanced support for metadata in the CINE format (PR 317)
- Enhanced documentation for PyAV video support (PR 334)
- Added multidimensional capabilities to ImageIOReader
  (see :doc:`multidimensional`) (PR 320)
- Added support for nd2reader as alternative reader for Nikon nd2 files (PR 272)
- Removed fallback to scipy.misc.imread, which was removed from scipy (PR 359)
- Fixed compatibility with PyAV 0.4.0 and newer (PR 300)
- Fixed compatibility and bugs in PyAV export (PR 283, PR 313)
- Fixed opening of PyAV videos with audio (PR 322)
- Fixed compatibility with newer versions of tifffile (PR 314, PR 339)
- Fixed response to missing ImageIO (PR 333)


v0.4
----
- API: N-dimensional readers now smartly use ``get_frame`` methods; depending on
  the ``bundle_axes`` settings, the reader optimizes which ``get_frame`` methods is
  optimal to use. Readers that derive from ``FramesSequenceND`` will need to call
  ``FramesSequenceND.__init__()`` on initialization, and also will need to register
  ``get_frame`` methods using ``_register_get_frame(method, axes)``. (PR 227)
- API: Swap elements of ``frame_shape`` in ``SpeStack`` to match frames' ``shape``. (PR 241)
- API: Swap elements of ``frame_shape`` in ``PyAVVideoReader`` to match frames' ``shape``. (PR 251)
- API: Reimplement the ``PyAVVideoReader`` (or: ``Video``) into a reader that
  uses the frame timestamps and reader frame rate to compute the frame index. The
  new reader is named ``PyAVReaderTimed`` and the names ``PyAVVideoReader`` and
  ``Video`` refer to it. For the case when the video misses timestamps, the
  old implementation is available under ``PyAVReaderIndexed``.
- API: The video exporter (``export``) takes more arguments. The argument ``'format'``
  has been renamed to ``'pixel_format'``, while ``'format'`` now refers to the
  container format. (PR 257)
- Fixed filename sorting when list is provided explicitely to ``ImageSequence`` (PR 252)
- Fixed ``plot_to_frame`` with ``savefig.bbox == 'tight'`` (PR 248)
- Added a reader that wraps ImageIO ``ImageIOReader`` (PR 233)
- Added a reader that wraps MoviePy ``MoviePyReader`` (PR 233)
- Added a reader for single images ``ImageReader`` and ``ImageReaderND`` (PR 249)
- Added a reader that stacks readers into a multidimensional reader ``ReaderSequence`` (PR 249)
- Added a video exporter based on MoviePy (PR 233)
- Added ``BioformatsReader.metadata.fields`` that lists all metadata fields. (PR 230)

v0.3.4
------
- API: Swap elements of ``frame_shape`` in ``SpeStack`` to match frames'
  ``shape``.

v0.3.3
------
- Fix compatibility with Pillow v0.3.0 (PR 204)
- API: ``plot_to_frame`` and ``plots_to_frame`` now take ``fig_size_inches`` and ``bbox_inches`` as keyword arguments instead of passing on all keyword arguments to ``imsave`` (PR 206)
- Fix zipfile handling in Py3 (PR 199)
- Fix CINE reader for Py2 (PR 219)
- Support non-monochrome and packed-bits images in NorpixSeq (PR 218)
- Update documentation
- Update slicerator dependency to v0.9.7 (fixes pipeline nesting)
- Update bioformats version to v5.1.7 (PR 224)

v0.3.2
------
- Bug fixes
- Build fixes and fewer required deps

v0.3.1
------
- Fix build-related mistakes in v0.3.0.

v0.3.0
------

* Refactor the slicing logic into a separate package, slicerator.
* Extend the slicing logic to allow nested lazy slicing.
* Add pipeline feature; deprecate process_func.
* Support multispectral and multidimensional images.
* Add Norpix reader.

v0.2.2
------
This is a simple maintenance release, introducing no functionality changes. The
packaging and installation are simplified by adopting tifffile as a dependency
rather than directly including the source PIMS.

Henceforth, to install PIMS on any platform, we recommend
``conda install -c soft-matter pims``, but ``pip install pims`` is also supported.

v0.2.1
------

* Use PyAV for handling video files
* Ships with Christoph Gohlke's tifffile
* Added support for .cine files
* Added prototype of universal open function which tries to guess the correct class to use to handle a given file based on the extension
* Added ability to create an ImageSequence from a list of paths
