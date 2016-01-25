Release notes
=============

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
