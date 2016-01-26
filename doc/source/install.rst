Installation
============

.. seealso:: What has been changed recently? Check out the :doc:`release_notes`.

PIMS is easy to install on Windows, OSX, or Linux. Its dependencies are:

* `numpy <http://www.numpy.org/>`_
* `slicerator <https://github.com/soft-matter/slicerator/>`_

For basic image reading one of the following is required:

* `scikit-image <http://scikit-image.org/>`_
* `matplotlib <http://matplotlib.org/>`_
* `scipy <http://www.scipy.org/>`_

Depending on what file formats you want to read, you will also need:

-  `ffmpeg <https://www.ffmpeg.org/>`__ and
   `PyAV <http://mikeboers.github.io/PyAV/>`__ (video formats such as
   AVI, MOV)
-  `jpype <http://jpype.readthedocs.org/en/latest/>`__ (interface with
   bioformats to support
   `many <https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html>`__
   microscopy formats)
-  `Pillow <http://pillow.readthedocs.org/en/latest/>`__ (improved TIFF
   support)
-  `libtiff <https://code.google.com/p/pylibtiff/>`__ (alternative TIFF
   support)
-  `tifffile <http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`__
   (alterative TIFF support)
-  `pims\_nd2 <https://github.com/soft-matter/pims_nd2>`__ (improved
   Nikon .nd2 support)

Recommended: conda
------------------

.. note::

   To get started with Python on any platform, download and install
   `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.
   It comes with the common scientific Python packages built in.

With `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ 
(or `Miniconda <http://conda.pydata.org/miniconda.html>`_) installed,
type the following into a Terminal to install PIMS. That's "Terminal" on a Mac,
and "Start > Applications > Command Prompt" on Windows. Type these lines:

.. code-block:: bash

   conda install -c soft-matter pims

The above installs the latest stable release. Finally, to try it out, type:

.. code-block:: bash

    ipython notebook


Development version
-------------------
To install the version under active development, with the latest tested code,
use the development channel.

.. code-block:: bash

   conda config --add channels soft-matter
   conda install -c soft-matter/channel/dev pims


pip
---

PIMS can also be installed using pip.

.. code-block:: bash

   pip install pims

source
------

If you plan to edit the code, you can install PIMS manually.

.. code-block:: bash

   git clone https://github.com/soft-matter/pims
   cd pims
   python setup.py develop
