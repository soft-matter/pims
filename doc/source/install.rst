Installation
============

PIMS is easy to install on Windows, OSX, or Linux.

Recommended: conda 
------------------


.. note::

   To get started with Python on any platform, download and install
   `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.
   It comes with the common scientific Python packages built in.

With `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ 
(or `Miniconda <http://conda.pydata.org/miniconda.html>`_) installed,
type the following into a Terminal to install PIMS.

.. code-block:: bash

   conda install -c soft-matter pims

The above installs the latest stable to release. To install the version under
active development, with the latest tested code, use the development channel.

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

