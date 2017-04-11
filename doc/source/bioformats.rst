Bioformats
==========

.. seealso:: The section :doc:`multidimensional` describes how to deal with multidimensional files.

The ``Bioformats`` reader interfaces with Bio-Formats, an open-source Java
library for reading and writing multidimensional image data, especially from
file formats used in microscopy.

See the `list of supported formats <https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html>`_.

Dependencies
------------
To interface with the java library, we use
`JPype <https://github.com/originell/jpype>`_, which allows fast and easy access
to all java functions. JRE or JDK are not required.

For `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ users,
platforms, jpype is available via the ``conda-forge`` channel:

.. code-block:: bash

    conda install jpype1 -c conda-forge


For installation with pip, type in the following into the terminal:

.. code-block:: bash

    pip install jpype1

Or, for windows users,
download the binary from `Christoph Gohlke's website <http://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype>`_.

On first use of pims.Bioformats(filename), the required java library
:file:`loci_tools.jar` will be automatically downloaded from
`openmicroscopy.org <http://downloads.openmicroscopy.org/bio-formats/>`__.

Special functions
-----------------
Some files contain multiple experiments. The ``series`` argument or property
switches between them:

.. code-block:: python

   # open a multi-experiment file and read the first experiment
   reader = pims.BioformatsReader('path/to/file', series=0)
   # switch to the third experiment
   reader.series = 2

Very large files may need more Java memory. If you ever encounter a memory error,
open a file with for instance 1 GB of java memory:

.. code-block:: python

   reader = BioformatsReader('path/to/file', java_memory='1024m')

Metadata
--------

The ``Bioformats`` reader can be used to access the metadata stored in the image,
including physical dimensions pixel, instrument parameters, and other useful information.
For performance increase, this function may be toggled off using the ``meta=False``
keyword argument.

.. code-block:: python

    meta = images.metadata

    image_count = meta.ImageCount()
    print('Total number of images: {}'.format(image_count))

    for i in range(image_count):
        print('Dimensions for image {}'.format(i))
        shape = (meta.PixelsSizeX(i), meta.PixelsSizeY(i), meta.PixelsSizeZ(i))
        dxyz = (meta.PixelsPhysicalSizeX(i),
                meta.PixelsPhysicalSizeY(i),
                meta.PixelsPhysicalSizeZ(i))
        print('\tShape: {} x {} x {}'.format(*shape))
        print('\tDxyz:  {:2.2f} x {:2.2f} x {:2.2f}'.format(*dxyz))

See the documentation for the `Metadata retrieve API <http://www.openmicroscopy.org/site/support/bio-formats5.1/developers/cpp/tutorial.html>`_ for more details.
