Bioformats
==========

.. seealso:: The section :doc:`multidimensional` describes how to deal with multidimensional files.

The ``Bioformats`` reader interfaces with Bio-Formats, an open-source Java
library for reading and writing multidimensional image data, especially from
file formats used in microscopy.

See the `list of supported formats <https://docs.openmicroscopy.org/bio-formats/6.5.0/supported-formats.html>`_.

Dependencies
------------
To interface with the java library, we use
`JPype <https://github.com/originell/jpype>`_, which allows fast and easy access
to all java functions. JRE or JDK are not required.

For `Anaconda <https://docs.continuum.io/anaconda/>`_ users,
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

See the documentation for the `Metadata retrieve API <https://www-legacy.openmicroscopy.org/site/support/bio-formats5.1/developers/cpp/tutorial.html>`_ for more details.

Updating bioformats
-------------------

To update the version of bioformats you are using in pims:

1. Find the version number for the latest bioformats release at: https://downloads.openmicroscopy.org/bio-formats/
2. Run this pims command, replacing the version number with the one you want:

.. code-block:: python

    pims.bioformats.download_jar(version='6.5')

Now you should be able to use pims with the updated bioformats version.

Note: This pims command downloads a bioformats file named `loci_tools.jar`
to your computer. There are a few possible locations where it might be stored.
The precedence order is (highest priority first):

1. pims package location
2. PROGRAMDATA/pims/loci_tools.jar
3. LOCALAPPDATA/pims/loci_tools.jar
4. APPDATA/pims/loci_tools.jar
5. /etc/loci_tools.jar
6. ~/.config/pims/loci_tools.jar

If you encounter problems updating to the latest version of bioformats,
you may wish to manually remove `loci_tools.jar` from each of the six locations
and re-run the `pims.bioformats.download_jar` command again.
