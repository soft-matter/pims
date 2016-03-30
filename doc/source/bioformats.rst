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

For installation with pip, type in the following into the terminal:

.. code-block:: bash

    pip install jpype1

Or, for windows users,
download the binary from `Christoph Gohlke's website <http://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype>`_.

On first use of pims.Bioformats(filename), the required java library
:file:`loci_tools.jar` will be automatically downloaded from
`openmicroscopy.org <http://downloads.openmicroscopy.org/bio-formats/>`__.

Metadata
--------

The ``Bioformats`` reader can be used to access the metadata stored in the image, including physical dimensions pixel, instrument parameters, and other useful information:

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
