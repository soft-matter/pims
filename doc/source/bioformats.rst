Bioformats
==========

The ``Bioformats`` reader interfaces with Bio-Formats, an open-source Java
library for reading and writing multidimensional image data, especially from
file formats used in microscopy.

See the `list of supported formats <https://www.openmicroscopy.org/site/support/bio-formats5.1/supported-formats.html>`_.

Depedencies
-----------
To interface with the java library, we use `JPype <https://github.com/originell/jpype>`_, which allows fast and easy access to all java functions. JRE or JDK are not required.

On first use of pims.Bioformats(filename), the required java library :file:`loci_tools.jar` will be automatically downloaded from openmicroscopy.org.
