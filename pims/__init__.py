from pims.api import *

from pims.version import version as __version__

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
