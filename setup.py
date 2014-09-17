from __future__ import print_function
# This downloads and install setuptools if it is not installed.
from ez_setup import use_setuptools
use_setuptools()

import os
import numpy
from numpy.distutils.core import setup, Extension

import warnings

# try bootstrapping setuptools if it doesn't exist
try:
    import pkg_resources
    try:
        pkg_resources.require("setuptools>=0.6c5")
    except pkg_resources.VersionConflict:
        from ez_setup import use_setuptools
        use_setuptools(version="0.6c5")
    from setuptools import setup, Extension
    _have_setuptools = True
except ImportError:
    # no setuptools installed
    from numpy.distutils.core import setup
    _have_setuptools = False

import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'pims/_version.py'
versioneer.versionfile_build = 'pims/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = '.'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup_parameters = dict(
    name="PIMS",
    version=FULLVERSION,
    description="Python Image Sequence",
    ext_modules=[Extension('_tifffile', ['pims/extern/tifffile.c'],
                    include_dirs=[numpy.get_include()])],
    author="Daniel Allan",
    install_requires=['numpy'],
    author_email="dallan@pha.jhu.edu",
    url="https://github.com/soft-matter/pims",
    packages=['pims', 'pims.extern'])

try:
    setup(**setup_parameters)
except SystemExit:
    warnings.warn(
        """DON'T PANIC! Compiling C is not working, so I will
skip the components that need a C compiler.""")
    # Try again without ext_modules.
    del setup_parameters['ext_modules']
    setup(**setup_parameters)
