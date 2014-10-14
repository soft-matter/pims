from __future__ import print_function
import os
import warnings
import setuptools
from setuptools import setup, Extension
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
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    description="Python Image Sequence",
    author="PIMS Contributors",
    install_requires=['six>=1.8', 'numpy>=1.7', 'tifffile>=0.3.1'],
    author_email="dallan@pha.jhu.edu",
    url="https://github.com/soft-matter/pims",
    packages=['pims'],
    long_description=read('README.md'))

setup(**setup_parameters)
