from __future__ import print_function
import os
from setuptools import setup
import versioneer


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup_parameters = dict(
    name="PIMS",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python Image Sequence",
    author="PIMS Contributors",
    install_requires=['six>=1.8', 'numpy>=1.7'],
    author_email="dallan@pha.jhu.edu",
    url="https://github.com/soft-matter/pims",
    packages=['pims',
              'pims.utils',
              'pims.tests'],
    long_description=read('README.md'))

setup(**setup_parameters)
