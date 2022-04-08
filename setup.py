import os
from setuptools import setup
import versioneer


try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except IOError:
    descr = ''

setup_parameters = dict(
    name="PIMS",
    license="BSD-3-clause",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python Image Sequence",
    author="PIMS Contributors",
    python_requires=">=3.7",
    install_requires=[
        'imageio',
        'numpy>=1.19',
        'slicerator>=0.9.8',
    ],
    author_email="dallan@pha.jhu.edu",
    url="https://github.com/soft-matter/pims",
    packages=["pims", "pims.utils", "pims.tests"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    long_description=descr,
    long_description_content_type='text/markdown',
)

setup(**setup_parameters)
