from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose)

import pims
from test_common import _image_single, _image_series

try:
    import javabridge
    import bioformats
    BIOFORMATS_INSTALLED = True
except ImportError:
    BIOFORMATS_INSTALLED = False

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data', 'bioformats')


def _skip_if_no_bioformats():
    if not BIOFORMATS_INSTALLED:
        raise nose.SkipTest('Bioformats and/or javabridge not installed. Skipping.')


def assert_image_equal(actual, expected):
    if np.issubdtype(actual.dtype, np.integer):
        assert_equal(actual, expected)
    else:
        if np.issubdtype(expected.dtype, np.integer):
            expected = expected/float(np.iinfo(expected.dtype).max)
        assert_allclose(actual, expected, atol=1/256.)


class _image_stack(unittest.TestCase):
    def check_skip(self):
        pass

    def test_getting_stack(self):
        self.check_skip()
        assert_image_equal(self.v[0][0], self.frame0)
        assert_image_equal(self.v[0][1], self.frame1)

    def test_sizeZ(self):
        self.check_skip()
        assert_equal(self.v.sizes['Z'], self.expected_Z)


class _image_multichannel(unittest.TestCase):
    def check_skip(self):
        pass

    def test_change_channel(self):
        self.check_skip()
        self.v.channel = (0, 1)
        channel0, channel1 = self.v[0][0], self.v[0][1]
        self.v.channel = 0
        assert_image_equal(self.v[0], channel0)
        self.v.channel = 1
        assert_image_equal(self.v[0], channel1)

    def test_sizeC(self):
        self.check_skip()
        assert_equal(self.v.sizes['C'], self.expected_C)


class TestND2(_image_series, _image_multichannel):
    # Nikon NIS-Elements ND2
    # 38 x 31 pixels, 16 bits, 2 channels, 3 time points, 10 focal planes
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'cluster.nd2')
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.frame0 = np.load(os.path.join(path, 'nd2_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'nd2_frame1.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (31, 38)
        self.expected_len = 3
        self.expected_Z = 10
        self.expected_C = 2

    def tearDown(self):
        self.v.close()


class TestIPL(_image_single):
    # IPLab format, 650 x 515 pixels, 8 bits per sample, RGB
    # Scanalytics has provided a sample multi-channel image in IPLab format.
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'blend_final.ipl')
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.frame0 = np.load(os.path.join(path, 'blend_final.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (515, 650, 3)
        self.expected_len = 1

    def tearDown(self):
        self.v.close()


class TestSEQ(_image_single, _image_stack):
    # Image-Pro sequence format
    # 512 x 512 pixels, 8 bits per sample, 30 focal planes
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'heart.seq')
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.frame0 = np.load(os.path.join(path, 'heart_plane0.npy'))
        self.frame1 = np.load(os.path.join(path, 'heart_plane1.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 1
        self.expected_Z = 30

    def tearDown(self):
        self.v.close()


class TestLEI(_image_single, _image_stack):
    # Leica format
    # 256 x 256 pixels, 8 bits per sample, 3 focal planes
    # Clay Glennon of the Wisconsin National Primate Research Center at the
    # UW-Madison has provided a dataset in Leica format.
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'leica_stack.lei')
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.frame0 = np.load(os.path.join(path, 'leica_stack_plane0.npy'))
        self.frame1 = np.load(os.path.join(path, 'leica_stack_plane1.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (256, 256)
        self.expected_len = 1
        self.expected_Z = 3

    def tearDown(self):
        self.v.close()


class TestICS(_image_single):
    # Image Cytometry Standard format
    # 256 x 256 pixels, 8 bits per sample
    # Nico Stuurman of the Department of Cellular and Molecular Pharmacology at
    # the University of California-San Francisco has provided an image in Image
    # Cytometry Standard (ICS) format.
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'qdna1.ics')
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.frame0 = np.load(os.path.join(path, 'qdna1_frame0.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (256, 256)
        self.expected_len = 1

    def tearDown(self):
        self.v.close()


class TestMetadata(unittest.TestCase):
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'cluster.nd2')
        self.klass = pims.Bioformats

    def tearDown(self):
        self.v.close()

    def test_metadataretrieve(self):
        # tests using the metadata object are combined in one, to reduce the
        # amount of log output.
        self.v = self.klass(self.filename, meta=True, C=0)
        # test fields directly
        assert_equal(self.v.metadata.getChannelCount(0), 2)
        assert_equal(self.v.metadata.getChannelName(0, 0), '5-FAM/pH 9.0')
        assert_almost_equal(self.v.metadata.getPixelsPhysicalSizeX(0),
                            0.167808983)
        # test metadata in Frame objects
        assert_almost_equal(self.v[0].metadata['T'], 0.445083498)
        assert_equal(self.v[0].metadata['indexT'], 0)
        # test changing frame_metadata
        del self.v.frame_metadata['T']
        assert 'T' not in self.v[0].metadata
        self.v.frame_metadata['T'] = 'getPlaneDeltaT'
        assert 'T' in self.v[0].metadata
        # test colors field
        assert_allclose(self.v[0].metadata['colors'][0], [0.47, 0.91, 0.06],
                        atol=0.01)

    def test_metadata_raw(self):
        self.v = self.klass(self.filename, meta=False, C=0)
        metadata = self.v.get_metadata_raw('dict')
        assert_equal(metadata['ChannelCount'], '2')
        assert_equal(metadata['CH2ChannelDyeName'], '5-FAM/pH 9.0')
        assert_equal(metadata['dCalibration'], '0.16780898323268245')

    def test_metadata_omexml(self):
        self.v = self.klass(self.filename, meta=False, C=0)
        omexml = self.v.get_metadata_omexml()
        assert_equal(omexml.image().Pixels.SizeC, 2)
        assert_equal(omexml.image().Pixels.Channel(0).Name, '5-FAM/pH 9.0')


class zzzKillVM(unittest.TestCase):
    def check_skip(self):
        _skip_if_no_bioformats()

    def test_kill_javaVM(self):
        self.check_skip()
        pims.kill_vm()


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
