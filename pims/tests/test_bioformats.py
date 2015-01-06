from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal, assert_allclose)

import sys
sys.path.append(r'E:\Documents\Scripts\GitHub\pims')
import pims
try:
    from javabridge import kill_vm
    from bioformats import JARS
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


class _image_single(unittest.TestCase):
    def check_skip(self):
        pass

    def test_getting_single_frame(self):
        self.check_skip()
        assert_image_equal(self.v[0], self.frame0)

    def test_shape(self):
        self.check_skip()
        assert_equal(self.v.frame_shape, self.expected_shape)

    def test_count(self):
        self.check_skip()
        assert_equal(len(self.v), self.expected_len)

    def test_bool(self):
        self.check_skip()
        pass

    def test_repr(self):
        self.check_skip()
        # simple smoke test, values not checked
        repr(self.v)


class _image_series(_image_single):
    def check_skip(self):
        pass

    def test_getting_slice(self):
        self.check_skip()
        tmp = list(self.v[0:2])
        frame0, frame1 = tmp
        assert_image_equal(frame0, self.frame0)
        assert_image_equal(frame1, self.frame1)

    def test_getting_list(self):
        self.check_skip()
        actual = list(self.v[[1, 0, 0, 1, 1]])
        expected = [self.frame1, self.frame0, self.frame0, self.frame1,
                    self.frame1]
        [assert_image_equal(a, b) for a, b in zip(actual, expected)]

    def test_integer_attributes(self):
        self.check_skip()
        assert_equal(len(self.v.frame_shape), len(self.expected_shape))
        self.assertTrue(isinstance(self.v.frame_shape[0], int))
        self.assertTrue(isinstance(self.v.frame_shape[1], int))
        self.assertTrue(isinstance(len(self.v), int))

    def test_simple_negative_index(self):
        self.check_skip()
        self.v[-1]
        list(self.v[[0, -1]])

    def test_frame_number_present(self):
        self.check_skip()
        for frame_no in [0, 1, 2, 1]:
            self.assertTrue(hasattr(self.v[frame_no], 'frame_no'))
            not_none = self.v[frame_no].frame_no is not None
            self.assertTrue(not_none)

    def test_frame_number_accurate(self):
        self.check_skip()
        for frame_no in [0, 1, 2, 1]:
            self.assertEqual(self.v[frame_no].frame_no, frame_no)

    def test_iterator(self):
        self.check_skip()
        i = iter(self.v)
        assert_image_equal(next(i), self.frame0)
        assert_image_equal(next(i), self.frame1)


class _image_stack(_image_single):
    def check_skip(self):
        pass

    def test_getting_single_frame(self):
        self.check_skip()
        assert_image_equal(self.v[0][0], self.frame0)
        assert_image_equal(self.v[0][1], self.frame1)

    def test_sizeZ(self):
        self.check_skip()
        assert_equal(self.v.sizes['Z'], self.expected_Z)


class TestND2(_image_series):
    # Nikon NIS-Elements ND2
    # 38 x 31 pixels, 16 bits per sample, 3 time points, 10 focal planes
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'cluster.nd2')
        self.klass = pims.Bioformats3D
        self.kwargs = {'C': (0, 1)}
        self.frame0 = np.load(os.path.join(path, 'nd2_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'nd2_frame1.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (38, 31)
        self.expected_len = 3

    def tearDown(self):
        self.v.close()


class TestIPL(_image_single):
    # IPLab format, 650 x 515 pixels, 8 bits per sample, 3 channels
    # Scanalytics has provided a sample multi-channel image in IPLab format.
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'blend_final.ipl')
        self.klass = pims.Bioformats3D
        self.kwargs = {}
        self.frame0 = np.load(os.path.join(path, 'blend_final.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (650, 515)
        self.expected_len = 1

    def tearDown(self):
        self.v.close()


class TestSEQ(_image_stack):
    # Image-Pro sequence format
    # 512 x 512 pixels, 8 bits per sample, 30 focal planes
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'heart.seq')
        self.klass = pims.Bioformats3D
        self.kwargs = {}
        self.frame0 = np.load(os.path.join(path, 'heart_plane0.npy'))
        self.frame1 = np.load(os.path.join(path, 'heart_plane1.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 1
        self.expected_Z = 30

    def tearDown(self):
        self.v.close()


class TestLEI(_image_stack):
    # Leica format
    # 256 x 256 pixels, 8 bits per sample, 3 focal planes
    # Clay Glennon of the Wisconsin National Primate Research Center at the
    # UW-Madison has provided a dataset in Leica format.
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'leica_stack.lei')
        self.klass = pims.Bioformats3D
        self.kwargs = {}
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
        self.klass = pims.Bioformats3D
        self.kwargs = {}
        self.frame0 = np.load(os.path.join(path, 'qdna1_frame0.npy'))
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (256, 256)
        self.expected_len = 1

    def tearDown(self):
        self.v.close()


class zzzKillVM(unittest.TestCase):
    def check_skip(self):
        _skip_if_no_bioformats()

    def test_kill_javaVM(self):
        self.check_skip()
        kill_vm()


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
