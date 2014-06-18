from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal)
import pims

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def _skip_if_no_ffmpeg():
    import pims.ffmpeg_reader
    if not pims.ffmpeg_reader.available():
        raise nose.SkipTest('FFmpeg not found. Skipping.')


def _skip_if_no_libtiff():
    try:
        import libtiff
    except ImportError:
        raise nose.SkipTest('libtiff not installed. Skipping.')


class _base_klass(unittest.TestCase):
    def check_skip(self):
        pass

    def test_getting_slice(self):
        self.check_skip()
        tmp = list(self.v[0:2])
        frame0, frame1 = tmp
        assert_equal(frame0, self.frame0)
        assert_equal(frame1, self.frame1)

    def test_getting_single_frame(self):
        self.check_skip()
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[1], self.frame1)
        assert_equal(self.v[1], self.frame1)

    def test_getting_list(self):
        self.check_skip()
        actual = list(self.v[[1, 0, 0, 1, 1]])
        expected = [self.frame1, self.frame0, self.frame0, self.frame1,
                    self.frame1]
        [assert_equal(a, b) for a, b in zip(actual, expected)]

    def test_bool(self):
        self.check_skip()
        pass

    def test_integer_attributes(self):
        self.check_skip()
        assert_equal(len(self.v.frame_shape), 2)
        self.assertTrue(isinstance(self.v.frame_shape[0], int))
        self.assertTrue(isinstance(self.v.frame_shape[1], int))
        self.assertTrue(isinstance(len(self.v), int))

    def test_simple_negative_index(self):
        self.v[-1]
        list(self.v[[0, -1]])

    def test_repr(self):
        # simple smoke test, values not checked
        repr(self.v)

    def test_frame_number_present(self):
        for frame_no in [0, 1, 2, 1]:
            self.assertTrue(hasattr(self.v[frame_no], 'frame_no'))
            not_none = self.v[frame_no].frame_no is not None
            self.assertTrue(not_none)

    def test_frame_number_accurate(self):
        for frame_no in [0, 1, 2, 1]:
            self.assertEqual(self.v[frame_no].frame_no, frame_no)

    def test_process_func(self):
        # Use a trivial identity function to verify the process_func exists.
        f = lambda x: x
        self.klass(self.filename, process_func=f, **self.kwargs)
        
        # Also, it should be the second positional arg for each class.
        # This is verified more directly in later tests, too.
        self.klass(self.filename, f, **self.kwargs)

    def test_inversion_process_func(self):
        def invert(image):
            max_value = np.iinfo(image.dtype).max
            image = image ^ max_value
            return image

        v_raw = self.klass(self.filename, **self.kwargs)
        v = self.klass(self.filename, invert, **self.kwargs)
        assert_equal(v[0], invert(v_raw[0]))

    def test_grayscale_process_func(self):
        # Note: Some, but not all, of the files are already grayscale
        # so in some cases this function does nothing.
        def grayscale(image):
            if image.ndim == 3:
                image = image[:, :, 0]
                assert image.ndim == 2
            return image

        v_raw = self.klass(self.filename, **self.kwargs)
        v = self.klass(self.filename, grayscale, **self.kwargs)
        assert_equal(v[0], grayscale(v_raw[0]))


class _frame_base_klass(_base_klass):
    def test_iterator(self):
        self.check_skip()
        i = iter(self.v)
        assert_equal(next(i), self.frame0)
        assert_equal(next(i), self.frame1)


class TestVideo(_frame_base_klass):
    def check_skip(self):
        _skip_if_no_ffmpeg()

    def setUp(self):
        _skip_if_no_ffmpeg()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.Video
        self.kwargs = dict(use_cache=False)
        self.v = self.klass(self.filename, **self.kwargs)

    def test_shape(self):
        _skip_if_no_ffmpeg()
        assert_equal(self.v.frame_shape, (640, 424))

    def test_count(self):
        _skip_if_no_ffmpeg()
        assert_equal(len(self.v), 480)

    def tearDown(self):
        os.remove(self.filename + '.pims_buffer')
        os.remove(self.filename + '.pims_meta')


class TestTiffStack_libtiff(_base_klass):
    def check_skip(self):
        _skip_if_no_libtiff()

    def setUp(self):
        _skip_if_no_libtiff()
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.klass = pims.TiffStack_libtiff
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)

    def test_shape(self):
        _skip_if_no_libtiff()
        assert_equal(self.v.frame_shape, (512, 512))

    def test_count(self):
        _skip_if_no_libtiff()
        assert_equal(len(self.v), 5)


class TestImageSequence(_frame_base_klass):
    def setUp(self):
        self.filename = os.path.join(path, 'image_sequence')
        self.frame0 = np.load(os.path.join(path, 'seq_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'seq_frame1.npy'))
        self.klass = pims.ImageSequence
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)

    def test_shape(self):
        assert_equal(self.v.frame_shape, (424, 640))

    def test_count(self):
        assert_equal(len(self.v), 5)


class TestTiffStack_pil(_base_klass):
    def check_skip(self):
        pass

    def setUp(self):
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy')).T[::-1]
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy')).T[::-1]
        self.klass = pims.TiffStack_pil
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)

    def test_shape(self):
        assert_equal(self.v.frame_shape, (512, 512))

    def test_count(self):
        assert_equal(len(self.v), 5)
