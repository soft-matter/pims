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
        raise nose.SkipTest('OpenCV not installed. Skipping.')


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


class _frame_base_klass(_base_klass):
    def test_iterator(self):
        self.check_skip()
        i = iter(self.v)
        assert_equal(i.next(), self.frame0)
        assert_equal(i.next(), self.frame1)


class TestVideo(_frame_base_klass):
    def check_skip(self):
        _skip_if_no_ffmpeg()

    def setUp(self):
        _skip_if_no_ffmpeg()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.v = pims.Video(self.filename)

    def test_shape(self):
        _skip_if_no_ffmpeg()
        assert_equal(self.v.frame_shape, (640, 424))

    def test_count(self):
        _skip_if_no_ffmpeg()
        assert_equal(len(self.v), 480)


class TestTiffStack_libtiff(_base_klass):
    def check_skip(self):
        _skip_if_no_libtiff()

    def setUp(self):
        _skip_if_no_libtiff()
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.v = pims.TiffStack_libtiff(self.filename)

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
        self.v = pims.ImageSequence(self.filename)

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
        self.v = pims.TiffStack_pil(self.filename)

    def test_shape(self):
        assert_equal(self.v.frame_shape, (512, 512))

    def test_count(self):
        assert_equal(len(self.v), 5)
