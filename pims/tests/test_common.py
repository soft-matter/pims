from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from itertools import product
import os
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal, assert_allclose)
import pims
import pims.sprite

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def _skip_if_no_PyAV():
    import pims.pyav_reader
    if not pims.pyav_reader.available():
        raise nose.SkipTest('PyAV not found. Skipping.')


def _skip_if_no_libtiff():
    try:
        import libtiff
    except ImportError:
        raise nose.SkipTest('libtiff not installed. Skipping.')


def assert_image_equal(actual, expected):
    if np.issubdtype(actual.dtype, np.integer):
        assert_equal(actual, expected)
    else:
        if np.issubdtype(expected.dtype, np.integer):
            expected = expected/float(np.iinfo(expected.dtype).max)
        assert_allclose(actual, expected, atol=1/256.)


class _base_klass(unittest.TestCase):
    def check_skip(self):
        pass

    def test_getting_slice(self):
        self.check_skip()
        tmp = list(self.v[0:2])
        frame0, frame1 = tmp
        assert_image_equal(frame0, self.frame0)
        assert_image_equal(frame1, self.frame1)

    def test_getting_single_frame(self):
        self.check_skip()
        assert_image_equal(self.v[0], self.frame0)
        assert_image_equal(self.v[0], self.frame0)
        assert_image_equal(self.v[1], self.frame1)
        assert_image_equal(self.v[1], self.frame1)

    def test_getting_list(self):
        self.check_skip()
        actual = list(self.v[[1, 0, 0, 1, 1]])
        expected = [self.frame1, self.frame0, self.frame0, self.frame1,
                    self.frame1]
        [assert_image_equal(a, b) for a, b in zip(actual, expected)]

    def test_bool(self):
        self.check_skip()
        pass

    def test_integer_attributes(self):
        self.check_skip()
        assert_equal(len(self.v.frame_shape), len(self.expected_shape))
        self.assertTrue(isinstance(self.v.frame_shape[0], int))
        self.assertTrue(isinstance(self.v.frame_shape[1], int))
        self.assertTrue(isinstance(len(self.v), int))

    def test_shape(self):
        self.check_skip()
        assert_equal(self.v.frame_shape, self.expected_shape)

    def test_count(self):
        self.check_skip()
        assert_equal(len(self.v), self.expected_len)

    def test_simple_negative_index(self):
        self.check_skip()
        self.v[-1]
        list(self.v[[0, -1]])

    def test_repr(self):
        self.check_skip()
        # simple smoke test, values not checked
        repr(self.v)

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

    def test_dtype_conversion(self):
        self.check_skip()
        v8 = self.klass(self.filename, dtype='uint8', **self.kwargs)
        v16 = self.klass(self.filename, dtype='uint16', **self.kwargs)
        type8 = v8[0].dtype
        type16 = v16[0].dtype
        self.assertEqual(type8, np.uint8)
        self.assertEqual(type16, np.uint16)

    def test_process_func(self):
        self.check_skip()
        # Use a trivial identity function to verify the process_func exists.
        f = lambda x: x
        self.klass(self.filename, process_func=f, **self.kwargs)

        # Also, it should be the second positional arg for each class.
        # This is verified more directly in later tests, too.
        self.klass(self.filename, f, **self.kwargs)

    def test_inversion_process_func(self):
        self.check_skip()
        def invert(image):
            if np.issubdtype(image.dtype, np.integer):
                max_value = np.iinfo(image.dtype).max
                image = image ^ max_value
            else:
                image = 1 - image
            return image

        v_raw = self.klass(self.filename, **self.kwargs)
        v = self.klass(self.filename, invert, **self.kwargs)
        assert_image_equal(v[0], invert(v_raw[0]))

    def test_greyscale_process_func(self):
        self.check_skip()
        # Note: Some, but not all, of the files are already greyscale
        # so in some cases this function does nothing.
        def greyscale(image):
            if image.ndim == 3:
                image = image[:, :, 0]
                assert image.ndim == 2
            return image

        v_raw = self.klass(self.filename, **self.kwargs)
        v = self.klass(self.filename, greyscale, **self.kwargs)
        assert_image_equal(v[0], greyscale(v_raw[0]))

    def test_as_grey(self):
        self.check_skip()
        v = self.klass(self.filename, as_grey=True, **self.kwargs)
        ndim = v[0].ndim
        self.assertEqual(ndim, 2)


class _frame_base_klass(_base_klass):
    def test_iterator(self):
        self.check_skip()
        i = iter(self.v)
        assert_image_equal(next(i), self.frame0)
        assert_image_equal(next(i), self.frame1)


class TestVideo(_frame_base_klass):
    def check_skip(self):
        _skip_if_no_PyAV()

    def setUp(self):
        _skip_if_no_PyAV()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.Video
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (640, 424, 3)
        self.expected_len = 480


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
        self.expected_shape = (512, 512)
        self.expected_len = 5


class TestImageSequenceWithPIL(_frame_base_klass):
    def setUp(self):
        self.filename = os.path.join(path, 'image_sequence')
        self.frame0 = np.load(os.path.join(path, 'seq_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'seq_frame1.npy'))
        self.kwargs = dict(plugin='pil')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640)
        self.expected_len = 5

    def test_bad_path_raises(self):
        raises = lambda: pims.ImageSequence('this/path/does/not/exist/*.jpg')
        self.assertRaises(IOError, raises)


class TestImageSequenceWithMPL(_frame_base_klass):
    def setUp(self):
        self.filename = os.path.join(path, 'image_sequence')
        self.frame0 = np.load(os.path.join(path, 'seq_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'seq_frame1.npy'))
        self.kwargs = dict(plugin='matplotlib')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640)
        self.expected_len = 5


class TestImageSequenceAcceptsList(_frame_base_klass):
    def setUp(self):
        filenames = ['T76S3F00001.png', 'T76S3F00002.png', 'T76S3F00003.png',
                     'T76S3F00004.png', 'T76S3F00005.png']
        self.filename = [os.path.join(path, 'image_sequence', fn)
                         for fn in filenames]
        self.frame0 = np.load(os.path.join(path, 'seq_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'seq_frame1.npy'))
        self.kwargs = dict(plugin='matplotlib')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640)
        self.expected_len = len(filenames)


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
        self.expected_shape = (512, 512)
        self.expected_len = 5


class TestTiffStack_tifffile(_base_klass):
    def check_skip(self):
        pass

    def setUp(self):
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy')).T[::-1]
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy')).T[::-1]
        self.klass = pims.TiffStack_tifffile
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 5


def test_sprite_roundtrip():
    klass = pims.sprite.SpriteSheet
    r, c = 10, 12

    im_sz = 360, 480
    sheet = np.zeros((r*im_sz[0], c*im_sz[1]))
    for _r, _c in product(range(r), range(c)):
        slc_r = slice(_r*im_sz[0], (_r+1)*im_sz[0])
        slc_c = slice(_c*im_sz[1], (_c+1)*im_sz[1])
        sheet[slc_r, slc_c] = _r * c + _c

    sprites = klass(sheet, r, c)


def test_open_pngs():
    pims.open(os.path.join(path, 'image_sequence', '*.png'))


def test_open_mov():
    _skip_if_no_PyAV()
    pims.open(os.path.join(path, 'bulk-water.mov'))


def test_open_tiff():
    pims.open(os.path.join(path, 'stuck.tif'))
