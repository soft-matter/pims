import gc
import os
import random
import pickle
import types
import unittest

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pims
import pytest

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def _skip_if_no_PyAV():
    import pims.pyav_reader
    if not pims.pyav_reader.available():
        raise unittest.SkipTest('PyAV not found. Skipping.')


def _skip_if_no_MoviePy():
    import pims.moviepy_reader
    if not pims.moviepy_reader.available():
        raise unittest.SkipTest('MoviePy not found. Skipping.')


def _skip_if_no_ImageIO_ffmpeg():
    import pims.imageio_reader
    if not pims.imageio_reader.ffmpeg_available():
        raise unittest.SkipTest('ImageIO and ffmpeg not found. Skipping.')


def _skip_if_no_tifffile():
    import pims.tiff_stack
    if not pims.tiff_stack.tifffile_available():
        raise unittest.SkipTest('tifffile not installed. Skipping.')


def _skip_if_no_skimage():
    try:
        import skimage
    except ImportError:
        raise unittest.SkipTest('skimage not installed. Skipping.')


def assert_image_equal(actual, expected):
    if np.issubdtype(actual.dtype, np.integer):
        assert_equal(actual, expected)
    else:
        if np.issubdtype(expected.dtype, np.integer):
            expected = expected/float(np.iinfo(expected.dtype).max)
        assert_allclose(actual, expected, atol=1/256.)


def save_dummy_png(filepath, filenames, shape):
    from PIL import Image
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    frames = []
    for f in filenames:
        dummy = np.random.randint(0, 255, shape).astype('uint8')
        im = Image.fromarray(dummy)
        im.save(os.path.join(filepath, f), 'png')
        frames.append(dummy)
        im.close()
    return frames


def clean_dummy_png(filepath, filenames):
    for f in filenames:
        os.remove(os.path.join(filepath, f))
    if os.listdir(filepath) == []:
        os.rmdir(filepath)


class _image_single(object):
    def check_skip(self):
        pass

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
        assert_equal(self.v[0].shape, self.expected_shape)

    def test_count(self):
        self.check_skip()
        assert_equal(len(self.v), self.expected_len)

    def test_repr(self):
        self.check_skip()
        # simple smoke test, values not checked
        repr(self.v)

    def tearDown(self):
        if hasattr(self, 'v'):
            self.v.close()
            # This helps avoiding a ResourceWarning in imageio-ffmpeg teardown.
            gc.collect()


def box(letter):
    return pims.Frame(np.array(letter))


def assert_letters_equal(actual, expected):
    for actual_, expected_ in zip(actual, expected):
        # This contrived reader has weird shape behavior,
        # but that's not what I'm testing here.
        assert_equal(actual_.reshape((1, 1)), box(expected_).reshape((1, 1)))


def compare_slice_to_list(actual, expected):
    assert_letters_equal(actual, expected)
    # test lengths
    actual_len = len(actual)
    assert_equal(actual_len, len(expected))
    indices = list(range(len(actual)))
    for i in indices:
        # test positive indexing
        assert_letters_equal(actual[i], expected[i])
        # test negative indexing
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # in reverse order
    for i in indices[::-1]:
        assert_letters_equal(actual[i], expected[i])
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # in shuffled order (using a consistent random seed)
    r = random.Random(5)
    r.shuffle(indices)
    for i in indices:
        assert_letters_equal(actual[i], expected[i])
        assert_letters_equal(actual[-i + 1], expected[-i + 1])
    # test list indexing
    some_indices = [r.choice(indices) for _ in range(2)]
    assert_letters_equal(actual[some_indices],
                         np.array(expected)[some_indices])
    # mixing positive and negative indices
    some_indices = [r.choice(indices + [-i-1 for i in indices])
                    for _ in range(2)]
    assert_letters_equal(actual[some_indices],
                         np.array(expected)[some_indices])
    # test slices
    assert_letters_equal(actual[::2], expected[::2])
    assert_letters_equal(actual[1::2], expected[1::2])
    assert_letters_equal(actual[::3], expected[::3])
    assert_letters_equal(actual[1:], expected[1:])
    assert_letters_equal(actual[:], expected[:])
    assert_letters_equal(actual[:-1], expected[:-1])


class TestRecursiveSlicing(unittest.TestCase):

    def setUp(self):
        class DemoReader(pims.ImageSequence):
            def imread(self, filename, **kwargs):
                return np.array([[filename]])

        self.v = DemoReader(list('abcdefghij'))

    def test_slice_of_slice(self):
        slice1 = self.v[4:]
        compare_slice_to_list(slice1, list('efghij'))
        slice2 = slice1[-3:]
        compare_slice_to_list(slice2, list('hij'))
        slice1a = self.v[[3, 4, 5, 6, 7, 8, 9]]
        compare_slice_to_list(slice1a, list('defghij'))
        slice2a = slice1a[::2]
        compare_slice_to_list(slice2a, list('dfhj'))
        slice2b = slice1a[::-1]
        compare_slice_to_list(slice2b, list('jihgfed'))
        slice2c = slice1a[::-2]
        compare_slice_to_list(slice2c, list('jhfd'))
        # print('slice2d')
        slice2d = slice1a[:0:-1]
        compare_slice_to_list(slice2d, list('jihgfe'))
        slice2e = slice1a[-1:1:-1]
        compare_slice_to_list(slice2e, list('jihgf'))
        slice2f = slice1a[-2:1:-1]
        compare_slice_to_list(slice2f, list('ihgf'))
        slice2g = slice1a[::-3]
        compare_slice_to_list(slice2g, list('jgd'))
        slice2h = slice1a[[5, 6, 2, -1, 3, 3, 3, 0]]
        compare_slice_to_list(slice2h, list('ijfjgggd'))


    def test_slice_of_slice_of_slice(self):
        slice1 = self.v[4:]
        compare_slice_to_list(slice1, list('efghij'))
        slice2 = slice1[1:-1]
        compare_slice_to_list(slice2, list('fghi'))
        slice2a = slice1[[2, 3, 4]]
        compare_slice_to_list(slice2a, list('ghi'))
        slice3 = slice2[1::2]
        compare_slice_to_list(slice3, list('gi'))

    def test_slice_of_slice_of_slice_of_slice(self):
        # Take the red pill. It's slices all the way down!
        slice1 = self.v[4:]
        compare_slice_to_list(slice1, list('efghij'))
        slice2 = slice1[1:-1]
        compare_slice_to_list(slice2, list('fghi'))
        slice3 = slice2[1:]
        compare_slice_to_list(slice3, list('ghi'))
        slice4 = slice3[1:]
        compare_slice_to_list(slice4, list('hi'))

        # Give me another!
        slice1 = self.v[2:]
        compare_slice_to_list(slice1, list('cdefghij'))
        slice2 = slice1[0::2]
        compare_slice_to_list(slice2, list('cegi'))
        slice3 = slice2[:]
        compare_slice_to_list(slice3, list('cegi'))
        # print('define slice4')
        slice4 = slice3[:-1]
        # print('compare slice4')
        compare_slice_to_list(slice4, list('ceg'))
        # print('define slice4a')
        slice4a = slice3[::-1]
        # print('compare slice4a')
        compare_slice_to_list(slice4a, list('igec'))

    def test_slice_with_generator(self):
        slice1 = self.v[1:]
        compare_slice_to_list(slice1, list('bcdefghij'))
        slice2 = slice1[(i for i in range(2,5))]
        assert_letters_equal(slice2, list('def'))
        self.assertTrue(isinstance(slice2, types.GeneratorType))


def _rescale(img):
    return (img - img.min()) / np.ptp(img)


def _color_channel(img, channel):
    if img.ndim == 3:
        return img[:, :, channel]
    else:
        return img


class _image_series(_image_single):
    def test_iterator(self):
        self.check_skip()
        i = iter(self.v)
        assert_image_equal(next(i), self.frame0)
        assert_image_equal(next(i), self.frame1)

    def test_getting_slice(self):
        self.check_skip()
        tmp = list(self.v[0:2])
        frame0, frame1 = tmp
        assert_image_equal(frame0, self.frame0)
        assert_image_equal(frame1, self.frame1)

    def test_slice_of_slice(self):
        # More thorough recursive slicing tests, making use of more than
        # the two frames available for these tests, are elsewhere:
        # see test_recursive_slicing.
        self.check_skip()
        tmp = self.v[0:2]
        tmp1 = tmp[1:]
        frame1 = tmp1[0]
        assert_image_equal(frame1, self.frame1)

        # Do the same thing again, show that the generators are not dead.
        tmp1 = tmp[1:]
        frame1 = tmp1[0]
        assert_image_equal(frame1, self.frame1)

        frame0 = tmp[0]
        assert_image_equal(frame0, self.frame0)

        # Show that we can listify the slice twice.
        frame0, frame1 = list(tmp)
        assert_image_equal(frame0, self.frame0)
        assert_image_equal(frame1, self.frame1)
        frame0, frame1 = list(tmp)
        assert_image_equal(frame0, self.frame0)
        assert_image_equal(frame1, self.frame1)

    def test_pipeline_simple(self):
        rescale = pims.pipeline(_rescale)
        rescaled_v = rescale(self.v[:1])

        assert_image_equal(rescaled_v[0], _rescale(self.frame0))

    def test_pipeline_with_args(self):
        color_channel = pims.pipeline(_color_channel)
        red = color_channel(self.v, 0)
        green = color_channel(self.v, 1)

        assert_image_equal(red[0], _color_channel(self.frame0, 0))
        assert_image_equal(green[0], _color_channel(self.frame0, 1))

        # Multiple pipelines backed by the same data are indep,
        # so this call to red is unaffected by green above.
        assert_image_equal(red[0], _color_channel(self.frame0, 0))

    def test_composed_pipelines(self):
        color_channel = pims.pipeline(_color_channel)
        rescale = pims.pipeline(_rescale)

        composed = rescale(color_channel(self.v, 0))

        expected = _rescale(_color_channel(self.v[0], 0))
        assert_image_equal(composed[0], expected)

    def test_as_grey(self):
        gr = pims.as_grey(self.v)
        assert len(gr[0].shape) == 2
        # Calling a second time does nothing
        gr2 = pims.as_grey(gr)
        assert_image_equal(gr[0], gr2[0])

        # Alternate spelling accepted
        gr = pims.as_gray(self.v)
        assert len(gr[0].shape) == 2

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

    def test_simple_negative_index(self):
        self.check_skip()
        self.v[-1]
        list(self.v[[0, -1]])


class TestImageReaderTIFF(_image_single, unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.klass = pims.ImageReader
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (5, 512, 512)
        self.expected_len = 1


class TestImageReaderPNG(_image_single, unittest.TestCase):
    def setUp(self):
        self.klass = pims.ImageReader
        self.kwargs = dict()
        self.expected_shape = (10, 11)
        self.expected_len = 1

        save_dummy_png(path, ['dummy.png'], self.expected_shape)
        self.v = self.klass(os.path.join(path, 'dummy.png'), **self.kwargs)
        clean_dummy_png(path, ['dummy.png'])


class TestImageReaderND(_image_single, unittest.TestCase):
    def setUp(self):
        self.klass = pims.ImageReaderND
        self.kwargs = dict()
        self.expected_shape = (10, 11, 3)
        self.expected_len = 1

        save_dummy_png(path, ['dummy.png'], self.expected_shape)
        self.v = self.klass(os.path.join(path, 'dummy.png'), **self.kwargs)
        clean_dummy_png(path, ['dummy.png'])


class TestVideo_PyAV_timed(_image_series, unittest.TestCase):
    def check_skip(self):
        _skip_if_no_PyAV()

    def setUp(self):
        _skip_if_no_PyAV()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.PyAVReaderTimed
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640, 3)
        self.expected_len = 480


class TestVideo_PyAV_indexed(_image_series, unittest.TestCase):
    def check_skip(self):
        _skip_if_no_PyAV()

    def setUp(self):
        _skip_if_no_PyAV()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.PyAVReaderIndexed
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640, 3)
        self.expected_len = 480


class TestVideo_ImageIO(_image_series, unittest.TestCase):
    def check_skip(self):
        _skip_if_no_ImageIO_ffmpeg()

    def setUp(self):
        _skip_if_no_ImageIO_ffmpeg()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.ImageIOReader
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640, 3)
        self.expected_len = 480


class TestVideo_MoviePy(_image_series, unittest.TestCase):
    def check_skip(self):
        _skip_if_no_MoviePy()

    def setUp(self):
        _skip_if_no_MoviePy()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.MoviePyReader
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (424, 640, 3)
        self.expected_len = 480


class _tiff_image_series(_image_series):
    def test_metadata(self):
        m = self.v[0].metadata
        pkl_path = os.path.join(path, 'stuck_metadata_py3.pkl')
        with open(pkl_path, 'rb') as p:
            d = pickle.load(p)
        assert_equal(m, d)


class TestTiffStack_pil(_tiff_image_series, unittest.TestCase):
    def check_skip(self):
        pass

    def setUp(self):
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.klass = pims.TiffStack_pil
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 5


class TestTiffStack_tifffile(_tiff_image_series, unittest.TestCase):
    def check_skip(self):
        pass

    def setUp(self):
        _skip_if_no_tifffile()
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.klass = pims.TiffStack_tifffile
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 5


class TestSpeStack(_image_series, unittest.TestCase):
    def check_skip(self):
        pass

    def setUp(self):
        self.filename = os.path.join(path, 'spestack_test.spe')
        self.frame0 = np.load(os.path.join(path, 'spestack_test_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'spestack_test_frame1.npy'))
        self.klass = pims.SpeStack
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (128, 128)
        self.expected_len = 5

    def test_metadata(self):
        m = self.v.metadata
        with open(os.path.join(path, 'spestack_test_metadata.pkl'), 'rb') as p:
            d = pickle.load(p, encoding="latin1")
            #spare4 is actually a byte array
            d["spare4"] = d["spare4"].encode("latin1")

        assert_equal(m, d)


class TestOpenFiles(unittest.TestCase):
    def test_open_png(self):
        self.filenames = ['dummy_png.png']
        shape = (10, 11)
        save_dummy_png(path, self.filenames, shape)
        pims.open(os.path.join(path, 'dummy_png.png')).close()
        clean_dummy_png(path, self.filenames)

    def test_open_pngs(self):
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F00001.png', 'T76S3F00002.png',
                          'T76S3F00003.png', 'T76S3F00004.png',
                          'T76S3F00005.png']
        shape = (10, 11)
        save_dummy_png(self.filepath, self.filenames, shape)
        pims.open(os.path.join(path, 'image_sequence', '*.png')).close()
        clean_dummy_png(self.filepath, self.filenames)

    def test_open_mov(self):
        _skip_if_no_PyAV()
        pims.open(os.path.join(path, 'bulk-water.mov')).close()

    def test_open_tiff(self):
        _skip_if_no_tifffile()
        pims.open(os.path.join(path, 'stuck.tif')).close()


if __name__ == '__main__':
    unittest.main()
