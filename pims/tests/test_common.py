from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import tempfile
import zipfile
import sys
import random
import types
import unittest
import pickle
from io import BytesIO
import nose
import numpy as np
from numpy.testing import (assert_equal, assert_allclose)
from nose.tools import assert_true
import pims

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def _skip_if_no_PyAV():
    import pims.pyav_reader
    if not pims.pyav_reader.available():
        raise nose.SkipTest('PyAV not found. Skipping.')


def _skip_if_no_MoviePy():
    import pims.moviepy_reader
    if not pims.moviepy_reader.available():
        raise nose.SkipTest('MoviePy not found. Skipping.')


def _skip_if_no_ImageIO():
    import pims.imageio_reader
    if not pims.imageio_reader.available():
        raise nose.SkipTest('ImageIO not found. Skipping.')


def _skip_if_no_libtiff():
    try:
        import libtiff
    except ImportError:
        raise nose.SkipTest('libtiff not installed. Skipping.')


def _skip_if_no_tifffile():
    try:
        import tifffile
    except ImportError:
        raise nose.SkipTest('tifffile not installed. Skipping.')


def _skip_if_no_imread():
    if pims.image_sequence.imread is None:
        raise nose.SkipTest('ImageSequence requires either scipy, matplotlib or'
                            ' scikit-image. Skipping.')


def _skip_if_no_skimage():
    try:
        import skimage
    except ImportError:
        raise nose.SkipTest('skimage not installed. Skipping.')


def _skip_if_no_PIL():
    try:
        from PIL import Image
    except ImportError:
        raise nose.SkipTest('PIL/Pillow not installed. Skipping.')


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
        self.assertTrue(isinstance(self.v.frame_shape[0], six.integer_types))
        self.assertTrue(isinstance(self.v.frame_shape[1], six.integer_types))
        self.assertTrue(isinstance(len(self.v), six.integer_types))

    def test_shape(self):
        self.check_skip()
        assert_equal(self.v.frame_shape, self.expected_shape)

    def test_count(self):
        self.check_skip()
        assert_equal(len(self.v), self.expected_len)

    def test_repr(self):
        self.check_skip()
        # simple smoke test, values not checked
        repr(self.v)


class _deprecated_functions(object):
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
        _skip_if_no_imread()
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
        print('slice2d')
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
        print('define slice4')
        slice4 = slice3[:-1]
        print('compare slice4')
        compare_slice_to_list(slice4, list('ceg'))
        print('define slice4a')
        slice4a = slice3[::-1]
        print('compare slice4a')
        compare_slice_to_list(slice4a, list('igec'))

    def test_slice_with_generator(self):
        slice1 = self.v[1:]
        compare_slice_to_list(slice1, list('bcdefghij'))
        slice2 = slice1[(i for i in range(2,5))]
        assert_letters_equal(slice2, list('def'))
        assert_true(isinstance(slice2, types.GeneratorType))


class TestMultidimensional(unittest.TestCase):
    def setUp(self):
        class IndexReturningReader(pims.FramesSequenceND):
            @property
            def pixel_type(self):
                pass

            def __init__(self, **dims):
                self._init_axis('x', len(dims))
                self._init_axis('y', 1)
                for k in dims:
                    self._init_axis(k, dims[k])

            def get_frame_2D(self, **ind):
                return np.array([[ind[i] for i in sorted(ind)]])

        self.v = IndexReturningReader(c=3, m=5, t=100, z=20)

    def test_iterate(self):
        self.v.iter_axes = 't'
        for i in [0, 1, 15]:
            assert_equal(self.v[i], [[0, 0, i, 0]])
        self.v.iter_axes = 'm'
        for i in [0, 1, 3]:
            assert_equal(self.v[i], [[0, i, 0, 0]])
        self.v.iter_axes = 'zc'
        assert_equal(self.v[0], [[0, 0, 0, 0]])
        assert_equal(self.v[2], [[2, 0, 0, 0]])
        assert_equal(self.v[30], [[0, 0, 0, 10]])
        self.v.iter_axes = 'cz'
        assert_equal(self.v[0], [[0, 0, 0, 0]])
        assert_equal(self.v[4], [[0, 0, 0, 4]])
        assert_equal(self.v[21], [[1, 0, 0, 1]])
        self.v.iter_axes = 'tzc'
        assert_equal(self.v[0], [[0, 0, 0, 0]])
        assert_equal(self.v[4], [[1, 0, 0, 1]])
        assert_equal(self.v[180], [[0, 0, 3, 0]])
        assert_equal(self.v[210], [[0, 0, 3, 10]])
        assert_equal(self.v[212], [[2, 0, 3, 10]])

    def test_default(self):
        self.v.iter_axes = 't'
        self.v.default_coords['m'] = 2
        for i in [0, 1, 3]:
            assert_equal(self.v[i], [[0, 2, i, 0]])
        self.v.default_coords['m'] = 0
        for i in [0, 1, 3]:
            assert_equal(self.v[i], [[0, 0, i, 0]])

    def test_bundle(self):
        self.v.bundle_axes = 'zyx'
        assert_equal(self.v[0].shape, (20, 1, 4))
        self.v.bundle_axes = 'cyx'
        assert_equal(self.v[0].shape, (3, 1, 4))
        self.v.bundle_axes = 'czyx'
        assert_equal(self.v[0].shape, (3, 20, 1, 4))
        self.v.bundle_axes = 'zcyx'
        assert_equal(self.v[0].shape, (20, 3, 1, 4))

    def test_frame_no(self):
        self.v.iter_axes = 't'
        for i in np.random.randint(0, 100, 10):
            assert_equal(self.v[i].frame_no, i)
        self.v.iter_axes = 'zc'
        for i in np.random.randint(0, 3*20, 10):
            assert_equal(self.v[i].frame_no, i)

    def test_metadata(self):
        # if no metadata is provided by the reader, metadata should be {}
        assert_equal(self.v[0].metadata, {})

        class MetadataReturningReader(pims.FramesSequenceND):
            @property
            def pixel_type(self):
                pass

            def __init__(self, **dims):
                self._init_axis('x', len(dims))
                self._init_axis('y', 1)
                for k in dims:
                    self._init_axis(k, dims[k])

            def get_frame_2D(self, **ind):
                metadata = {i: ind[i] for i in ind}
                im = np.array([[ind[i] for i in sorted(ind)]])
                return pims.Frame(im, metadata=metadata)

        self.v_md = MetadataReturningReader(c=3, m=5, t=100, z=20)
        self.v_md.iter_axes = 't'
        self.v_md.bundle_axes = 'czyx'
        md = self.v_md[15].metadata

        # if metadata is provided, it should have the correct shape
        assert_equal(md['z'].shape, (3, 20))  # shape 'c', 'z'
        assert_equal(md['z'][:, 5], 5)
        assert_equal(md['c'][1, :], 1)

        # if a metadata field is equal for all frames, it should be a scalar
        assert_equal(md['t'], 15)

def _rescale(img):
    print(type(img))
    return (img - img.min()) / img.ptp()

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


class _image_rgb(_image_single):
    # Only include these tests for 2D RGB files.
    def test_greyscale_process_func(self):
        self.check_skip()
        def greyscale(image):
            assert image.ndim == 3
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


class TestVideo_PyAV(_image_series, _image_rgb, _deprecated_functions,
                     unittest.TestCase):
    def check_skip(self):
        _skip_if_no_PyAV()

    def setUp(self):
        _skip_if_no_PyAV()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.klass = pims.PyAVVideoReader
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (640, 424, 3)  # (x, y), wrong convention?
        self.expected_len = 480


class TestVideo_ImageIO(_image_series, unittest.TestCase):
    def check_skip(self):
        _skip_if_no_ImageIO()

    def setUp(self):
        _skip_if_no_ImageIO()
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


class _tiff_image_series(_image_series, _deprecated_functions):
    def test_metadata(self):
        m = self.v[0].metadata
        if sys.version_info.major < 3:
            pkl_path = os.path.join(path, 'stuck_metadata_py2.pkl')
        else:
            pkl_path = os.path.join(path, 'stuck_metadata_py3.pkl')
        with open(pkl_path, 'rb') as p:
            d = pickle.load(p)
        assert_equal(m, d)


class TestTiffStack_libtiff(_tiff_image_series, _deprecated_functions,
                            unittest.TestCase):
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


class TestImageSequenceWithPIL(_image_series, _deprecated_functions,
                               unittest.TestCase):
    def setUp(self):
        _skip_if_no_skimage()
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F00001.png', 'T76S3F00002.png',
                          'T76S3F00003.png', 'T76S3F00004.png',
                          'T76S3F00005.png']
        shape = (10, 11)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = os.path.join(self.filepath, '*.png')
        self.frame0 = frames[0]
        self.frame1 = frames[1]
        self.kwargs = dict(plugin='pil')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = shape
        self.expected_len = 5
        self.tempdir = tempfile.mkdtemp()
        self.tempfile = os.path.join(self.tempdir, 'test.zip')

        with zipfile.ZipFile(self.tempfile, 'w') as archive:
            for fn in self.filenames:
                archive.write(os.path.join(self.filepath, fn))

    def test_bad_path_raises(self):
        raises = lambda: pims.ImageSequence('this/path/does/not/exist/*.jpg')
        self.assertRaises(IOError, raises)

    def test_zipfile(self):
        pims.ImageSequence(self.tempfile)[0]

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)
        os.remove(self.tempfile)
        os.rmdir(self.tempdir)


class TestImageSequenceWithMPL(_image_series, _deprecated_functions,
                               unittest.TestCase):
    def setUp(self):
        _skip_if_no_skimage()
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F00001.png', 'T76S3F00002.png',
                          'T76S3F00003.png', 'T76S3F00004.png',
                          'T76S3F00005.png']
        shape = (10, 11)
        frames = save_dummy_png(self.filepath, self.filenames, shape)
        self.filename = os.path.join(self.filepath, '*.png')
        self.frame0 = frames[0]
        self.frame1 = frames[1]
        self.kwargs = dict(plugin='matplotlib')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = shape
        self.expected_len = 5

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)

class TestImageSequenceAcceptsList(_image_series, _deprecated_functions,
                                   unittest.TestCase):
    def setUp(self):
        _skip_if_no_imread()
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F00001.png', 'T76S3F00002.png',
                          'T76S3F00003.png', 'T76S3F00004.png',
                          'T76S3F00005.png']
        shape = (10, 11)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = [os.path.join(self.filepath, fn)
                         for fn in self.filenames]
        self.frame0 = frames[0]
        self.frame1 = frames[1]
        self.kwargs = dict(plugin='matplotlib')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = shape
        self.expected_len = len(self.filenames)

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)

class TestImageSequenceNaturalSorting(_image_series, _deprecated_functions,
                                      unittest.TestCase):
    def setUp(self):
        _skip_if_no_imread()
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F1.png', 'T76S3F20.png',
                     'T76S3F3.png', 'T76S3F4.png',
                     'T76S3F50.png', 'T76S3F10.png']
        shape = (10, 11)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = [os.path.join(self.filepath, fn)
                         for fn in self.filenames]
        self.frame0 = frames[0]
        self.frame1 = frames[2]
        self.kwargs = dict(plugin='matplotlib')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = shape
        self.expected_len = len(self.filenames)

        sorted_files = ['T76S3F1.png',
                        'T76S3F3.png',
                        'T76S3F4.png',
                        'T76S3F10.png',
                        'T76S3F20.png',
                        'T76S3F50.png']

        assert sorted_files == [x.split(os.path.sep)[-1] for x in self.v._filepaths]

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)

class TestTiffStack_pil(_tiff_image_series, _deprecated_functions,
                        unittest.TestCase):
    def check_skip(self):
        pass

    def setUp(self):
        _skip_if_no_PIL()
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.klass = pims.TiffStack_pil
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 5


class TestTiffStack_tifffile(_tiff_image_series, _deprecated_functions,
                             unittest.TestCase):
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


class TestSpeStack(_image_series, _deprecated_functions,
                   unittest.TestCase):
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
            if sys.version_info.major < 3:
                d = pickle.load(p)
            else:
                d = pickle.load(p, encoding="latin1")
                #spare4 is actually a byte array
                d["spare4"] = d["spare4"].encode("latin1")

        assert_equal(m, d)



class TestOpenFiles(unittest.TestCase):
    def setUp(self):
        _skip_if_no_PIL()

    def test_open_pngs(self):
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F00001.png', 'T76S3F00002.png',
                          'T76S3F00003.png', 'T76S3F00004.png',
                          'T76S3F00005.png']
        shape = (10, 11)
        save_dummy_png(self.filepath, self.filenames, shape)
        pims.open(os.path.join(path, 'image_sequence', '*.png'))
        clean_dummy_png(self.filepath, self.filenames)

    def test_open_mov(self):
        _skip_if_no_PyAV()
        pims.open(os.path.join(path, 'bulk-water.mov'))

    def test_open_tiff(self):
        _skip_if_no_tifffile()
        pims.open(os.path.join(path, 'stuck.tif'))


class ImageSequenceND(_image_series, _deprecated_functions, unittest.TestCase):
    def setUp(self):
        _skip_if_no_imread()
        self.filepath = os.path.join(path, 'image_sequence3d')
        self.filenames = ['file_t001_z001_c1.png',
                          'file_t001_z001_c2.png',
                          'file_t001_z002_c1.png',
                          'file_t001_z002_c2.png',
                          'file_t002_z001_c1.png',
                          'file_t002_z001_c2.png',
                          'file_t002_z002_c1.png',
                          'file_t002_z002_c2.png',
                          'file_t003_z001_c1.png',
                          'file_t003_z001_c2.png',
                          'file_t003_z002_c1.png',
                          'file_t003_z002_c2.png']
        shape = (10, 11)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = os.path.join(self.filepath, '*.png')
        self.frame0 = np.array([frames[0], frames[2]])
        self.frame1 = np.array([frames[4], frames[6]])
        self.klass = pims.ImageSequenceND
        self.kwargs = dict(axes_identifiers='tzc')
        self.v = self.klass(self.filename, **self.kwargs)
        self.v.default_coords['c'] = 0
        self.expected_len = 3
        self.expected_Z = 2
        self.expected_C = 2
        self.expected_shape = (self.expected_Z,) + shape

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)

    def test_filename_tzc(self):
        tzc = pims.image_sequence.filename_to_indices('file_t01_z005_c4.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_indices('t01file_t01_z005_c4.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_indices('file_z005_c4_t01.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_indices(u'file\u03BC_z05_c4_t01.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_indices('file_t9415_z005.png')
        self.assertEqual(tzc, [9415, 5, 0])
        tzc = pims.image_sequence.filename_to_indices('file_t47_c34.png')
        self.assertEqual(tzc, [47, 0, 34])
        tzc = pims.image_sequence.filename_to_indices('file_z4_c2.png')
        self.assertEqual(tzc, [0, 4, 2])
        tzc = pims.image_sequence.filename_to_indices('file_p4_c2_q5_r1.png',
                                                      ['p', 'q', 'r'])
        self.assertEqual(tzc, [4, 5, 1])

    def test_sizeZ(self):
        self.check_skip()
        assert_equal(self.v.sizes['z'], self.expected_Z)

    def test_sizeC(self):
        self.check_skip()
        assert_equal(self.v.sizes['c'], self.expected_C)


class ImageSequenceND_RGB(_image_series, _deprecated_functions,
                          unittest.TestCase):
    def setUp(self):
        _skip_if_no_imread()
        self.filepath = os.path.join(path, 'image_sequence3d')
        self.filenames = ['file_t001_z001_c1.png',
                          'file_t001_z002_c1.png',
                          'file_t002_z001_c1.png',
                          'file_t002_z002_c1.png',
                          'file_t003_z001_c1.png',
                          'file_t003_z002_c1.png']
        shape = (10, 11, 3)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = os.path.join(self.filepath, '*.png')
        self.frame0 = np.array([frames[0][:, :, 0], frames[1][:, :, 0]])
        self.frame1 = np.array([frames[2][:, :, 0], frames[3][:, :, 0]])
        self.klass = pims.ImageSequenceND
        self.kwargs = dict(axes_identifiers='tz')
        self.v = self.klass(self.filename, **self.kwargs)
        self.v.default_coords['c'] = 0
        self.expected_len = 3
        self.expected_Z = 2
        self.expected_C = 3
        self.expected_shape = (2, 10, 11)

    def test_sizeZ(self):
        self.check_skip()
        assert_equal(self.v.sizes['z'], self.expected_Z)

    def test_sizeC(self):
        self.check_skip()
        assert_equal(self.v.sizes['c'], self.expected_C)

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
