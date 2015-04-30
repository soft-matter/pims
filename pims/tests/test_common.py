from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import types
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal, assert_allclose)
from nose.tools import assert_true
import pims
from PIL import Image

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


def save_dummy_png(filepath, filenames, shape):
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


class _image_single(unittest.TestCase):
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
    return pims.Frame(np.array(letter).reshape(1, 1))

def assert_letters_equal(actual, expected):
    for actual_, expected_ in zip(actual, expected):
        assert_equal(actual_, box(expected_))


class TestRecursiveSlicing(unittest.TestCase):

    def setUp(self):
        class DemoReader(pims.ImageSequence):
            def imread(self, filename, **kwargs):
                return np.array([[filename]])

        self.v = DemoReader(list('abcdefghij'))

    def test_slice_of_slice(self):
        slice1 = self.v[4:]
        assert_letters_equal(slice1, list('efghij'))

        slice2 = slice1[-3:]
        assert_letters_equal(slice2, list('hij'))

    def test_slice_of_slice_of_slice(self):
        slice1 = self.v[4:]
        assert_letters_equal(slice1, list('efghij'))
        slice2 = slice1[1:-1]
        assert_letters_equal(slice2, list('fghi'))
        slice3 = slice2[1::2]  # gi
        assert_letters_equal(slice3, list('gi'))

    def test_slice_of_slice_of_slice_of_slice(self):
        # Take the red pill. It's slices all the way down!
        slice1 = self.v[4:]
        assert_letters_equal(slice1, list('efghij'))
        slice2 = slice1[1:-1]
        assert_letters_equal(slice2, list('fghi'))
        slice3 = slice2[1:]
        assert_letters_equal(slice3, list('ghi'))
        slice4 = slice3[1:]
        assert_letters_equal(slice4, list('hi'))

        # We should be able to iterate all these again.
        assert_letters_equal(slice4, list('hi'))
        assert_letters_equal(slice3, list('ghi'))
        assert_letters_equal(slice2, list('fghi'))
        assert_letters_equal(slice1, list('efghij'))
        # ... in any order
        assert_letters_equal(slice4, list('hi'))
        assert_letters_equal(slice2, list('fghi'))
        assert_letters_equal(slice3, list('ghi'))
        assert_letters_equal(slice1, list('efghij'))
        assert_letters_equal(slice3, list('ghi'))

        # Give me another!
        slice1 = self.v[2:]
        assert_letters_equal(slice1, list('cdefghij'))
        slice2 = slice1[0::2]
        assert_letters_equal(slice2, list('cegi'))
        slice3 = slice2[:]
        assert_letters_equal(slice3, list('cegi'))
        slice4 = slice3[:-1]
        assert_letters_equal(slice4, list('ceg'))

        assert_letters_equal(slice1, list('cdefghij'))
        assert_letters_equal(slice2, list('cegi'))
        assert_letters_equal(slice3, list('cegi'))
        assert_letters_equal(slice4, list('ceg'))
        assert_letters_equal(slice3, list('cegi'))
        assert_letters_equal(slice4, list('ceg'))
        assert_letters_equal(slice2, list('cegi'))
        assert_letters_equal(slice1, list('cdefghij'))

    def test_slice_with_generator(self):
        slice1 = self.v[1:]
        assert_letters_equal(slice1, list('bcdefghij'))
        slice2 = slice1[(i for i in range(2,5))]
        assert_letters_equal(slice2, list('def'))
        assert_true(isinstance(slice2, types.GeneratorType))


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


class _image_rgb(unittest.TestCase):
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


class TestVideo(_image_series, _image_rgb):
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


class TestTiffStack_libtiff(_image_series):
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


class TestImageSequenceWithPIL(_image_series):
    def setUp(self):
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

    def test_bad_path_raises(self):
        raises = lambda: pims.ImageSequence('this/path/does/not/exist/*.jpg')
        self.assertRaises(IOError, raises)

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)


class TestImageSequenceWithMPL(_image_series):
    def setUp(self):
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

class TestImageSequenceAcceptsList(_image_series):
    def setUp(self):
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

class TestImageSequenceNaturalSorting(_image_series):
    def setUp(self):
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

class TestTiffStack_pil(_image_series):
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


class TestTiffStack_tifffile(_image_series):
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


class TestOpenFiles(unittest.TestCase):
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
        pims.open(os.path.join(path, 'stuck.tif'))


class ImageSequence3D(_image_series):
    def check_skip(self):
        pass

    def setUp(self):
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
        self.frame0 = [frames[0], frames[2]]
        self.frame1 = [frames[4], frames[6]]
        self.klass = pims.ImageSequence3D
        self.kwargs = dict()
        self.v = self.klass(self.filename, **self.kwargs)
        self.v.channel = 0
        self.expected_shape = shape
        self.expected_len = 3
        self.expected_Z = 2
        self.expected_C = 2

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)

    def test_filename_tzc(self):
        tzc = pims.image_sequence.filename_to_tzc('file_t01_z005_c4.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_tzc('t01file_t01_z005_c4.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_tzc('file_z005_c4_t01.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_tzc(u'file\u03BC_z05_c4_t01.png')
        self.assertEqual(tzc, [1, 5, 4])
        tzc = pims.image_sequence.filename_to_tzc('file_t9415_z005.png')
        self.assertEqual(tzc, [9415, 5, 0])
        tzc = pims.image_sequence.filename_to_tzc('file_t47_c34.png')
        self.assertEqual(tzc, [47, 0, 34])
        tzc = pims.image_sequence.filename_to_tzc('file_z4_c2.png')
        self.assertEqual(tzc, [0, 4, 2])
        tzc = pims.image_sequence.filename_to_tzc('file_x4_c2_y5_z1.png',
                                                  ['x', 'y', 'z'])
        self.assertEqual(tzc, [4, 5, 1])

    def test_sizeZ(self):
        self.check_skip()
        assert_equal(self.v.sizes['Z'], self.expected_Z)

    def test_sizeC(self):
        self.check_skip()
        assert_equal(self.v.sizes['C'], self.expected_C)

    def test_change_channels(self):
        self.check_skip()
        self.v.channel = (0, 1)
        assert_equal(self.v[0].shape, (2, 2, self.expected_shape[0],
                                       self.expected_shape[1]))


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
