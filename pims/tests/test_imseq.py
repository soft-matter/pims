import os
import tempfile
import zipfile
import unittest
import numpy as np
from numpy.testing import (assert_equal, assert_allclose)
import pims

from pims.tests.test_common import (_image_series,
                                    clean_dummy_png, save_dummy_png,
                                    _skip_if_no_skimage)


path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


class TestImageSequenceWithPIL(_image_series, unittest.TestCase):
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


class TestImageSequenceWithMPL(_image_series, unittest.TestCase):
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


class TestImageSequenceAcceptsList(_image_series, unittest.TestCase):
    def setUp(self):
        _skip_if_no_skimage()
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


class TestImageSequenceNaturalSorting(_image_series, unittest.TestCase):
    def setUp(self):
        _skip_if_no_skimage()
        self.filepath = os.path.join(path, 'image_sequence')
        self.filenames = ['T76S3F1.png', 'T76S3F20.png',
                          'T76S3F3.png', 'T76S3F4.png',
                          'T76S3F50.png', 'T76S3F10.png']
        shape = (10, 11)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = os.path.join(self.filepath, 'T76*.png')
        self.frame0 = frames[0]
        self.frame1 = frames[2]
        self.kwargs = dict(plugin='matplotlib')
        self.klass = pims.ImageSequence
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = shape
        self.expected_len = len(self.filenames)

    def test_sorting(self):
        sorted_files = ['T76S3F1.png',
                        'T76S3F3.png',
                        'T76S3F4.png',
                        'T76S3F10.png',
                        'T76S3F20.png',
                        'T76S3F50.png']

        assert sorted_files == [x.split(os.path.sep)[-1]
                                for x in self.v._filepaths]

        # provide a list: there should be no sorting
        self.filename = [os.path.join(self.filepath, fn) for fn in
                         self.filenames]
        v_unsorted = self.klass(self.filename, **self.kwargs)

        assert self.filenames == [x.split(os.path.sep)[-1]
                                  for x in v_unsorted._filepaths]


    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)


class ImageSequenceND(_image_series, unittest.TestCase):
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


class ImageSequenceND_RGB(_image_series, unittest.TestCase):
    def setUp(self):
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


class ReaderSequence(_image_series, unittest.TestCase):
    def setUp(self):
        self.filepath = os.path.join(path, 'image_sequence3d')
        self.filenames = ['file001.png',
                          'file002.png',
                          'file003.png',
                          'file004.png']
        shape = (10, 11, 3)
        frames = save_dummy_png(self.filepath, self.filenames, shape)

        self.filename = os.path.join(self.filepath, '*.png')
        self.frame0 = frames[0]
        self.frame1 = frames[1]
        self.klass = pims.ReaderSequence
        self.kwargs = dict(reader_cls=pims.ImageReaderND, axis_name='t')
        self.v = self.klass(self.filename, **self.kwargs)
        self.v.bundle_axes = 'yxc'
        self.v.iter_axes = 't'
        self.expected_len = 4
        self.expected_C = 3
        self.expected_shape = (10, 11, 3)

    def test_sizeC(self):
        self.check_skip()
        assert_equal(self.v.sizes['c'], self.expected_C)

    def tearDown(self):
        clean_dummy_png(self.filepath, self.filenames)
