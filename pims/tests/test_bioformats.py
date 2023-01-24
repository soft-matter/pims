"""These tests are mostly based on files from http://loci.wisc.edu/software/sample-data
Please download and extract them to pims/tests/data/bioformats, or use the
provided python script in download_bioformats_test.py."""

import os
import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose

import pims

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def _skip_if_no_bioformats():
    if not pims.bioformats.available():
        raise unittest.SkipTest('JPype is not installed. Skipping.')


def assert_image_equal(actual, expected):
    if np.issubdtype(actual.dtype, np.integer):
        assert_equal(actual, expected)
    else:
        if np.issubdtype(expected.dtype, np.integer):
            expected = expected/float(np.iinfo(expected.dtype).max)
        assert_allclose(actual, expected, atol=1/256.)


class _image_single(object):
    def check_skip(self):
        pass

    def test_bool(self):
        self.check_skip()
        pass

    def test_open(self):
        self.v.close()
        self.v = pims.open(self.filename)

    def tearDown(self):
        if hasattr(self, 'v'):
            self.v.close()

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

    def test_repr(self):
        self.check_skip()
        # simple smoke test, values not checked
        repr(self.v)


class _image_series(_image_single):
    def test_iterator(self):
        self.check_skip()
        iter(self.v)

    def test_getting_slice(self):
        self.check_skip()
        tmp = list(self.v[0:2])
        frame0, frame1 = tmp

    def test_getting_single_frame(self):
        self.v[0]
        self.v[0]
        self.v[1]
        self.v[1]

    def test_getting_list(self):
        self.check_skip()
        list(self.v[[1, 0, 0, 1, 1]])

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


class _image_stack(object):
    def check_skip(self):
        pass

    def test_getting_stack(self):
        self.check_skip()
        assert_equal(self.v[0].shape[-3], self.expected_Z)

    def test_sizeZ(self):
        self.check_skip()
        assert_equal(self.v.sizes['z'], self.expected_Z)


class _image_multichannel(object):
    def check_skip(self):
        pass

    def test_change_channel(self):
        self.check_skip()
        self.v.bundle_axes = 'cyx'
        channel0, channel1 = self.v[0][0], self.v[0][1]
        self.v.bundle_axes = 'yx'
        self.v.default_coords['c'] = 0
        assert_image_equal(self.v[0], channel0)
        self.v.default_coords['c'] = 1
        assert_image_equal(self.v[0], channel1)

    def test_sizeC(self):
        self.check_skip()
        assert_equal(self.v.sizes['c'], self.expected_C)


class TestBioformatsTiff(_image_series, unittest.TestCase):
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        _skip_if_no_bioformats()
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (512, 512)
        self.expected_len = 5

    def test_getting_single_frame(self):  # overwrite, to check with .npys
        self.check_skip()
        assert_image_equal(self.v[0], self.frame0)
        assert_image_equal(self.v[1], self.frame1)


class TestBioformatsND2(_image_series, _image_multichannel, unittest.TestCase):
    # Nikon NIS-Elements ND2
    # 38 x 31 pixels, 16 bits, 2 channels, 3 time points, 10 focal planes
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'cluster.nd2')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (10, 31, 38)
        self.expected_len = 3
        self.expected_Z = 10
        self.expected_C = 2


class TestBioformatsMOV(_image_series, unittest.TestCase):
    # QuickTime movie format, 320 x 240 pixels, 8 bits per sample
    # 108 time points, grayscale image stored as interleaved RGB
    # A sample timelapse dataset in QuickTime movie format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'wtembryo.mov')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (240, 320)
        self.expected_len = 108
        self.expected_C = 3


class TestBioformatsIPW(_image_series, _image_stack, _image_multichannel,
                        unittest.TestCase):
    # Image-Pro workspace format, 256 x 256 pixels, 8 bits per sample
    # 7 time points, 24 focal planes, 2 channels
    # A sample 4D series with intensity and transmitted channels in Image-Pro
    # workspace format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'mitosis-test.ipw')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (24, 256, 256)
        self.expected_len = 7
        self.expected_C = 2
        self.expected_Z = 24


class TestBioformatsDM3(_image_single, unittest.TestCase):
    # Image-Pro workspace format, 4096 x 4096 pixels, 16 bits per sample
    # Jay Campbell of UW-Madison's John White Laboratory has provided an image
    # in Gatan Digital Micrograph (DM3) format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'dnasample1.dm3')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (4096, 4096)
        self.expected_len = 1


class TestBioformatsLSM(_image_series, _image_stack, _image_multichannel,
                        unittest.TestCase):
    # Zeiss Laser Scanning Microscopy format, 400 x 300 pixels, 8 bits per sample
    # 19 time points, 21 focal planes, 2 channels
    # Zeiss has provided a sample multi-channel 4D series in Zeiss LSM format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', '2chZT.lsm')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (21, 300, 400)
        self.expected_len = 19
        self.expected_C = 2
        self.expected_Z = 21


class TestBioformatsAndorTiff(_image_series, _image_stack, _image_multichannel,
                              unittest.TestCase):
    # Andor Bio-imaging Division TIFF format, 256 x 256 pixels, 16 bits per sample
    # 5 time points, 4 focal planes, 2 channels
    # Mark Browne of Andor Technology's Bio-imaging Division has provided a
    # multifield, 2-channel Z-T series in ABD TIFF format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'MF-2CH-Z-T.tif')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (4, 256, 256)
        self.expected_len = 5
        self.expected_C = 2
        self.expected_Z = 4


class TestBioformatsOlympusTiff(_image_series, _image_stack, unittest.TestCase):
    # Olympus Fluoview TIFF format, 512 x 512 pixels, 16 bits per sample
    # 16 time points, 21 focal planes
    # Timothy Gomez of the Department of Anatomy at the UW-Madison has provided
    # a 4D series in Fluoview TIFF format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', '10-31 E1.tif')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (21, 512, 512)
        self.expected_len = 16
        self.expected_Z = 21


class TestBioformatsLIFseries1(_image_single, _image_stack, _image_multichannel,
                               unittest.TestCase):
    # Leica LIF format, 512 x 512 pixels, 16 bits per sample
    # Series 1: XYZ, 25 focal planes, 4 channels
    # Jean-Yves Tinevez of the PFID Imagopole at Institut Pasteur has provided
    # a mouse kidney section from Invitrogen's FluoCell prepared slide #3. The
    # section is stained with Alexafluor 488 WGA, Alexafluor 568 phalloidin and
    # DAPI, and is imaged as two series: an XYZ stack, and an XZY version of
    # the same zone.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'mouse-kidney.lif')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False, 'series': 0}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (25, 512, 512)
        self.expected_len = 1
        self.expected_C = 4
        self.expected_Z = 25

    def test_count_series(self):
        assert_equal(self.v.size_series, 2)

    def test_switch_series(self):
        self.v.series = 1
        assert_equal(self.v.sizes['z'], 46)


class TestBioformatsLIFseries2(_image_single, _image_stack, _image_multichannel,
                               unittest.TestCase):
    # Series 2: XZY, 46 focal planes, 4 channels
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'mouse-kidney.lif')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False, 'series': 1}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (46, 512, 512)
        self.expected_len = 1
        self.expected_C = 4
        self.expected_Z = 46


class TestBioformatsIPL(_image_single, unittest.TestCase):
    # IPLab format, 650 x 515 pixels, 8 bits per sample, 3 channels
    # Scanalytics has provided a sample multi-channel image in IPLab format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'Blend_Final.IPL')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (515, 650)
        self.expected_len = 1


class TestBioformatsSEQ(_image_single, _image_stack, unittest.TestCase):
    # Image-Pro sequence format
    # 512 x 512 pixels, 8 bits per sample, 30 focal planes
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'HEART.SEQ')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (30, 512, 512)
        self.expected_len = 1
        self.expected_Z = 30


class TestBioformatsLEI(_image_single, _image_stack, unittest.TestCase):
    # Leica format
    # 256 x 256 pixels, 8 bits per sample, 3 focal planes
    # Clay Glennon of the Wisconsin National Primate Research Center at the
    # UW-Madison has provided a dataset in Leica format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'leica_stack.lei')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (3, 256, 256)
        self.expected_len = 1
        self.expected_Z = 3


class TestBioformatsICS(_image_single, unittest.TestCase):
    # Image Cytometry Standard format
    # 256 x 256 pixels, 8 bits per sample
    # Nico urman of the Department of Cellular and Molecular Pharmacology at
    # the University of California-San Francisco has provided an image in Image
    # Cytometry Standard (ICS) format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'qdna1.ics')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (256, 256)
        self.expected_len = 1


class TestBioformatsZPO(_image_stack, _image_multichannel, unittest.TestCase):
    # PerkinElmer format, 672 x 512 pixels
    # 1 time point, 29 focal planes, 3 channels
    # Kevin O'Connell of NIH/NIDDK's Laboratory of Biochemistry and Genetics
    # has provided a multichannel 4D series in PerkinElmer format.
    def check_skip(self):
        _skip_if_no_bioformats()
        if not os.path.isfile(self.filename):
            raise unittest.SkipTest('File missing. Skipping.')

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'KEVIN2-3.zpo')
        self.check_skip()
        self.klass = pims.Bioformats
        self.kwargs = {'meta': False}
        self.v = self.klass(self.filename, **self.kwargs)
        self.expected_shape = (29, 672, 512)
        self.expected_C = 3
        self.expected_Z = 29


class TestBioformatsMetadataND2(unittest.TestCase):
    def check_skip(self):
        _skip_if_no_bioformats()

    def setUp(self):
        self.filename = os.path.join(path, 'bioformats', 'cluster.nd2')
        self.check_skip()
        self.klass = pims.Bioformats

    def tearDown(self):
        if hasattr(self, 'v'):
            self.v.close()

    def test_metadataretrieve(self):
        # tests using the metadata object are combined in one, to reduce the
        # amount of log output.
        self.v = self.klass(self.filename, meta=True)
        self.v.bundle_axes = 'yx'
        # test fields directly
        assert_equal(self.v.metadata.ChannelCount(0), 2)
        assert_equal(self.v.metadata.ChannelName(0, 0), '5-FAM/pH 9.0')
        assert_almost_equal(self.v.metadata.PixelsPhysicalSizeX(0),
                            0.167808983)
        assert_almost_equal(self.v.calibration, 0.167808983)
        # test metadata in Frame objects
        assert_almost_equal(self.v[0].metadata['t_s'], 0.445083498)
        assert_equal(self.v[0].metadata['t'], 0)
        # test changing frame_metadata
        del self.v.frame_metadata['t_s']
        assert 't_s' not in self.v[0].metadata
        self.v.frame_metadata['t_s'] = 'PlaneDeltaT'
        assert 't_s' in self.v[0].metadata
        # test colors field
        assert_allclose(self.v[0].metadata['colors'][0], [0.47, 0.91, 0.06],
                        atol=0.01)

    def test_metadata_raw(self):
        self.v = self.klass(self.filename, meta=False)
        metadata = self.v.get_metadata_raw('dict')
        assert_equal(metadata['ChannelCount'], '2')
        assert_equal(metadata['CH2ChannelDyeName'], '5-FAM/pH 9.0')
        assert_almost_equal(float(metadata['dCalibration']), 0.16780898323268245)

    def test_metadata_tags(self):
        self.v = self.klass(self.filename, meta=True)
        fields = self.v.metadata.fields
        assert 'PixelsPhysicalSizeX' in fields


if __name__ == '__main__':
    unittest.runmodule(argv=[__file__, '-vvs'],
                   exit=False)
