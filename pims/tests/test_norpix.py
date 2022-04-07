# Tests for norpix_reader.py

import os
from datetime import datetime
import unittest
import numpy as np
import pims
import pims.norpix_reader

tests_path, _ = os.path.split(os.path.abspath(__file__))

class _common_norpix_sample_tests(object):
    """Test the Norpix .seq reader on a sample file."""
    def setUp(self):
        self.seq = pims.open(self.sample_filename, **self.options)

    def tearDown(self):
        if hasattr(self, 'seq'):
            self.seq.close()

    def test_type(self):
        assert isinstance(self.seq, pims.norpix_reader.NorpixSeq)

    def test_metadata(self):
        s = self.seq
        assert isinstance(len(s), int)
        assert np.issubdtype(s.pixel_type, np.number)
        assert s.width > 0
        assert s.height > 0
        assert len(s.filename)
        assert len(repr(s))
        for k in ('description', 'bit_depth_real', 'origin',
                  'suggested_frame_rate', 'width', 'height',
                  'gamut'):
            assert k in s.metadata

    def test_get_frame(self):
        s = self.seq
        hashes = set()  # Check that each frame is unique
        for i in range(len(s)):
            fr = s[i]
            fhash = hash(np.array(fr).tobytes())
            assert fhash not in hashes
            hashes.add(fhash)

            for k in ('time', 'time_float', 'gamut'):
                assert k in fr.metadata

            assert fr.shape[1] == s.width
            assert fr.shape[0] == s.height

    def test_get_time(self):
        """Check all 3 ways to get time of a frame."""
        s = self.seq
        maxtime = 0.
        for i in range(len(s)):
            tdt = s.get_time(i)
            tfloat = s.get_time_float(i)
            fr = s[i]

            assert isinstance(tdt, datetime)

            assert tfloat > maxtime
            maxtime = tfloat

            assert fr.metadata['time'] == tdt
            assert fr.metadata['time_float'] == tfloat

    def test_get_time_mapped(self):
        s = self.seq
        sli = s[4:]
        assert s.get_time(4) == sli.get_time(0)
        assert s.get_time_float(4) == list(sli.get_time_float[:])[0]

    def test_dump_times(self):
        assert isinstance(self.seq.dump_times_float(), np.ndarray)

    def test_repr(self):
        assert len(repr(self.seq))


class test_norpix5_sample(_common_norpix_sample_tests, unittest.TestCase):
    def setUp(self):
        self.sample_filename = os.path.join(tests_path, 'data',
                                            'sample_norpix5.seq')
        if not os.path.exists(self.sample_filename):
            raise unittest.SkipTest('(Large) Norpix v5 sample file not found. '
                                    'Skipping.')

        self.options = {}
        super(test_norpix5_sample, self).setUp()


class _norpix6_sample_tests(_common_norpix_sample_tests):
    def setUp(self):
        self.sample_filename = os.path.join(tests_path, 'data',
                                            'sample_norpix6.seq')
        super(_norpix6_sample_tests, self).setUp()

    def test_specific_file_info(self):
        """Tests based on the specific file in the repo."""
        s = self.seq
        assert len(s) == 6
        assert s.height == 32
        assert s.width == 36


class test_defaults(_norpix6_sample_tests, unittest.TestCase):
    def setUp(self):
        self.options = {}
        super(test_defaults, self).setUp()

    def test_specific_file_dtype(self):
        """Based on the specific file in the repo."""
        assert self.seq[0].dtype == np.uint8


# class test_dtype(_norpix6_sample_tests, unittest.TestCase):
#     def setUp(self):
#         self.options = {}
#         self.dtype = np.float_
#         self.options['dtype'] = self.dtype
#         super(test_dtype, self).setUp()
#
#     def test_dtype(self):
#         fr = self.seq[0]
#         assert fr.dtype == self.dtype
#
#
# class test_process_func(_norpix6_sample_tests, unittest.TestCase):
#     def setUp(self):
#         self.options = {}
#         self.options['dtype'] = np.float_
#         self.options['process_func'] = lambda x: -x
#         super(test_process_func, self).setUp()
#
#     def test_process_func(self):
#         fr = self.seq[0]
#         assert np.all(fr <= 0)


class test_as_raw(_norpix6_sample_tests, unittest.TestCase):
    def setUp(self):
        self.options = {'as_raw': True}
        super(test_as_raw, self).setUp()
    #
    # def test_post_hoc_process_func(self):
    #     testbyte = self.seq[0][0,0]
    #     self.seq.set_process_func(lambda x: 255 - x)
    #     assert self.seq[0][0,0] == 255 - testbyte
