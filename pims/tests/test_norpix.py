# Tests for norpix_reader.py

import os
import unittest
import numpy as np
import pims
import pims.norpix_reader

tests_path, _ = os.path.split(os.path.abspath(__file__))
sample_filename = os.path.join(tests_path, 'data', 'sample_norpix.seq')

class common_norpix_sample_tests(object):
    """Test the Norpix .seq reader on a sample file."""
    def setUp(self):
        self.seq = pims.open(sample_filename, **self.options)

    def tearDown(self):
        self.seq.close()

    def test_type(self):
        assert isinstance(self.seq, pims.norpix_reader.NorpixSeq)

    def test_metadata(self):
        s = self.seq
        assert isinstance(len(s), int)
        assert np.issubdtype(s.pixel_type, np.dtype)
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
        frames = {0, 1, len(s) - 1}  # In case of length 1 or 2
        hashes = set()  # Check that each frame is unique
        for i in sorted(frames):
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
        times = set()
        for i in (0, 1, len(s) - 1):
            tdt = s.get_time(i)
            tfloat = s.get_time_float(i)
            fr = s[i]

            assert tfloat not in times
            times.add(tfloat)

            assert fr.metadata['time'] == tdt
            assert fr.metadata['time_float'] == tfloat

    def test_dump_times(self):
        assert isinstance(self.seq.dump_times_float(), np.ndarray)


class test_defaults(common_norpix_sample_tests, unittest.TestCase):
    def setUp(self):
        self.options = {}
        super(test_defaults, self).setUp()


class test_dtype(common_norpix_sample_tests, unittest.TestCase):
    def setUp(self):
        self.options = {}
        self.dtype = np.float_
        self.options['dtype'] = self.dtype
        super(test_dtype, self).setUp()

    def test_dtype(self):
        fr = self.seq[0]
        assert fr.dtype == self.dtype


class test_process_func(common_norpix_sample_tests, unittest.TestCase):
    def setUp(self):
        self.options = {}
        self.options['dtype'] = np.float_
        self.options['process_func'] = lambda x: -x
        super(test_process_func, self).setUp()

    def test_process_func(self):
        fr = self.seq[0]
        assert np.all(fr <= 0)


