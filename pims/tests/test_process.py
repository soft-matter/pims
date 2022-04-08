import os
import unittest
import numpy as np
from numpy.testing import assert_equal
from pims import FramesSequence, Frame
from pims.process import crop


class RandomReader(FramesSequence):
    def __init__(self, length=10, shape=(128, 128), dtype='uint8'):
        self._len = length
        self._dtype = dtype
        self._shape = shape
        data_shape = (length,) + shape
        if np.issubdtype(self._dtype, float):
            self._data = np.random.random(data_shape).astype(self._dtype)
        else:
            self._data = np.random.randint(0, np.iinfo(self._dtype).max,
                                           data_shape).astype(self._dtype)

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype

    def get_frame(self, i):
        return Frame(self._data[i], frame_no=i)


class PipelinesCommon(object):
    def test_on_frame(self):
        assert_equal(self.pipeline(self.rdr[0]), self.first_frame)

    def test_on_reader(self):
        assert_equal(self.pipeline(self.rdr)[0], self.first_frame)

    def test_on_random_frame(self):
        i = np.random.randint(0, len(self.rdr))
        assert_equal(self.pipeline(self.rdr)[i], self.pipeline(self.rdr[i]))


class TestCrop(PipelinesCommon, unittest.TestCase):
    def setUp(self):
        self.rdr = RandomReader(length=10, shape=(32, 33))
        self.pipeline = lambda x: crop(x, ((5, 32-26), (7, 33-27)))
        self.first_frame = self.rdr[0][5:26, 7:27]

    def test_attrs(self):
        proc = self.pipeline(self.rdr)
        assert_equal(self.rdr.pixel_type, proc.pixel_type)
        assert_equal(len(self.rdr), len(proc))
        assert_equal(self.rdr.frame_shape, (32, 33))
        assert_equal(proc.frame_shape, (21, 20))
