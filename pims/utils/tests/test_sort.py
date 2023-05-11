import unittest

from pims.utils.sort import natural_keys


class TestNaturalSort(unittest.TestCase):
    def setUp(self):
        pass

    def test_natural_keys(self):
        alist = ["something1", "something12", "something17", "something2"]
        alist.sort(key=natural_keys)
        assert alist == ['something1', 'something2', 'something12', 'something17']
