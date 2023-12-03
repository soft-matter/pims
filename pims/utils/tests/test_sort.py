import unittest

from pims.utils.sort import natural_keys


class TestNaturalSort(unittest.TestCase):
    def setUp(self):
        pass

    def test_natural_keys(self):
        alist = ["something1", "something12", "something17", "something2"]
        alist.sort(key=natural_keys)
        assert alist == ['something1', 'something2', 'something12', 'something17']

    def test_natural_keys_with_spaces(self):
        paths = [
            '/data/meh/img-   19.tiff',
            '/data/meh/img-   181.tiff',
            '/data/meh/img-   20.tiff',
            '/data/meh/img-    0.tiff',
            '/data/meh/img-    1.tiff',
            '/data/meh/img-    2.tiff',
        ]

        paths.sort(key=natural_keys)

        assert paths == [
            '/data/meh/img-    0.tiff',
            '/data/meh/img-    1.tiff',
            '/data/meh/img-    2.tiff',
            '/data/meh/img-   19.tiff',
            '/data/meh/img-   20.tiff',
            '/data/meh/img-   181.tiff',

        ]
