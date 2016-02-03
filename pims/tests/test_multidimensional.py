from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import unittest
import nose
from itertools import chain, permutations
import numpy as np
from numpy.testing import assert_equal
import pims
from pims import FramesSequenceND, reads_axes


class TestMultidimensional(unittest.TestCase):
    def setUp(self):
        class IndexReturningReader(FramesSequenceND):
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


class TestFramesSequenceND(unittest.TestCase):
    def test_reader_x(self):
        class RandomReader_x(FramesSequenceND):
            def __init__(self, **sizes):
                for key in sizes:
                    self._init_axis(key, sizes[key])

            @reads_axes('x')
            def _get_frame(self, **ind):
                return np.random.randint(0, 255, (128,)).astype(np.uint8)

            @property
            def pixel_type(self):
                return np.uint8

        sizes = dict(x=128, y=64, c=3, z=10)
        all_modes = chain(*[permutations(sizes, x)
                            for x in range(1, len(sizes) + 1)])
        reader = RandomReader_x(**sizes)
        for bundle in all_modes:
            reader.bundle_axes = bundle
            assert_equal(reader[0].shape, [sizes[k] for k in bundle])

    def test_reader_yx(self):
        class RandomReader_yx(FramesSequenceND):
            def __init__(self, **sizes):
                for key in sizes:
                    self._init_axis(key, sizes[key])

            @reads_axes('yx')
            def _get_frame(self, **ind):
                return np.random.randint(0, 255, (64, 128)).astype(np.uint8)

            @property
            def pixel_type(self):
                return np.uint8

        sizes = dict(x=128, y=64, c=3, z=10)
        all_modes = chain(*[permutations(sizes, x)
                            for x in range(1, len(sizes) + 1)])
        reader = RandomReader_yx(**sizes)
        for bundle in all_modes:
            reader.bundle_axes = bundle
            assert_equal(reader[0].shape, [sizes[k] for k in bundle])

    def test_reader_yxc(self):
        class RandomReader_yxc(FramesSequenceND):
            def __init__(self, **sizes):
                for key in sizes:
                    self._init_axis(key, sizes[key])

            @reads_axes('yxc')
            def _get_frame(self, **ind):
                return np.random.randint(0, 255, (64, 128, 3)).astype(np.uint8)

            @property
            def pixel_type(self):
                return np.uint8

        sizes = dict(x=128, y=64, c=3, z=10)
        all_modes = chain(*[permutations(sizes, x)
                            for x in range(1, len(sizes) + 1)])
        reader = RandomReader_yxc(**sizes)
        for bundle in all_modes:
            reader.bundle_axes = bundle
            assert_equal(reader[0].shape, [sizes[k] for k in bundle])

    def test_reader_compatibility(self):
        class RandomReader_2D(FramesSequenceND):
            def __init__(self, **sizes):
                for key in sizes:
                    self._init_axis(key, sizes[key])

            def get_frame_2D(self, **ind):
                return np.random.randint(0, 255, (64, 128)).astype(np.uint8)

            @property
            def pixel_type(self):
                return np.uint8

        sizes = dict(x=128, y=64, c=3, z=10)
        all_modes = chain(*[permutations(sizes, x)
                            for x in range(1, len(sizes) + 1)])
        reader = RandomReader_2D(**sizes)
        for bundle in all_modes:
            reader.bundle_axes = bundle
            assert_equal(reader[0].shape, [sizes[k] for k in bundle])

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
