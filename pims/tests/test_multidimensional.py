import unittest
from itertools import chain, permutations
import numpy as np
from numpy.testing import assert_equal
from pims import FramesSequenceND, Frame


class TestMultidimensional(unittest.TestCase):
    def setUp(self):
        class IndexReturningReader(FramesSequenceND):
            @property
            def pixel_type(self):
                return np.uint8

            def __init__(self, c, m, t, z):
                super(IndexReturningReader, self).__init__()
                self._init_axis('x', 6)
                self._init_axis('y', 1)
                self._init_axis('z', z)
                self._init_axis('c', c)
                self._init_axis('m', m)
                self._init_axis('t', t)
                self.bundle_axes = 'yx'

            def get_frame_2D(self, **ind):
                result_order = ['c', 'm', 't', 'x', 'y', 'z']
                result = [ind[a] for a in result_order]
                metadata = {i: ind[i] for i in result_order}
                return Frame([result], metadata=metadata).astype(np.uint8)

        self.v = IndexReturningReader(c=3, m=5, t=100, z=20)

    def test_iterate(self):
        # index order c, m, t, x, y, z
        self.v.iter_axes = 't'
        for i in [0, 1, 15]:
            assert_equal(self.v[i], [[0, 0, i, 0, 0, 0]])
        self.v.iter_axes = 'm'
        for i in [0, 1, 3]:
            assert_equal(self.v[i], [[0, i, 0, 0, 0, 0]])
        self.v.iter_axes = 'zc'
        assert_equal(self.v[0], [[0, 0, 0, 0, 0, 0]])
        assert_equal(self.v[2], [[2, 0, 0, 0, 0, 0]])
        assert_equal(self.v[30], [[0, 0, 0, 0, 0, 10]])
        self.v.iter_axes = 'cz'
        assert_equal(self.v[0], [[0, 0, 0, 0, 0, 0]])
        assert_equal(self.v[4], [[0, 0, 0, 0, 0, 4]])
        assert_equal(self.v[21], [[1, 0, 0, 0, 0, 1]])
        self.v.iter_axes = 'tzc'
        assert_equal(self.v[0], [[0, 0, 0, 0, 0, 0]])
        assert_equal(self.v[4], [[1, 0, 0, 0, 0, 1]])
        assert_equal(self.v[180], [[0, 0, 3, 0, 0, 0]])
        assert_equal(self.v[210], [[0, 0, 3, 0, 0, 10]])
        assert_equal(self.v[212], [[2, 0, 3, 0, 0, 10]])

    def test_default(self):
        self.v.iter_axes = 't'
        self.v.default_coords['m'] = 2
        for i in [0, 1, 3]:
            assert_equal(self.v[i], [[0, 2, i, 0, 0, 0]])
        self.v.default_coords['m'] = 0
        for i in [0, 1, 3]:
            assert_equal(self.v[i], [[0, 0, i, 0, 0, 0]])

    def test_bundle(self):
        self.v.bundle_axes = 'zyx'
        assert_equal(self.v[0].shape, (20, 1, 6))
        self.v.bundle_axes = 'cyx'
        assert_equal(self.v[0].shape, (3, 1, 6))
        self.v.bundle_axes = 'czyx'
        assert_equal(self.v[0].shape, (3, 20, 1, 6))
        self.v.bundle_axes = 'zcyx'
        assert_equal(self.v[0].shape, (20, 3, 1, 6))

    def test_frame_no(self):
        self.v.iter_axes = 't'
        for i in np.random.randint(0, 100, 10):
            assert_equal(self.v[i].frame_no, i)
        self.v.iter_axes = 'zc'
        for i in np.random.randint(0, 3*20, 10):
            assert_equal(self.v[i].frame_no, i)

    def test_metadata(self):
        self.v.iter_axes = 't'
        self.v.bundle_axes = 'czyx'
        md = self.v[15].metadata

        assert_equal(md['axes'], tuple('czyx'))
        assert_equal(md['coords']['t'], 15)
        assert_equal(md['coords']['m'], 0)
        for ax in 'czyx':
            assert ax not in md['coords']

        # if metadata is provided, it should have the correct shape
        assert_equal(md['z'].shape, (3, 20))  # shape 'c', 'z'
        assert_equal(md['z'][:, 5], 5)
        assert_equal(md['c'][1, :], 1)

        # if a metadata field is equal for all frames, it should be a scalar
        assert_equal(md['t'], 15)

    def test_mutability(self):
        # test for issues that may arise when properties return mutable objects

        # the list bundle_axes
        self.v.bundle_axes = ['z', 'y', 'x']
        temp = self.v.bundle_axes
        temp = []
        assert_equal(self.v[0].shape, (20, 1, 6))

        # the list iter_axes
        self.v.iter_axes = ['t']
        temp = self.v.iter_axes
        temp = []
        assert_equal(len(self.v), 100)

        # the dict default_coords
        # changing an nonexisting item on a copy does nothing
        a = dict(self.v.default_coords)
        a['non_existing'] = 0
        # but assigning it again raises
        with self.assertRaises(ValueError) as cm:
            self.v.default_coords = a

        # changing an nonexisting item should raise
        with self.assertRaises(ValueError) as cm:
            self.v.default_coords['non_existing'] = 0


class RandomReaderFlexible(FramesSequenceND):
    def __init__(self, reads_axes, **sizes):
        super(RandomReaderFlexible, self).__init__()
        for key in sizes:
            self._init_axis(key, sizes[key])
        self._gf_shape = tuple([sizes[a] for a in reads_axes])
        self._register_get_frame(self._get_frame, reads_axes)

    def _get_frame(self, **ind):
        return np.empty(self._gf_shape, dtype=np.uint8)

    @property
    def pixel_type(self):
        return np.uint8


class TestFramesSequenceND(unittest.TestCase):
    def test_flexible_get_frame(self):
        sizes = dict(x=128, y=64, c=3, z=10)
        all_modes = list(chain(*[permutations(sizes, x)
                               for x in range(1, len(sizes) + 1)]))
        for read_mode in all_modes:
            reader = RandomReaderFlexible(read_mode, **sizes)
            for bundle in all_modes:
                reader.bundle_axes = bundle
                assert_equal(reader[0].shape, [sizes[k] for k in bundle])

    def test_flexible_get_frame_compatibility(self):
        class RandomReader_2D(FramesSequenceND):
            def __init__(self, **sizes):
                super(RandomReader_2D, self).__init__()
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
    unittest.main()
