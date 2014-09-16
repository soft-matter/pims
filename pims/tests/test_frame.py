from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from pims.frame import Frame
from nose.tools import assert_true, assert_equal


def test_scalar_casting():
    tt = Frame(np.ones((5, 3)), frame_no=42)
    sum1 = tt.sum()
    assert_true(np.isscalar(sum1))
    sum2 = tt.sum(keepdims=True)
    assert_equal(sum2.ndim, 2)
    assert_equal(sum2.frame_no, tt.frame_no)


def test_creation_md():
    md_dict = {'a': 1}
    frame_no = 42
    tt = Frame(np.ones((5, 3)), frame_no=frame_no, metadata=md_dict)
    assert_equal(tt.metadata, md_dict)
    assert_equal(tt.frame_no, frame_no)


def test_repr_png():
    # This confims a bugfix, where 16-bit images would raise
    # an error.
    Frame(10000*np.ones((50, 50), dtype=np.uint16))._repr_png_()
