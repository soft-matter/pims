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
