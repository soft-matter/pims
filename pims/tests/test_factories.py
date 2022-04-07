import numpy as np
import unittest
from numpy.testing import assert_array_equal
from pims.image_sequence import customize_image_sequence


def test_customize_image_sequence():
    dummy = lambda filename, **kwargs: np.zeros((1, 1))
    reader = customize_image_sequence(dummy)
    result = reader(['a', 'b', 'c'])
    actual_len = len(result)
    assert actual_len == 3
    for frame in result:
        assert_array_equal(frame, np.zeros((1, 1)))
    reader2 = customize_image_sequence(dummy, 'my_name')
    assert reader2.__name__ == 'my_name'
