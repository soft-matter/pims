import unittest
import numpy as np
from pims.frame import Frame


def _skip_if_no_jinja2():
    try:
        import jinja2
    except ImportError:
        raise unittest.SkipTest('Jinja2 not installed. Skipping.')


def test_scalar_casting():
    tt = Frame(np.ones((5, 3)), frame_no=42)
    sum1 = tt.sum()
    assert np.isscalar(sum1)
    sum2 = tt.sum(keepdims=True)
    assert sum2.ndim == 2
    assert sum2.frame_no == tt.frame_no


def test_creation_md():
    md_dict = {'a': 1}
    frame_no = 42
    tt = Frame(np.ones((5, 3)), frame_no=frame_no, metadata=md_dict)
    assert tt.metadata == md_dict
    assert tt.frame_no == frame_no


def test_repr_html_():
    _skip_if_no_jinja2()
    # This confims a bugfix, where 16-bit images would raise
    # an error.
    Frame(10000*np.ones((50, 50), dtype=np.uint16))._repr_html_()


def test_copy():
    md_dict = {'a': 1}
    frame_no = 42
    tt_base = Frame(np.ones((5, 3)), frame_no=frame_no, metadata=md_dict)
    tt = Frame(tt_base)
    assert tt.metadata == md_dict
    assert tt.frame_no == frame_no


def test_copy_override_frame():
    frame_no = 42
    tt_base = Frame(np.ones((5, 3)), frame_no=frame_no)
    frame_no_2 = 123
    tt = Frame(tt_base, frame_no=frame_no_2)
    assert tt.frame_no == frame_no_2


def test_copy_update_md():
    frame_no = 42
    md_dict = {'a': 1}
    md_dict2 = {'b': 1}
    md_dict3 = {'a': 2, 'c': 3}
    tt_base = Frame(np.ones((5, 3)), frame_no=frame_no, metadata=md_dict)

    tt = Frame(tt_base, frame_no=frame_no, metadata=md_dict2)
    target_dict = dict(md_dict)
    target_dict.update(md_dict2)
    # print(target_dict)
    # print(tt.metadata)
    assert tt.metadata == target_dict

    tt2 = Frame(tt_base, frame_no=frame_no, metadata=md_dict3)
    assert tt2.metadata == md_dict3
