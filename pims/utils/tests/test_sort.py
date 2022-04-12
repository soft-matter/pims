from pims.utils.sort import natural_keys


def test_natural_keys():
    alist = ["something1", "something12", "something17", "something2"]
    alist.sort(key=natural_keys)
    assert alist == ['something1', 'something2', 'something12', 'something17']
