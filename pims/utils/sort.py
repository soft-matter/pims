import re

__all__ = ["natural_keys"]


def _atoi(text: str):
    return int(text) if text.isdigit() else text


def natural_keys(text: str):
    """Natural sort order function for strings.
    See: http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-
    inside
    Examples
    --------
    >>> alist=["something1", "something12", "something17", "something2"]
    >>> alist.sort(key=natural_keys)
    >>> print(alist)
    ['something1', 'something2', 'something12', 'something17']
    """
    return [_atoi(c) for c in re.split(r'(\d+)', text.replace(' ', ''))]