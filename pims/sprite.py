from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import itertools
import numpy as np
from pims.frame import Frame
from pims.base_frames import FramesSequence


class SpriteSheet(FramesSequence):
    """
    This is a class for providing an easy interface into
    a sprite sheet of uniformly sized images.

    Parameters
    ----------
    sheet : ndarray
        The sprite sheet.  It should consist of N paneled images.  In
        this version all possible positions have an image, this may be changed
        to limit the number of acessable images in the sheet to be less than
        the possible number.

    rows : int
    cols : int
        The number of rows and columns of sprites.
        The sprite size is computed from these + the shape of the sheet.

    process_func : callable or None
        Pre-processing to be done

    dtype : np.dtype or None
        dtype of the returned array.  Defaults to the type of sheet

    as_gray : bool
        If the data should be converted to gray scale or not.
    """
    @classmethod
    def class_exts(cls):
        # does not know how to read files
        return {}

    def __init__(self, sheet, rows, cols, process_func=None, dtype=None,
                 as_grey=False):
        self._sheet = sheet
        sheet_height, sheet_width = sheet.shape
        if sheet_width % cols != 0:
            raise ValueError("Sheet width not evenly divisible by cols")
        if sheet_height % rows != 0:
            raise ValueError("Sheet height not evenly divisible by rows")

        self._sheet_shape = (rows, cols)

        self._im_sz = sheet_height // rows, sheet_width // cols
        self._sprite_height, self._sprite_width = self._im_sz

        if dtype is None:
            self._dtype = sheet.dtype
        else:
            self._dtype = dtype

        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return np.prod(self._sheet_shape)

    def get_frame(self, n):
        r, c = np.unravel_index(n, self._sheet_shape)
        slc_r = slice(r*self._sprite_height, (r+1)*self._sprite_height)
        slc_c = slice(c*self._sprite_width, (c+1)*self._sprite_width)
        tmp = self._sheet[slc_r, slc_c]
        return Frame(self.process_func(tmp).astype(self._dtype),
                     frame_no=n)
