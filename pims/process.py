from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np

from slicerator import pipeline

@pipeline
def as_grey(frame):
    """Convert a 2D image or PIMS reader to greyscale.

    This weights the color channels according to their typical
    response to white light.

    It does nothing if the input is already greyscale.
    """
    if len(frame.shape) == 2:
        return frame
    else:
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        return 0.2125 * red + 0.7154 * green + 0.0721 * blue

# "Gray" is the more common spelling
as_gray = as_grey