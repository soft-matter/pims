from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
from pims.frame import Frame
from PIL import Image


def plot_to_frame(fig, dpi, **imsave_kwargs):
    """ Renders a matplotlib figure or axes object into a numpy array
    containing RGBA data of the rendered image.

    Parameters
    ----------
    fig : matplotlib Figure or Axes object
    dpi : number, dots per inch used in figure rendering
    imsave_kwargs : keyword arguments passed to `Figure.imsave(...)`

    Returns
    -------
    pims.Frame object containing RGBA values (dtype uint8)
    """
    import matplotlib as mpl
    buffer = six.BytesIO()
    if isinstance(fig, mpl.axes.Axes):
        fig = fig.figure
    fig.savefig(buffer, format='tif', dpi=dpi, **imsave_kwargs)
    buffer.seek(0)
    im = np.asarray(Image.open(buffer))
    buffer.close()
    return Frame(im)


def plots_to_frame(figures, width=512, **imsave_kwargs):
    """ Renders an iterable of matplotlib figures or axes objects into a
    pims Frame object, that will be displayed as scrollable stack in IPython.

    Parameters
    ----------
    figures : iterable of matplotlib Figure or Axes objects
    width : integer, width in pixels
    imsave_kwargs : keyword arguments passed to `Figure.imsave(...)`

    Returns
    -------
    pims.Frame object containing a stack of RGBA values (dtype uint8)
    """
    import matplotlib as mpl
    if 'dpi' in imsave_kwargs or 'format' in imsave_kwargs:
        raise ValueError('Do not specify dpi or format imsave kwargs.')
    if isinstance(figures, mpl.axes.Axes) or \
       isinstance(figures, mpl.figure.Figure):
        raise ValueError('Use plot_to_frame for single figures, or supply '
                         'an iterable of figures to plots_to_frame.')

    # render first image to calculate the correct dpi and image size
    size = plot_to_frame(figures[0], 100, **imsave_kwargs).shape
    dpi = width * 100 / size[1]
    h = width * size[0] / size[1]

    frames = []
    for n, fig in enumerate(figures):
        im = plot_to_frame(fig, dpi, **imsave_kwargs)
        # make the image the same size as the first image
        if (im.shape[0] != h) or (im.shape[1] != width):
            im = np.pad(im[:h, :width], ((0, max(0, h - im.shape[0])),
                                         (0, max(0, width - im.shape[1])),
                                         (0, 0)), mode=b'constant')
        frames.append(im)

    return Frame(np.array(frames))
