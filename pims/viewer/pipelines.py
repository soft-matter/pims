import numpy as np
from pims.viewer import ViewerPipeline, Slider, ViewerPlotting


def convert_to_grey(img, r, g, b):
    color_axis = img.shape.index(3)
    img = np.rollaxis(img, color_axis, 3)
    grey = (img * [r, g, b]).sum(2)
    return grey.astype(img.dtype)  # coerce to original dtype


def add_noise(img, noise_level):
    return img + np.random.random(img.shape) * noise_level

def tp_locate(image, radius, minmass, separation, noise_size, ax):
    _plot_style = dict(markersize=15, markeredgewidth=2,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none')
    from trackpy import locate
    f = locate(image, radius * 2 + 1, minmass, None, separation, noise_size)
    if len(f) == 0:
        return None
    else:
        return ax.plot(f['x'], f['y'], **_plot_style)

# TODO: this does not work because it changes the image shape
# RGBToGrey = ViewerPipeline(convert_to_grey, 'RGB to Grey', dock='right') + \
#             Slider('r', 0, 1, 0.2125, orientation='vertical') + \
#             Slider('g', 0, 1, 0.7154, orientation='vertical') + \
#             Slider('b', 0, 1, 0.0721, orientation='vertical')
AddNoise = ViewerPipeline(add_noise, 'Add noise', dock='right') + \
           Slider('noise_level', 0, 100, 0, orientation='vertical')
Locate = ViewerPlotting(tp_locate, 'Locate', dock='right') + \
       Slider('radius', 1, 20, 7, value_type='int', orientation='vertical') + \
       Slider('separation', 1, 20, 7, value_type='float', orientation='vertical') + \
       Slider('noise_size', 1, 20, 1, value_type='int', orientation='vertical') + \
       Slider('minmass', 1, 10000, 100, value_type='int', orientation='vertical')