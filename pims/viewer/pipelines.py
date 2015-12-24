import numpy as np
from pims.viewer import ViewerPipeline, Slider


def convert_to_grey(img, r, g, b):
    color_axis = img.shape.index(3)
    img = np.rollaxis(img, color_axis, 3)
    grey = (img * [r, g, b]).sum(2)
    return grey.astype(img.dtype)  # coerce to original dtype


def add_noise(img, noise_level):
    return img + np.random.random(img.shape) * noise_level

# TODO: this does not work because it changes the image shape
# RGBToGrey = ViewerPipeline(convert_to_grey, 'RGB to Grey', dock='right') + \
#             Slider('r', 0, 1, 0.2125, orientation='vertical') + \
#             Slider('g', 0, 1, 0.7154, orientation='vertical') + \
#             Slider('b', 0, 1, 0.0721, orientation='vertical')
AddNoise = ViewerPipeline(add_noise, 'Add noise', dock='right') + \
           Slider('noise_level', 0, 100, 0, orientation='vertical')
