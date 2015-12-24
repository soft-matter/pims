import numpy as np
from pims import ViewerPipeline
from pims.viewer import Slider

def convert_to_grey(img, r, g, b):
    color_axis = img.shape.index(3)
    img = np.rollaxis(img, color_axis, 3)
    grey = (img * [r, g, b]).sum(2)
    return grey.astype(img.dtype)  # coerce to original dtype

def add_noise(img, noise_level):
    return img + np.random.random(img.shape) * noise_level

RGBToGrey = ViewerPipeline(convert_to_grey) + Slider('r', 0, 1, 0.2125) + \
            Slider('g', 0, 1, 0.7154) + Slider('b', 0, 1, 0.0721)
AddNoise = ViewerPipeline(add_noise) + Slider('noise_level', 0, 100, 0)
