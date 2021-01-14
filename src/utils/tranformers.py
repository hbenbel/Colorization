import numpy as np
from skimage import color
from skimage.transform import resize
from torch import from_numpy


class RGB2LAB(object):
    def __call__(self, image):
        lab_image = color.rgb2lab(image)
        return lab_image


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return resize(image, self.size)


class NormalizeImage(object):
    def __call__(self, image):
        min_val = image.min()
        max_val = image.max()
        return 2*((image - min_val)/(max_val - min_val))-1, min_val, max_val


class ToTensor(object):
    def __call__(self, normalized_image_info):
        image = normalized_image_info[0]
        min_val = normalized_image_info[1]
        max_val = normalized_image_info[2]
        return from_numpy(np.moveaxis(image, -1, 0)), min_val, max_val
