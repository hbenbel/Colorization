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
        image[:, :, 0] /= 100
        image[:, :, 1] /= 128
        image[:, :, 2] /= 128
        return image


class ToTensor(object):
    def __call__(self, image):
        return from_numpy(np.moveaxis(image, -1, 0))
