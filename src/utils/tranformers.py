import numpy as np
from skimage import color
from skimage.transform import resize


class RGB2LAB(object):
    def __call__(self, image):
        lab_image = color.rgb2lab(image)
        return np.moveaxis(lab_image, -1, 0)


class LAB2RGB(object):
    def __call__(self, image):
        channel_last = np.moveaxis(image, 0, -1)
        return color.lab2rgb(channel_last)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return resize(image, self.size)


class NormalizeImage(object):
    def __call__(self, image):
        return 2*((image - np.min(image))/(np.max(image) - np.min(image)))-1
