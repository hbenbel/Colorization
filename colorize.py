import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import color, io
from skimage.transform import resize
from torch import from_numpy

from src.models import Generator32, Generator256


def preprocess(image):
    image = color.rgb2lab(image)
    image[:, :, 0] /= 100
    image[:, :, 1] /= 128
    image[:, :, 2] /= 128

    return from_numpy(np.moveaxis(image, -1, 0))


def postprocess(in_image, prediction):
    in_image = in_image[0, :, :, :]
    prediction = prediction[0, :, :, :]

    predicted_image = np.vstack([in_image, prediction])
    predicted_image = np.moveaxis(predicted_image, 0, -1)
    predicted_image[:, :, 0] *= 100
    predicted_image[:, :, 1] *= 128
    predicted_image[:, :, 2] *= 128
    predicted_image = color.lab2rgb(predicted_image)

    return predicted_image


def getModel(image_size):
    if image_size == 32:
        return Generator32()
    return Generator256()


def main(config):
    image_path = config['image_path']
    generator_path = config['generator_path']
    save_path = config['save_path']
    image_size = config['image_size']

    assert image_size == 32 or image_size == 256, "image_size should be equal to 32 or 256 for the training :("

    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    image = image[:, :, :3]
    original_image = resize(image, (image_size, image_size))
    image = preprocess(original_image)

    model = getModel(image_size).double()
    model.load_state_dict(
        torch.load(
            generator_path,
            map_location=torch.device('cpu')
        )
    )
    model.eval()

    in_image = image[0, :, :].unsqueeze(0).unsqueeze(0).double()
    prediction = model(in_image).cpu().data.numpy()

    post_processed = postprocess(in_image, prediction)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image)
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(post_processed)
    ax2.axis('off')

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Colorize Grayscale Image'
            )

    parser.add_argument(
        '--config',
        '-c',
        type=str,
        help='Path to the json config file',
        required=True
    )

    args = parser.parse_args()
    config_path = args.config

    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    main(config)
