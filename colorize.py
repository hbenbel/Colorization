import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import color, io
from skimage.transform import resize
from torch import from_numpy

from src.models import Generator


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


def main(config):
    image_path = config['image_path']
    generator_path = config['generator_path']
    save_path = config['save_path']

    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    image = image[:, :, :3]
    original_image = resize(image, (256, 256))
    image = preprocess(original_image)

    model = Generator().double()
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

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(post_processed)

    fig.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Train Image Colorization Algorithm'
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
