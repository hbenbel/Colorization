import argparse
import json

import numpy as np
import torch
from skimage import color, io
from skimage.transform import resize
from torch import from_numpy

from src.models import Generator


def preprocess(image):
    image = resize(image, (256, 256))
    image = color.rgb2lab(image)
    min_val = image.min()
    max_val = image.max()
    image = 2*((image - min_val)/(max_val - min_val))-1

    return from_numpy(np.moveaxis(image, -1, 0)), min_val, max_val


def postprocess(in_image, prediction, min_val, max_val):
    in_image = in_image[0, :, :, :]
    prediction = prediction[0, :, :, :]

    predicted_image = np.vstack([in_image, prediction])
    predicted_image = np.moveaxis(predicted_image, 0, -1)
    predicted_image += 1
    predicted_image /= 2
    predicted_image *= (max_val - min_val)
    predicted_image += min_val
    predicted_image = color.lab2rgb(predicted_image)

    return predicted_image


def main(config):
    image_path = config['image_path']
    generator_path = config['generator_path']
    save_path = config['save_path']

    image = io.imread(image_path)[:, :, :3]
    image, min_val, max_val = preprocess(image)

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

    post_processed = postprocess(in_image, prediction, min_val, max_val)

    io.imsave(
        save_path,
        post_processed
    )


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
