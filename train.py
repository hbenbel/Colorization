import argparse
import glob
import json
import os
import random

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.datasets import ImageColorizationDataset
from src.models import Discriminator, Generator
from src.trainer import DCGANTrainer
from src.utils import RGB2LAB, NormalizeImage, Resize


def save_list(data, save_path, name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, name)

    with open(file_path, 'w') as f:
        for image_path in data:
            f.write(image_path + "\n")


def splitData(data_path, save_path, train=0.8, test=0.1, shuffle=True):
    assert os.path.exists(data_path), "data_path given to splitData doesn't exists :("

    data = glob.glob(os.path.join(data_path, '*'))

    if shuffle is True:
        data = random.shuffle(data)

    data_size = len(data)
    train_size = int(data_size * train)
    test_size = int(data_size * test)

    train_data = data[:train_size]
    test_data = data[train_size:train_size+test_size]
    validation_data = data[train_size+test_size:]

    save_list(train_data, save_path, 'train.txt')
    save_list(test_data, save_path, 'test.txt')
    save_list(validation_data, save_path, 'validation_data.txt')

    return train_data, test_data, validation_data


def main(config):
    train_data, test_data, validation_data = splitData(
                                                data_path=config['data_path'],
                                                save_path=config['save_path'],
                                                shuffle=config['shuffle_data']
                                            )

    height, width = config['image_size'][0], config['image_size'][1]
    transforms = torchvision.transforms.Compose([
                    Resize(size=(height, width)),
                    RGB2LAB(),
                    NormalizeImage()
                ])

    train_data_loader = DataLoader(
                            ImageColorizationDataset(
                                dataset=train_data,
                                transforms=transforms
                            ),
                            batch_size=config['batch_size'],
                            shuffle=True
                        )

    test_data_loader = DataLoader(
                            ImageColorizationDataset(
                                dataset=test_data,
                                transforms=transforms
                            ),
                            shuffle=False
                        )

    validation_data_loader = DataLoader(
                                ImageColorizationDataset(
                                    dataset=validation_data,
                                    transforms=transforms
                                ),
                                shuffle=False
                            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g_model = Generator().to(device)
    d_model = Discriminator().to(device)

    g_optimizer = Adam(
                    params=list(g_model.parameters()),
                    lr=config['learning_rate'],
                    betas=config['betas']
                )

    d_optimizer = Adam(
                    params=list(d_model.parameters()),
                    lr=config['learning_rate'],
                    betas=config['betas']
                )

    trainer = DCGANTrainer(
                g_model=g_model,
                d_model=d_model,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                config=config,
                train_data_loader=train_data_loader,
                validation_data_loader=validation_data_loader,
                device=device
            )

    trainer.train()


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
