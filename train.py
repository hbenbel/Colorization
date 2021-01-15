import argparse
import glob
import json
import os
import random

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader, dataloader

from src.datasets import ImageColorizationDataset
from src.models import Discriminator, Generator
from src.tester import DCGANTester
from src.trainer import DCGANTrainer
from src.utils import RGB2LAB, NormalizeImage, Resize, ToTensor


def save_list(data, save_path, name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, name)

    with open(file_path, 'w') as f:
        for image_path in data:
            f.write(image_path + "\n")


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)


def splitData(data_path, save_path, train, test, shuffle):
    assert os.path.exists(data_path), "data_path given to splitData doesn't exists :("
    assert train + test < 1, "train percentage and test percentage should summup to be < 1 to keep some data for validation :("

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
    batch_size = config['batch_size']
    betas = config['betas']
    data_path = config['data_path']
    height, width = config['image_size'][0], config['image_size'][1]
    learning_rate = config['learning_rate']
    save_path = config['save_path']
    shuffle_data = config['shuffle_data']
    test_model = config['test_model']
    train_percentage = config['train_percentage']
    test_percentage = config['test_percentage']

    train_data, test_data, validation_data = splitData(
                                                data_path=data_path,
                                                save_path=save_path,
                                                train=train_percentage,
                                                test=test_percentage,
                                                shuffle=shuffle_data
                                            )

    transforms = torchvision.transforms.Compose([
                    Resize(size=(height, width)),
                    RGB2LAB(),
                    NormalizeImage(),
                    ToTensor()
                ])

    train_data_loader = DataLoader(
                            ImageColorizationDataset(
                                dataset=train_data,
                                transforms=transforms
                            ),
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=my_collate
                        )

    test_data_loader = DataLoader(
                            ImageColorizationDataset(
                                dataset=test_data,
                                transforms=transforms
                            ),
                            shuffle=False,
                            collate_fn=my_collate
                        )

    validation_data_loader = DataLoader(
                                ImageColorizationDataset(
                                    dataset=validation_data,
                                    transforms=transforms
                                ),
                                shuffle=False,
                                collate_fn=my_collate
                            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g_model = Generator().to(device)
    d_model = Discriminator().to(device)

    g_optimizer = Adam(
                    params=list(g_model.parameters()),
                    lr=learning_rate,
                    betas=betas
                )

    d_optimizer = Adam(
                    params=list(d_model.parameters()),
                    lr=learning_rate,
                    betas=betas
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

    if test_model is True:
        tester = DCGANTester(
                    g_model=g_model,
                    config=config,
                    test_data_loader=test_data_loader,
                    device=device
                )

        tester.test()


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
