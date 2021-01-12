import os
from sys import maxsize

import torch
import torch.nn as nn
from tqdm import tqdm


class DCGANTrainer:
    def __init__(self, g_model, d_model, g_optimizer, d_optimizer, config,
                 train_data_loader, validation_data_loader, device):
        self.device = device
        self.g_model = g_model.double()
        self.d_model = d_model.double()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_criterion1 = nn.BCEWithLogitsLoss()
        self.g_criterion2 = nn.L1Loss()
        self.d_criterion = nn.BCEWithLogitsLoss()
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.l1_lambda = config['lambda']
        self.save_path = config['save_path']
        self.height = config['image_size'][0]
        self.width = config['image_size'][1]

    def _train_generator(self, l_images, ab_images):
        self.g_optimizer.zero_grad()

        prediction = self.g_model(l_images)

        generator_loss1 = self.g_criterion1(
                            prediction,
                            torch.ones(
                                (self.batch_size, 2, self.height, self.width),
                                dtype=torch.double
                            ).to(self.device)
                        ).to(self.device)

        generator_loss2 = self.g_criterion2(
                            prediction,
                            ab_images
                        ).to(self.device)

        generator_loss = generator_loss1 + self.l1_lambda * generator_loss2
        generator_loss.backward()
        self.g_optimizer.step()

        return generator_loss

    def _train_discriminator(self, l_images, ab_images, fake_ab_images):
        self.d_optimizer.zero_grad()

        real_prediction = self.d_model(torch.cat([l_images, ab_images], 1))

        fake_prediciton = self.d_model(
                            torch.cat([l_images, fake_ab_images], 1)
                        )

        real_loss = self.d_criterion(
                        real_prediction,
                        torch.ones(
                            self.batch_size,
                            dtype=torch.double
                        ).to(self.device)
                    ).to(self.device)

        fake_loss = self.d_criterion(
                        fake_prediciton,
                        torch.zeros(
                            self.batch_size,
                            dtype=torch.double
                        ).to(self.device)
                    ).to(self.device)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.d_optimizer.step()

        return discriminator_loss

    def _train_epoch(self):
        discriminator_loss = 0
        generator_loss = 0

        for images in tqdm(self.train_data_loader, desc="Training"):
            l_images = images[:, 0, :, :].unsqueeze(1).double()
            ab_images = images[:, 1:, :, :].double()

            l_images = l_images.to(self.device)
            ab_images = ab_images.to(self.device)

            fake_ab_images = self.g_model(l_images)

            discriminator_loss += self._train_discriminator(
                                    l_images,
                                    ab_images,
                                    fake_ab_images
                                )

            generator_loss += self._train_generator(
                                l_images,
                                ab_images
                            )

        return discriminator_loss, generator_loss

    def _validate_epoch(self):
        validation_loss = 0

        self.g_model.eval()

        with torch.no_grad():
            for images in tqdm(self.validation_data_loader, desc="Validation"):
                l_images = images[:, 0, :, :].unsqueeze(1).double()
                ab_images = images[:, 1:, :, :].double()

                l_images = l_images.to(self.device)
                ab_images = ab_images.to(self.device)

                fake_ab_images = self.g_model(l_images)

                validation_loss += self.g_criterion2(fake_ab_images, ab_images)

        return validation_loss

    def _save_model(self, epoch):
        saving_path = os.path.join(self.save_path, str(epoch))
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        torch.save(
            self.g_model.state_dict(),
            os.path.join(saving_path, 'generator.pth')
        )

        torch.save(
            self.d_model.state_dict(),
            os.path.join(saving_path, 'discriminator.pth')
        )

    def train(self):
        min_loss = maxsize

        for epoch in tqdm(range(self.epochs), desc="Epoch"):
            training_loss_d, training_loss_g = self._train_epoch()
            validation_loss = self._validate_epoch()

            if validation_loss < min_loss:
                min_loss = validation_loss
                self._save_model(epoch)

            training_loss_d = training_loss_d / len(self.train_data_loader)
            training_loss_g = training_loss_g / len(self.train_data_loader)
            validation_loss = validation_loss / len(self.validation_data_loader)

            print("Epoch: {}, train loss d: {}, train loss g: {}, validation loss: {}"
                  .format(epoch + 1, training_loss_d, training_loss_g, validation_loss))
