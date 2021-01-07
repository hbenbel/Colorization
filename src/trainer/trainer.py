import os
from sys import maxsize

import torch
import torch.nn as nn


class DCGANTrainer:
    def __init__(self, g_model, d_model, g_optimizer, d_optimizer, config,
                 training_data_loader, validation_data_loader, device):
        self.device = torch.device(device)
        self.g_model = g_model.to(self.device)
        self.d_model = d_model.to(self.device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_criterion1 = nn.BCELoss()
        self.g_criterion2 = nn.L1Loss()
        self.d_criterion = nn.BCELoss()
        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.l1_lambda = config['lambda']
        self.save_path = config['save_path']

    def _train_generator(self, l_images, ab_images, fake_ab_images):
        self.g_optimizer.zero_grad()

        prediction = self.g_model(torch.cat([l_images, fake_ab_images], 1))

        generator_loss1 = self.g_criterion1(
                            prediction,
                            torch.ones(self.batch_size)
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
                        torch.ones(self.batch_size)
                    ).to(self.device)

        fake_loss = self.d_criterion(
                        fake_prediciton,
                        torch.zeros(self.batch_size)
                    ).to(self.device)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.d_optimizer.step()

        return discriminator_loss

    def _train_epoch(self):
        discriminator_loss = 0
        generator_loss = 0

        for images in self.training_data_loader:
            l_images = images[:, 0, :, :]
            ab_images = images[:, 1:, :, :]

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
                                ab_images,
                                fake_ab_images
                            )

        return discriminator_loss, generator_loss

    def _validate_epoch(self):
        validation_loss = 0

        self.g_model.eval()
        for images in self.validation_data_loader:
            l_images = images[:, 0, :, :]
            ab_images = images[:, 1:, :, :]

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

        for epoch in range(self.epochs):
            training_loss_d, training_loss_g = self._train_epoch()
            validation_loss = self._validate_epoch()

            if validation_loss < min_loss:
                min_loss = validation_loss
                self._save_model(epoch)

            training_loss_d = training_loss_d / len(self.training_data_loader)
            training_loss_g = training_loss_g / len(self.training_data_loader)
            validation_loss = validation_loss / len(self.validation_data_loader)

            print("Epoch: {}, train loss d: {}, train loss g: {}, validation loss: {}"
                  .format(epoch + 1, training_loss_d, training_loss_g, validation_loss))
