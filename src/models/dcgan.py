import torch
import torch.nn as nn


def _down_sample(in_channels, out_channels, kernel_size=4, stride=2, negative_slope=0.2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope)
    )


def _up_sample(in_channels, out_channels, kernel_size=4, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        down_sample1 = _down_sample(1, 64)
        down_sample2 = _down_sample(64, 128)
        down_sample3 = _down_sample(128, 256)
        down_sample4 = _down_sample(256, 512)
        down_sample5 = _down_sample(512, 512)
        down_sample6 = _down_sample(512, 512)
        down_sample7 = _down_sample(512, 512)

        bottleneck = _down_sample(512, 512)

        up_sample1 = _up_sample(512, 512)
        up_sample2 = _up_sample(512, 512)
        up_sample3 = _up_sample(512, 512)
        up_sample4 = _up_sample(512, 256)
        up_sample5 = _up_sample(256, 128)
        up_sample6 = _up_sample(128, 64)
        up_sample7 = _up_sample(64, 64)

        output = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.down_sample1(x)
        x2 = self.down_sample2(x1)
        x3 = self.down_sample3(x2)
        x4 = self.down_sample4(x3)
        x5 = self.down_sample5(x4)
        x6 = self.down_sample6(x5)
        x7 = self.down_sample7(x6)

        x = self.bottleneck(x7)
        x = self.up_sample(x)

        x = self.up_sample(torch.cat((x7, x), 1))
        x = self.up_sample(torch.cat((x6, x), 1))
        x = self.up_sample(torch.cat((x5, x), 1))
        x = self.up_sample(torch.cat((x4, x), 1))
        x = self.up_sample(torch.cat((x3, x), 1))
        x = self.up_sample(torch.cat((x2, x), 1))
        x = self.up_sample(torch.cat((x1, x), 1))

        x = self.output(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        down_sample1 = _down_sample(3, 64)
        down_sample2 = _down_sample(64, 128)
        down_sample3 = _down_sample(128, 256)
        down_sample4 = _down_sample(256, 512)

        output = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down_sample1(x)
        x = self.down_sample2(x)
        x = self.down_sample3(x)
        x = self.down_sample4(x)

        x = self.output(x)

        return x
