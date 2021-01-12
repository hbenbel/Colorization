import torch
import torch.nn as nn


def _down_sample(in_channels, out_channels, padding=1, kernel_size=4, stride=2,
                 negative_slope=0.2):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope)
    )


def _up_sample(in_channels, out_channels, padding=1, kernel_size=4, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def _conv(in_channels, out_channels, stride=1, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down_sample1 = _down_sample(1, 64, kernel_size=3, stride=1)
        self.down_sample2 = _down_sample(64, 64)
        self.down_sample3 = _down_sample(64, 128)
        self.down_sample4 = _down_sample(128, 256)
        self.down_sample5 = _down_sample(256, 512)
        self.down_sample6 = _down_sample(512, 512)
        self.down_sample7 = _down_sample(512, 512)
        self.down_sample8 = _down_sample(512, 512)

        self.up_sample1 = _up_sample(512, 512)
        self.up_sample2 = _up_sample(512, 512)
        self.up_sample3 = _up_sample(512, 512)
        self.up_sample4 = _up_sample(512, 256)
        self.up_sample5 = _up_sample(256, 128)
        self.up_sample6 = _up_sample(128, 64)
        self.up_sample7 = _up_sample(64, 64)

        self.conv1 = _conv(1024, 512)
        self.conv2 = _conv(1024, 512)
        self.conv3 = _conv(1024, 512)
        self.conv4 = _conv(512, 256)
        self.conv5 = _conv(256, 128)
        self.conv6 = _conv(128, 64)
        self.conv7 = _conv(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 2, 1, 1),
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
        x8 = self.down_sample8(x7)

        x = self.conv1(torch.cat([x7, self.up_sample1(x8)], 1))
        x = self.conv2(torch.cat([x6, self.up_sample2(x)], 1))
        x = self.conv3(torch.cat([x5, self.up_sample3(x)], 1))
        x = self.conv4(torch.cat([x4, self.up_sample4(x)], 1))
        x = self.conv5(torch.cat([x3, self.up_sample5(x)], 1))
        x = self.conv6(torch.cat([x2, self.up_sample6(x)], 1))
        x = self.conv7(torch.cat([x1, self.up_sample7(x)], 1))

        x = self.output(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down_sample1 = _down_sample(3, 64)
        self.down_sample2 = _down_sample(64, 128)
        self.down_sample3 = _down_sample(128, 256)
        self.down_sample4 = _down_sample(256, 512, stride=1, kernel_size=3)

        self.conv = nn.Conv2d(512, 1, 1, 1)

        self.output = nn.Linear(32 * 32, 1)

    def forward(self, x):
        x = self.down_sample1(x)
        x = self.down_sample2(x)
        x = self.down_sample3(x)
        x = self.down_sample4(x)

        x = self.conv(x)
        x = x.view(-1, 32 * 32)
        x = self.output(x)
        x = x.squeeze(-1)

        return x
