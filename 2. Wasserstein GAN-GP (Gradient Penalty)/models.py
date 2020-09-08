import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*8, self.out_channel, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(-1, 1).squeeze(1)
        return out


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self):
        super(Generator, self).__init__()

        self.nz = 100
        self.ngf = 64
        self.out_channel = 3

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf, self.out_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out