import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder Network"""
    def __init__(self):
        super(Encoder, self).__init__()

        self.in_channel = 3
        self.dim = 64
        self.h_dim = 64

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channel, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2, 2),


            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim * 2, self.dim * 2, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2, 2),


            nn.Conv2d(self.dim * 2, self.dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim * 2, self.dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim * 2, self.dim * 3, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2, 2),


            nn.Conv2d(self.dim * 3, self.dim * 3, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim * 3, self.dim * 3, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim * 3, self.dim * 3, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * self.dim * 3, self.h_dim)
        )

    def forward(self, x):
        x = self.main(x)
        out = self.fc(x)
        return out


class Decoder(nn.Module):
    """Decoder Network"""
    def __init__(self):
        super(Decoder, self).__init__()

        self.h_dim = 64
        self.dim = 64
        self.out_channel = 3

        self.fc = nn.Linear(in_features=self.h_dim, out_features=8 * self.dim * 8)

        self.main = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.Conv2d(self.dim, self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.dim, 8, 8)
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self):
        super(Generator, self).__init__()

        self.generator = Decoder()

    def forward(self, x):
        out = self.generator(x)
        return out