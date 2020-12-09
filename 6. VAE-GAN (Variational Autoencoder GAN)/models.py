import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder"""
    def __init__(self):
        super(Encoder, self).__init__()

        self.in_channels = 3
        self.dim = 64
        self.fc_dim = 2048
        self.noise_dim = 100

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, self.dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.dim, self.dim*2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.dim*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.dim*2, self.dim*4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.dim*4),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=8*8*self.dim*4, out_features=self.fc_dim),
            nn.BatchNorm1d(self.fc_dim),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Linear(self.fc_dim, self.noise_dim)
        self.logvar = nn.Linear(self.fc_dim, self.noise_dim)

    def forward(self, x):
        out = self.main(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        mu, logvar = self.mu(out), self.logvar(out)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder"""
    def __init__(self):
        super(Decoder, self).__init__()

        self.noise_dim = 100
        self.dim = 64
        self.out_channels = 3

        self.fc = nn.Sequential(
            nn.Linear(self.noise_dim, 8*8*self.dim*4),
            nn.BatchNorm1d(8*8*self.dim*4),
            nn.ReLU(inplace=True),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim * 4, self.dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.dim * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.dim*2, self.dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.dim, int(self.dim/2), kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(int(self.dim/2)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(self.dim/2), self.out_channels, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(len(x), self.dim * 4, 8, 8)
        out = self.main(out)
        return out


class Generator(nn.Module):
    """Generator"""
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu

    def generate(self, z):
        self.eval()
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        self.eval()
        mu, logvar = self.encoder(x)
        x_hat = self.decoder(mu)
        return x_hat


class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels = 3
        self.dim = 64
        self.fc_dim = 512
        self.out_channels = 1

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, int(self.dim//2), kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int(self.dim//2), self.dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim, self.dim*2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dim*2, self.dim*4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.dim*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=8*8*self.dim*4, out_features=self.fc_dim),
            nn.BatchNorm1d(self.fc_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.fc_dim, self.out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(-1, 8*8*self.dim*4)
        out = self.fc(out)
        return out

    def feature(self, x):
        out = self.main(x)
        return out