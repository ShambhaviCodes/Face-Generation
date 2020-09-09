import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Self_Attention(nn.Module):
    """Self Attention Module"""
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        # 1x1 Convolution #
        self.query_conv1x1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, stride=1, padding=0)
        self.key_conv1x1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, stride=1, padding=0)
        self.value_conv1x1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)

        # Softmax #
        self.softmax = nn.Softmax(dim=-1)

        # For y_i = gamma * o_i + x_i #
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        inputs:
            x: input feature maps (B x C x W x H)
        outputs:
            out: self attention value + input features
            attention: B x N x N (where N = W X H)
        """
        batch_size, C, W, H = x.size()

        # Projection Query and Key
        projection_query = self.query_conv1x1(x)
        projection_query = projection_query.view(batch_size, -1, W * H)
        projection_query = projection_query.permute(0, 2, 1)

        projection_key = self.key_conv1x1(x)
        projection_key = projection_key.view(batch_size, -1, W * H)

        # projection_query_T * projection_key #
        S = torch.bmm(projection_query, projection_key)

        # Softmax #
        attention_map = self.softmax(S)
        attention = attention_map.permute(0, 2, 1)

        # Projection Value #
        projection_value = self.value_conv1x1(x).view(batch_size, -1, W * H)

        # Calculate Output #
        out = torch.bmm(projection_value, attention)
        out = out.view(batch_size, C, W, H)
        out = x + self.gamma * out

        return out, attention


class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channel = 3
        self.ndf = 64
        self.out_channel = 1

        self.layer_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.in_channel, self.ndf, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer_4 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer_5 = nn.Conv2d(self.ndf * 8, self.out_channel, 4, 1, 0)

        self.attention_1 = Self_Attention(in_dim=self.ndf * 4)
        self.attention_2 = Self_Attention(in_dim=self.ndf * 8)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out, att_1 = self.attention_1(out)
        out = self.layer_4(out)
        out, att_2 = self.attention_2(out)
        out = self.layer_5(out)
        return out, att_1, att_2


class Generator(nn.Module):
    """Generator Network"""
    def __init__(self):
        super(Generator, self).__init__()

        self.z_dim = 100
        self.ngf = 64
        self.out_channels = 3

        self.layer_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.z_dim, self.ngf * 8, 4, 1, 0)),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True)
        )

        self.layer_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1)),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True)
        )

        self.layer_3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1)),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True)
        )

        self.layer_4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1)),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True)
        )

        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.out_channels, 4, 2, 1),
            nn.Tanh()
        )

        self.attention_1 = Self_Attention(in_dim=self.ngf * 2)
        self.attention_2 = Self_Attention(in_dim=self.ngf)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out, att_1 = self.attention_1(out)
        out = self.layer_4(out)
        out, att_2 = self.attention_2(out)
        out = self.layer_5(out)
        return out, att_1, att_2