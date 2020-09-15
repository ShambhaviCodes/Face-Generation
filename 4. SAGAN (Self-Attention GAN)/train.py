import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from celeba import get_celeba_loader
from models import Discriminator, Generator
from utils import *

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():

    # Fix Seed for Reproducibility #
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Samples, Weights and Results Path #
    paths = [config.samples_path, config.weights_path, config.plots_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    celeba_loader = get_celeba_loader(path=config.celeba_path, batch_size=config.batch_size)
    total_batch = len(celeba_loader)

    # Prepare Networks #
    D = Discriminator().to(device)
    G = Generator().to(device)

    # Optimizer #
    D_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=config.D_lr, betas=(0.0, 0.9))
    G_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=config.G_lr, betas=(0.0, 0.9))

    D_optim_scheduler = get_lr_scheduler(D_optim)
    G_optim_scheduler = get_lr_scheduler(G_optim)

    # Fixed Noise #
    fixed_noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)

    # Lists #
    D_losses, G_losses = [], []

    # Train #
    print("Training has started with total epoch of {}.".format(config.num_epochs))

    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(celeba_loader):

            # Data Preparation #
            images = images.to(device)
            noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)

            # Initialize Optimizers #
            D_optim.zero_grad()
            G_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            # Hinge Loss using Real Image #
            prob_real = D(images)[0]
            D_real_loss = nn.ReLU()(1.0-prob_real).mean()

            # Hinge Loss using Generated Image #
            fake_image = G(noise)[0]
            prob_fake = D(fake_image.detach())[0]
            D_fake_loss = nn.ReLU()(1.0+prob_fake).mean()

            # Calculate Total Discriminator Loss #
            D_loss = D_real_loss + D_fake_loss

            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

            ###################
            # Train Generator #
            ###################

            # Hinge Loss using Generated Image #
            fake_image = G(noise)[0]
            prob_fake = D(fake_image)[0]

            # Calculate Total Generator Loss #
            G_loss = - prob_fake.mean()

            # Back Propagation and Update #
            G_loss.backward()
            G_optim.step()

            # Contain Losses #
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i+1) % config.print_every == 0:
                print("Epochs [{}/{}] | Iterations [{}/{}] | D Loss {:.4f} | G Loss {:.4f}"
                      .format(epoch+1, config.num_epochs, i+1, total_batch, np.mean(D_losses), np.mean(G_losses)))

        # Sample Images #
        sample_images(G, fixed_noise, epoch)

        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights #
        if (epoch + 1) % config.save_every == 0:
            torch.save(G.state_dict(),
                       os.path.join(config.weights_path, 'Face_Generator_Epoch_{}.pkl'.format(epoch + 1)))

    # Make a GIF file #
    make_gifs_train("Face_Generation", config.samples_path)

    # Plot Losses #
    plot_losses(D_losses, G_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
