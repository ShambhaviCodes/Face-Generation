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
    paths = [config.samples_gen_path, config.samples_recon_path, config.weights_path, config.plots_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data Loader #
    celeba_loader = get_celeba_loader(path=config.celeba_path, batch_size=config.batch_size)
    total_batch = len(celeba_loader)

    # Prepare Networks #
    D = Discriminator().to(device)
    G = Generator().to(device)

    # Loss Function #
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l2 = nn.MSELoss()

    # Optimizer #
    Disc_optim = torch.optim.Adam(D.parameters(), lr=config.D_lr, betas=(0.5, 0.999))
    Enc_optim = torch.optim.Adam(G.encoder.parameters(), lr=config.G_lr, betas=(0.5, 0.999))
    Dec_optim = torch.optim.Adam(G.decoder.parameters(), lr=config.G_lr, betas=(0.5, 0.999))

    Disc_optim_scheduler = get_lr_scheduler(Disc_optim)
    Enc_optim_scheduler = get_lr_scheduler(Enc_optim)
    Dec_optim_scheduler = get_lr_scheduler(Dec_optim)

    # Labels #
    real_labels = torch.ones(config.batch_size, 1).to(device)
    fake_labels = torch.zeros(config.batch_size, 1).to(device)

    # Fixed Noise #
    fixed_noise = torch.randn(config.batch_size, config.noise_dim).to(device)

    # Lists #
    Disc_losses, Enc_losses, Dec_losses = list(), list(), list()

    # Train #
    print("Training has started with total epoch of {}.".format(config.num_epochs))

    for epoch in range(config.num_epochs):
        for i, (images, labels) in enumerate(celeba_loader):

            # Data Preparation #
            images = images.to(device)

            # Initialize Optimizers #
            Disc_optim.zero_grad()
            Enc_optim.zero_grad()
            Dec_optim.zero_grad()

            #######################
            # Train Discriminator #
            #######################

            # Noise and Generate Fake Images #
            noise = torch.randn(images.size(0), config.noise_dim).to(device)
            fake_images, mu, log_var = G(images)

            # Discriminator GAN Loss #
            prob_real = D(images)
            Disc_real_loss = criterion_bce(prob_real, real_labels)

            prob_fake = D(fake_images)
            Disc_fake_loss = criterion_bce(prob_fake, fake_labels)

            prob_p_fake = D(G.decoder(noise))
            Disc_fake_p_loss = criterion_bce(prob_p_fake, fake_labels)

            # Calculate Total Discriminator Loss #
            Disc_loss = Disc_real_loss + Disc_fake_loss + Disc_fake_p_loss

            # Back Propagation and Update #
            Disc_loss.backward()
            Disc_optim.step()

            #################
            # Train Decoder #
            #################

            # Generate Fake Images #
            fake_images, mu, log_var = G(images)

            # Decoder GAN Loss #
            prob_real = D(images)
            Dec_real_loss = criterion_bce(prob_real, real_labels)

            prob_fake = D(fake_images)
            Dec_fake_loss = criterion_bce(prob_fake, fake_labels)

            prob_p_fake = D(G.decoder(noise))
            Dec_fake_p_loss = criterion_bce(prob_p_fake, fake_labels)

            # Calculate Total Decoder GAN Loss #
            Dec_gan_loss = -1 * (Dec_fake_loss + Dec_fake_p_loss + Dec_real_loss)

            # Decoder Reconstruction Loss #
            Dec_recon_loss = criterion_l2(D.feature(fake_images), D.feature(images))

            # Calculate Total Decoder Loss #
            Dec_loss = Dec_gan_loss + config.gamma * Dec_recon_loss

            # Back Propagation and Update #
            Dec_loss.backward()
            Dec_optim.step()

            #################
            # Train Encoder #
            #################

            # Generate Fake Images #
            fake_images, mu, log_var = G(images)

            # Encoder Prior Loss #
            Enc_prior_loss = -0.5 * torch.mean(1 + log_var - torch.pow(mu, 2) - torch.exp(log_var))

            # Encoder Reconstruction Loss #
            Enc_recon_loss = criterion_l2(D.feature(fake_images), D.feature(images))

            # Calculate Total Encoder Loss #
            Enc_loss = Enc_prior_loss + config.beta * Enc_recon_loss

            # Back Propagation and Update #
            Enc_loss.backward()
            Enc_optim.step()

            # Add items to Lists #
            Disc_losses.append(Disc_loss.item())
            Enc_losses.append(Enc_loss.item())
            Dec_losses.append(Dec_loss.item())

            ####################
            # Print Statistics #
            ####################

            if (i + 1) % config.print_every == 0:
                print("Epochs [{}/{}] | Iterations [{}/{}] | Disc Loss {:.4f} | Enc Loss {:.4f} | Dec Loss {:.4f}".
                      format(epoch + 1, config.num_epochs, i + 1, total_batch, np.average(Disc_losses), np.average(Enc_losses), np.average(Dec_losses)))

                # Sample Images #
                sample_images(G, celeba_loader, fixed_noise, epoch)

        # Adjust Learning Rate #
        Disc_optim_scheduler.step()
        Enc_optim_scheduler.step()
        Dec_optim_scheduler.step()

        # Save Model Weights #
        if (epoch + 1) % config.save_every == 0:
            torch.save(G.state_dict(), os.path.join(config.weights_path, 'Face_Generator_Epoch_{}.pkl'.format(epoch + 1)))

    # Make a GIF file #
    make_gifs_train("Face_Generation", config.samples_gen_path)
    make_gifs_train("Face_Reconstruction", config.samples_recon_path)

    # Plot Losses #
    plot_losses(Disc_losses, Enc_losses, Dec_losses, config.num_epochs, config.plots_path)

    print("Training finished.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()