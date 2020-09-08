import os
import imageio
from matplotlib import pyplot as plt

import torch
from torch.nn import init
from torch.autograd import grad
from torchvision.utils import save_image

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def init_weights_normal(m):
    """Normal Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)


def init_weights_xavier(m):
    """Xavier Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)


def init_weights_kaiming(m):
    """Kaiming He Weight Initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')


def get_lr_scheduler(optimizer):
    """Learning Rate Scheduler"""
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)
    elif config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def set_requires_grad(networks, requires_grad=False):
    """Prevent a Network from Updating"""
    for network in networks:
        for param in network.parameters():
            param.requires_grad = requires_grad


def get_gradient_penalty(real_image, fake_image, discriminator):
    """Compute Gradient Penalty"""
    alpha = torch.rand(real_image.size(0), 1, 1, 1).to(device)
    interpolate = (alpha * real_image.data + (1 - alpha) * fake_image.data).requires_grad_(True)
    prob_interpolate = discriminator(interpolate)
    weight = torch.ones(prob_interpolate.size()).to(device)

    gradients = grad(outputs=prob_interpolate,
                     inputs=interpolate,
                     grad_outputs=weight,
                     retain_graph=True,
                     create_graph=True,
                     only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_l2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    out = torch.mean((gradients_l2norm - 1)**2)

    return out

def denorm(x):
    """De-normalization"""
    out = (x+1) / 2
    return out.clamp(0, 1)


def sample_images(generator, fixed_noise, epoch):
    """Sample Images for Every Epoch"""
    fake_celeba = generator(fixed_noise.detach())
    save_image(denorm(fake_celeba[0:64].data),
               os.path.join(config.samples_path, 'Face_Generation_Epoch_%03d.png' % (epoch+1))
               )


def plot_losses(discriminator_losses, generator_losses, num_epochs, path):
    """Plot Losses After Training"""
    plt.figure(figsize=(10, 5))
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("Face Generation Losses over Epoch {}".format(num_epochs))
    plt.plot(discriminator_losses, label='Discriminator', alpha=0.5)
    plt.plot(generator_losses, label='Generator', alpha=0.5)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(os.path.join(path, 'Face_Generation_Losses_Epoch_{}.png'.format(num_epochs)))


def make_gifs_train(title, path):
    """Create a GIF file After Train"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_Epoch_%03d.png' % (title, i+1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_Train_Results.gif'.format(title), generated_images, fps=2)
    print("{} gif file is generated.".format(title))


def make_gifs_test(title, direction, path):
    """Create a GIF file After Inference"""
    images = os.listdir(path)
    generated_images = []

    for i in range(len(images)):
        file = os.path.join(path, '%s_%s_Results_%03d.png' % (title, direction, i + 1))
        generated_images.append(imageio.imread(file))

    imageio.mimsave(path + '{}_{}_Test_Results.gif'.format(title, direction), generated_images, fps=2)
    print("{} gif file is generated.".format(title))