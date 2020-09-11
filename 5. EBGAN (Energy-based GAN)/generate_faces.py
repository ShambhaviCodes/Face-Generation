import os

import torch
from torchvision.utils import save_image

from config import *
from models import *
from utils import make_dirs, denorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_faces():

    # Test Path #
    make_dirs(config.inference_path)

    # Prepare Generator #
    G = Generator().to(device)
    G.load_state_dict(torch.load(os.path.join(config.weights_path, 'Face_Generator_Epoch_{}.pkl'.format(config.num_epochs))))
    G.eval()

    # Start Generating Faces #
    count = 1

    while (True):

        # Prepare Fixed Noise and Generator #
        noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)
        generated = G(noise)

        for i in range(config.batch_size):
            save_image(denorm(generated[i].data),
                       os.path.join(config.inference_path, "Generated_CelebA_Faces_{}.png".format(count)),
                       )
            count += 1

        if count > config.limit:
            print("Generating fake CelebA faces is finished.")
            break


if __name__ == '__main__':
    torch.cuda.empty_cache()
    generate_faces()