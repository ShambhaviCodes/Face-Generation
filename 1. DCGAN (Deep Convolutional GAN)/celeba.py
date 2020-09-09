from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from config import *


def get_celeba_loader(path, batch_size):

    transform = transforms.Compose([
        transforms.Resize((config.crop_size, config.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    celeba_dataset = ImageFolder(root=path, transform=transform)
    celeba_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return celeba_loader


if __name__ == "__main__":
    path_1 = './results/inference/'
    path_1_dataset = ImageFolder(root=path_1)
    print(len(path_1_dataset))