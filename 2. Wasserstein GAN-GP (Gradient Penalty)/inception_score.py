import os
import numpy as np
from scipy.stats import entropy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from torchvision.datasets import ImageFolder

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inception_score(splits=10):
    """Calculate IS (Inception Score)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    generated_celeba = ImageFolder('./results/inference/', transform=transform)
    N = len(generated_celeba)
    celeba_loader = DataLoader(generated_celeba, batch_size=config.batch_size)

    # Load Inception Model #
    inception_model = inception_v3(pretrained=True, transform_input=False).type(torch.FloatTensor).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(torch.FloatTensor).to(device)

    def get_pred(x):
        x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    print("Calculation of Inception Score started...")
    for i, (image, label) in enumerate(celeba_loader, 0):
        image = image.type(torch.FloatTensor).to(device)
        batch_size_i = image.size()[0]

        preds[i*config.batch_size:i*config.batch_size + batch_size_i] = get_pred(image)

    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []

        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    score = inception_score(splits=10)
    print("Inception Score: Mean {} | Std {}".format(score[0], score[1]))
