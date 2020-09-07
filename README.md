# Face-Generation

- [ ] Add results
- [ ] Add link for celeba dataset and pre-trained files
- [ ] BEGAN
- [ ] SAGAN
- [ ] and so on.


### 0. Introduction

This repository contains implementation of various GANs using CelebA dataset, including DCGAN, LSGAN, Improved Techniques for Training GANs, WGAN-GP and such.

### 1. Dataset
Please check if the directory corresponds to below:
```
+---[data]
|   \---[celeba]
|       \---[img_align_celeba]
|           +---[000001.jpg]
|           |...
|           \---[202599.jpg]
+---celeba.py
+---config.py
|   ...
+---walking in the latent space.ipynb
```

### 2. DCGAN (Deep Convolutional GAN)

This folder includes implementations of DCGAN, LSGAN, Improved Techniques for training GAN (only smoothed labels) and WGAN-GP.

#### 1) Run DCGAN
```
python train.py
```

#### 2) Run LSGAN
```
python train.py --ls_gan True
```

#### 3) Run Improved Techniques for Training GANs
```
python train.py --ls_gan True --smoothed True
```

#### 4) Run WGAN-GP
```
python train.py --ls_gan True --wgan_gp True
```

#### 5) A list of relevant papers
- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434.pdf)
- LSGAN: [Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)
- Improved Techniques for Training GANs: [Improved Techniques for Training GANs](https://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)
- WGAN-GP: [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1611.04076.pdf)

