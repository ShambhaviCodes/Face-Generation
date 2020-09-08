# Face-Generation

- [ ] Add results
- [ ] Add link for celeba dataset and pre-trained files
- [ ] BEGAN
- [ ] SAGAN

### 0. Introduction

This repository contains implementation of various GANs using CelebA dataset, including DCGAN, LSGAN, Improved Techniques for Training GANs, WGAN-GP and such.

### 1. Dataset
Download the CelebA dataset from this [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
After downloading the dataset, please check if the directory corresponds to below:
```
+---[data]
|   \---[celeba]
|       \---[img_align_celeba]
|           +---[000001.jpg]
|           |...
|           +---[202599.jpg]
+---celeba.py
+---config.py
|   ...
+---walking in the latent space.ipynb
```

### 2. Qualitative Analysis

### 3. Quantitative Analysis
| Model | IS | FID |
|:-----:|:-----:|:-----:|
| DCGAN | 2.827 ± 0.0164 | - |
| WGAN-GP | - | - |
