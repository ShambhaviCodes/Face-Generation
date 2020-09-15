# Face-Generation

### 0. Introduction

This repository contains implementation of various GANs using CelebA dataset, including DCGAN, LSGAN, Improved Techniques for Training GANs, WGAN-GP and such.

### 2. Qualitative Analysis
| DCGAN | WGAN-GP | BEGAN | SAGAN | EBGAN |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| <img src = './1. DCGAN (Deep Convolutional GAN)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './2. Wasserstein GAN-GP (Gradient Penalty)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './3. BEGAN (Boundary Equilibrium GAN)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './4. SAGAN (Self-Attention GAN)/results/samples/Face_Generation_Epoch_100.png'> | - |

### 3. Quantitative Analysis
| Model | IS↑ | FID↓ |
|:-----:|:-----:|:-----:|
| DCGAN | 2.827 ± 0.0164 | 6.600 |
| WGAN-GP | 2.735 ± 0.0141 | 5.857 |
| BEGAN | 2.362 ± 0.0131 | 9.942 |
| SAGAN | 2.094 ± 0.0174 | 6.140 |
| EBGAN | - | - |
