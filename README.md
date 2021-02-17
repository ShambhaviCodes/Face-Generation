# Face-Generation

### 0. Overview

This repository contains implementations of various GANs using CelebA dataset. The list of papers are below,
- [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
- [LSGAN](https://arxiv.org/pdf/1611.04076.pdf)
- [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
- [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf)
- [BEGAN](https://arxiv.org/pdf/1703.10717.pdf)
- [SAGAN](https://arxiv.org/pdf/1805.08318.pdf)
- [EBGAN](https://arxiv.org/pdf/1609.03126.pdf)
- [VAE-GAN](https://arxiv.org/pdf/1512.09300.pdf)

### 1. Qualitative Analysis
#### 1) Generation
| DCGAN | WGAN-GP | BEGAN | SAGAN | EBGAN | VAE-GAN |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| <img src = './1. DCGAN (Deep Convolutional GAN)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './2. Wasserstein GAN-GP (Gradient Penalty)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './3. BEGAN (Boundary Equilibrium GAN)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './4. SAGAN (Self-Attention GAN)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './5. EBGAN (Energy-based GAN)/results/samples/Face_Generation_Epoch_100.png'> | <img src = './6. VAE-GAN (Variational Autoencoder GAN)/results/samples/generation/Face_Generation_Epoch_040.png'> |

#### 2) Interpolation
| GANs | Interpolation |
|:-----:|:-----:|
| DCGAN | <img src = './1. DCGAN (Deep Convolutional GAN)/results/interpolation/Generated_Face_Interpolation.png'> |
| WGAN-GP | <img src = './2. Wasserstein GAN-GP (Gradient Penalty)/results/interpolation/Generated_Face_Interpolation.png'> |
| BEGAN | <img src = './3. BEGAN (Boundary Equilibrium GAN)/results/interpolation/Generated_Face_Interpolation.png'> |
| SAGAN | <img src = './4. SAGAN (Self-Attention GAN)/results/interpolation/Generated_Face_Interpolation.png'> |
| EBGAN | <img src = './5. EBGAN (Energy-based GAN)/results/interpolation/Generated_Face_Interpolation.png'> |
| VAE-GAN | <img src = './6. VAE-GAN (Variational Autoencoder GAN)/results/interpolation/Generated_Face_Interpolation.png'> |

### 2. Quantitative Analysis
| Model | IS↑ | FID↓ |
|:-----:|:-----:|:-----:|
| DCGAN | 2.827 ± 0.0164 | 6.600 |
| WGAN-GP | 2.735 ± 0.0141 | 5.857 |
| BEGAN | 2.362 ± 0.0131 | 9.942 |
| SAGAN | 2.094 ± 0.0174 | 6.140 |
| EBGAN | 2.499 ± 0.0186 | 4.946 |
| VAE-GAN | 1.823 ± 0.0074 | 6.314 |

### 3. Acknowledgement
Thank you for inspiration and sharing codes for metrics of IS and FID!
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Inception Score Pytorch](https://github.com/sbarratt/inception-score-pytorch)
- [Fréchet Inception Distance (FID score) in PyTorch](https://github.com/mseitzer/pytorch-fid)

### Development Environment
```
- Ubuntu 18.04 LTS
- NVIDIA GFORCE GTX 1080 ti
- CUDA 10.2
- torch 1.5.1
- torchvision 0.5.0
- etc
```
