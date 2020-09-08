This folder contains the implementation of three papers: DCGAN, LSGAN and Smoothed labels.

### 0. Introduction
- DCGAN: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434.pdf)
- LSGAN: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076.pdf)
- Smoothed labels: [Improved Techniques for Training GAN](https://arxiv.org/abs/1606.03498.pdf)


### 1. Results
1) Qualitative Analysis

2) Quantative Analysis

| Model | IS↑ | FID↓ |
|:-----:|:-----:|:-----:|
| DCGAN | 2.827 ± 0.0164 | 6.60 |


### 2. Run the Codes
1) Train
```
python train.py
```

The default is set to DCGAN. If you want to train different papers, you can type:
```
python train.py --ls_gan True
```
Or,
```
python train.py --smoothed True
```

2) Interpolation (so-called 'Walking in the latent space')

Prepare the pre-trained weights and run `walking in the latent space.ipynb`.

3) Generate faces (for inference)
```
python generate_faces.py
```

4) Inception Score
```
python inception_score.py
```

4) Calculate FID (Frechet Inception Distance) Score
```
python get_FIDScore.py './data/celeba/img_align_celeba/ './results/inference/inference/'
```

