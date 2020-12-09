This folder contains the implementation of VAEGAN.

### 0. Introduction
- VAEGAN: [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf)


### 1. Results
1) Qualitative Analysis

#### Generation
<img src = './results/samples/generation/Face_Generation_Epoch_40.png'>

#### Reconstruction
<img src = './results/samples/reconstruction/Face_Generation_Epoch_40.png'>


2) Quantative Analysis

| Model | IS↑ | FID↓ |
|:-----:|:-----:|:-----:|
| VAEGAN | 1.823 ± 0.0074 | 6.314 |

3) Interpolation

<img src = './results/interpolation/Generated_Face_Interpolation.png'>

### 1. Dataset
Download the CelebA dataset from this [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
After downloading the dataset, please check if the directory corresponds to below:
```
+---[celeba]
|     \---[img_align_celeba]
|         +---[000001.jpg]
|         |...
|         +---[202599.jpg]
+---celeba.py
+---config.py
|   ...
+---walking in the latent space.ipynb
```

### 2. Run the Codes
1) Train
```
python train.py
```

2) Interpolation (so-called 'Walking in the latent space')

Place the pre-trained weights to `./results/weights` and run `walking in the latent space.ipynb`.

3) Generate faces (for inference)
```
python generate_faces.py
```

4) Inception Score
```
python inception_score.py
```

4) FID (Frechet Inception Distance) Score
```
python fid_score.py './celeba/img_align_celeba/' './results/inference/inference/'
```

### 3. Plot Loss
<img src = './results/plots/Face_Generation_Losses_Epoch_40.png'>
