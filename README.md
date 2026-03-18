# <img src="https://github.com/user-attachments/assets/56acb64f-9f7c-40df-9373-81ca7a7ed0ad" style="width: 7%;" alt="image"> TensorTouch

## Calibration of Tactile Sensors for High Resolution Stress Tensor and Deformation for Dexterous Manipulation

<img src="demo.gif" width="75%"/>

#### [Won-Kyung Do](https://wonkyungdo.github.io/website_wkdo/), [Matthew Strong](https://peasant98.github.io/), [Aiden Swann](https://aidenswann.com/), [Boshu Lei](https://scholar.google.com/citations?user=Jv88S-IAAAAJ&hl=en), and [Monroe Kennedy III](https://monroekennedy3.com/)
Stanford University | University of Pennsylvania


[![Project](https://img.shields.io/badge/Project_Page-TensorTouch-blue)](https://tensor-touch.github.io/)
[![ArXiv](https://img.shields.io/badge/Arxiv-TensorTouch-red)](https://arxiv.org/abs/2506.08291v1) 


## TO-DO (Release dates are at the latest -- we aim to release as soon as possible!)
- [X] Release model on PyTorch Hub (**July 17th, 2025**)
- [X] Release model inference (**Sept 16, 2025**)
- [ ] Release model training code (**September 30th, 2025**)
- [ ] Release the datasets (**September 30th, 2025**)
- [ ] Release the data collection code (**October 15th, 2025**)
- [ ] Release FEM simulation pipeline code (**October 15th, 2025**)


## Model Inference

Use Torch Hub to load our models! It's easy!

We have released the training code, but have yet to release the datasets, which will be released next week.


```sh
pip install torch torchvision yacs timm matplotlib
```

### Available Models

| Model | Entry Point | Params | Encoder Dim | PSNR | Description |
|---|---|---|---|---|---|
| Hiera Base | `hiera` | 56M | 768 | — | Original base model |
| Hiera Large v1 | `hiera_large_v1` | 268M | 1152 | 45.32 | Large model (aggressive augmentation) |
| Hiera Large v2 | `hiera_large_v2` | 268M | 1152 | 52.22 | Large model (best) |

### Hiera Base (original)
```python
import torch
model = torch.hub.load('peasant98/DenseTact-Model', 'hiera', pretrained=True, map_location='cpu', trust_repo=True)
model = model.cuda()
```

### Hiera Large (recommended)
```python
import torch

# Best model (epoch 43, PSNR 52.22)
model = torch.hub.load('peasant98/DenseTact-Model', 'hiera_large_v2', pretrained=True, map_location='cpu', trust_repo=True)
model = model.cuda()

# Earlier checkpoint (epoch 22, PSNR 45.32) — trained with more aggressive augmentation
model_v1 = torch.hub.load('peasant98/DenseTact-Model', 'hiera_large_v1', pretrained=True, map_location='cpu', trust_repo=True)
```

### Running Inference

```python
model.eval()

# Input: 6-channel tensor (deformed RGB + undeformed RGB), 256x256
x = torch.randn(1, 6, 256, 256).cuda()

with torch.no_grad():
    output = model(x)  # Shape: (1, 15, 256, 256)
    # 15 channels = 5 modalities x 3 directions (x, y, z)
    # Modalities: disp, cnorm, stress1, stress2, shear
```

### Using the Encoder

```python
# Extract vision token embeddings from the encoder
z, intermediates = model.encoder(x, None, return_intermediates=True)
# Hiera Base:  z shape is (1, 256, 768)  — 256 vision tokens, 768-dim
# Hiera Large: z shape is (1, 256, 1152) — 256 vision tokens, 1152-dim
```

### Demo

We have a demo to run on sample images that compares all three models:

```sh
python3 model/test_hub.py
```

## Model Lessons

We detail some lessons about training these kinds of models for some insights.

- With an optimized training paradigm for different architectures, sometimes certain architectures are just straight up better. We found that a compact hierarchical ViT "Hiera" was exceedingly better than the other models. Thanks SAM2 for the inspiration!

- Pretraining gets you very far for ViTs. ViTs become slightly overrated when you don't pretrain them compared to the tried and true Resnets of the world. The attention operation is quite expensive (On^2), warranting the use of patches (16 by 16) for ViTs. There are optimizations that can be made with attention (flash or deformable attention), but we didn't get to them.

- Don't get fancy with optimizers and learning rates; if you are trying to get your dense prediction model to work with tiny adjustments to the learning rate, you should look into things like dataset/architecture/etc.

- Sparse prediction in vision is pretty vicious compared to dense prediction. No one seems to have "won" sparse prediction, but it looks like dense prediction scales nicely with 1. a simple architecture with a simple loss function (L1), 2. good and curated data, and 3. a massive amount of that data.

- More quality, diverse data has a bigger effect than you would think for this kind of training. Models like Pi3 easily clear VGGT probably because they absolutely sent it with dynamic data.



