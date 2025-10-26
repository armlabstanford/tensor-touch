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
```python
>>> import torch
>>> model = torch.hub.load('peasant98/DenseTact-Model', 'hiera', pretrained=True, map_location='cpu', trust_repo=True)
>>> model = model.cuda()
```

We have a demo to run on sample images:

We also provide steps for running the encoder, which can be found in the file!

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



