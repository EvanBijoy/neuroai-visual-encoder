# NeuroAI Visual Encoder

Predicting human brain responses to visual stimuli using deep neural networks and vision-language models. Built on the Algonauts 2023 Challenge dataset (Natural Scenes Dataset).

## Project Overview

This project explores how well different AI architectures can predict fMRI brain activity when humans view natural images. We compare traditional CNNs, modern vision-language models (VLMs), and analyze layer-wise representations in transformers.

## Objectives

### 1. CNN-Based Encoding Models
- Extract features from CNN architectures (ResNet-50, EfficientNet, SqueezeNet)
- Train encoding models (ridge regression) to predict brain responses per ROI
- Compare which architectures best explain activity in different visual regions

### 2. Vision-Language Model Encoding
- Use modern VLMs (CLIP, BLIP-2, LLaVA, etc.) for feature extraction
- Extract both visual embeddings and text/semantic embeddings from image captions
- Build encoding models using:
  - Visual-only features
  - Semantic-only features  
  - Joint visual + semantic features
- Ablate different prompt strategies (object-centric, scene-centric, etc.)

### 3. Layer-wise Transformer Analysis
- Extract representations from multiple transformer layers (ViT, DINOv2)
- Test if layer depth corresponds to cortical hierarchy
- Perform Representational Similarity Analysis (RSA) and CKA alignment

## Dataset

**Algonauts 2023 Challenge** - fMRI responses from 8 subjects viewing natural scenes. The dataset was downloaded from the [link](https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/alreadyresponded).

| Data | Description |
|------|-------------|
| Stimuli | Natural scene images |
| fMRI | Vertex-wise responses (LH: ~19K, RH: ~20K vertices) |
| ROIs | Early visual (V1-V4), body/face/place/word-selective regions, anatomical streams |


## Methods

- **Feature Extraction**: CNN final layers, VLM image/text encoders, transformer intermediate layers
- **Encoding Models**: Ridge regression, banded ridge regression (multi-feature-space)
- **Evaluation**: Pearson correlation between predicted and actual neural responses
- **Alignment Analysis**: RSA (RDM correlation), linear CKA

## Requirements

```
torch
torchvision
transformers
numpy
scipy
scikit-learn
matplotlib
seaborn
```

## Credits

Gifford AT, Lahner B, Saba-Sadiya S, Vilas MG, Lascelles A, Oliva A, Kay K, Roig G, Cichy RM. 2023. The Algonauts Project 2023 Challenge: How the Human Brain Makes Sense of Natural Scenes. arXiv preprint, arXiv:2301.03198. DOI: https://doi.org/10.48550/arXiv.2301.03198 

Allen EJ, St-Yves G, Wu Y, Breedlove JL, Prince JS, Dowdle LT, Nau M, Caron B, Pestilli F, Charest I, Hutchinson JB, Naselaris T, Kay K. 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and computational intelligence. Nature Neuroscience, 25(1):116–126. DOI: https://doi.org/10.1038/s41593-021-00962-x 