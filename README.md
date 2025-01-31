This repository is for educational purposes only. Any use of this code, dataset, or methodology for research, commercial, or academic purposes is strictly prohibited without prior written consent from the authors.

# Hybrid Random Forest and CNN Framework for Hyperspectral Oil-Water Classification

This repository contains the implementation, datasets (hosted on Hugging Face), and results for the paper **"A Hybrid Random Forest and CNN Framework for Tile-Wise Oil-Water Classification in Hyperspectral Images"**, authored by Mehdi Nickzamir and S. Mohammad Sheikh Ahmadi.





## Overview

Oil spills are devastating for the environment and human activities, requiring timely detection to mitigate their impacts. This project proposes a hybrid framework that combines the strengths of **Random Forest (RF)** for pixel-wise classification and **Convolutional Neural Networks (CNN)** for incorporating spatial context to enhance oil-water classification in hyperspectral images (HSI).

### Features:
- Hyperspectral Oil Spill Detection (HOSD) dataset preprocessing pipeline.
- Implementation of the hybrid RF+CNN framework.
- Comparison with baseline and state-of-the-art models.
- Pretrained models and metrics for evaluation.

## Repository Structure

### 1. **Files and Scripts**
| File/Folder                 | Description                                                                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `train.ipynb`               | Training script for the hybrid RF+CNN model.                                                                                                  |
| `test.ipynb`                | Testing script for evaluating the model.                                                                                                     |
| `preprocessing.ipynb`       | Preprocessing pipeline including noisy channel removal, normalization, PCA, tiling, and augmentation.                                         |
| `ADSP.pdf`                  | Full research paper detailing the methodology, experiments, and results.                                                                      |

### 2. **Datasets**
The datasets and models are hosted on [Hugging Face](https://huggingface.co). Links to the recourses:
- [Original HOSD Dataset](https://huggingface.co/datasets/smsag99/OIL_SPILL_HSI)
- [Preprocessed Dataset](https://huggingface.co/datasets/smsag99/OIL_SPILL_HSI_AUGMENTED)
- [Saved Models](https://huggingface.co/smsag99/OIL_SPILL_Model)

These datasets include:
1. **Original Dataset**: Raw HOSD dataset with 18 hyperspectral images.
2. **Preprocessed Dataset**: Dataset after noisy channel removal, normalization, PCA, and tiling.
