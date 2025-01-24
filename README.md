# Hybrid Random Forest and CNN Framework for Hyperspectral Oil-Water Classification

This repository contains the implementation, datasets, and results for the paper **"A Hybrid Random Forest and CNN Framework for Tile-Wise Oil-Water Classification in Hyperspectral Images"**, authored by Mehdi Nickzamir and S. Mohammad Sheikh Ahmadi.

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
| `saved_models/`             | Contains trained models in `.keras` format.                                                                                                   |
| `original_datasets/`        | Raw HOSD dataset used in the project.                                                                                                         |
| `preprocessed_datasets/`    | Preprocessed datasets ready for model training and evaluation.                                                                                |
| `ADSP.pdf`                  | Full research paper detailing the methodology, experiments, and results.                                                                      |

### 2. **Key Sections**
- **Data Preprocessing**: Explained in `preprocessing.ipynb`.
  - Removes noisy channels.
  - Normalizes and reduces dimensionality using PCA.
  - Splits and tiles datasets.
  - Applies augmentation to balance class distributions.
- **Model Architecture**: Explained in `train.ipynb`.
  - Random Forest for initial pixel-wise classification.
  - CNN for refining spatial context.
- **Evaluation**: Results and metrics are detailed in `test.ipynb` and the paper (`ADSP.pdf`).

## Getting Started

### Prerequisites
- Python 3.8 or later
- TensorFlow, scikit-learn, numpy, pandas, and other libraries (see `requirements.txt`).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hybrid-rf-cnn-oil-spill-classification.git
