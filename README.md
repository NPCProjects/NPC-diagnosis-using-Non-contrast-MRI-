Here is the updated `README.md` including the newly uploaded files:

---

# README

## Overview

This repository contains Python scripts for processing, training, and evaluating a deep learning model designed for multi-modal medical image analysis. The code focuses on leveraging MRI data for diagnosing nasopharyngeal carcinoma (NPC). The project includes various components such as data preprocessing, model training, evaluation, visualization, and loss function implementations.

### Core Files:

1. **`preprocess_improved.py`**: Preprocesses raw MRI volumes by loading, cropping, normalizing, and saving the data.
2. **`train_improved.py`**: Trains the deep learning model using the preprocessed MRI data.
3. **`test_improved.py`**: Tests the trained model on test data and computes performance metrics.
4. **`predict_roc_improved.py`**: Evaluates the model's performance through ROC curve analysis.
5. **`CAM_improved.py`**: Generates and visualizes Class Activation Maps (CAMs) to interpret model predictions.
6. **`ctrans.py`**: Defines the cross-attention mechanism and various model layers, including fusion and residual blocks.
7. **`layers.py`**: Contains helper functions and classes for constructing convolution layers with different normalization techniques.
8. **`net.py`**: Defines the overall model architecture, including the encoder-decoder structure, attention mechanisms, and the fusion module.
9. **`transformer.py`**: Implements Transformer-based self-attention and feedforward layers.
10. **`tsf.py`**: Contains classes for layer normalization, self-attention, and hyper-parameterized convolution used in model blocks.
11. **`__init__.py`**: Initializes the module and imports necessary components.
12. **`criterions.py`**: Defines various loss functions, including Dice loss, Focal loss, and softmax-weighted loss.
13. **`generate.py`**: Contains utilities for generating snapshots with highlighted regions based on output and target comparisons.
14. **`lr_scheduler.py`**: Implements learning rate schedulers with polynomial decay and utilities for temperature-based adjustments.
15. **`parser.py`**: Provides configuration parsing and management, as well as logging setup.
16. **`str2bool.py`**: Converts string inputs to boolean values, useful for argument parsing.

## Prerequisites

Before running the scripts, ensure the following dependencies are installed:

* Python 3.x
* Libraries:

  * `torch`
  * `numpy`
  * `medpy`
  * `scikit-learn`
  * `imblearn`
  * `tqdm`
  * `matplotlib`

You can install all required dependencies by running:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── preprocess_improved.py         # Preprocessing MRI data
├── train_improved.py              # Training the deep learning model
├── test_improved.py               # Testing the trained model
├── predict_roc_improved.py        # Evaluating model performance (ROC)
├── CAM_improved.py                # Class Activation Map generation
├── ctrans.py                      # Cross-attention and model layers
├── layers.py                      # Layer definitions
├── net.py                         # Model architecture
├── transformer.py                 # Transformer-based attention layers
├── tsf.py                         # Attention-based normalization and convolution
├── __init__.py                    # Module initialization
├── criterions.py                  # Loss functions (Dice, Focal, etc.)
├── generate.py                    # Snapshot generation based on output vs target comparison
├── lr_scheduler.py                # Learning rate scheduler with polynomial decay
├── parser.py                      # Configuration parsing and logging setup
├── str2bool.py                    # Converts string values to boolean
└── requirements.txt               # Python dependencies
```

## Script Details

### 1. `preprocess_improved.py`

Preprocesses MRI data by:

* **Loading**: Loads the T1c, T1, and T2 MRI sequences for each sample.
* **Cropping**: Removes background regions to focus on the area of interest.
* **Normalizing**: Performs zero-mean and unit-variance normalization on the data.
* **Saving**: Saves the processed volumes as `.npy` files.

#### Usage

```bash
python preprocess_improved.py
```

### 2. `train_improved.py`

Trains the model by:

* **Initializing**: Setting up the model, optimizer, and scheduler.
* **Training**: Running the training loop with data augmentation and checkpointing.
* **Evaluating**: Evaluating model performance during training on validation data.

#### Usage

```bash
python train_improved.py --batch_size 8 --lr 2e-4 --num_epochs 300 --savepath ./model
```

### 3. `test_improved.py`

Tests the model on the test dataset and computes performance metrics:

* **Loading** the preprocessed data.
* **Evaluating** the model predictions.
* **Computing** performance metrics like accuracy, sensitivity, and specificity.

#### Usage

```bash
python test_improved.py
```

### 4. `predict_roc_improved.py`

Generates the ROC curve and computes the Area Under Curve (AUC) for model evaluation.

#### Usage

```bash
python predict_roc_improved.py
```

### 5. `CAM_improved.py`

Generates and saves Class Activation Maps (CAMs) for model predictions, helping to interpret the regions of interest the model focuses on for diagnosis.

#### Usage

```bash
python CAM_improved.py --gpu 1
```

### 6. `ctrans.py`, `layers.py`, `net.py`, `transformer.py`, `tsf.py`

These files define the underlying model architecture, including:

* **Self-attention** mechanisms, residual blocks, and convolution layers.
* **Transformer-based** layers for processing multi-modal data.
* **Hyper-parameterized convolution** layers for advanced feature extraction and fusion.
* **Normalization layers** for stable model training.

These scripts are utilized by `train_improved.py` and `test_improved.py` to build and train the model.

### 7. `criterions.py`

Defines various loss functions including:

* **Dice loss**: Measures the overlap between predicted and true segmentations.
* **Focal loss**: A modification of cross-entropy to address class imbalance.
* **Softmax-weighted loss**: Applies weights based on the distribution of the target labels.

### 8. `generate.py`

Contains utilities for generating snapshots comparing the model's predictions and ground truth. This includes the creation of empty figures to highlight discrepancies between the predicted and true labels.

### 9. `lr_scheduler.py`

Implements learning rate scheduling with polynomial decay and functions for dynamic temperature adjustments.

### 10. `parser.py`

Handles configuration parsing, management, and logging setup for the training and evaluation scripts.

### 11. `str2bool.py`

Converts string inputs (such as `'yes'`, `'no'`) to boolean values, useful for argument parsing and configuration management.

---

## Running the Scripts

1. **Preprocessing**: First, preprocess the MRI data using `preprocess_improved.py` to convert raw NIfTI files into a format suitable for model training.
2. **Training**: Then, run `train_improved.py` to start training the model with the preprocessed data.
3. **Evaluation**: Finally, test the trained model with `test_improved.py` and analyze the performance metrics with `predict_roc_improved.py`.
4. **Visualization**: Use `CAM_improved.py` to generate Class Activation Maps to interpret the model’s decision-making process.
