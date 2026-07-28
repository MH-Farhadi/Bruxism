# Real-time-Detection-of-Bruxism-Using-EMG-Signals

## Overview

This repository contains a machine learning pipeline for real-time detection and classification of bruxism (teeth grinding) and various jaw movement conditions using electromyography (EMG) signals and microphone (MIC) data. The project implements multiple deep learning and traditional machine learning approaches to classify EMG data from 4-channel sensors into different movement conditions, with a focus on detecting natural bruxism events.
The following flowchart illustrates the complete data processing and classification pipeline used in our best model, in the **`run_new_wavelet_training.py`** script:

![Pipeline Flowchart](flowchart.png)


The primary goal of this project is to develop an accurate and robust system for classifying jaw movements and detecting bruxism episodes from EMG signals. The system is designed to work with small datasets and can distinguish between multiple conditions including:
- Various jaw movements (opening/closing, lateral deviations, protrusion/retrusion)
- Different clenching patterns (molar, incisor, left/right bite)
- Natural bruxism events
- Chewing tasks (cheese, carrots, gum)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training, but CPU is also fully supported)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/csc505-f25/Real-time-Detection-of-Bruxism-Using-EMG-Signals>
   cd Real-time-Detection-of-Bruxism-Using-EMG-Signals
   ```

2. **Install dependencies:**

   **Option A: Using pip**
   ```bash
   pip install -r requirements.txt
   ```

   **Option B: Using conda**
   ```bash
   conda env create -f requirements.yaml
   conda activate bruxism-detection
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; import numpy; import pywt; print('All dependencies installed successfully!')"
   ```

## Repository Structure

### Core Dataset and Preprocessing

- **`bruxism_dataset.py`**: 
  - Main dataset class (`BruxismDataset`) that loads EMG and MIC data from CSV files
  - Segments signals into sliding windows with configurable window size and stride
  - Provides labeled samples for training with subject-based train/test splitting
  - Supports filtering by subjects and conditions
  - Includes `create_train_test_split()` function for easy dataset preparation

- **`preprocessing_utils.py`**: 
  - Signal preprocessing utilities for both EMG and MIC signals:
    - **Bandpass filtering** (20-450 Hz for EMG, configurable for MIC)
    - **Notch filtering** (60 Hz power line interference removal)
    - **Baseline drift removal** using high-pass filtering
    - **Independent Component Analysis (ICA)** for artifact removal (EMG only)

### Model Architectures

- **`wavelet_cnn.py`**: 
  - Inlcudes auxiliary functions for building the training loop.
  - Hybrid neural network architectures:
    - `WaveletCNN`: Processes wavelet coefficients with multi-scale CNN branches
    - `WaveletFeatureCNN`: Processes extracted wavelet features
  - Multi-resolution frequency analysis using wavelet transforms
  - Attention mechanism for feature weighting
  - Designed to be around ~15 parameters

- **`run_new_wavelet_training.py`** (Main Training Script):
  - Implements the dual-branch architecture shown in the flowchart
  - `ImprovedDualBranchWaveletCNN`: Separate processing for EMG (4 channels) and MIC (1 channel)
  - Modality-specific wavelet transforms (db4 for EMG, coif5 for MIC)
  - Three parallel CNN branches per modality (low/mid/high frequency)
  - Feature fusion MLP for final classification
  - Includes Focal Loss for handling class imbalance
  - Data augmentation for minority class balancing

### Feature Extraction Utilities

- **`wavelet_features.py`**: 
  - Functions for extracting statistical features from wavelet coefficients
  - Used by traditional ML approaches

- **`training_improvements.py`**: 
  - `FeatureBasedDataset`: Dataset class for handcrafted features (RMS, MAV, variance, waveform length, zero crossings)
  - `ReducedClassDataset`: Wrapper for class reduction strategies
  - `AugmentedWaveletDataset`: Data augmentation wrapper
  - Early stopping mechanisms

### Other Training Scripts


- **`run_wavelet_training.py`**: 
  - Trains `WaveletCNN` and `WaveletFeatureCNN` models
  - Processes Only the EMG signals with wavelet decomposition
  - an ablation study to see whether processing only the EMG signal without the Mic channels improves performance (we found including Mic increased our performance by ~15%)
  - Includes smaller variants for overfitting mitigation

- **`run_feature_based_training.py`**: 
  - Trains MLP classifiers (`SimpleFeatureClassifier`) on handcrafted statistical features
  - Well-suited for very small datasets
  - Uses RMS, MAV, variance, waveform length, and zero crossing features

- **`run_random_forest_training.py`**: 
  - Trains Random Forest classifiers on extracted features
  - Provides feature importance insights
  - Good baseline for comparison with deep learning approaches

### Dataset Validation

- **`sanity_check.py`**: 
  - Data validation and visualization tools
  - Verifies dataset quality and preprocessing effectiveness
  - Generates sample visualizations of preprocessed signals


## Usage

### Data Preparation

1. **Organize your data directory structure:**

   The dataset used to train these models contains sensitive patient information and is not included in this repository to comply with privacy regulations. Access to the de-identified data may be granted upon request, Please contact [mh.farhadi@uri.edu]. Once you have obtained the data, please organize it in the root directory as follows to ensure the scripts run correctly:
   ```
   Data/
   ├── Subject_1/
   │   ├── rest.csv
   │   ├── open_close.csv
   │   ├── natural_bruxing.csv
   │   └── ...
   ├── Subject_2/
   └── ...
   ```

2. **Update data path in training scripts:**
   - Edit the `data_root` variable in each training script (e.g., line 878 in `run_new_wavelet_training.py`)
   - Default path: `"Depo/Brusxism_data"` (update this according to your data path)


### Data Validation

Before training, you can visually validate your data to make sure it looks right:

```bash
python sanity_check.py
```

This will:
- Load and visualize sample windows
- Verify preprocessing pipeline
- Check for data quality issues
- Generate sanity check figures in `sanity_check_figures/`


### Running Training Scripts

#### 1. Recommended: Dual-Branch Wavelet CNN (EMG + MIC)

This is our best pipeline. run the training script using:

```bash
python run_new_wavelet_training.py
```

**Config Options**
- `EMG_WAVELET = 'db4'`: Wavelet type for EMG
- `EMG_WAVELET_LEVEL = 4`: Decomposition levels for EMG
- `MIC_WAVELET = 'coif5'`: Wavelet type for MIC
- `MIC_WAVELET_LEVEL = 5`: Decomposition levels for MIC
- `USE_ICA_EMG = False`: Enable ICA preprocessing for EMG
- `reduction_strategy = '4_classes'`: Class reduction strategy
- `batch_size = 16`: Training batch size
- `num_epochs = 100`: Maximum training epochs

**Output:**
- Best model saved as: `best_model_new_wavelet.pth`
- Training curves: `new_wavelet_training_results.png`
- Confusion matrix: `confusion_matrix_new_wavelet_4classes.png`

#### 2. Wavelet CNN Training (EMG only)

```bash
python run_wavelet_training.py
```

**Output:**
- Model checkpoints and training visualizations
- Confusion matrices for different class configurations

#### 3. Feature-Based MLP Training

```bash
python run_feature_based_training.py
```

**Output:**
- `feature_based_training_results.png`
- `confusion_matrix_feature_mlp_4classes.png`

#### 4. Random Forest Training

```bash
python run_random_forest_training.py
```

**Output:**
- `confusion_matrix_rf_4classes.png`
- Feature importance rankings




## Troubleshooting


1. **CUDA Out of Memory:**
   - Reduce `batch_size` in training scripts (try 8 or 4)
   - Reduce `window_size` if possible
   - Use CPU: Set `device = torch.device('cpu')` in scripts

2. **Data Path Errors:**
   - Verify `data_root` path points to your data directory
   - Ensure CSV files are in correct subject/condition folders
   - Check file naming matches expected format

3. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version >= 3.8

4. **Poor Model Performance:**
   - Check class imbalance (script will report this)
   - Try different class reduction strategies
   - Adjust learning rate or use different optimizers
   - Enable data augmentation for minority classes
   - Try different wavelet types or decomposition levels
