"""
This is an improved version of the wavelet-based training pipeline for bruxism detection using EMG + MIC signals.

Trains a multi-branch WaveletCNN model that processes EMG and microphone signals separately
with modality-specific wavelet transforms. Uses different wavelet types and decomposition
levels optimized for each signal type to better capture bruxism-related patterns.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import pywt
from collections import Counter
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA
import warnings

from bruxism_dataset import create_train_test_split

# Suppress FastICA convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.decomposition._fastica')


# ICA Configuration
USE_ICA_EMG = False   # Apply ICA to EMG signals (helps separate muscle activity from noise)
USE_ICA_MIC = False   # Apply ICA to MIC signals - (We disabled ICA for MIC as we had only
                      # one mic channel and using time delayed version of the same signal   
                      # would not help in separating artifacts from the actual sound.  


# Wavelet Configuration - SEPARATE FOR EMG AND MIC
EMG_WAVELET = 'db4'      # Daubechies 4 - good for EMG signals
EMG_WAVELET_LEVEL = 4    # 4 levels for EMG

MIC_WAVELET = 'coif5'    
MIC_WAVELET_LEVEL = 5    

# Note: Different wavelets and levels for EMG and MIC because:
# - EMG: Higher frequency content (20-450 Hz), benefits from db4
# - MIC: Audio characteristics (50-550 Hz), benefits from coiflets (better frequency localization)
# - MIC may benefit from more decomposition levels for finer frequency resolution


def bandpass_filter(data, lowcut=20, highcut=450, fs=1200, order=4):
    """
    Apply bandpass filter to extract relevant EMG frequency range.
    
    Args:
        data (np.ndarray): Input signal data.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order.
    
    Returns:
        np.ndarray: Bandpass filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def notch_filter(data, freq=60, fs=1200, quality=30):
    """
    Remove power line interference using notch filter.
    
    Args:
        data (np.ndarray): Input signal data.
        freq (float): Frequency to notch out (typically 60 Hz).
        fs (float): Sampling frequency in Hz.
        quality (float): Filter quality factor.
    
    Returns:
        np.ndarray: Notch filtered signal.
    """
    nyquist = 0.5 * fs
    freq_norm = freq / nyquist
    b, a = iirnotch(freq_norm, quality)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def remove_baseline_drift(data, fs=1200, cutoff=5):
    """
    Remove baseline drift using high-pass filter.
    
    Args:
        data (np.ndarray): Input signal data.
        fs (float): Sampling frequency in Hz.
        cutoff (float): High-pass cutoff frequency in Hz.
    
    Returns:
        np.ndarray: Signal with baseline drift removed.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def apply_ica(data, n_components=None, random_state=42):
    """
    Apply Independent Component Analysis to separate signal sources.
    
    Args:
        data (np.ndarray): Input signal data (samples, channels).
        n_components (int, optional): Number of components to extract.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (sources, ica) where sources are separated components and ica is the fitted model.
    """
    if n_components is None:
        n_components = min(data.shape)
    
    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=3000,
        tol=0.001,
        fun='logcosh',
        whiten_solver='eigh'
    )
    sources = ica.fit_transform(data)
    return sources, ica


def reconstruct_from_ica(sources, ica, components_to_keep=None):
    """
    Reconstruct signal from ICA sources, optionally keeping only selected components.
    
    Args:
        sources (np.ndarray): ICA source components.
        ica: Fitted ICA model.
        components_to_keep (list, optional): Indices of components to keep.
    
    Returns:
        np.ndarray: Reconstructed signal.
    """
    if components_to_keep is not None:
        sources_filtered = sources.copy()
        mask = np.ones(sources.shape[1], dtype=bool)
        mask[components_to_keep] = False
        sources_filtered[:, mask] = 0
        reconstructed = ica.inverse_transform(sources_filtered)
    else:
        reconstructed = ica.inverse_transform(sources)
    return reconstructed


def preprocess_emg_signal(data, fs=1200, apply_ica_flag=True):
    """
    Preprocess EMG signal with filtering and optional ICA.
    
    Applies the following steps defined above: notch filter (60 Hz), bandpass filter (20-450 Hz), baseline removal,
    and optionally ICA for source separation.
    
    Args:
        data (np.ndarray): Raw EMG signal data.
        fs (float): Sampling frequency in Hz.
        apply_ica_flag (bool): Whether to apply ICA preprocessing.
    
    Returns:
        np.ndarray: Preprocessed EMG signal.
    """
    data_filtered = notch_filter(data, freq=60, fs=fs)
    data_filtered = bandpass_filter(data_filtered, lowcut=20, highcut=450, fs=fs)
    data_filtered = remove_baseline_drift(data_filtered, fs=fs, cutoff=5)
    
    if apply_ica_flag and data.shape[1] > 1:
        sources, ica = apply_ica(data_filtered)
        data_filtered = reconstruct_from_ica(sources, ica)
    
    return data_filtered


def preprocess_mic_signal(data, fs=1200):
    """
    Minimal preprocessing for microphone signal.
    Only removes DC offset (mean subtraction). No filters applied.
    Normalization is handled separately in the dataset class.
    
    Args:
        data: 1D microphone signal
        fs: Sampling frequency (not used, kept for compatibility)
    """
    # Ensure data is 1D
    if len(data.shape) > 1:
        data = data.flatten()
    
    # Remove DC offset only (subtract mean)
    data_filtered = data - np.mean(data)
    
    return data_filtered


CLASS_REDUCTION_STRATEGIES = {
    '4_classes': {
        'movement': 0,  # open_close, deviation_left_right, protrusion_retrusion
        'clenching': 1,  # bite_left, bite_right, molar_clench, incisor_clench
        'bruxing': 2,    # natural_bruxing
        'chewing': 3,    # cheese, carrots, gum
    },
    '5_classes': {
        'rest': 0,
        'movement': 1,  # open_close, deviation_left_right, protrusion_retrusion
        'clenching': 2,  # bite_left, bite_right, molar_clench, incisor_clench
        'bruxing': 3,    # natural_bruxing
        'chewing': 4,    # cheese, carrots, gum
    },
    '6_classes': {
        'rest': 0,
        'movement': 1,  # open_close, deviation_left_right, protrusion_retrusion
        'biting': 2,     # bite_left, bite_right
        'clenching': 3,  # molar_clench, incisor_clench
        'bruxing': 4,    # natural_bruxing
        'chewing': 5,    # cheese, carrots, gum
    },
}

ORIGINAL_CLASSES = {
    'rest': 0,
    'open_close': 1,
    'deviation_left_right': 2,
    'protrusion_retrusion': 3,
    'bite_left': 4,
    'bite_right': 5,
    'molar_clench': 6,
    'incisor_clench': 7,
    'natural_bruxing': 8,
    'cheese': 9,
    'carrots': 10,
    'gum': 11
}


class ImprovedDualBranchWaveletCNN(nn.Module):
    """
    Separate processing for EMG and mic signals.
    Each branch uses its own wavelet decomposition method with modality-specific parameters.
    Features are then fused for final classification.
    """
    def __init__(
        self,
        num_classes=4,
        emg_channels=4,
        window_size=1200,
        emg_wavelet='db4',
        emg_wavelet_level=4,
        mic_wavelet='coif2',
        mic_wavelet_level=5,
        dropout=0.6
    ):
        super(ImprovedDualBranchWaveletCNN, self).__init__()
        
        self.emg_channels = emg_channels
        self.window_size = window_size
        self.emg_wavelet = emg_wavelet
        self.emg_wavelet_level = emg_wavelet_level
        self.mic_wavelet = mic_wavelet
        self.mic_wavelet_level = mic_wavelet_level
        
        # === EMG Branch (4 channels) ===
        self.emg_low_freq_branch = nn.Sequential(
            nn.Conv1d(emg_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.emg_mid_freq_branch = nn.Sequential(
            nn.Conv1d(emg_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.emg_high_freq_branch = nn.Sequential(
            nn.Conv1d(emg_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # === MIC Branch (1 channel) - optimized for audio signals ===
        self.mic_low_freq_branch = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.mic_mid_freq_branch = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.mic_high_freq_branch = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # === Feature Fusion ===
        # EMG: 3 branches * 16 = 48 features
        # MIC: 3 branches * 8 = 24 features
        # Total: 72 features
        self.fusion = nn.Sequential(
            nn.Linear(72, 48),  # 48 + 24 = 72
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(32, num_classes)
    
    def _wavelet_decompose_emg(self, x):
        """
        Perform wavelet decomposition on EMG signal using EMG-specific parameters.
        
        Args:
            x: EMG signal tensor, shape (batch, channels, time)
        
        Returns:
            approx: Approximation coefficients (low freq)
            details: List of detail coefficients at each level
        """
        batch_size, n_channels, n_samples = x.shape
        device = x.device
        
        # Optimize: Move to CPU only for wavelet transform
        if not x.is_contiguous():
            x = x.contiguous()
        
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
        
        approx_list = []
        details_list = [[] for _ in range(self.emg_wavelet_level)]
        
        for b in range(batch_size):
            for c in range(n_channels):
                sig = x_np[b, c, :]
                coeffs = pywt.wavedec(sig, self.emg_wavelet, level=self.emg_wavelet_level)
                approx_list.append(coeffs[0])
                for i, cD in enumerate(coeffs[1:]):
                    details_list[i].append(cD)
        
        approx = torch.from_numpy(
            np.array(approx_list).reshape(batch_size, n_channels, -1)
        ).to(device=device, dtype=x.dtype, non_blocking=True)
        
        details = []
        for d_list in details_list:
            d_tensor = torch.from_numpy(
                np.array(d_list).reshape(batch_size, n_channels, -1)
            ).to(device=device, dtype=x.dtype, non_blocking=True)
            details.append(d_tensor)
        
        return approx, details
    
    def _wavelet_decompose_mic(self, x):
        """
        Perform wavelet decomposition on MIC signal using MIC-specific parameters.
        Uses different wavelet and level optimized for audio characteristics.
        
        Args:
            x: MIC signal tensor, shape (batch, 1, time)
        
        Returns:
            approx: Approximation coefficients (low freq)
            details: List of detail coefficients at each level
        """
        batch_size, n_channels, n_samples = x.shape
        device = x.device
        
        # Optimize: Move to CPU only for wavelet transform
        if not x.is_contiguous():
            x = x.contiguous()
        
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
        
        approx_list = []
        details_list = [[] for _ in range(self.mic_wavelet_level)]
        
        for b in range(batch_size):
            for c in range(n_channels):
                sig = x_np[b, c, :]
                coeffs = pywt.wavedec(sig, self.mic_wavelet, level=self.mic_wavelet_level)
                approx_list.append(coeffs[0])
                for i, cD in enumerate(coeffs[1:]):
                    details_list[i].append(cD)
        
        approx = torch.from_numpy(
            np.array(approx_list).reshape(batch_size, n_channels, -1)
        ).to(device=device, dtype=x.dtype, non_blocking=True)
        
        details = []
        for d_list in details_list:
            d_tensor = torch.from_numpy(
                np.array(d_list).reshape(batch_size, n_channels, -1)
            ).to(device=device, dtype=x.dtype, non_blocking=True)
            details.append(d_tensor)
        
        return approx, details
    
    def forward(self, emg, mic):
        # Ensure mic is in the correct shape (batch, 1, time)
        if mic.dim() == 2:
            mic = mic.unsqueeze(1)
        
        # === Process EMG with EMG-specific wavelet decomposition ===
        emg_approx, emg_details = self._wavelet_decompose_emg(emg)
        # Select appropriate detail levels for EMG
        emg_detail_idx_mid = min(2, len(emg_details) - 1)  # Use level 2 or last available
        emg_detail_idx_high = 0  # Use level 0 (highest frequency)
        
        emg_low = self.emg_low_freq_branch(emg_approx).squeeze(-1)  # (batch, 16)
        emg_mid = self.emg_mid_freq_branch(emg_details[emg_detail_idx_mid]).squeeze(-1)  # (batch, 16)
        emg_high = self.emg_high_freq_branch(emg_details[emg_detail_idx_high]).squeeze(-1)  # (batch, 16)
        emg_features = torch.cat([emg_low, emg_mid, emg_high], dim=1)  # (batch, 48)
        
        # === Process MIC with MIC-specific wavelet decomposition ===
        mic_approx, mic_details = self._wavelet_decompose_mic(mic)
        # Select appropriate detail levels for MIC (may have more levels)
        mic_detail_idx_mid = min(3, len(mic_details) - 1)  # Use level 3 or last available
        mic_detail_idx_high = 0  # Use level 0 (highest frequency)
        
        mic_low = self.mic_low_freq_branch(mic_approx).squeeze(-1)  # (batch, 8)
        mic_mid = self.mic_mid_freq_branch(mic_details[mic_detail_idx_mid]).squeeze(-1)  # (batch, 8)
        mic_high = self.mic_high_freq_branch(mic_details[mic_detail_idx_high]).squeeze(-1)  # (batch, 8)
        mic_features = torch.cat([mic_low, mic_mid, mic_high], dim=1)  # (batch, 24)
        
        # === Fuse features ===
        combined = torch.cat([emg_features, mic_features], dim=1)  # (batch, 72)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance by focusing on hard examples.
    Down-weights easy examples and emphasizes difficult misclassified samples,
    making the model learn more effectively from minority classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Compute base cross-entropy loss per sample
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate probability of true class (higher = easier example)
        pt = torch.exp(-ce_loss)
        
        # Apply focusing term: (1 - pt)^gamma down-weights easy examples
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedWaveletCoefficientDataset(Dataset):
    """
    Dataset that provides wavelet coefficients for dual-branch CNN processing.
    Handles both EMG and mic signals with separate preprocessing and normalization.
    Supports separate wavelet parameters for EMG and MIC.
    """
    def __init__(
        self,
        base_dataset,
        apply_preprocessing=True,
        use_ica=False,
        emg_wavelet='db4',
        emg_wavelet_level=4,
        mic_wavelet='coif2',
        mic_wavelet_level=5,
        emg_mean=None,
        emg_std=None,
        mic_mean=None,
        mic_std=None
    ):
        """
        Args:
            base_dataset: Base dataset to process
            apply_preprocessing: Whether to apply signal preprocessing
            use_ica: Whether to use ICA preprocessing for EMG (MIC doesn't use ICA)
            emg_wavelet: Wavelet type for EMG decomposition
            emg_wavelet_level: Wavelet decomposition level for EMG
            mic_wavelet: Wavelet type for MIC decomposition
            mic_wavelet_level: Wavelet decomposition level for MIC
            emg_mean, emg_std, mic_mean, mic_std: Normalization statistics from training set.
                                                   If None, compute from this dataset (for training).
                                                   If provided, use these (for test set).
        """
        self.base_dataset = base_dataset
        self.apply_preprocessing = apply_preprocessing
        self.use_ica = use_ica  # For EMG only
        self.emg_wavelet = emg_wavelet
        self.emg_wavelet_level = emg_wavelet_level
        self.mic_wavelet = mic_wavelet
        self.mic_wavelet_level = mic_wavelet_level
        
        print("Preparing wavelet coefficients for EMG and MIC signals...")
        print(f"   EMG: {emg_wavelet}, level {emg_wavelet_level}")
        print(f"   MIC: {mic_wavelet}, level {mic_wavelet_level}")
        self.samples = []
        
        # Collect all EMG and MIC data for global normalization
        all_emg = []
        all_mic = []
        
        for idx in range(len(base_dataset)):
            emg, mic, label = base_dataset[idx]
            emg_np = emg.numpy()
            mic_np = mic.numpy() if isinstance(mic, torch.Tensor) else mic
            
            # Preprocess EMG
            if self.apply_preprocessing:
                emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
            
            # Preprocess MIC 
            if self.apply_preprocessing:
                mic_np = preprocess_mic_signal(mic_np, fs=1200)
            
            # Store preprocessed signals (wavelet decomposition done in model with separate methods)
            self.samples.append({
                'emg': emg_np.astype(np.float32),
                'mic': mic_np.astype(np.float32) if len(mic_np.shape) == 1 else mic_np.flatten().astype(np.float32),
                'label': label
            })
            
            # Collect for normalization statistics
            all_emg.append(emg_np)
            all_mic.append(mic_np.flatten())
        
        # Compute or use provided normalization statistics
        if emg_mean is None or emg_std is None:
            # Compute from this dataset (training set)
            all_emg_array = np.concatenate(all_emg, axis=0)
            self.emg_mean = np.mean(all_emg_array, axis=0, keepdims=True)
            self.emg_std = np.std(all_emg_array, axis=0, keepdims=True) + 1e-8
            print("   Computed global EMG normalization statistics from training data")
        else:
            # Use provided statistics (test set)
            self.emg_mean = emg_mean
            self.emg_std = emg_std
            print("   Using provided EMG normalization statistics from training set")
        
        if mic_mean is None or mic_std is None:
            # Compute from this dataset (training set)
            all_mic_array = np.concatenate(all_mic)
            self.mic_mean = np.mean(all_mic_array)
            self.mic_std = np.std(all_mic_array) + 1e-8
            print("   Computed global MIC normalization statistics from training data")
        else:
            # Use provided statistics (test set)
            self.mic_mean = mic_mean
            self.mic_std = mic_std
            print("   Using provided MIC normalization statistics from training set")
        
        print(f"Prepared {len(self.samples)} samples for improved dual-branch wavelet-CNN")
        print(f"   EMG mean shape: {self.emg_mean.shape}, EMG std shape: {self.emg_std.shape}")
        print(f"   MIC mean: {self.mic_mean:.4f}, MIC std: {self.mic_std:.4f}")
    
    def get_normalization_stats(self):
        """Return normalization statistics for use with test set."""
        return {
            'emg_mean': self.emg_mean,
            'emg_std': self.emg_std,
            'mic_mean': self.mic_mean,
            'mic_std': self.mic_std
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        emg = torch.from_numpy(sample['emg'])  # Shape: (time, channels) = (1200, 4)
        mic = torch.from_numpy(sample['mic'])   # Shape: (time,) = (1200,)
        label = sample['label']
        
        # Apply global normalization (using statistics computed from all training samples)
        emg_mean_tensor = torch.from_numpy(self.emg_mean).float()
        emg_std_tensor = torch.from_numpy(self.emg_std).float()
        emg = (emg - emg_mean_tensor) / emg_std_tensor
        
        # Normalize mic (it's 1D, so scalar normalization)
        mic = (mic - self.mic_mean) / self.mic_std
        
        # Return mic as 1D tensor (time,) - will be reshaped in training loop
        # EMG is (time, channels) - will be transposed in training loop
        return emg, mic, label


class ReducedWaveletCoefficientDataset(Dataset):
    """
    Dataset wrapper that maps original 12-class labels to reduced class sets.
    
    Combines semantically similar classes (e.g., different clench types) into
    single categories to reduce complexity and improve generalization on small datasets.
    """
    def __init__(self, base_dataset, reduction_strategy='4_classes'):
        self.base_dataset = base_dataset
        self.reduction_map = CLASS_REDUCTION_STRATEGIES[reduction_strategy]
        self.reverse_map = {v: k for k, v in self.reduction_map.items()}
        self.num_classes = len(self.reduction_map)
        
        # Build mapping from original class indices to reduced class indices
        # Strategy: Group semantically similar activities into single classes
        self.original_to_reduced = {}
        
        if reduction_strategy == '4_classes':
            for orig_name, orig_idx in ORIGINAL_CLASSES.items():
                if orig_name == 'rest':
                    continue
                elif orig_name in ['open_close', 'deviation_left_right', 'protrusion_retrusion']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['movement']
                elif orig_name in ['bite_left', 'bite_right', 'molar_clench', 'incisor_clench']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['clenching']
                elif orig_name == 'natural_bruxing':
                    self.original_to_reduced[orig_idx] = self.reduction_map['bruxing']
                else:  # cheese, carrots, gum
                    self.original_to_reduced[orig_idx] = self.reduction_map['chewing']
        elif reduction_strategy == '5_classes':
            for orig_name, orig_idx in ORIGINAL_CLASSES.items():
                if orig_name == 'rest':
                    self.original_to_reduced[orig_idx] = self.reduction_map['rest']
                elif orig_name in ['open_close', 'deviation_left_right', 'protrusion_retrusion']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['movement']
                elif orig_name in ['bite_left', 'bite_right', 'molar_clench', 'incisor_clench']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['clenching']
                elif orig_name == 'natural_bruxing':
                    self.original_to_reduced[orig_idx] = self.reduction_map['bruxing']
                else:  # cheese, carrots, gum
                    self.original_to_reduced[orig_idx] = self.reduction_map['chewing']
        elif reduction_strategy == '6_classes':
            for orig_name, orig_idx in ORIGINAL_CLASSES.items():
                if orig_name == 'rest':
                    self.original_to_reduced[orig_idx] = self.reduction_map['rest']
                elif orig_name in ['open_close', 'deviation_left_right', 'protrusion_retrusion']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['movement']
                elif orig_name in ['bite_left', 'bite_right']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['biting']
                elif orig_name in ['molar_clench', 'incisor_clench']:
                    self.original_to_reduced[orig_idx] = self.reduction_map['clenching']
                elif orig_name == 'natural_bruxing':
                    self.original_to_reduced[orig_idx] = self.reduction_map['bruxing']
                else:  # cheese, carrots, gum
                    self.original_to_reduced[orig_idx] = self.reduction_map['chewing']
        
        # Filter samples: keep only those with valid class mappings
        # (e.g., remove 'rest' class if it's excluded from reduction strategy)
        self.valid_indices = []
        for idx in range(len(self.base_dataset)):
            _, _, original_label = self.base_dataset[idx]
            if original_label in self.original_to_reduced:
                self.valid_indices.append(idx)
        
        print(f"\nClass Reduction Mapping ({reduction_strategy}):")
        print(f"Original 12 classes -> Reduced {self.num_classes} classes")
        if 'rest' in ORIGINAL_CLASSES and ORIGINAL_CLASSES['rest'] not in self.original_to_reduced:
            print(f"  [REST CLASS REMOVED - filtered out {len(self.base_dataset) - len(self.valid_indices)} samples]")
        for reduced_name, reduced_idx in sorted(self.reduction_map.items(), key=lambda x: x[1]):
            original_classes = [name for name, orig_idx in ORIGINAL_CLASSES.items() 
                              if orig_idx in self.original_to_reduced and self.original_to_reduced[orig_idx] == reduced_idx]
            print(f"  {reduced_idx}: {reduced_name:15s} <- {', '.join(original_classes)}")
        
        print(f"Valid samples: {len(self.valid_indices)}/{len(self.base_dataset)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        emg, mic, original_label = self.base_dataset[actual_idx]
        reduced_label = self.original_to_reduced[original_label]
        return emg, mic, reduced_label
    
    def get_class_names(self):
        return [self.reverse_map[i] for i in range(self.num_classes)]


class AugmentedWaveletDataset(Dataset):
    """
    Dataset wrapper that applies data augmentation to balance class distribution.
    
    Identifies minority classes and applies random augmentations (scaling, noise, time shifts)
    only to underrepresented classes to improve training balance.
    """
    def __init__(self, base_dataset, augment_prob=0.5, augment_minority_only=True):
        
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob
        self.augment_minority_only = augment_minority_only
        
        # Determine number of classes from base dataset
        if hasattr(base_dataset, 'num_classes'):
            self.num_classes = base_dataset.num_classes
        else:
            labels = []
            for i in range(len(self.base_dataset)):
                sample = self.base_dataset[i]
                labels.append(sample[2] if len(sample) == 3 else sample[1])
            self.num_classes = len(set(labels))
        
        # Identify minority classes (those with < 70% of maximum class count)
        labels = []
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            labels.append(sample[2] if len(sample) == 3 else sample[1])
        
        self.class_counts = Counter(labels)
        if len(self.class_counts) > 0:
            self.max_count = max(self.class_counts.values())
            self.minority_classes = {
                class_idx for class_idx, count in self.class_counts.items()
                if count < 0.7 * self.max_count
            }
            print(f"Augmentation: Minority classes (will be augmented): {self.minority_classes}")
        else:
            self.minority_classes = set()
    
    def get_class_names(self):
    
        if hasattr(self.base_dataset, 'get_class_names'):
            return self.base_dataset.get_class_names()
        else:
            return [f'class_{i}' for i in range(self.num_classes)]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _augment_emg(self, emg):
        """
        Apply random augmentations to EMG signal to increase data diversity.
        
        Augmentations include: random scaling (0.9-1.1x), additive Gaussian noise,
        and circular time shifts. Applied probabilistically to preserve signal characteristics.
        
        Args:
            torch.Tensor: EMG signal tensor.
        
        Returns:
            torch.Tensor: Augmented EMG signal.
        """
        if torch.rand(1).item() > self.augment_prob:
            return emg
        
        augmented = emg.clone()
        
        # Random amplitude scaling to simulate different muscle activation levels
        if torch.rand(1).item() < 0.5:
            scale = torch.rand(1).item() * 0.2 + 0.9
            augmented = augmented * scale
        
        # Add small Gaussian noise to simulate sensor noise
        if torch.rand(1).item() < 0.3:
            noise_level = 0.02 * augmented.std()
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise
        
        # Circular time shift to simulate temporal variations
        if torch.rand(1).item() < 0.3:
            shift = torch.randint(-50, 50, (1,)).item()
            augmented = torch.roll(augmented, shift, dims=0)
        
        return augmented
    
    def _augment_mic(self, mic):
        """
        Apply random augmentations to microphone signal to increase data diversity.
        
        Uses same augmentation strategy as EMG: scaling, noise, and time shifts.
        
        Args:
            mic (torch.Tensor): Microphone signal tensor.
        
        Returns:
            torch.Tensor: Augmented microphone signal.
        """
        if torch.rand(1).item() > self.augment_prob:
            return mic
        
        augmented = mic.clone()
        
        # Random amplitude scaling
        if torch.rand(1).item() < 0.5:
            scale = torch.rand(1).item() * 0.2 + 0.9
            augmented = augmented * scale
        
        # Add small Gaussian noise
        if torch.rand(1).item() < 0.3:
            noise_level = 0.02 * augmented.std()
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise
        
        # Circular time shift
        if torch.rand(1).item() < 0.3:
            shift = torch.randint(-50, 50, (1,)).item()
            augmented = torch.roll(augmented, shift, dims=0)
        
        return augmented
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        if len(sample) == 3:
            emg, mic, label = sample
            # Augment only minority classes if flag is set
            if not self.augment_minority_only or label in self.minority_classes:
                emg = self._augment_emg(emg)
                mic = self._augment_mic(mic)
            return emg, mic, label
        else:
            features, label = sample
            return features, label


def main():
    """
    Main training pipeline for improved dual-branch wavelet CNN.
    
    Trains a model that processes EMG and microphone signals separately using
    modality-specific wavelet transforms, then fuses features for classification.
    """


    data_root = r"C:\Users\mhfar\Desktop\Depo\Brusxism_data"
    
    print("1. Loading base datasets.")
    # Skip initial seconds to avoid sensor initialization artifacts
    skip_initial_seconds = 3.0
    print(f"   Skipping first {skip_initial_seconds} seconds of each recording to avoid initialization artifacts")
    train_dataset, test_dataset = create_train_test_split(
        data_root=data_root,
        train_subjects=[1, 2, 3, 4],
        test_subjects=[5],
        window_size=1200,
        stride=600,
        normalize=False,
        skip_initial_seconds=skip_initial_seconds
    )
    
    print(f"\nRaw train dataset: {len(train_dataset)} samples")
    print(f"Raw test dataset: {len(test_dataset)} samples")
    
    # Reduce to 4 classes by removing 'rest' (contaminated/poor quality)
    reduction_strategy = '4_classes'
    print(f"\nUsing reduction strategy: {reduction_strategy}")
    
    print("\n2. Preparing improved wavelet coefficient datasets (EMG + MIC)...")
    print(f"   ICA Configuration: EMG={USE_ICA_EMG}, MIC=False (no ICA for MIC)")
    print(f"   Wavelet Configuration:")
    print(f"     EMG: {EMG_WAVELET}, level {EMG_WAVELET_LEVEL}")
    print(f"     MIC: {MIC_WAVELET}, level {MIC_WAVELET_LEVEL}")
    
    train_wavelet_base = ImprovedWaveletCoefficientDataset(
        train_dataset,
        apply_preprocessing=True,
        use_ica=USE_ICA_EMG,
        emg_wavelet=EMG_WAVELET,
        emg_wavelet_level=EMG_WAVELET_LEVEL,
        mic_wavelet=MIC_WAVELET,
        mic_wavelet_level=MIC_WAVELET_LEVEL
    )
    
    # Compute normalization statistics from training set only to prevent data leakage
    norm_stats = train_wavelet_base.get_normalization_stats()
    
    # Apply training statistics to test set for consistent preprocessing
    test_wavelet_base = ImprovedWaveletCoefficientDataset(
        test_dataset,
        apply_preprocessing=True,
        use_ica=USE_ICA_EMG,  # Use same ICA settings as training (EMG only)
        emg_wavelet=EMG_WAVELET,
        emg_wavelet_level=EMG_WAVELET_LEVEL,
        mic_wavelet=MIC_WAVELET,
        mic_wavelet_level=MIC_WAVELET_LEVEL,
        emg_mean=norm_stats['emg_mean'],
        emg_std=norm_stats['emg_std'],
        mic_mean=norm_stats['mic_mean'],
        mic_std=norm_stats['mic_std']
    )
    
    print("\n2b. Applying class reduction...")
    train_wavelet_reduced = ReducedWaveletCoefficientDataset(train_wavelet_base, reduction_strategy=reduction_strategy)
    test_wavelet = ReducedWaveletCoefficientDataset(test_wavelet_base, reduction_strategy=reduction_strategy)
    
    # Add data augmentation to balance classes
    print("\n2c. Adding data augmentation for class balancing...")
    train_wavelet = AugmentedWaveletDataset(train_wavelet_reduced, augment_prob=0.4, augment_minority_only=True)
    num_classes = train_wavelet.num_classes
    class_names = train_wavelet.get_class_names()
    
    # Analyze class distribution to assess imbalance
    print("\n2d. Analyzing class distribution...")
    train_labels = [train_wavelet_reduced[i][2] for i in range(len(train_wavelet_reduced))]
    test_labels = [test_wavelet[i][2] for i in range(len(test_wavelet))]
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print("Train set class distribution (before augmentation):")
    for class_idx in range(num_classes):
        count = train_counts.get(class_idx, 0)
        percentage = 100 * count / len(train_wavelet_reduced) if len(train_wavelet_reduced) > 0 else 0
        print(f"  {class_names[class_idx]:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    print("\nTest set class distribution:")
    for class_idx in range(num_classes):
        count = test_counts.get(class_idx, 0)
        percentage = 100 * count / len(test_wavelet) if len(test_wavelet) > 0 else 0
        print(f"  {class_names[class_idx]:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Detect class imbalance to determine if weighted loss is needed
    train_class_counts = [train_counts.get(i, 0) for i in range(num_classes)]
    if len(train_class_counts) > 0:
        max_count = max(train_class_counts)
        min_count = min(train_class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
        if imbalance_ratio > 3.0:
            print("⚠️  Significant class imbalance detected! Using class weights in loss function.")
    
    print("\n3. Creating ImprovedDualBranchWaveletCNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = ImprovedDualBranchWaveletCNN(
        num_classes=num_classes,
        emg_channels=4,
        window_size=1200,
        emg_wavelet=EMG_WAVELET,
        emg_wavelet_level=EMG_WAVELET_LEVEL,
        mic_wavelet=MIC_WAVELET,
        mic_wavelet_level=MIC_WAVELET_LEVEL,
        dropout=0.6
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Compute inverse-frequency class weights to balance training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=np.array(train_labels)
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    batch_size = 16
    
    print("\n4. Creating data loaders with GPU optimization...")
    # Enable pin_memory to accelerate CPU->GPU data transfers
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_wavelet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_wavelet,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    if pin_memory:
        print(f"   Using pin_memory=True for faster GPU transfer")
    
    print("\n5. Setting up training...")
    # Select loss function: Focal Loss handles class imbalance better than weighted CrossEntropy
    use_focal_loss = True
    if use_focal_loss and class_weight_dict:
        weight_tensor = torch.tensor([class_weight_dict[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=weight_tensor, gamma=1.5)
        print("   Using Focal Loss with class weights (gamma=1.5)")
    elif class_weight_dict:
        weight_tensor = torch.tensor([class_weight_dict[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        try:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.15)
            print("   Using CrossEntropyLoss with class weights and label_smoothing=0.15")
        except TypeError:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print("   Using CrossEntropyLoss with class weights")
    else:
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
            print("   Using CrossEntropyLoss with label_smoothing=0.15")
        except TypeError:
            criterion = nn.CrossEntropyLoss()
            print("   Using standard CrossEntropyLoss")
    
    # Conservative learning rate and weight decay for stable training on small dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00007, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)
    
    print("\n6. Training (with early stopping)...")
    
    # Training loop parameters
    num_epochs = 100  # Maximum epochs
    min_epochs = 30  # Minimum epochs to train
    patience = 30  # for early stopping
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"Starting training... (min epochs: {min_epochs}, patience: {patience})\n")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for batch in train_loader:
            emg_batch, mic_batch, labels_batch = batch
            
            # Transpose EMG: (batch, time, channels) -> (batch, channels, time)
            emg_input = emg_batch.transpose(1, 2).to(device, non_blocking=True)
            
            # Handle mic shape: should be (batch, 1, time) for Conv1d
            # mic_batch could be (batch, time, 1) or (batch, time) or (batch, 1, time)
            if mic_batch.dim() == 2:
                # (batch, time) -> (batch, 1, time)
                mic_input = mic_batch.unsqueeze(1).to(device, non_blocking=True)
            elif mic_batch.dim() == 3:
                # (batch, time, 1) -> (batch, 1, time)
                if mic_batch.shape[2] == 1:
                    mic_input = mic_batch.transpose(1, 2).to(device, non_blocking=True)
                else:
                    # Already (batch, 1, time)
                    mic_input = mic_batch.to(device, non_blocking=True)
            else:
                mic_input = mic_batch.to(device, non_blocking=True)
            
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(emg_input, mic_input)  # Pass both signals
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                emg_batch, mic_batch, labels_batch = batch
                
                # Transpose EMG: (batch, time, channels) -> (batch, channels, time)
                emg_input = emg_batch.transpose(1, 2).to(device, non_blocking=True)
                
                # Handle mic shape: should be (batch, 1, time) for Conv1d
                # mic_batch could be (batch, time, 1) or (batch, time) or (batch, 1, time)
                if mic_batch.dim() == 2:
                    # (batch, time) -> (batch, 1, time)
                    mic_input = mic_batch.unsqueeze(1).to(device, non_blocking=True)
                elif mic_batch.dim() == 3:
                    # (batch, time, 1) -> (batch, 1, time)
                    if mic_batch.shape[2] == 1:
                        mic_input = mic_batch.transpose(1, 2).to(device, non_blocking=True)
                    else:
                        # Already (batch, 1, time)
                        mic_input = mic_batch.to(device, non_blocking=True)
                else:
                    mic_input = mic_batch.to(device, non_blocking=True)
                
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                outputs = model(emg_input, mic_input)  # Pass both signals
                loss = criterion(outputs, labels_batch)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        val_loss = running_loss / len(test_loader)
        val_acc = 100 * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Save best model when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_new_wavelet.pth')
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% [SAVED]")
        else:
            patience_counter += 1
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping: wait for minimum epochs, then stop if no improvement
        if epoch + 1 >= min_epochs and patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1} (patience: {patience} epochs without improvement)")
            break
    
    print("\n7. Analyzing results...")
    best_val_acc_from_history = max(val_accs)
    best_val_epoch = val_accs.index(best_val_acc_from_history) + 1
    final_train_acc = train_accs[val_accs.index(best_val_acc_from_history)]
    gap = final_train_acc - best_val_acc_from_history
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_val_epoch})")
    print(f"Corresponding Train Accuracy: {final_train_acc:.2f}%")
    print(f"Train-Val Gap: {gap:.2f}%")
    print(f"Total Epochs Run: {len(train_losses)}")
    
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Improved Wavelet: EMG + MIC)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy (Improved Wavelet: EMG + MIC)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('new_wavelet_training_results.png', dpi=150, bbox_inches='tight')
    print("Saved plot to 'new_wavelet_training_results.png'")
    plt.show()
    
    print("\n9. Evaluating on test set...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(torch.load('best_model_new_wavelet.pth'))
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            emg_batch, mic_batch, labels_batch = batch
            
            # Transpose EMG: (batch, time, channels) -> (batch, channels, time)
            emg_input = emg_batch.transpose(1, 2).to(device, non_blocking=True)
            
            # Handle mic shape: should be (batch, 1, time) for Conv1d
            # mic_batch could be (batch, time, 1) or (batch, time) or (batch, 1, time)
            if mic_batch.dim() == 2:
                # (batch, time) -> (batch, 1, time)
                mic_input = mic_batch.unsqueeze(1).to(device, non_blocking=True)
            elif mic_batch.dim() == 3:
                # (batch, time, 1) -> (batch, 1, time)
                if mic_batch.shape[2] == 1:
                    mic_input = mic_batch.transpose(1, 2).to(device, non_blocking=True)
                else:
                    # Already (batch, 1, time)
                    mic_input = mic_batch.to(device, non_blocking=True)
            else:
                mic_input = mic_batch.to(device, non_blocking=True)
            
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            outputs = model(emg_input, mic_input)  # Pass both signals
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    test_acc = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
    
    print(f"\n{'='*70}")
    print(f"FINAL TEST METRICS")
    print(f"{'='*70}")
    print(f"Accuracy:  {test_acc:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    condition_names = class_names
    
    print("\nPer-Class Performance:")
    print(classification_report(all_labels, all_predictions, target_names=condition_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Improved Wavelet Model (EMG + MIC, {num_classes} classes)')
    plt.colorbar()
    tick_marks = np.arange(len(condition_names))
    plt.xticks(tick_marks, condition_names, rotation=45, ha='right')
    plt.yticks(tick_marks, condition_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_new_wavelet_{num_classes}classes.png', dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to 'confusion_matrix_new_wavelet_{num_classes}classes.png'")
    plt.show()
    
if __name__ == "__main__":
    main()

