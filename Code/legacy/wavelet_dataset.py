"""
Dataset wrappers for wavelet-based features and coefficients.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from wavelet_features import extract_wavelet_features, extract_wavelet_coefficients


class WaveletFeatureDataset(Dataset):
    """
    Dataset that extracts wavelet features from EMG signals.
    Uses global normalization to prevent overfitting.
    """
    def __init__(self, base_dataset, apply_preprocessing=True, use_ica=False, wavelet='db4', max_level=5, scaler=None):
        """
        Initialize wavelet feature dataset.
        
        Args:
            base_dataset: Base dataset to process.
            apply_preprocessing: Apply signal preprocessing (default: True).
            use_ica: Use ICA preprocessing (default: False).
            wavelet: Wavelet type (default: 'db4').
            max_level: Maximum decomposition level (default: 5).
            scaler: Pre-fitted StandardScaler. None = fit on this dataset (training).
        """
        self.base_dataset = base_dataset
        self.apply_preprocessing = apply_preprocessing
        self.use_ica = use_ica
        self.wavelet = wavelet
        self.max_level = max_level
        
        print("Extracting wavelet features from all samples...")
        self.features = []
        self.labels = []
        
        for idx in range(len(base_dataset)):
            emg, _, label = base_dataset[idx]
            emg_np = emg.numpy()
            
            if self.apply_preprocessing:
                from preprocessing_utils import preprocess_emg_signal
                emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
            
            feat = extract_wavelet_features(emg_np, wavelet=wavelet, max_level=max_level, fs=1200)
            self.features.append(feat)
            self.labels.append(label)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Normalize features using global statistics
        from sklearn.preprocessing import StandardScaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            print("   Fitted StandardScaler on training features")
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
            print("   Using provided StandardScaler from training set")
        
        print(f"Extracted {self.features.shape[1]} wavelet features per sample")
    
    def get_scaler(self):
        """
        Return fitted scaler for test set normalization.
        
        Returns:
            Fitted StandardScaler object.
        """
        return self.scaler
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get sample at index.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (features tensor, label).
        """
        return torch.from_numpy(self.features[idx]), self.labels[idx]


class WaveletCoefficientDataset(Dataset):
    """
    Dataset that provides wavelet coefficients for CNN processing.
    Uses global normalization to prevent overfitting.
    """
    def __init__(self, base_dataset, apply_preprocessing=True, use_ica=False, wavelet='db4', level=4, 
                 emg_mean=None, emg_std=None, mic_mean=None, mic_std=None):
        """
        Initialize wavelet coefficient dataset.
        
        Args:
            base_dataset: Base dataset to process.
            apply_preprocessing: Apply signal preprocessing (default: True).
            use_ica: Use ICA preprocessing (default: False).
            wavelet: Wavelet type (default: 'db4').
            level: Decomposition level (default: 4).
            emg_mean, emg_std, mic_mean, mic_std: Normalization stats from training set.
                None = compute from this dataset (training).
        """
        self.base_dataset = base_dataset
        self.apply_preprocessing = apply_preprocessing
        self.use_ica = use_ica
        self.wavelet = wavelet
        self.level = level
        
        print("Preparing wavelet coefficients...")
        self.samples = []
        
        # Collect all EMG and MIC data for global normalization
        all_emg = []
        all_mic = []
        
        for idx in range(len(base_dataset)):
            emg, mic, label = base_dataset[idx]
            emg_np = emg.numpy()
            
            if self.apply_preprocessing:
                from preprocessing_utils import preprocess_emg_signal
                emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
            
            # Store preprocessed signal (wavelet decomposition done in model)
            self.samples.append({
                'emg': emg_np.astype(np.float32),
                'mic': mic.numpy() if isinstance(mic, torch.Tensor) else mic,
                'label': label
            })
            
            all_emg.append(emg_np)
            mic_np = mic.numpy() if isinstance(mic, torch.Tensor) else mic
            all_mic.append(mic_np.flatten())
        
        # Compute or use provided normalization statistics
        if emg_mean is None or emg_std is None:
            all_emg_array = np.concatenate(all_emg, axis=0)
            self.emg_mean = np.mean(all_emg_array, axis=0, keepdims=True)
            self.emg_std = np.std(all_emg_array, axis=0, keepdims=True) + 1e-8
            print("   Computed global EMG normalization statistics from training data")
        else:
            self.emg_mean = emg_mean
            self.emg_std = emg_std
            print("   Using provided EMG normalization statistics from training set")
        
        if mic_mean is None or mic_std is None:
            all_mic_array = np.concatenate(all_mic)
            self.mic_mean = np.mean(all_mic_array)
            self.mic_std = np.std(all_mic_array) + 1e-8
            print("   Computed global MIC normalization statistics from training data")
        else:
            self.mic_mean = mic_mean
            self.mic_std = mic_std
            print("   Using provided MIC normalization statistics from training set")
        
        print(f"Prepared {len(self.samples)} samples for wavelet-CNN")
        print(f"   EMG mean shape: {self.emg_mean.shape}, EMG std shape: {self.emg_std.shape}")
        print(f"   MIC mean: {self.mic_mean:.4f}, MIC std: {self.mic_std:.4f}")
    
    def get_normalization_stats(self):
        """
        Return normalization statistics for test set normalization.
        
        Returns:
            Dict with emg_mean, emg_std, mic_mean, mic_std.
        """
        return {
            'emg_mean': self.emg_mean,
            'emg_std': self.emg_std,
            'mic_mean': self.mic_mean,
            'mic_std': self.mic_std
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get sample at index.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (emg tensor, mic tensor, label).
        """
        sample = self.samples[idx]
        
        emg = torch.from_numpy(sample['emg'])
        mic = torch.from_numpy(sample['mic']).unsqueeze(-1) if len(sample['mic'].shape) == 1 else torch.from_numpy(sample['mic'])
        label = sample['label']
        
        # Apply global normalization using statistics from all training samples
        emg_mean_tensor = torch.from_numpy(self.emg_mean).float()
        emg_std_tensor = torch.from_numpy(self.emg_std).float()
        emg = (emg - emg_mean_tensor) / emg_std_tensor
        
        mic = (mic - self.mic_mean) / self.mic_std
        
        return emg, mic, label

