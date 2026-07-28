#This script trains the WaveletCNN and WaveletFeatureCNN models.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np

from bruxism_dataset import create_train_test_split
from wavelet_dataset import WaveletFeatureDataset, WaveletCoefficientDataset
from wavelet_cnn import WaveletCNN, WaveletFeatureCNN
from training_improvements import ReducedClassDataset
from wavelet_features import get_wavelet_feature_dimension
from torch.utils.data import Dataset

class SmallerWaveletCNN(nn.Module):
    """
    Reduced-parameter version of WaveletCNN to mitigate overfitting.
    Uses fewer convolutional channels and simpler fusion layers while maintaining
    the multi-scale frequency bands.
    """
    def __init__(self, num_classes=4, input_channels=4, window_size=1200, wavelet='db4', dropout=0.6):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes for classification.
            input_channels (int): Number of input EMG channels.
            window_size (int): Length of input signal window in samples.
            wavelet (str): Wavelet type for decomposition (e.g., 'db4').
            dropout (float): Dropout probability for regularization.
        """
        super(SmallerWaveletCNN, self).__init__()
        import pywt
        import numpy as np
        
        self.input_channels = input_channels
        self.window_size = window_size
        self.wavelet = wavelet
        self.wavelet_level = 4
        
        # Reduced channel capacities to  8->16 instead of 16->32 to prevent overfitting
        self.low_freq_branch = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.mid_freq_branch = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.high_freq_branch = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Reduced fusion layer: 48 features (3 branches * 16) -> 32 -> 16
        # Original used 96 -> 64 -> 32, this reduces parameters by ~75%
        self.fusion = nn.Sequential(
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(16, num_classes)
    
    def _wavelet_decompose(self, x):
        """
        Decompose input signal into wavelet coefficients with GPU optimization.
        Performs multi-level wavelet transform, and also optimizing CPU-GPU transfers by
        using non-blocking transfers.
        
        Args:
            x (torch.Tensor): Input signal tensor of shape (batch, channels, samples).
        
        Returns:
            tuple: (approx, details) where:
                - approx (torch.Tensor): Approximation coefficients.
                - details (list): List of detail coefficient tensors.
        """
        import pywt
        import numpy as np
        batch_size, n_channels, n_samples = x.shape
        device = x.device
        
        # Ensure contiguous memory layout for faster CPU-GPU transfers
        if not x.is_contiguous():
            x = x.contiguous()
        
        # PyWavelets requires numpy arrays, so we detach the tensors and move them to CPU temporarily
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
        
        approx_list = []
        details_list = [[] for _ in range(self.wavelet_level)]
        
        # Decompose each channel independently
        for b in range(batch_size):
            for c in range(n_channels):
                sig = x_np[b, c, :]
                coeffs = pywt.wavedec(sig, self.wavelet, level=self.wavelet_level)
                approx_list.append(coeffs[0])
                for i, cD in enumerate(coeffs[1:]):
                    details_list[i].append(cD)
        
        # Reconstruct tensors on target device using non-blocking transfer
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
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input signal tensor of shape (batch, channels, samples).
        
        Returns:
            torch.Tensor: Class logits of shape (batch, num_classes).
        """
        approx, details = self._wavelet_decompose(x)
        
        # Extract features from each frequency band
        low_feat = self.low_freq_branch(approx).squeeze(-1)
        mid_feat = self.mid_freq_branch(details[2]).squeeze(-1)
        high_feat = self.high_freq_branch(details[0]).squeeze(-1)
        
        # Combine multi-scale features
        combined = torch.cat([low_feat, mid_feat, high_feat], dim=1)
        
        # Fuse features and classify
        features = self.fusion(combined)
        logits = self.classifier(features)
        
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance by focusing on hard examples.
    Down-weights easy examples and emphasizes difficult misclassified samples,
    making the model learn more effectively from minority classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss function.
        
        Args:
            alpha (torch.Tensor, optional): Class weights for balancing.
            gamma (float): Focusing parameter - higher values increase emphasis on hard examples.
            reduction (str): Reduction method - 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth class indices.
        Returns:
            torch.Tensor: Computed focal loss value.
        """
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


class AugmentedWaveletDataset(Dataset):
    """
    Dataset wrapper that applies data augmentation to rebalance class distribution.
    Identifies minority classes and applies random augmentations (scaling, noise, time shifts)
    only to underrepresented classes to class representation.
    """
    def __init__(self, base_dataset, augment_prob=0.5, augment_minority_only=True):
       
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob
        self.augment_minority_only = augment_minority_only
        
        # Determine number of classes from base dataset
        if hasattr(base_dataset, 'num_classes'):
            self.num_classes = base_dataset.num_classes
        else:
            from collections import Counter
            labels = []
            for i in range(len(self.base_dataset)):
                sample = self.base_dataset[i]
                labels.append(sample[2] if len(sample) == 3 else sample[1])
            self.num_classes = len(set(labels))
        
        # Identify minority classes (those with < 70% of maximum class cardinality)
        from collections import Counter
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
        
        #Get class names from base dataset or generate some generic names.
        
        if hasattr(self.base_dataset, 'get_class_names'):
            return self.base_dataset.get_class_names()
        else:
            return [f'class_{i}' for i in range(self.num_classes)]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _augment_emg(self, emg):
        """
        Apply random augmentations to EMG signal to increase data diversity and further prevent overfitting.
        random scaling (0.9-1.1x), additive Gaussian noise,
        and circular time shifts. Applied probabilistically to preserve signal characteristics.
        
        Args:
            torch.Tensor: raw EMG signal tensor.
        
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
    
    def __getitem__(self, idx):

        sample = self.base_dataset[idx]
        
        if len(sample) == 3:
            emg, mic, label = sample
            # Apply augmentation only to minority classes if flag is set
            if not self.augment_minority_only or label in self.minority_classes:
                emg = self._augment_emg(emg)
            return emg, mic, label
        else:
            features, label = sample
            return features, label


class ReducedWaveletCoefficientDataset(Dataset):
    """
    Dataset wrapper that maps original 12-class labels to reduced class sets.
    Combines semantically similar classes (e.g., different clench types) into
    single categories to reduce complexity and improve generalization on small datasets.
    """
    def __init__(self, base_dataset, reduction_strategy='4_classes'):
        """
        Initialize reduced class dataset.
        Args:
            base_dataset (Dataset): Base dataset with original 12-class labels.
            reduction_strategy (str): Reduction strategy - '4_classes', '5_classes', or '6_classes'.
        """
        from training_improvements import CLASS_REDUCTION_STRATEGIES, ORIGINAL_CLASSES
        
        self.base_dataset = base_dataset
        self.reduction_map = CLASS_REDUCTION_STRATEGIES[reduction_strategy]
        self.reverse_map = {v: k for k, v in self.reduction_map.items()}
        self.num_classes = len(self.reduction_map)
        """
        We basically build a mapping from original class indices to reduced class indices
        by grouping semantically similar activities into single classes
        - Movement: jaw movements (open/close, deviation, protrusion)
        - Clenching: various clench types (bite, molar, incisor)
        - Bruxing: natural bruxism episodes
        - Chewing: food-related activities (cheese, carrots, gum)
        - Rest: inactive state (may be excluded in 4-class strategy)
        """
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
        """
        Get sample with reduced class label.
        
        Args:
            idx (int): Sample index in reduced dataset.
        
        Returns:
            tuple: (emg, mic, reduced_label) where label is mapped to reduced class set.
        """
        actual_idx = self.valid_indices[idx]
        emg, mic, original_label = self.base_dataset[actual_idx]
        reduced_label = self.original_to_reduced[original_label]
        return emg, mic, reduced_label
    
    def get_class_names(self):
        """
        Get names of reduced classes.
        
        Returns:
            list: List of reduced class name strings.
        """
        return [self.reverse_map[i] for i in range(self.num_classes)]


def main():
    """
    Main training pipeline for wavelet-based bruxism detection models.
    
    Supports two approaches:
    1. 'coefficients': Uses raw wavelet coefficients with WaveletCNN
    2. 'features': Uses pre-extracted wavelet features with WaveletFeatureCNN
    
    Applies class reduction, data augmentation, and handles class imbalance
    through focal loss or weighted cross-entropy.
    """
    print("="*70)
    print("WAVELET-BASED TRAINING PIPELINE")
    print("="*70)
    print("\nThis approach uses wavelet transforms for multi-resolution analysis.")
    print("Wavelets are particularly well-suited for EMG signals and bruxism detection.\n")
    
    data_root = r"C:\Users\mhfar\Desktop\Depo\Brusxism_data"
    
    print("1. Loading base datasets...")
    train_dataset, test_dataset = create_train_test_split(
        data_root=data_root,
        train_subjects=[1, 2, 3, 4],
        test_subjects=[5],
        window_size=1200,
        stride=600,
        normalize=False
    )
    
    print(f"\nRaw train dataset: {len(train_dataset)} samples")
    print(f"Raw test dataset: {len(test_dataset)} samples")
    
    # Reduce from 12 classes to 4 by removing 'rest' (contaminated/poor quality)
    # and grouping semantically similar classes
    reduction_strategy = '4_classes'
    print(f"\nUsing reduction strategy: {reduction_strategy}")
    print("Note: REMOVING 'rest' class - it's contaminated and causing confusion.")
    print("      Focusing on active states only: movement, clenching, bruxing, chewing.\n")
    
    # Choose processing approach
    approach = 'coefficients'
    
    if approach == 'coefficients':
        print("\n2. Preparing wavelet coefficient datasets (for WaveletCNN)...")
        train_wavelet_base = WaveletCoefficientDataset(
            train_dataset,
            apply_preprocessing=True,
            use_ica=False,
            wavelet='db4',
            level=4
        )
        
        # Compute normalization statistics from training set only
        # to prevent data leakage from test set
        norm_stats = train_wavelet_base.get_normalization_stats()
        
        # Apply training statistics to test set for consistent preprocessing
        test_wavelet_base = WaveletCoefficientDataset(
            test_dataset,
            apply_preprocessing=True,
            use_ica=False,
            wavelet='db4',
            level=4,
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
        
        # Analyze class distribution before augmentation to see true data balance
        print("\n2d. Analyzing class distribution...")
        from collections import Counter
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
                print("   Note: SMOTE is only available for feature-based approach (not coefficient-based).")
        
        print("\n3. Creating WaveletCNN model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Use reduced-parameter model to prevent overfitting on small dataset
        use_smaller_model = True
        if use_smaller_model:
            model = SmallerWaveletCNN(
                num_classes=num_classes,
                input_channels=4,
                window_size=1200,
                wavelet='db4',
                dropout=0.6
            )
            print("   Using SmallerWaveletCNN (reduced parameters to prevent overfitting)")
        else:
            model = WaveletCNN(
                num_classes=num_classes,
                input_channels=4,
                window_size=1200,
                wavelet='db4',
                use_learnable_wavelet=False,
                dropout=0.6
            )
            print("   Using standard WaveletCNN")
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Compute inverse-frequency class weights to balance training
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=np.array(train_labels)
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        batch_size = 16
        
    else:  # features approach
        print("\n2. Extracting wavelet features (for WaveletFeatureCNN)...")
        train_wavelet_base = WaveletFeatureDataset(
            train_dataset,
            apply_preprocessing=True,
            use_ica=False,
            wavelet='db4',
            max_level=5
        )
        
        # Fit scaler on training set only to prevent data leakage
        scaler = train_wavelet_base.get_scaler()
        
        # Apply training scaler to test set for consistent feature scaling
        test_wavelet_base = WaveletFeatureDataset(
            test_dataset,
            apply_preprocessing=True,
            use_ica=False,
            wavelet='db4',
            max_level=5,
            scaler=scaler
        )
        
        print("\n2b. Applying class reduction...")
        train_wavelet_reduced = ReducedClassDataset(train_wavelet_base, reduction_strategy=reduction_strategy)
        test_wavelet = ReducedClassDataset(test_wavelet_base, reduction_strategy=reduction_strategy)
        num_classes = train_wavelet_reduced.num_classes
        class_names = train_wavelet_reduced.get_class_names()
        
        # Add data augmentation to balance classes
        print("\n2c. Adding data augmentation for class balancing...")
        train_wavelet = AugmentedWaveletDataset(train_wavelet_reduced, augment_prob=0.4, augment_minority_only=True)
        
        # Analyze class distribution (use reduced dataset before augmentation for accurate counts)
        print("\n2d. Analyzing class distribution...")
        from collections import Counter
        train_labels = [train_wavelet_reduced[i][1] for i in range(len(train_wavelet_reduced))]
        test_labels = [test_wavelet[i][1] for i in range(len(test_wavelet))]
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
        
        # Check for imbalance
        train_class_counts = [train_counts.get(i, 0) for i in range(num_classes)]
        if len(train_class_counts) > 0:
            max_count = max(train_class_counts)
            min_count = min(train_class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
            if imbalance_ratio > 3.0:
                print("⚠️  Significant class imbalance detected! Using class weights in loss function.")
        
        print("\n3. Creating WaveletFeatureCNN model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        feature_dim = train_wavelet_base.features.shape[1]
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=np.array(train_labels)
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        # Use reduced-capacity model to prevent overfitting
        model = WaveletFeatureCNN(
            num_classes=num_classes,
            wavelet_feature_dim=feature_dim,
            hidden_dim=32,
            dropout=0.6
        )
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        batch_size = 64
    
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
    if use_focal_loss and 'class_weight_dict' in locals() and class_weight_dict:
        weight_tensor = torch.tensor([class_weight_dict[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=weight_tensor, gamma=1.5)
        print("   Using Focal Loss with class weights (gamma=1.5)")
    elif 'class_weight_dict' in locals() and class_weight_dict:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)
    
    print("\n6. Training (with early stopping)...")
    
    # Training configuration
    num_epochs = 5
    min_epochs = 2
    patience = 30
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
            if approach == 'coefficients':
                emg_batch, mic_batch, labels_batch = batch
                # Convert (batch, time, channels) -> (batch, channels, time) for Conv1d
                input_batch = emg_batch.transpose(1, 2).to(device, non_blocking=True)
            else:
                features_batch, labels_batch = batch
                input_batch = features_batch.to(device, non_blocking=True)
            
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(input_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            # Clip gradients to prevent instability
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
                if approach == 'coefficients':
                    emg_batch, mic_batch, labels_batch = batch
                    input_batch = emg_batch.transpose(1, 2).to(device, non_blocking=True)
                else:
                    features_batch, labels_batch = batch
                    input_batch = features_batch.to(device, non_blocking=True)
                
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                outputs = model(input_batch)
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
            torch.save(model.state_dict(), 'best_model_early_stopping.pth')
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
    
    print("\n8. Plotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Wavelet-Based)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy (Wavelet-Based)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelet_training_results.png', dpi=150, bbox_inches='tight')
    print("Saved plot to 'wavelet_training_results.png'")
    plt.show()
    
    print("\n9. Evaluating on test set...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(torch.load('best_model_early_stopping.pth'))
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
            for batch in test_loader:
                if approach == 'coefficients':
                    emg_batch, mic_batch, labels_batch = batch
                    input_batch = emg_batch.transpose(1, 2).to(device, non_blocking=True)
                else:
                    features_batch, labels_batch = batch
                    input_batch = features_batch.to(device, non_blocking=True)
                
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                outputs = model(input_batch)
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
    
    # Use reduced class names
    condition_names = class_names
    
    print("\nPer-Class Performance:")
    print(classification_report(all_labels, all_predictions, target_names=condition_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Wavelet-Based Model ({num_classes} classes)')
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
    plt.savefig(f'confusion_matrix_wavelet_{num_classes}classes.png', dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to 'confusion_matrix_wavelet_{num_classes}classes.png'")
    plt.show()
    

if __name__ == "__main__":
    main()

