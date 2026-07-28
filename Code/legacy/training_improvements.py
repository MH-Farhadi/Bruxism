import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


def calculate_signal_quality(emg_data):
    """
    Calculate quality metrics for EMG signal windows.
    """
    rms = np.sqrt(np.mean(emg_data**2, axis=0))
    variance = np.var(emg_data, axis=0)
    mean_abs = np.mean(np.abs(emg_data), axis=0)
    
    avg_rms = np.mean(rms)
    avg_variance = np.mean(variance)
    avg_mean_abs = np.mean(mean_abs)
    
    return avg_rms, avg_variance, avg_mean_abs


class FilteredBruxismDataset(Dataset):
    """
    Dataset that filters out low-activity windows.
    This removes segments with minimal muscle activity that don't contain useful information.
    Uses global normalization (computed from all training samples) to prevent overfitting.
    """
    def __init__(self, base_dataset, apply_preprocessing=True, normalize=True, 
                 filter_low_activity=True, activity_threshold_percentile=20, use_ica=False,
                 emg_mean=None, emg_std=None, mic_mean=None, mic_std=None):
        """
        Args:
            base_dataset: Base dataset to process
            apply_preprocessing: Whether to apply signal preprocessing
            normalize: Whether to normalize signals
            filter_low_activity: Whether to filter low-activity windows
            activity_threshold_percentile: Percentile threshold for filtering
            use_ica: Whether to use ICA preprocessing (default False for consistency)
            emg_mean, emg_std, mic_mean, mic_std: Normalization statistics from training set.
                                                   If None, compute from this dataset (for training).
                                                   If provided, use these (for test set).
        """
        self.base_dataset = base_dataset
        self.apply_preprocessing = apply_preprocessing
        self.normalize = normalize
        self.use_ica = use_ica  # Default to False for consistency across codebase
        
        if filter_low_activity:
            print("Analyzing signal quality and filtering low-activity windows...")
            self.valid_indices = self._filter_windows(activity_threshold_percentile)
            print(f"Kept {len(self.valid_indices)}/{len(base_dataset)} windows ({100*len(self.valid_indices)/len(base_dataset):.1f}%)")
        else:
            self.valid_indices = list(range(len(base_dataset)))
        
        # Compute or use provided normalization statistics
        if normalize:
            if emg_mean is None or emg_std is None:
                # Compute from this dataset (training set)
                all_emg = []
                all_mic = []
                for idx in self.valid_indices:
                    emg, mic, _ = base_dataset[idx]
                    emg_np = emg.numpy()
                    if self.apply_preprocessing:
                        from preprocessing_utils import preprocess_emg_signal
                        emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
                    all_emg.append(emg_np)
                    mic_np = mic.numpy() if isinstance(mic, torch.Tensor) else mic
                    all_mic.append(mic_np.flatten())
                
                all_emg_array = np.concatenate(all_emg, axis=0)
                self.emg_mean = np.mean(all_emg_array, axis=0, keepdims=True)
                self.emg_std = np.std(all_emg_array, axis=0, keepdims=True) + 1e-8
                
                all_mic_array = np.concatenate(all_mic)
                self.mic_mean = np.mean(all_mic_array)
                self.mic_std = np.std(all_mic_array) + 1e-8
                print("   Computed global normalization statistics from training data")
            else:
                # Use provided statistics (test set)
                self.emg_mean = emg_mean
                self.emg_std = emg_std
                self.mic_mean = mic_mean
                self.mic_std = mic_std
                print("   Using provided normalization statistics from training set")
        else:
            self.emg_mean = None
            self.emg_std = None
            self.mic_mean = None
            self.mic_std = None
    
    def get_normalization_stats(self):
        """Return normalization statistics for use with test set."""
        if self.normalize:
            return {
                'emg_mean': self.emg_mean,
                'emg_std': self.emg_std,
                'mic_mean': self.mic_mean,
                'mic_std': self.mic_std
            }
        return None
    
    def _filter_windows(self, threshold_percentile):
        """Filter out windows with low muscle activity."""
        valid_indices = []
        quality_metrics = []
        
        for idx in range(len(self.base_dataset)):
            emg, _, label = self.base_dataset[idx]
            emg_np = emg.numpy()
            
            _, variance, _ = calculate_signal_quality(emg_np)
            quality_metrics.append(variance)
        
        threshold = np.percentile(quality_metrics, threshold_percentile)
        
        for idx, metric in enumerate(quality_metrics):
            info = self.base_dataset.get_sample_info(idx)
            condition = info['condition']
            
            if condition == 'rest':
                valid_indices.append(idx)
            elif metric > threshold:
                valid_indices.append(idx)
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        from preprocessing_utils import preprocess_emg_signal
        
        real_idx = self.valid_indices[idx]
        emg, mic, label = self.base_dataset[real_idx]
        
        emg_np = emg.numpy()
        
        if self.apply_preprocessing:
            # Use ICA preprocessing (can be disabled by setting use_ica=False)
            emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
        
        emg = torch.from_numpy(emg_np.astype(np.float32))
        
        if self.normalize and self.emg_mean is not None:
            # Apply global normalization (using statistics computed from all training samples)
            emg_mean_tensor = torch.from_numpy(self.emg_mean).float()
            emg_std_tensor = torch.from_numpy(self.emg_std).float()
            emg = (emg - emg_mean_tensor) / emg_std_tensor
            mic = (mic - self.mic_mean) / self.mic_std
        
        return emg, mic, label


class AugmentedBruxismDataset(Dataset):
    """
    Dataset with safe EMG augmentation techniques.
    Applies random scaling, time shifting, and additive noise.
    Uses global normalization (computed from all training samples) to prevent overfitting.
    """
    def __init__(self, base_dataset, apply_preprocessing=True, normalize=True, 
                 augment=True, augment_prob=0.5, use_ica=False,
                 emg_mean=None, emg_std=None, mic_mean=None, mic_std=None):
        """
        Args:
            base_dataset: Base dataset to process
            apply_preprocessing: Whether to apply signal preprocessing
            normalize: Whether to normalize signals
            augment: Whether to apply augmentation
            augment_prob: Probability of applying augmentation
            use_ica: Whether to use ICA preprocessing (default False for consistency)
            emg_mean, emg_std, mic_mean, mic_std: Normalization statistics from training set.
                                                   If None, compute from this dataset (for training).
                                                   If provided, use these (for test set).
        """
        self.base_dataset = base_dataset
        self.apply_preprocessing = apply_preprocessing
        self.normalize = normalize
        self.augment = augment
        self.augment_prob = augment_prob
        self.use_ica = use_ica  # Default to False for consistency
        
        # Compute or use provided normalization statistics
        if normalize:
            if emg_mean is None or emg_std is None:
                # Compute from this dataset (training set)
                all_emg = []
                all_mic = []
                for idx in range(len(base_dataset)):
                    emg, mic, _ = base_dataset[idx]
                    emg_np = emg.numpy()
                    if self.apply_preprocessing:
                        from preprocessing_utils import preprocess_emg_signal
                        emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
                    all_emg.append(emg_np)
                    mic_np = mic.numpy() if isinstance(mic, torch.Tensor) else mic
                    all_mic.append(mic_np.flatten())
                
                all_emg_array = np.concatenate(all_emg, axis=0)
                self.emg_mean = np.mean(all_emg_array, axis=0, keepdims=True)
                self.emg_std = np.std(all_emg_array, axis=0, keepdims=True) + 1e-8
                
                all_mic_array = np.concatenate(all_mic)
                self.mic_mean = np.mean(all_mic_array)
                self.mic_std = np.std(all_mic_array) + 1e-8
                print("   Computed global normalization statistics from training data")
            else:
                # Use provided statistics (test set)
                self.emg_mean = emg_mean
                self.emg_std = emg_std
                self.mic_mean = mic_mean
                self.mic_std = mic_std
                print("   Using provided normalization statistics from training set")
        else:
            self.emg_mean = None
            self.emg_std = None
            self.mic_mean = None
            self.mic_std = None
    
    def get_normalization_stats(self):
        """Return normalization statistics for use with test set."""
        if self.normalize:
            return {
                'emg_mean': self.emg_mean,
                'emg_std': self.emg_std,
                'mic_mean': self.mic_mean,
                'mic_std': self.mic_std
            }
        return None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _augment_emg(self, emg_np):
        """Apply safe augmentation to EMG signal."""
        if np.random.rand() > self.augment_prob:
            return emg_np
        
        augmented = emg_np.copy()
        
        if np.random.rand() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented = augmented * scale_factor
        
        if np.random.rand() < 0.3:
            noise_level = 0.02 * np.std(augmented)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise
        
        if np.random.rand() < 0.3:
            shift = np.random.randint(-50, 50)
            augmented = np.roll(augmented, shift, axis=0)
        
        return augmented
    
    def __getitem__(self, idx):
        from preprocessing_utils import preprocess_emg_signal
        
        emg, mic, label = self.base_dataset[idx]
        
        emg_np = emg.numpy()
        
        if self.apply_preprocessing:
            emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
        
        if self.augment:
            emg_np = self._augment_emg(emg_np)
        
        emg = torch.from_numpy(emg_np.astype(np.float32))
        
        if self.normalize and self.emg_mean is not None:
            # Apply global normalization (using statistics computed from all training samples)
            emg_mean_tensor = torch.from_numpy(self.emg_mean).float()
            emg_std_tensor = torch.from_numpy(self.emg_std).float()
            emg = (emg - emg_mean_tensor) / emg_std_tensor
            mic = (mic - self.mic_mean) / self.mic_std
        
        return emg, mic, label


class SimplerBruxismCNN(nn.Module):
    """
    Simpler CNN with fewer parameters - better for small datasets.
    """
    def __init__(self, num_classes=12, input_channels=4, dropout=0.5):
        super(SimplerBruxismCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MinimalBruxismCNN(nn.Module):
    """
    Minimal CNN with very few parameters - for severe overfitting cases.
    ~20k parameters - minimal complexity for small datasets.
    """
    def __init__(self, num_classes=12, input_channels=4, dropout=0.7):
        super(MinimalBruxismCNN, self).__init__()
        
        # Only one conv layer
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)  # Aggressive pooling
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(16, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


class EarlyStoppingCallback:
    """
    Early stopping to prevent overfitting.
    Can monitor both loss and accuracy.
    """
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, monitor='loss', 
                 immediate_stop_threshold=None, min_epochs_before_stop=5):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor  # 'loss' or 'accuracy'
        self.immediate_stop_threshold = immediate_stop_threshold  # None to disable, or a value like 3.0
        self.min_epochs_before_stop = min_epochs_before_stop  # Don't apply immediate stop until this many epochs
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.best_weights = None
        self.should_stop = False
        self.epoch_count = 0
    
    def __call__(self, val_loss, val_acc, model):
        self.epoch_count += 1
        
        if self.monitor == 'loss':
            metric = val_loss
            if self.best_loss is None:
                self.best_loss = val_loss
                if self.restore_best_weights:
                    self.best_weights = model.state_dict().copy()
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = model.state_dict().copy()
            else:
                self.counter += 1
        else:  # monitor == 'accuracy'
            metric = val_acc
            if self.best_acc is None:
                self.best_acc = val_acc
                if self.restore_best_weights:
                    self.best_weights = model.state_dict().copy()
            elif val_acc > self.best_acc + self.min_delta:
                self.best_acc = val_acc
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = model.state_dict().copy()
            else:
                self.counter += 1
                # Immediate stop if accuracy decreases significantly (overfitting)
                # Only apply if threshold is set AND we've trained for at least min_epochs_before_stop epochs
                if (self.immediate_stop_threshold is not None and 
                    self.best_acc is not None and 
                    val_acc < self.best_acc - self.immediate_stop_threshold and
                    self.epoch_count >= self.min_epochs_before_stop):
                    self.should_stop = True
                    if self.restore_best_weights and self.best_weights is not None:
                        model.load_state_dict(self.best_weights)
                    print(f"\n⚠️ Early stopping: Val accuracy decreased from {self.best_acc:.2f}% to {val_acc:.2f}% (severe overfitting)")
                    return self.should_stop
        
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
        
        return self.should_stop


def train_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, 
                               scheduler, device, num_epochs=100, patience=15, monitor='accuracy',
                               use_early_stopping=True, immediate_stop_threshold=None, min_epochs_before_stop=5):
    """
    Training loop with early stopping.
    monitor: 'loss' or 'accuracy' - what to monitor for early stopping
    use_early_stopping: if False, disable early stopping entirely
    immediate_stop_threshold: if None, disable immediate stop on accuracy drop. If set (e.g., 3.0), 
                               stop immediately if accuracy drops by this amount (only after min_epochs_before_stop)
    min_epochs_before_stop: minimum epochs before applying immediate stop check
    """
    if use_early_stopping:
        early_stopping = EarlyStoppingCallback(
            patience=patience, 
            min_delta=0.001, 
            monitor=monitor,
            immediate_stop_threshold=immediate_stop_threshold,
            min_epochs_before_stop=min_epochs_before_stop
        )
    else:
        early_stopping = None
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_loss = float('inf')
    
    print("Starting training with early stopping...\n")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Handle both raw signal datasets (emg, mic, labels) and feature datasets (features, labels)
        for batch in train_loader:
            if len(batch) == 3:
                # Raw signal dataset: (emg, mic, labels)
                emg_batch, mic_batch, labels_batch = batch
                input_batch = emg_batch.to(device)
            else:
                # Feature-based dataset: (features, labels)
                input_batch, labels_batch = batch
                input_batch = input_batch.to(device)
            
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Handle both raw signal datasets and feature datasets
            for batch in test_loader:
                if len(batch) == 3:
                    # Raw signal dataset: (emg, mic, labels)
                    emg_batch, mic_batch, labels_batch = batch
                    input_batch = emg_batch.to(device)
                else:
                    # Feature-based dataset: (features, labels)
                    input_batch, labels_batch = batch
                    input_batch = input_batch.to(device)
                
                labels_batch = labels_batch.to(device)
                
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_early_stopping.pth')
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% [SAVED]")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Pass both loss and accuracy to early stopping
        if early_stopping is not None and early_stopping(val_loss, val_acc, model):
            print(f"Training stopped at epoch {epoch+1}")
            break
    
    return train_losses, train_accs, val_losses, val_accs


def cross_validate_subjects(data_root, all_subjects, window_size=1200, stride=600, 
                             model_class=None, num_epochs=50, batch_size=32):
    """
    Perform leave-one-subject-out cross-validation.
    This is crucial for small subject datasets to get reliable performance estimates.
    """
    from bruxism_dataset import create_train_test_split
    
    results = []
    
    for test_subject in all_subjects:
        train_subjects = [s for s in all_subjects if s != test_subject]
        
        print(f"\n{'='*60}")
        print(f"Fold: Testing on Subject {test_subject}, Training on {train_subjects}")
        print(f"{'='*60}\n")
        
        train_dataset, test_dataset = create_train_test_split(
            data_root=data_root,
            train_subjects=train_subjects,
            test_subjects=[test_subject],
            window_size=window_size,
            stride=stride,
            normalize=False
        )
        
        train_filtered = FilteredBruxismDataset(train_dataset, apply_preprocessing=True, 
                                                 normalize=True, filter_low_activity=True)
        test_filtered = FilteredBruxismDataset(test_dataset, apply_preprocessing=True, 
                                                normalize=True, filter_low_activity=True)
        
        train_loader = DataLoader(train_filtered, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_filtered, batch_size=batch_size, shuffle=False, num_workers=0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        _, _, _, val_accs = train_with_early_stopping(
            model, train_loader, test_loader, criterion, optimizer, 
            scheduler, device, num_epochs=num_epochs, patience=10
        )
        
        best_val_acc = max(val_accs)
        results.append({
            'test_subject': test_subject,
            'best_val_acc': best_val_acc
        })
        
        print(f"\nSubject {test_subject} best validation accuracy: {best_val_acc:.2f}%\n")
    
    mean_acc = np.mean([r['best_val_acc'] for r in results])
    std_acc = np.std([r['best_val_acc'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results:")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return results


def extract_features(emg_data, fs=1200):
    """
    Extract comprehensive time-domain and frequency-domain features from EMG.
    Enhanced version with more features for better classification.
    Includes rest-specific features for better rest detection.
    """
    from scipy import signal as scipy_signal
    from scipy.fft import fft
    
    features = []
    
    # Global features (across all channels) - important for rest detection
    all_channels_rms = []
    all_channels_var = []
    
    for channel in range(emg_data.shape[1]):
        sig = emg_data[:, channel]
        
        # === Time Domain Features ===
        # Basic statistical features
        rms = np.sqrt(np.mean(sig**2))
        mav = np.mean(np.abs(sig))
        var = np.var(sig)
        std = np.std(sig)
        
        # Store for global features
        all_channels_rms.append(rms)
        all_channels_var.append(var)
        
        # Waveform features
        wl = np.sum(np.abs(np.diff(sig)))  # Waveform length
        zc = np.sum(np.diff(np.sign(sig)) != 0)  # Zero crossings
        ssc = np.sum((np.diff(sig[:-1]) * np.diff(sig[1:])) < 0)  # Slope sign changes
        
        # Additional time domain features
        skewness = np.mean(((sig - np.mean(sig)) / (np.std(sig) + 1e-8))**3)
        kurtosis = np.mean(((sig - np.mean(sig)) / (np.std(sig) + 1e-8))**4)
        peak_value = np.max(np.abs(sig))
        
        # Mean power
        mean_power = np.mean(sig**2)
        
        features.extend([rms, mav, var, std, wl, zc, ssc, skewness, kurtosis, peak_value, mean_power])
        
        # === Frequency Domain Features ===
        # FFT
        fft_vals = np.abs(fft(sig))
        fft_freqs = np.fft.fftfreq(len(sig), 1/fs)
        
        # Use only positive frequencies
        positive_freq_mask = fft_freqs > 0
        fft_vals = fft_vals[positive_freq_mask]
        fft_freqs = fft_freqs[positive_freq_mask]
        
        if len(fft_vals) > 0:
            # Spectral features
            mean_freq = np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
            median_freq = np.sum(fft_freqs[fft_vals >= np.median(fft_vals)]) / (np.sum(fft_vals >= np.median(fft_vals)) + 1e-8)
            spectral_power = np.sum(fft_vals**2)
            
            # Band power (20-450 Hz typical EMG range)
            emg_band_mask = (fft_freqs >= 20) & (fft_freqs <= 450)
            if np.any(emg_band_mask):
                band_power = np.sum(fft_vals[emg_band_mask]**2)
            else:
                band_power = 0
            
            # Peak frequency
            peak_freq_idx = np.argmax(fft_vals)
            peak_freq = fft_freqs[peak_freq_idx] if peak_freq_idx < len(fft_freqs) else 0
            
            features.extend([mean_freq, median_freq, spectral_power, band_power, peak_freq])
        else:
            features.extend([0, 0, 0, 0, 0])
    
    # === Cross-Channel Features ===
    # Correlation between channels
    if emg_data.shape[1] > 1:
        correlations = []
        for i in range(emg_data.shape[1]):
            for j in range(i+1, emg_data.shape[1]):
                corr = np.corrcoef(emg_data[:, i], emg_data[:, j])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
        features.extend(correlations)
    
    # === Rest-Specific Features ===
    # These help distinguish rest from active states
    if len(all_channels_rms) > 0:
        mean_rms_across_channels = np.mean(all_channels_rms)
        min_rms_across_channels = np.min(all_channels_rms)
        max_rms_across_channels = np.max(all_channels_rms)
        rms_range = max_rms_across_channels - min_rms_across_channels
        
        mean_var_across_channels = np.mean(all_channels_var)
        min_var_across_channels = np.min(all_channels_var)
        var_range = np.max(all_channels_var) - min_var_across_channels
        
        # Rest typically has low and uniform activity across channels
        rms_uniformity = 1.0 - (rms_range / (mean_rms_across_channels + 1e-8))  # Closer to 1 = more uniform
        var_uniformity = 1.0 - (var_range / (mean_var_across_channels + 1e-8))
        
        # Low activity indicator (rest should have very low values)
        activity_level = mean_rms_across_channels
        
        features.extend([
            mean_rms_across_channels, min_rms_across_channels, max_rms_across_channels, rms_range,
            mean_var_across_channels, min_var_across_channels, var_range,
            rms_uniformity, var_uniformity, activity_level
        ])
    else:
        features.extend([0] * 10)
    
    return np.array(features)


class FeatureBasedDataset(Dataset):
    """
    Dataset that extracts handcrafted features instead of using raw signals.
    Can work better with very small datasets.
    Uses global normalization (StandardScaler) to prevent overfitting.
    """
    def __init__(self, base_dataset, apply_preprocessing=True, use_ica=False, scaler=None):
        """
        Args:
            base_dataset: Base dataset to process
            apply_preprocessing: Whether to apply signal preprocessing
            use_ica: Whether to use ICA preprocessing (default False for consistency)
            scaler: Pre-fitted StandardScaler from training set. If None, fit on this dataset (for training).
                    If provided, use it to transform (for test set).
        """
        self.base_dataset = base_dataset
        self.apply_preprocessing = apply_preprocessing
        self.use_ica = use_ica  # Option to disable ICA
        
        print("Extracting features from all samples...")
        self.features = []
        self.labels = []
        
        for idx in range(len(base_dataset)):
            emg, _, label = base_dataset[idx]
            emg_np = emg.numpy()
            
            if self.apply_preprocessing:
                from preprocessing_utils import preprocess_emg_signal
                emg_np = preprocess_emg_signal(emg_np, fs=1200, apply_ica_flag=self.use_ica)
            
            feat = extract_features(emg_np, fs=1200)
            self.features.append(feat)
            self.labels.append(label)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        from sklearn.preprocessing import StandardScaler
        if scaler is None:
            # Fit scaler on training data
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            print("   Fitted StandardScaler on training features")
        else:
            # Use provided scaler from training set
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
            print("   Using provided StandardScaler from training set")
        
        print(f"Extracted {self.features.shape[1]} features per sample")
    
    def get_scaler(self):
        """Return the fitted scaler for use with test set."""
        return self.scaler
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), self.labels[idx]


class SimpleFeatureClassifier(nn.Module):
    """
    Simple MLP for feature-based classification.
    """
    def __init__(self, input_dim, num_classes=12, hidden_dim=128, dropout=0.5):
        super(SimpleFeatureClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


# Class reduction mappings for reducing 12 classes to fewer
CLASS_REDUCTION_STRATEGIES = {
    '4_classes': {
        # REST REMOVED - contaminated/poor quality
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

# Original class mappings from bruxism_dataset
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


class ReducedClassDataset(Dataset):
    """
    Wrapper that maps original classes to reduced classes.
    This simplifies the classification problem for small datasets.
    """
    def __init__(self, base_dataset, reduction_strategy='5_classes'):
        self.base_dataset = base_dataset
        self.reduction_map = CLASS_REDUCTION_STRATEGIES[reduction_strategy]
        self.reverse_map = {v: k for k, v in self.reduction_map.items()}
        self.num_classes = len(self.reduction_map)
        
        # Create mapping from original class to reduced class
        self.original_to_reduced = {}
        
        if reduction_strategy == '4_classes':
            for orig_name, orig_idx in ORIGINAL_CLASSES.items():
                if orig_name == 'rest':
                    # REST REMOVED - skip rest samples (they won't be included in training)
                    # Don't assign a mapping - rest will be filtered out
                    continue
                elif orig_name in ['open_close', 'deviation_left_right', 'protrusion_retrusion']:
                    # Movement class (separate from clenching)
                    self.original_to_reduced[orig_idx] = self.reduction_map['movement']
                elif orig_name in ['bite_left', 'bite_right', 'molar_clench', 'incisor_clench']:
                    # Clenching class
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
        
        # Filter out samples that don't have a mapping (e.g., rest if removed)
        self.valid_indices = []
        for idx in range(len(self.base_dataset)):
            _, original_label = self.base_dataset[idx]
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
        
        print(f"Valid samples: {len(self.valid_indices)}/{len(self.base_dataset)} (rest filtered out)")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        features, original_label = self.base_dataset[actual_idx]
        reduced_label = self.original_to_reduced[original_label]
        return features, reduced_label
    
    def get_class_names(self):
        return [self.reverse_map[i] for i in range(self.num_classes)]

