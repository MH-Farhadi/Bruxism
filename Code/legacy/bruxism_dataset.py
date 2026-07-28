import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from typing import Optional, List, Tuple, Dict


class BruxismDataset(Dataset):
    """
    Our main Dataset class for bruxism EMG signal classification.
    Loads CSV data, segments into sliding windows, and provides labeled samples.
    """
    
    def __init__(
        self,
        data_root: str,
        window_size: int = 1200,
        stride: int = 600,
        subjects: Optional[List[int]] = None,
        conditions: Optional[List[str]] = None,
        normalize: bool = True,
        skip_initial_seconds: float = 0.0
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory with 'Data' folder.
            window_size: Samples per window (default: 1200).
            stride: Samples between windows (default: 600).
            subjects: Subject IDs to include. None = all.
            conditions: Condition names to include. None = all.
            normalize: Normalize signals (default: True).
            skip_initial_seconds: Seconds to skip at file start (default: 0.0).
        """
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.skip_initial_seconds = skip_initial_seconds
        
        self.condition_mapping = {
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
        
        self.samples = []
        self._load_data(subjects, conditions)
        
        print(f"Loaded {len(self.samples)} windows from {len(set([s['subject'] for s in self.samples]))} subjects")
        
    def _load_data(self, subjects: Optional[List[int]], conditions: Optional[List[str]]):
        """
        Load raw data files and create signal windows.
        
        Args:
            subjects: Subject IDs to include.
            conditions: Condition names to include.
        """
        data_dir = self.data_root / "Data"
        
        subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Subject_')])
        
        for subject_dir in subject_dirs:
            subject_id = int(subject_dir.name.split('_')[1])
            
            if subjects is not None and subject_id not in subjects:
                continue
            
            csv_files = list(subject_dir.glob('*.csv'))
            
            for csv_file in csv_files:
                filename = csv_file.stem
                
                # Extract condition from filename 
                condition_key = '_'.join(filename.split('_')[:-3])
                
                if conditions is not None and condition_key not in conditions:
                    continue
                
                if condition_key not in self.condition_mapping:
                    print(f"Warning: Unknown condition '{condition_key}' in file {csv_file.name}")
                    continue
                
                # Read sampling rate from metadata
                metadata_file = csv_file.with_name(csv_file.stem + '_metadata.txt')
                sampling_rate = 1200
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        for line in f:
                            if line.startswith('sampling_rate:'):
                                sampling_rate = int(line.split(':')[1].strip())
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    emg_cols = ['EMG1_1-2', 'EMG2_3-4', 'EMG3_5-6', 'EMG4_7-8']
                    mic_col = 'Mic'
                    
                    emg_data = df[emg_cols].values.astype(np.float32)
                    mic_data = df[mic_col].values.astype(np.float32)
                    
                    # We skip initial samples to avoid initialization artifacts
                    skip_samples = int(self.skip_initial_seconds * sampling_rate)
                    if skip_samples > 0:
                        emg_data = emg_data[skip_samples:]
                        mic_data = mic_data[skip_samples:]
                    
                    # Calculate number of overlapping windows
                    num_windows = (len(emg_data) - self.window_size) // self.stride + 1
                    if num_windows <= 0:
                        continue
                    
                    # Create overlapping windows with stride offset
                    for i in range(num_windows):
                        start_idx = i * self.stride
                        end_idx = start_idx + self.window_size
                        
                        self.samples.append({
                            'emg': emg_data[start_idx:end_idx],
                            'mic': mic_data[start_idx:end_idx],
                            'label': self.condition_mapping[condition_key],
                            'condition': condition_key,
                            'subject': subject_id,
                            'file': csv_file.name
                        })
                        
                except Exception as e:
                    print(f"Error loading {csv_file.name}: {e}")
    
    def __len__(self):
        """
        Return number of samples.
    
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get sample at the given index.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (emg tensor, mic tensor, label).
        """
        sample = self.samples[idx]
        
        emg = torch.from_numpy(sample['emg'])
        mic = torch.from_numpy(sample['mic']).unsqueeze(-1)
        label = sample['label']
        
        # Normalize to zero mean, unit variance
        if self.normalize:
            emg = (emg - emg.mean(dim=0, keepdim=True)) / (emg.std(dim=0, keepdim=True) + 1e-8)
            mic = (mic - mic.mean()) / (mic.std() + 1e-8)
        
        return emg, mic, label
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get sample metadata without loading signal data.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dict with condition, subject, label, file.
        """
        return {
            'condition': self.samples[idx]['condition'],
            'subject': self.samples[idx]['subject'],
            'label': self.samples[idx]['label'],
            'file': self.samples[idx]['file']
        }
    
    def get_label_counts(self) -> Dict[str, int]:
        """
        Count samples per condition.
        
        Returns:
            Dict mapping condition names to counts.
        """
        counts = {}
        for sample in self.samples:
            condition = sample['condition']
            counts[condition] = counts.get(condition, 0) + 1
        return counts


def create_train_test_split(
    data_root: str,
    train_subjects: List[int],
    test_subjects: List[int],
    window_size: int = 1200,
    stride: int = 600,
    normalize: bool = True,
    conditions: Optional[List[str]] = None,
    skip_initial_seconds: float = 0.0
) -> Tuple[BruxismDataset, BruxismDataset]:
    """
    Split dataset by subject IDs (prevents data leakage).
    
    Args:
        data_root: Root directory with 'Data' folder.
        train_subjects: Subject IDs for training.
        test_subjects: Subject IDs for testing.
        window_size: Samples per window (default: 1200).
        stride: Samples between windows (default: 600).
        normalize: Normalize and standardize signals (default: True).
        conditions: Condition names to include.
        skip_initial_seconds: Seconds to skip at file start.
    
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    train_dataset = BruxismDataset(
        data_root=data_root,
        subjects=train_subjects,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        conditions=conditions,
        skip_initial_seconds=skip_initial_seconds
    )
    
    test_dataset = BruxismDataset(
        data_root=data_root,
        subjects=test_subjects,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        conditions=conditions,
        skip_initial_seconds=skip_initial_seconds
    )
    
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: BruxismDataset,
    test_dataset: BruxismDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        train_dataset: Training dataset.
        test_dataset: Testing dataset.
        batch_size: Samples per batch (default: 32).
        num_workers: Worker processes (default: 0).
        shuffle_train: Shuffle training data (default: True).
    
    Returns:
        Tuple of (train_dataloader, test_dataloader).
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    data_root = r"C:\Users\mhfar\Desktop\Depo\Brusxism_data" # change this to the path of the dataset
    
    dataset = BruxismDataset(
        data_root=data_root,
        window_size=1200,
        stride=600,
        normalize=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"\nLabel distribution:")
    for condition, count in sorted(dataset.get_label_counts().items()):
        print(f"  {condition}: {count} windows")
    
    train_dataset, test_dataset = create_train_test_split(
        data_root=data_root,
        train_subjects=[1, 2, 3, 4],
        test_subjects=[5],
        window_size=1200,
        stride=600
    )
    
    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
