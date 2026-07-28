"""
Wavelet-CNN hybrid architecture for bruxism detection.
Combines wavelet transforms, 1D CNNs, and pseudo-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


class WaveletCNN(nn.Module):
    """
    Hybrid CNN architecture that processes EMG signals using wavelet decomposition.
    Decomposes input signals into frequency bands (low, mid, high) using discrete
    wavelet transform, processes each band with separate CNN branches, then applies
    attention weighting and fusion before classification.
    """
    
    def __init__(
        self,
        num_classes: int = 12,
        input_channels: int = 4,
        window_size: int = 1200,
        wavelet: str = 'db4',
        use_learnable_wavelet: bool = False,
        dropout: float = 0.6
    ):
        """
        Initialize WaveletCNN model.
        
        Args:
            num_classes (int): Number of output classes for classification.
            input_channels (int): Number of input EMG channels.
            window_size (int): Length of input signal window in samples.
            wavelet (str): Wavelet type for decomposition (e.g., 'db4').
            use_learnable_wavelet (bool): Whether to use learnable wavelets (unused).
            dropout (float): Dropout probability for regularization.
        """
        super(WaveletCNN, self).__init__()
        
        self.input_channels = input_channels
        self.window_size = window_size
        self.wavelet = wavelet
        self.use_learnable_wavelet = use_learnable_wavelet
        self.wavelet_level = 4
        
        # Fixed wavelets are more stable for small datasets than learnable variants
        self.wavelet_transform = None
        
        # Separate CNN branches process different frequency bands independently
        # Each branch extracts features from its assigned frequency range
        self.low_freq_branch = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.mid_freq_branch = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # High-frequency branch uses larger kernel and more aggressive pooling
        # to handle noisier, faster-varying signals
        self.high_freq_branch = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Attention mechanism learns relative importance of each frequency band
        # Input: 96 features (32 from each of 3 branches), Output: 3 weights
        self.attention = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 3),
            nn.Softmax(dim=1)
        )
        
        # Fusion layer combines weighted features and reduces dimensionality
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(32, num_classes)
        
    def _wavelet_decompose(self, x: torch.Tensor) -> tuple:
        """
        Decompose input signal into wavelet coefficients.
        
        Performs multi-level discrete wavelet transform to separate signal into
        approximation (low-frequency) and detail (high-frequency) coefficients.
        Each level halves the frequency resolution, creating a multi-scale representation.
        
        Args:
            Input signal tensor of shape (batch, channels, samples).
        
        Returns:
            tuple(approx, details) where:
                - approx (torch.Tensor): Approximation coefficients (batch, channels, samples).
                - details (list): List of detail coefficient tensors, one per decomposition level.
        """
        batch_size, n_channels, n_samples = x.shape
        
        # PyWavelets requires numpy arrays, so we convert temporarily to numpy array
        x_np = x.detach().cpu().numpy()
        
        approx_list = []
        details_list = [[] for _ in range(self.wavelet_level)]
        
        # Decompose each channel independently
        for b in range(batch_size):
            for c in range(n_channels):
                sig = x_np[b, c, :]
                coeffs = pywt.wavedec(sig, self.wavelet, level=self.wavelet_level)
                
                # First element is approximation, rest are detail coefficients
                approx_list.append(coeffs[0])
                for i, cD in enumerate(coeffs[1:]):
                    details_list[i].append(cD)
        
        # Reconstruct tensors maintaining batch and channel structure
        approx = torch.tensor(
            np.array(approx_list).reshape(batch_size, n_channels, -1),
            device=x.device,
            dtype=x.dtype
        )
        
        details = []
        for d_list in details_list:
            d_tensor = torch.tensor(
                np.array(d_list).reshape(batch_size, n_channels, -1),
                device=x.device,
                dtype=x.dtype
            )
            details.append(d_tensor)
        
        return approx, details
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        #Forward pass through the model.
        
        # Normalize input shape to (batch, channels, time) for Conv1d
        if x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)
        
        # Decompose signal into frequency bands
        approx, details = self._wavelet_decompose(x)
        
        # Extract features from low-frequency approximation coefficients
        low_feat = self.low_freq_branch(approx).squeeze(-1)
        
        # Combine detail levels 3 and 4 for mid-frequency processing
        # Falls back to available details or approximation if insufficient levels
        if len(details) >= 3:
            mid_input = details[2]
            if len(details) >= 4:
                mid_input = torch.cat([details[2], details[3]], dim=2)
        else:
            mid_input = details[-1] if details else approx
        
        mid_feat = self.mid_freq_branch(mid_input).squeeze(-1)
        
        # Combine detail levels 1 and 2 for high-frequency processing
        # These capture the highet frequency components
        if len(details) >= 2:
            high_input = torch.cat([details[0], details[1]], dim=2)
        else:
            high_input = details[0] if details else approx
        
        high_feat = self.high_freq_branch(high_input).squeeze(-1)
        
        # Attention mechanism to  learn optimal weighting for each frequency band
        combined = torch.cat([low_feat, mid_feat, high_feat], dim=1)
        attention_weights = self.attention(combined)
        
        # Apply learned attention weights to each frequency band
        low_feat_weighted = low_feat * attention_weights[:, 0:1]
        mid_feat_weighted = mid_feat * attention_weights[:, 1:2]
        high_feat_weighted = high_feat * attention_weights[:, 2:3]
        
        combined_weighted = torch.cat([low_feat_weighted, mid_feat_weighted, high_feat_weighted], dim=1)
        
        # Fuse weighted features and classify
        fused = self.fusion(combined_weighted)
        logits = self.classifier(fused)
        
        return logits


class WaveletFeatureCNN(nn.Module):
    """
    Lightweight CNN for classification using pre-extracted wavelet features.
    Processes fixed wavelet features rather than raw signals, resulting in
    fewer parameters (~5-10k) and a much more stable training.
    """
    
    def __init__(
        self,
        num_classes: int = 12,
        wavelet_feature_dim: int = 100,
        hidden_dim: int = 64,
        dropout: float = 0.6
    ):
        """
        Initialize WaveletFeatureCNN model.
        
        Args:
            num_classes (int): Number of output classes for classification.
            wavelet_feature_dim (int): Dimension of input wavelet features.
            hidden_dim (int): Hidden layer dimension for feature processing.
            dropout (float): Dropout probability for regularization.
        """
        super(WaveletFeatureCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(wavelet_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    print("Testing WaveletCNN...")
    model1 = WaveletCNN(num_classes=12, input_channels=4, window_size=1200)
    
    total_params = sum(p.numel() for p in model1.parameters())
    trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    x = torch.randn(4, 4, 1200)
    with torch.no_grad():
        output = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting WaveletFeatureCNN...")
    model2 = WaveletFeatureCNN(num_classes=12, wavelet_feature_dim=100)
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters: {total_params2:,}")
    
    x2 = torch.randn(4, 100)
    with torch.no_grad():
        output2 = model2(x2)
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {output2.shape}")

