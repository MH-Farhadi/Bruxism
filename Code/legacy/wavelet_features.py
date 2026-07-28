"""
These script has functions for the wavelet-based feature extraction
for EMG signals and captures time-frequency information for bruxism detection.
"""

import numpy as np
import pywt
from scipy import signal
from typing import Tuple, List, Optional


def extract_wavelet_features(
    
    emg_data: np.ndarray,
    wavelet: str = 'db4',
    max_level: int = 5,
    fs: int = 1200
) -> np.ndarray:
    """
    Extract wavelet features from EMG signals using DWT, WPT, and CWT.
    
    Args:
        emg_data: EMG signal array (samples x channels).
        wavelet: Wavelet family (default: 'db4').
        max_level: Maximum decomposition level (default: 5).
        fs: Sampling frequency in Hz (default: 1200).
    
    Returns:
        Feature array (n_features,).
    """


    features = []
    
    for channel in range(emg_data.shape[1]):
        sig = emg_data[:, channel]
        
        # DWT: decompose signal into approximation (low freq) and detail (high freq) coefficients
        coeffs = pywt.wavedec(sig, wavelet, level=max_level)
        cA = coeffs[0]
        cD_list = coeffs[1:]
        
        # Calculate energy at each decomposition level
        approx_energy = np.sum(cA**2)
        detail_energies = [np.sum(cD**2) for cD in cD_list]
        total_energy = approx_energy + sum(detail_energies)
        
        # Normalize energies to relative proportions
        if total_energy > 0:
            relative_approx_energy = approx_energy / total_energy
            relative_detail_energies = [e / total_energy for e in detail_energies]
        else:
            relative_approx_energy = 0
            relative_detail_energies = [0] * len(detail_energies)
        
        features.extend([
            approx_energy,
            total_energy,
            relative_approx_energy,
            *relative_detail_energies
        ])
        
        # Statistical features of coefficients
        approx_std = np.std(cA)
        approx_mean = np.abs(np.mean(cA))
        detail_stds = [np.std(cD) for cD in cD_list]
        detail_means = [np.abs(np.mean(cD)) for cD in cD_list]
        
        features.extend([
            approx_std,
            approx_mean,
            *detail_stds,
            *detail_means
        ])
        

        """
        We also use the Wavelet Packet Transform for more detailed
        frequency decomposition for each channel, since DWT only extracts   
        low frequency components.
        """
        
        try:
            wp = pywt.WaveletPacket(sig, wavelet, maxlevel=4)
            node_names = ['a', 'd']
            wpt_energies = []
            for node_name in node_names:
                try:
                    node = wp[node_name]
                    node_energy = np.sum(node.data**2)
                    wpt_energies.append(node_energy)
                except (KeyError, TypeError, AttributeError, ValueError):
                    wpt_energies.append(0)
            
            if sum(wpt_energies) > 0:
                wpt_relative_energies = [e / sum(wpt_energies) for e in wpt_energies]
            else:
                wpt_relative_energies = [0] * len(wpt_energies)
            
            features.extend(wpt_energies)
            features.extend(wpt_relative_energies)
        except Exception:
            features.extend([0] * 4)
        
        
        """
        CWT is a time-frequency representation that is more localized in time
        than the DWT and WPT. It is also Better suited for 
        non-stationary signals like EMG signals.
        """
        
        try:
            scales = np.arange(1, 50)
            cwt_coeffs, freqs = pywt.cwt(sig, scales, wavelet, 1.0/fs)
            
            # Energy per scale, dominant frequency, and concentration
            cwt_energy = np.sum(np.abs(cwt_coeffs)**2, axis=1)
            dominant_scale_idx = np.argmax(cwt_energy)
            dominant_scale = scales[dominant_scale_idx] if len(cwt_energy) > 0 else 0
            
            if np.sum(cwt_energy) > 0:
                energy_concentration = np.max(cwt_energy) / np.sum(cwt_energy)
            else:
                energy_concentration = 0
            
            features.extend([
                dominant_scale,
                energy_concentration,
                np.mean(cwt_energy),
                np.std(cwt_energy)
            ])
        except Exception:
            features.extend([0] * 4)
        
        # Wavelet entropy: measures signal complexity (higher = more irregular)
        if total_energy > 0:
            energy_dist = [approx_energy / total_energy] + [e / total_energy for e in detail_energies]
            energy_dist = [e for e in energy_dist if e > 0]
            
            if len(energy_dist) > 0:
                wavelet_entropy = -np.sum([e * np.log2(e) for e in energy_dist])
            else:
                wavelet_entropy = 0
        else:
            wavelet_entropy = 0
        
        features.append(wavelet_entropy)
        
        # Variance across scales indicates signal variability
        coeff_variances = [np.var(cA)] + [np.var(cD) for cD in cD_list]
        features.extend([
            np.mean(coeff_variances),
            np.std(coeff_variances),
            np.max(coeff_variances)
        ])
    
    # Cross-channel correlations to capture spatial relationships between electrodes
    if emg_data.shape[1] > 1:
        channel_correlations = []
        
        for i in range(emg_data.shape[1]):
            for j in range(i+1, emg_data.shape[1]):
                sig1 = emg_data[:, i]
                sig2 = emg_data[:, j]
                
                coeffs1 = pywt.wavedec(sig1, wavelet, level=max_level)
                coeffs2 = pywt.wavedec(sig2, wavelet, level=max_level)
                
                # Correlate approximation coefficients (low frequency correlation)
                cA1 = coeffs1[0]
                cA2 = coeffs2[0]
                
                if len(cA1) > 1 and len(cA2) > 1:
                    corr = np.corrcoef(cA1, cA2)[0, 1]
                    channel_correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    channel_correlations.append(0)
        
        features.extend(channel_correlations)
    
    return np.array(features)


def extract_wavelet_coefficients(
    emg_data: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Extract wavelet coefficients for neural network input.
    
    Args:
        emg_data: EMG signal array (samples x channels).
        wavelet: Wavelet family (default: 'db4').
        level: Decomposition level (default: 4).
    
    Returns:
        Tuple of (approximation coefficients, list of detail coefficients per level).
    """
    n_channels = emg_data.shape[1]
    approx_coeffs_list = []
    detail_coeffs_list = [[] for _ in range(level)]
    
    for ch in range(n_channels):
        sig = emg_data[:, ch]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        
        approx_coeffs_list.append(coeffs[0])
        for i, cD in enumerate(coeffs[1:]):
            detail_coeffs_list[i].append(cD)
    
    # Stack channels into arrays
    approx_coeffs = np.stack(approx_coeffs_list, axis=1)
    detail_coeffs = [np.stack(d, axis=1) for d in detail_coeffs_list]
    
    return approx_coeffs, detail_coeffs


def get_wavelet_feature_dimension(n_channels: int = 4, max_level: int = 5) -> int:
    
    #Calculate total feature dimension for model initialization.
    # Per channel: DWT (3 + 3*max_level) + WPT (4) + CWT (4) + entropy (1) + variance (3)
    dwt_features_per_channel = 3 + 3 * max_level
    wpt_features_per_channel = 4
    cwt_features_per_channel = 4
    entropy_per_channel = 1
    variance_per_channel = 3
    
    per_channel = dwt_features_per_channel + wpt_features_per_channel + cwt_features_per_channel + entropy_per_channel + variance_per_channel
    
    # Cross-channel correlations: n_channels * (n_channels - 1) / 2
    cross_channel = n_channels * (n_channels - 1) // 2
    
    total = per_channel * n_channels + cross_channel
    
    return total


if __name__ == "__main__":
    print("Testing wavelet feature extraction...")
    
    fs = 1200
    t = np.linspace(0, 1, fs)
    synthetic_emg = np.random.randn(fs, 4) * 0.1 + np.sin(2 * np.pi * 50 * t)[:, None]
    
    features = extract_wavelet_features(synthetic_emg, wavelet='db4', max_level=5, fs=fs)
    print(f"Extracted {len(features)} wavelet features")
    print(f"Feature dimension: {get_wavelet_feature_dimension(n_channels=4, max_level=5)}")
    
    approx, details = extract_wavelet_coefficients(synthetic_emg, wavelet='db4', level=4)
    print(f"Approximation coefficients shape: {approx.shape}")
    print(f"Number of detail levels: {len(details)}")
    for i, d in enumerate(details):
        print(f"  Detail level {i+1} shape: {d.shape}")

