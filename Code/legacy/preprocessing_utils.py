"""
Preprocessing utilities for EMG signal processing.

Provides filtering functions (notch, bandpass, baseline removal) and ICA
for noise reduction and artifact removal in EMG signals.
"""

import numpy as np
import warnings
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA

# Suppress FastICA convergence warnings for small windows
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.decomposition._fastica')


def bandpass_filter(data, lowcut=20, highcut=450, fs=1200, order=4):
    """
    Apply bandpass filter to extract EMG frequency range.
    
    Removes frequencies outside the EMG signal band (typically 20-450 Hz)
    to eliminate noise and focus on muscle activity.
    
    Args:
        data (np.ndarray): Input signal array (samples x channels).
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.
    
    Returns:
        np.ndarray: Bandpass filtered signal with same shape as input.
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
    
    Eliminates 60 Hz (or specified frequency) interference from power lines
    that commonly contaminates EMG recordings.
    
    Args:
        data (np.ndarray): Input signal array.
        freq (float): Frequency to notch out in Hz (typically 60 Hz).
        fs (float): Sampling rate in Hz.
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
    
    Eliminates slow DC offset and baseline wander that can contaminate
    EMG signals, typically caused by electrode-skin interface or movement artifacts.
    
    Args:
        data (np.ndarray): Input signal array.
        fs (float): Sampling rate in Hz.
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
    
    Separates mixed EMG signals into independent components, allowing
    isolation of muscle activity from noise and artifacts. Configured with
    parameters optimized for small EMG windows.
    
    Args:
        data (np.ndarray): Input signal data (samples, channels).
        n_components (int, optional): Number of components to extract.
                                      If None, uses minimum of data dimensions.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: (sources, ica) where sources are separated components and
               ica is the fitted model.
    """
    if n_components is None:
        n_components = min(data.shape)
    
    # Configure for small windows: higher max_iter, lenient tolerance, stable solver
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
    
    Allows selective reconstruction using only desired components (e.g., muscle activity)
    while excluding noise or artifact components.
    
    Args:
        sources (np.ndarray): ICA source signals (samples x components).
        ica: Fitted ICA model object.
        components_to_keep (list, optional): Indices of components to keep.
                                             If None, keeps all components.
    
    Returns:
        np.ndarray: Reconstructed signal array.
    """
    if components_to_keep is not None:
        # Zero out components not in components_to_keep
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
    Apply full preprocessing pipeline to EMG signal.
    
    Applies sequential filtering steps: notch filter (60 Hz), bandpass filter (20-450 Hz),
    baseline removal, and optionally ICA for source separation. This pipeline removes
    common artifacts and noise while preserving muscle activity signals.
    
    Args:
        data (np.ndarray): Raw EMG signal data (samples x channels).
        fs (float): Sampling rate in Hz.
        apply_ica_flag (bool): Whether to apply ICA preprocessing.
                              Requires multiple channels (data.shape[1] > 1).
    
    Returns:
        np.ndarray: Preprocessed EMG signal.
    """
    # Remove power line interference
    data_filtered = notch_filter(data, freq=60, fs=fs)
    
    # Extract EMG frequency band
    data_filtered = bandpass_filter(data_filtered, lowcut=20, highcut=450, fs=fs)
    
    # Remove baseline drift
    data_filtered = remove_baseline_drift(data_filtered, fs=fs, cutoff=5)
    
    # Optionally apply ICA for source separation (requires multiple channels)
    if apply_ica_flag and data.shape[1] > 1:
        sources, ica = apply_ica(data_filtered)
        data_filtered = reconstruct_from_ica(sources, ica)
    
    return data_filtered

