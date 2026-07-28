"""
Sanity check visualization for preprocessing pipeline.

Loads data windows from recordings and visualizes raw signals alongside
each preprocessing step. Uses more aggressive filter parameters than training
to make preprocessing effects more visible. Matches preprocessing methodology
from run_new_wavelet_training.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from bruxism_dataset import BruxismDataset
from preprocessing_utils import (
    bandpass_filter,
    notch_filter,
    remove_baseline_drift,
    apply_ica,
    reconstruct_from_ica
)

# Match hyperparameters from run_new_wavelet_training.py
USE_ICA_EMG = False  # Set to True if ICA is enabled in training


def get_raw_data_from_dataset(dataset, idx):
    """
    Extract raw data from dataset without normalization.
    
    Args:
        dataset: Dataset object containing samples.
        idx (int): Sample index.
    
    Returns:
        tuple: (emg_raw, mic_raw, sample_info) where:
            - emg_raw (np.ndarray): Raw EMG signal (1200, 4).
            - mic_raw (np.ndarray): Raw microphone signal (1200,).
            - sample_info (dict): Sample metadata.
    """
    sample = dataset.samples[idx]
    emg_raw = sample['emg'].copy()
    mic_raw = sample['mic'].copy()
    return emg_raw, mic_raw, sample


def aggressive_notch_filter(data, freq=60, fs=1200, quality=60):
    """
    Apply aggressive notch filter for visualization.
    
    Uses higher quality factor than training to create sharper notch and
    make filtering effects more visible in plots.
    
    Args:
        data (np.ndarray): Input signal array.
        freq (float): Frequency to notch out in Hz.
        fs (float): Sampling rate in Hz.
        quality (float): Filter quality factor (higher = sharper notch).
    
    Returns:
        np.ndarray: Notch filtered signal.
    """
    from scipy.signal import iirnotch, filtfilt
    nyquist = 0.5 * fs
    freq_norm = freq / nyquist
    b, a = iirnotch(freq_norm, quality)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def aggressive_bandpass_filter(data, lowcut=20, highcut=450, fs=1200, order=8):
    """
    Apply aggressive bandpass filter for visualization.
    
    Uses higher filter order than training to create sharper frequency cutoffs
    and make filtering effects more visible in plots.
    
    Args:
        data (np.ndarray): Input signal array.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order (higher = sharper cutoff).
    
    Returns:
        np.ndarray: Bandpass filtered signal.
    """
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def aggressive_baseline_removal(data, fs=1200, cutoff=3):
    """
    Apply aggressive baseline removal for visualization.
    
    Uses lower cutoff frequency and higher filter order than training to
    make baseline removal effects more visible in plots.
    
    Args:
        data (np.ndarray): Input signal array.
        fs (float): Sampling rate in Hz.
        cutoff (float): High-pass cutoff frequency in Hz.
    
    Returns:
        np.ndarray: Signal with baseline drift removed.
    """
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='high')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def apply_preprocessing_steps_emg(emg_raw, fs=1200, apply_ica_flag=None):
    """
    Apply EMG preprocessing steps sequentially for visualization.
    
    Uses more aggressive filter parameters than training to make preprocessing
    effects more visible. Returns dictionary with signal after each step.
    
    Args:
        emg_raw (np.ndarray): Raw EMG signal (samples, channels).
        fs (float): Sampling rate in Hz.
        apply_ica_flag (bool, optional): Whether to apply ICA. If None, uses USE_ICA_EMG.
    
    Returns:
        dict: Dictionary mapping step names to processed signals:
            - 'Raw': Original signal
            - 'After Notch Filter (60Hz)': After notch filtering
            - 'After Bandpass (20-450Hz)': After bandpass filtering
            - 'After Baseline Removal': After baseline removal
            - 'After ICA': After ICA (if enabled)
            - 'After Normalization': Final normalized signal
    """
    if apply_ica_flag is None:
        apply_ica_flag = USE_ICA_EMG
    
    steps = {}
    
    # Capture signal after each preprocessing step
    steps['Raw'] = emg_raw.copy()
    
    # Remove power line interference
    steps['After Notch Filter (60Hz)'] = aggressive_notch_filter(emg_raw, freq=60, fs=fs, quality=60)
    
    # Extract EMG frequency band
    steps['After Bandpass (20-450Hz)'] = aggressive_bandpass_filter(
        steps['After Notch Filter (60Hz)'], 
        lowcut=20, highcut=450, fs=fs, order=8
    )
    
    # Remove baseline drift
    steps['After Baseline Removal'] = aggressive_baseline_removal(
        steps['After Bandpass (20-450Hz)'], 
        fs=fs, cutoff=3
    )
    
    # Apply ICA if enabled and multiple channels available
    if apply_ica_flag and emg_raw.shape[1] > 1:
        sources, ica = apply_ica(steps['After Baseline Removal'])
        steps['After ICA'] = reconstruct_from_ica(sources, ica)
    else:
        steps['After ICA'] = steps['After Baseline Removal'].copy()
    
    # Normalize using per-window statistics (for visualization)
    # Training uses global normalization from all training samples
    emg_mean = np.mean(steps['After ICA'], axis=0, keepdims=True)
    emg_std = np.std(steps['After ICA'], axis=0, keepdims=True) + 1e-8
    steps['After Normalization'] = (steps['After ICA'] - emg_mean) / emg_std
    
    return steps


def apply_preprocessing_steps_mic(mic_raw, fs=1200):
    """
    Apply MIC preprocessing steps sequentially for visualization.
    
    Training uses only DC offset removal. This function adds additional filtering
    steps for visualization purposes only - these are NOT used in actual training.
    
    Args:
        mic_raw (np.ndarray): Raw microphone signal.
        fs (float): Sampling rate in Hz.
    
    Returns:
        dict: Dictionary mapping step names to processed signals:
            - 'Raw': Original signal
            - 'After DC Offset Removal': Training-equivalent preprocessing
            - 'After Baseline Removal (VISUALIZATION)': Visualization only
            - 'After Bandpass 50-550Hz (VISUALIZATION)': Visualization only
            - 'After Notch Filter 60Hz (VISUALIZATION)': Visualization only
            - 'After Normalization (Training Equivalent)': Normalized training-equivalent
            - 'After Normalization (Filtered, VISUALIZATION)': Normalized filtered version
    """
    steps = {}
    
    steps['Raw'] = mic_raw.copy()
    
    # Training-equivalent preprocessing: DC offset removal only
    if len(mic_raw.shape) > 1:
        mic_1d = mic_raw.flatten()
    else:
        mic_1d = mic_raw.copy()
    
    steps['After DC Offset Removal'] = mic_1d - np.mean(mic_1d)
    
    # Additional filtering steps for visualization (not used in training)
    steps['After Baseline Removal (VISUALIZATION)'] = aggressive_baseline_removal(
        steps['After DC Offset Removal'].reshape(-1, 1), 
        fs=fs, cutoff=3
    ).flatten()
    
    # Bandpass filter optimized for audio frequency range
    nyquist = 0.5 * fs
    highcut = min(550, nyquist * 0.9)
    steps['After Bandpass 50-550Hz (VISUALIZATION)'] = aggressive_bandpass_filter(
        steps['After Baseline Removal (VISUALIZATION)'].reshape(-1, 1),
        lowcut=50, highcut=highcut, fs=fs, order=8
    ).flatten()
    
    # Notch filter for power line interference
    steps['After Notch Filter 60Hz (VISUALIZATION)'] = aggressive_notch_filter(
        steps['After Bandpass 50-550Hz (VISUALIZATION)'].reshape(-1, 1),
        freq=60, fs=fs, quality=60
    ).flatten()
    
    # Normalize training-equivalent signal (matches training pipeline)
    mic_mean = np.mean(steps['After DC Offset Removal'])
    mic_std = np.std(steps['After DC Offset Removal']) + 1e-8
    steps['After Normalization (Training Equivalent)'] = (
        steps['After DC Offset Removal'] - mic_mean
    ) / mic_std
    
    # Also normalize filtered version for comparison
    mic_mean_filtered = np.mean(steps['After Notch Filter 60Hz (VISUALIZATION)'])
    mic_std_filtered = np.std(steps['After Notch Filter 60Hz (VISUALIZATION)']) + 1e-8
    steps['After Normalization (Filtered, VISUALIZATION)'] = (
        steps['After Notch Filter 60Hz (VISUALIZATION)'] - mic_mean_filtered
    ) / mic_std_filtered
    
    return steps


def create_figure_name(window_num, idx, condition, subject, description, output_dir):
    """
    Generate descriptive filename for saved figure.
    
    Args:
        window_num (int): Window number in sequence.
        idx (int): Dataset index of the window.
        condition (str): Activity condition name.
        subject (int): Subject number.
        description (str): Figure description (e.g., 'EMG_raw', 'MIC_preprocessed').
        output_dir (str): Output directory path.
    
    Returns:
        str: Full file path for the figure.
    """
    safe_condition = condition.replace(' ', '_').replace('/', '_')
    filename = f"window_{window_num:02d}_idx_{idx:05d}_{safe_condition}_subject_{subject}_{description}.png"
    return os.path.join(output_dir, filename)


def visualize_window(emg_raw, mic_raw, emg_steps, mic_steps, sample_info, window_num, idx, output_dir):
    """
    Create visualization figures for raw and preprocessed signals.
    
    Generates separate figures showing: EMG raw, EMG preprocessed, EMG preprocessing
    steps per channel, MIC raw, MIC preprocessed, and MIC preprocessing steps.
    
    Args:
        emg_raw (np.ndarray): Raw EMG signal.
        mic_raw (np.ndarray): Raw microphone signal.
        emg_steps (dict): Dictionary of EMG signals after each preprocessing step.
        mic_steps (dict): Dictionary of MIC signals after each preprocessing step.
        sample_info (dict): Sample metadata (condition, subject, file).
        window_num (int): Window number in sequence.
        idx (int): Dataset index.
        output_dir (str): Directory to save figures.
    
    Returns:
        list: List of saved figure file paths.
    """
    fs = 1200
    time_axis = np.arange(len(emg_raw)) / fs
    emg_channel_names = ['EMG1_1-2', 'EMG2_3-4', 'EMG3_5-6', 'EMG4_7-8']
    colors_emg = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    condition = sample_info['condition']
    subject = sample_info['subject']
    
    saved_files = []
    
    # ========================================================================
    # 1. EMG: Raw Signals (All 4 channels)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for ch in range(4):
        ax.plot(time_axis, emg_steps['Raw'][:, ch], 
                label=emg_channel_names[ch], 
                color=colors_emg[ch], alpha=0.8, linewidth=1.5)
    ax.set_title(f'EMG Raw Signals', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    save_path = create_figure_name(window_num, idx, condition, subject, 'EMG_raw', output_dir)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    saved_files.append(save_path)
    plt.close(fig)
    
    # ========================================================================
    # 2. EMG: Final Preprocessed Signals (All 4 channels)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for ch in range(4):
        ax.plot(time_axis, emg_steps['After Normalization'][:, ch],
                label=emg_channel_names[ch],
                color=colors_emg[ch], alpha=0.8, linewidth=1.5)
    ax.set_title(f'EMG Preprocessed Signals', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    save_path = create_figure_name(window_num, idx, condition, subject, 'EMG_preprocessed', output_dir)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    saved_files.append(save_path)
    plt.close(fig)
    
    # ========================================================================
    # 3. EMG: Preprocessing Steps for Each Channel (Separate figure per channel)
    # ========================================================================
    step_names_short = {
        'Raw': 'Raw',
        'After Notch Filter (60Hz)': 'Notch (60 Hz)',
        'After Bandpass (20-450Hz)': 'Bandpass (20-450 Hz)',
        'After Baseline Removal': 'Baseline Removal',
        'After Normalization': 'Normalized'
    }
    
    for ch in range(4):
        fig, ax = plt.subplots(figsize=(12, 7))
        step_idx = 0
        for step_name in ['Raw', 'After Notch Filter (60Hz)', 'After Bandpass (20-450Hz)', 
                          'After Baseline Removal', 'After Normalization']:
            if step_name in emg_steps:
                y_data = emg_steps[step_name][:, ch]
                # Normalize for visualization (to see differences)
                if step_name != 'After Normalization':
                    y_data = (y_data - np.mean(y_data)) / (np.std(y_data) + 1e-8)
                ax.plot(time_axis, y_data + step_idx * 3.5, 
                        label=step_names_short[step_name], linewidth=2, alpha=0.8)
                step_idx += 1
        ax.set_title(f'EMG Channel {ch+1} - Preprocessing Pipeline', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Normalized Amplitude', fontsize=12)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.2, linestyle='--')
        plt.tight_layout()
        save_path = create_figure_name(window_num, idx, condition, subject, f'EMG_ch{ch+1}_steps', output_dir)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        saved_files.append(save_path)
        plt.close(fig)
    
    # ========================================================================
    # 4. MIC: Raw Signal
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, mic_steps['Raw'], 
            label='MIC', color='purple', linewidth=2, alpha=0.8)
    ax.set_title(f'Microphone Raw Signal', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    save_path = create_figure_name(window_num, idx, condition, subject, 'MIC_raw', output_dir)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    saved_files.append(save_path)
    plt.close(fig)
    
    # ========================================================================
    # 5. MIC: Final Preprocessed Signal (Training Equivalent)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_axis, mic_steps['After Normalization (Training Equivalent)'],
            label='MIC', color='purple', linewidth=2, alpha=0.8)
    ax.set_title(f'Microphone Preprocessed Signal', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    save_path = create_figure_name(window_num, idx, condition, subject, 'MIC_preprocessed', output_dir)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    saved_files.append(save_path)
    plt.close(fig)
    
    # ========================================================================
    # 6. MIC: Preprocessing Steps (Training Only)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    # Only show the actual training steps: Raw, DC Offset Removal, Normalized
    mic_plot_steps = [
        ('Raw', 'Raw', 'purple', '-', 0.8),
        ('After DC Offset Removal', 'DC Offset Removal', 'green', '-', 0.9),
        ('After Normalization (Training Equivalent)', 'Normalized', 'green', '-', 0.9)
    ]
    
    for i, (step_key, step_label, color, linestyle, alpha) in enumerate(mic_plot_steps):
        if step_key in mic_steps:
            y_data = mic_steps[step_key]
            # Normalize for visualization (to see differences)
            if 'Normalization' not in step_key:
                y_data = (y_data - np.mean(y_data)) / (np.std(y_data) + 1e-8)
            ax.plot(time_axis, y_data + i * 3.5,
                    label=step_label, linewidth=2, alpha=alpha, color=color, linestyle=linestyle)
    
    ax.set_title(f'Microphone Preprocessing Pipeline', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    save_path = create_figure_name(window_num, idx, condition, subject, 'MIC_steps', output_dir)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    saved_files.append(save_path)
    plt.close(fig)
    
    return saved_files


def main():
    """
    Main function to run sanity check visualization.
    
    Loads random windows from dataset, applies preprocessing steps with aggressive
    filters for visualization, and generates figures showing raw vs preprocessed signals.
    Matches preprocessing methodology from run_new_wavelet_training.py but uses more
    aggressive filter parameters to make effects more visible.
    """
    print("="*70)
    print("SANITY CHECK: Raw vs Preprocessed Signal Windows")
    print("="*70)
    print("\nThis script uses AGGRESSIVE filters for better visualization:")
    print(f"  - EMG: Aggressive filters (higher order, sharper cutoffs) for visibility")
    print(f"  - EMG ICA: {USE_ICA_EMG}")
    print(f"  - MIC: Shows training steps (DC offset only) + visualization filters")
    print(f"  - NOTE: Training script uses standard filters; this uses aggressive ones")
    print()
    
    # Create output directory
    output_dir = Path("sanity_check_figures")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Data path
    data_root = r"C:\Users\mhfar\Desktop\Depo\Brusxism_data"
    
    # Load dataset without normalization to access raw signals
    print("\n1. Loading dataset (raw, no normalization)...")
    dataset = BruxismDataset(
        data_root=data_root,
        window_size=1200,
        stride=600,
        normalize=False,
        skip_initial_seconds=3.0
    )
    
    print(f"   Loaded {len(dataset)} windows")
    
    # Select random windows from middle portion to avoid initialization artifacts
    num_windows_to_show = 5
    start_idx = len(dataset) // 4
    end_idx = 3 * len(dataset) // 4
    
    print(f"\n2. Selecting {num_windows_to_show} random windows from middle part of dataset...")
    print(f"   Selecting from indices {start_idx} to {end_idx} (middle 50%)")
    
    random_indices = random.sample(range(start_idx, end_idx), 
                                   min(num_windows_to_show, end_idx - start_idx))
    
    print(f"   Selected indices: {random_indices}")
    
    all_saved_files = []
    
    # Process each window
    for i, idx in enumerate(random_indices):
        print(f"\n3. Processing window {i+1}/{len(random_indices)} (index {idx})...")
        
        # Extract raw signals from dataset
        emg_raw, mic_raw, sample_info = get_raw_data_from_dataset(dataset, idx)
        
        print(f"   Condition: {sample_info['condition']}")
        print(f"   Subject: {sample_info['subject']}")
        print(f"   File: {sample_info['file']}")
        print(f"   EMG shape: {emg_raw.shape}, MIC shape: {mic_raw.shape}")
        
        # Apply preprocessing steps with aggressive filters for visualization
        print(f"   Applying EMG preprocessing steps (ICA={USE_ICA_EMG})...")
        emg_steps = apply_preprocessing_steps_emg(emg_raw, fs=1200, apply_ica_flag=USE_ICA_EMG)
        
        print("   Applying MIC preprocessing steps (DC offset removal only)...")
        mic_steps = apply_preprocessing_steps_mic(mic_raw, fs=1200)
        
        # Generate visualization figures
        print(f"   Creating visualizations...")
        saved_files = visualize_window(emg_raw, mic_raw, emg_steps, mic_steps, 
                                      sample_info, i+1, idx, str(output_dir))
        all_saved_files.extend(saved_files)
        print(f"   Created {len(saved_files)} figures for this window")
    
    print("\n" + "="*70)
    print("SANITY CHECK COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  ✓ Loaded raw data windows from middle parts of recordings")
    print(f"  ✓ Applied AGGRESSIVE preprocessing steps for VISUALIZATION:")
    print(f"    - EMG: Aggressive notch (60Hz, Q=60) → Aggressive bandpass (20-450Hz, order=8)")
    print(f"           → Aggressive baseline removal (3Hz, order=6) → ", end="")
    if USE_ICA_EMG:
        print(f"ICA → Normalization")
    else:
        print(f"Normalization (ICA disabled)")
    print(f"      NOTE: More aggressive than training for better visualization")
    print(f"    - MIC: DC offset removal (training) + visualization filters:")
    print(f"           → Baseline removal → Bandpass (50-550Hz) → Notch (60Hz)")
    print(f"      NOTE: Training uses DC offset only; filters shown for visualization")
    print(f"  ✓ Visualized raw vs preprocessed signals with enhanced visibility")
    print(f"  ✓ Generated {len(all_saved_files)} separate figure files")
    print(f"  ✓ All figures saved to: {output_dir.absolute()}")
    print(f"\nFigures per window:")
    print(f"  - EMG raw (all 4 channels)")
    print(f"  - EMG preprocessed (all 4 channels)")
    print(f"  - EMG preprocessing steps (1 figure per channel, 4 total)")
    print(f"  - MIC raw")
    print(f"  - MIC preprocessed")
    print(f"  - MIC preprocessing steps")
    print(f"  Total: 8 figures per window × {num_windows_to_show} windows = {len(all_saved_files)} figures")


if __name__ == "__main__":
    main()

