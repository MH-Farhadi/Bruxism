"""Filters, wavelets and features -- checked against independent references."""

from __future__ import annotations

import numpy as np
import pytest
import pywt
from scipy.signal import butter, sosfiltfilt, welch

from bruxism.features.time_frequency import (
    FeatureConfig,
    FeatureExtractor,
    median_frequency,
    wavelet_packet_energies,
    zero_crossing_rate,
)
from bruxism.preprocessing.filters import (
    FilterChainConfig,
    FilterDesignError,
    FilterStage,
    apply_filter_chain,
    design_bandpass,
    validate_cutoffs,
)
from bruxism.preprocessing.wavelets import (
    WaveletConfig,
    WaveletError,
    band_frequencies,
    band_index,
    decompose,
)

FS = 1200.0


# ------------------------------------------------------------------- filters ---


def test_bandpass_matches_a_direct_scipy_reference(rng):
    signal = rng.standard_normal((6000, 2))
    config = FilterChainConfig(
        emg_stages=(
            FilterStage(kind="bandpass", low_hz=20.0, high_hz=450.0, order=4, rationale="test"),
        ),
        mic_stages=(),
    )
    produced = apply_filter_chain(signal, config, FS, modality="emg")
    reference = sosfiltfilt(
        butter(4, [20.0 / (FS / 2), 450.0 / (FS / 2)], btype="band", output="sos"),
        signal,
        axis=0,
    )
    assert np.allclose(produced, reference)


@pytest.mark.parametrize(
    "cutoffs",
    [600.0, 700.0, 0.0, -5.0, (450.0, 20.0), (20.0, 600.0), float("nan")],
)
def test_invalid_cutoffs_are_rejected(cutoffs):
    with pytest.raises(FilterDesignError):
        validate_cutoffs(cutoffs, FS)


def test_nyquist_boundary_is_rejected_not_clipped():
    with pytest.raises(FilterDesignError, match="Nyquist"):
        design_bandpass(20.0, FS / 2, FS)


def test_filtering_a_single_window_is_refused_as_too_short():
    """Zero-phase filtering needs the continuous recording, not a 1 s window."""
    config = FilterChainConfig()
    with pytest.raises(FilterDesignError, match="too short for"):
        apply_filter_chain(np.zeros((20, 4)), config, FS, modality="emg")


def test_filtering_per_window_corrupts_the_window(rng):
    """Filtering each window separately is materially different from the pipeline order.

    The pipeline filters the continuous recording and then cuts windows. Filtering a
    1 s window in isolation produces enormous edge transients -- several times the signal
    standard deviation in the first samples -- and, because the 60 Hz notch is narrow
    (Q=30) and therefore long in time, a residual error across the whole window.
    """
    recording = rng.standard_normal((12000, 4))
    config = FilterChainConfig()
    continuous = apply_filter_chain(recording, config, FS, modality="emg")[3000:4200]
    per_window = apply_filter_chain(recording[3000:4200], config, FS, modality="emg")

    scale = float(np.abs(continuous).std())
    error = np.abs(continuous - per_window).max(axis=1) / scale

    # The leading edge is not approximately right -- it is wrong by more than the signal.
    assert error[:50].max() > 1.0
    assert error[-50:].max() > 0.5
    # The centre is much better but still not clean, because the notch rings for a long time.
    assert error[200:600].mean() < 0.1
    assert error[:50].mean() > 5 * error[200:600].mean()


def test_zero_phase_is_declared_as_not_supporting_realtime():
    assert FilterChainConfig(zero_phase=True).realtime_claim_supported is False
    assert FilterChainConfig(zero_phase=False).realtime_claim_supported is True
    assert "acausal" in FilterChainConfig().describe()["mode"]


def test_filter_config_roundtrips():
    original = FilterChainConfig()
    assert FilterChainConfig.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_default_chain_has_no_redundant_lowfrequency_highpass():
    """The prototype's 5 Hz high-pass after a 20 Hz bandpass was a no-op; it is gone."""
    kinds = [s.kind for s in FilterChainConfig().emg_stages]
    assert kinds == ["notch", "bandpass"]
    assert all(s.rationale for s in FilterChainConfig().emg_stages)


# ------------------------------------------------------------------ wavelets ---


@pytest.mark.parametrize(
    ("band", "level", "index"),
    [
        ("A4", 4, 0),
        ("D4", 4, 1),
        ("D3", 4, 2),
        ("D2", 4, 3),
        ("D1", 4, 4),
        ("A5", 5, 0),
        ("D1", 5, 5),
    ],
)
def test_band_index_matches_pywt_ordering(band, level, index):
    assert band_index(band, level) == index


def test_pywt_coefficient_order_is_what_we_assume(rng):
    """pywt.wavedec returns [cA_L, cD_L, ..., cD_1]; positional indexing is a trap."""
    signal = rng.standard_normal(1200)
    coefficients = pywt.wavedec(signal, "db4", level=4)
    selected = decompose(signal, WaveletConfig("db4", 4, ("A4", "D4", "D3", "D2", "D1")))
    for position, band in enumerate(("A4", "D4", "D3", "D2", "D1")):
        assert np.allclose(selected[band], coefficients[position])


@pytest.mark.parametrize(
    ("band", "low", "high"),
    [
        ("A4", 0.0, 37.5),
        ("D4", 37.5, 75.0),
        ("D3", 75.0, 150.0),
        ("D2", 150.0, 300.0),
        ("D1", 300.0, 600.0),
    ],
)
def test_nominal_band_frequencies_at_1200hz(band, low, high):
    assert band_frequencies(band, 4, FS) == (low, high)


@pytest.mark.parametrize(
    ("tone_hz", "expected_band"),
    [(10.0, "A4"), (50.0, "D4"), (100.0, "D3"), (200.0, "D2"), (450.0, "D1")],
)
def test_a_pure_tone_lands_in_its_nominal_band(tone_hz, expected_band):
    t = np.arange(4800) / FS
    config = WaveletConfig("db4", 4, ("A4", "D4", "D3", "D2", "D1"))
    coefficients = decompose(np.sin(2 * np.pi * tone_hz * t), config)
    energies = {b: float(np.sum(coefficients[b] ** 2)) for b in config.bands}
    assert max(energies, key=energies.get) == expected_band


def test_the_prototypes_positional_assumptions_were_wrong():
    """details[0] is the LOWEST-frequency detail, and details[2] at level 4 is D2."""
    level = 4
    details_index_0 = level - band_index("D4", level) + 1  # -> D4
    assert band_index("D4", level) == 1, "details[0] == coeffs[1] == D4, not D1"
    assert band_index("D2", level) == 3, "details[2] == coeffs[3] == D2, not D3"
    assert band_frequencies("D4", level, FS) == (37.5, 75.0)
    assert details_index_0 == 4


def test_continuous_wavelet_name_is_rejected_for_wavedec():
    with pytest.raises(WaveletError, match="not a discrete wavelet"):
        WaveletConfig(wavelet="morl", level=3, bands=("A3",))


def test_band_outside_the_decomposition_depth_is_rejected():
    with pytest.raises(WaveletError, match="outside a level-3"):
        WaveletConfig(wavelet="db4", level=3, bands=("D5",))
    with pytest.raises(WaveletError, match="does not exist"):
        WaveletConfig(wavelet="db4", level=3, bands=("A4",))


def test_non_finite_input_raises_instead_of_being_zero_filled():
    bad = np.full(1200, np.nan)
    with pytest.raises(WaveletError):
        decompose(bad, WaveletConfig("db4", 4, ("A4",)))


# ------------------------------------------------------------------ features ---


def test_median_frequency_on_a_known_two_tone_spectrum():
    """Equal power at 100 Hz and 300 Hz puts the spectral median between them."""
    t = np.arange(12000) / FS
    signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t)
    value = median_frequency(signal, FS, nperseg=2048)
    assert 100.0 < value < 300.0

    # A single tone puts the median at that tone.
    single = median_frequency(np.sin(2 * np.pi * 150 * t), FS, nperseg=2048)
    assert single == pytest.approx(150.0, abs=8.0)


def test_median_frequency_agrees_with_a_direct_welch_computation(rng):
    signal = rng.standard_normal(8192)
    value = median_frequency(signal, FS, nperseg=256)
    freqs, psd = welch(signal, fs=FS, nperseg=256)
    total = np.trapezoid(psd, freqs)
    below = np.trapezoid(psd[freqs <= value], freqs[freqs <= value])
    assert below == pytest.approx(total / 2, rel=0.05)


def test_median_frequency_of_a_silent_signal_raises_rather_than_returning_zero():
    with pytest.raises(ValueError, match="zero total power"):
        median_frequency(np.zeros(2048), FS)


def test_zero_crossing_rate_of_a_known_signal():
    assert zero_crossing_rate(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(1.0)
    assert zero_crossing_rate(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_wavelet_packet_uses_every_node_at_the_requested_depth(rng):
    energies = wavelet_packet_energies(rng.standard_normal(2048), max_level=3)
    assert len(energies) == 2**3, "the prototype only read the two first-level nodes"
    assert sum(energies.values()) == pytest.approx(1.0)


def test_wavelet_packet_of_a_degenerate_window_raises():
    with pytest.raises(ValueError, match="zero energy"):
        wavelet_packet_energies(np.zeros(1024), max_level=3)


def test_feature_extraction_is_ordered_named_and_finite(rng):
    extractor = FeatureExtractor(FeatureConfig(sampling_rate=FS))
    emg, mic = rng.standard_normal((1200, 4)), rng.standard_normal(1200)
    first = extractor.extract(emg, mic)
    names = extractor.feature_names
    assert len(first) == len(names) == len(set(names))
    assert np.isfinite(first).all()
    second = extractor.extract(rng.standard_normal((1200, 4)), rng.standard_normal(1200))
    assert len(second) == len(first)


def test_feature_extraction_raises_on_a_degenerate_window_rather_than_zero_filling():
    extractor = FeatureExtractor(FeatureConfig(sampling_rate=FS))
    with pytest.raises(ValueError):
        extractor.extract(np.zeros((1200, 4)), np.zeros(1200))
