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
    emg_stages,
    mains_harmonics,
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


def _band_power(signal: np.ndarray, freq_hz: float, *, half_width_hz: float = 3.0) -> float:
    """Power within +/- ``half_width_hz`` of ``freq_hz``, from a Welch PSD."""
    freqs, psd = welch(np.asarray(signal, dtype=np.float64).ravel(), fs=FS, nperseg=4096)
    band = (freqs >= freq_hz - half_width_hz) & (freqs <= freq_hz + half_width_hz)
    return float(np.trapezoid(psd[band], freqs[band]))


@pytest.mark.parametrize("harmonic_hz", [120.0, 180.0, 240.0, 300.0, 360.0, 420.0])
def test_production_chain_attenuates_every_mains_harmonic(rng, harmonic_hz):
    """Every mains harmonic inside the EMG passband must be attenuated by >= 20 dB.

    This is the regression test for the defect diagnosed in ``cause.md``: the acquisition
    hardware already removed the 60 Hz fundamental (``notch_filter: Index 9`` in every
    metadata sidecar), so the odd harmonics at 180/300/420 Hz -- measured at 37,000x to
    846,000x the local noise floor -- were the entire interference, and a chain that
    notched only 60 Hz passed them straight through. It fails against the pre-2026-08-03
    ``_default_emg_stages``.
    """
    t = np.arange(24_000) / FS
    broadband = rng.standard_normal(t.size)
    interference = 50.0 * np.sin(2 * np.pi * harmonic_hz * t)
    signal = broadband + interference

    filtered = apply_filter_chain(signal, FilterChainConfig(), FS, modality="emg")

    attenuation_db = 10 * np.log10(
        _band_power(signal, harmonic_hz) / _band_power(filtered, harmonic_hz)
    )
    assert attenuation_db >= 20.0, (
        f"{harmonic_hz:g} Hz attenuated by only {attenuation_db:.1f} dB; the production EMG "
        f"chain must remove every mains harmonic inside its passband"
    )


def test_production_chain_preserves_emg_between_the_harmonics(rng):
    """The notch bank must not cost the broadband EMG it is protecting.

    A 90 Hz tone sits midway between two harmonics; it must survive essentially intact, or
    the cure removes the signal along with the interference.
    """
    t = np.arange(24_000) / FS
    signal = rng.standard_normal(t.size) + 50.0 * np.sin(2 * np.pi * 90.0 * t)
    filtered = apply_filter_chain(signal, FilterChainConfig(), FS, modality="emg")
    loss_db = 10 * np.log10(_band_power(signal, 90.0) / _band_power(filtered, 90.0))
    assert abs(loss_db) < 1.0, f"90 Hz lost {loss_db:.2f} dB; the notches are too wide"


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
    assert "highpass" not in kinds
    assert kinds[-1] == "bandpass", "the bandpass runs last, after the mains notches"
    assert all(s.rationale for s in FilterChainConfig().emg_stages)


def test_default_chain_notches_every_mains_multiple_in_the_passband():
    """60, 120, 180, 240, 300, 360, 420 Hz -- not just the fundamental."""
    stages = FilterChainConfig().emg_stages
    notched = [s.freq_hz for s in stages if s.kind == "notch"]
    assert notched == [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0]
    bandpass = next(s for s in stages if s.kind == "bandpass")
    assert (bandpass.low_hz, bandpass.high_hz) == (20.0, 450.0)
    # 480 Hz is a mains multiple but sits above the passband, so it is not notched.
    assert 480.0 not in notched


def test_mains_harmonics_enumeration():
    assert mains_harmonics(60.0, 1200.0, low_hz=20.0, high_hz=450.0) == (
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
    )
    # Nyquist bounds the list even when no upper edge is given.
    assert mains_harmonics(60.0, 1200.0)[-1] == 540.0
    # A 50 Hz supply is expressible without touching code.
    assert mains_harmonics(50.0, 1200.0, low_hz=20.0, high_hz=450.0)[:3] == (50.0, 100.0, 150.0)


def test_notches_are_constant_width_not_constant_q():
    """All seven notches are equally wide in Hz, because the interference spread is.

    Constant Q would give 2 Hz at 60 Hz and 14 Hz at 420 Hz -- narrowest exactly where the
    hardware notch left a residue, widest where nothing survives.
    """
    notches = [s for s in FilterChainConfig().emg_stages if s.kind == "notch"]
    widths = [s.freq_hz / s.quality for s in notches]
    assert all(width == pytest.approx(widths[0]) for width in widths)
    assert widths[0] == pytest.approx(8.0)

    constant_q = [s for s in emg_stages(notch_bandwidth_hz=None, quality=30.0) if s.kind == "notch"]
    q_widths = [s.freq_hz / s.quality for s in constant_q]
    assert q_widths[0] == pytest.approx(2.0) and q_widths[-1] == pytest.approx(14.0)


def test_wide_fundamental_notch_still_leaves_the_emg_band_usable():
    """An 8 Hz notch at 60 Hz must not swallow the EMG either side of it."""
    t = np.arange(24_000) / FS
    for tone_hz in (45.0, 80.0):
        signal = rng_signal(t) + 50.0 * np.sin(2 * np.pi * tone_hz * t)
        filtered = apply_filter_chain(signal, FilterChainConfig(), FS, modality="emg")
        loss_db = 10 * np.log10(_band_power(signal, tone_hz) / _band_power(filtered, tone_hz))
        assert abs(loss_db) < 1.0, f"{tone_hz:g} Hz lost {loss_db:.2f} dB"


def rng_signal(t: np.ndarray) -> np.ndarray:
    """Deterministic broadband carrier for the notch-width tests."""
    return np.random.default_rng(0).standard_normal(t.size)


def test_the_superseded_single_notch_chain_is_still_expressible():
    """The pre-2026-08-03 chain must remain configurable, for regression comparisons."""
    stages = emg_stages(notch_harmonics=False)
    assert [s.freq_hz for s in stages if s.kind == "notch"] == [60.0]


@pytest.mark.parametrize("variant", ["notch_bank", "comb", "spectral_interpolation"])
def test_every_mains_removal_variant_suppresses_180hz(rng, variant):
    """The three candidate strategies are interchangeable at the interface."""
    t = np.arange(24_000) / FS
    signal = rng.standard_normal(t.size) + 50.0 * np.sin(2 * np.pi * 180.0 * t)
    bandpass = FilterStage(kind="bandpass", low_hz=20.0, high_hz=450.0, order=4, rationale="test")
    if variant == "notch_bank":
        stages = tuple(emg_stages())
    elif variant == "comb":
        stages = (
            FilterStage(kind="comb", freq_hz=60.0, quality=30.0, rationale="test"),
            bandpass,
        )
    else:
        stages = (
            FilterStage(
                kind="spectral_interpolation",
                freq_hz=60.0,
                low_hz=20.0,
                high_hz=450.0,
                half_width_hz=1.5,
                rationale="test",
            ),
            bandpass,
        )
    config = FilterChainConfig(emg_stages=stages, mic_stages=())
    filtered = apply_filter_chain(signal, config, FS, modality="emg")
    attenuation_db = 10 * np.log10(_band_power(signal, 180.0) / _band_power(filtered, 180.0))
    assert attenuation_db >= 20.0, f"{variant} attenuated 180 Hz by only {attenuation_db:.1f} dB"


def test_spectral_interpolation_is_declared_acausal_even_without_zero_phase():
    """It reads the whole recording, so no lfilter setting makes it streamable."""
    stage = FilterStage(
        kind="spectral_interpolation", freq_hz=60.0, low_hz=20.0, high_hz=450.0, rationale="t"
    )
    config = FilterChainConfig(emg_stages=(stage,), mic_stages=(), zero_phase=False)
    assert config.is_causal is False
    assert config.realtime_claim_supported is False


def test_spectral_interpolation_refuses_a_single_window():
    """A 1 s window resolves 1 Hz, too coarse to isolate a +/-1.5 Hz band."""
    stage = FilterStage(
        kind="spectral_interpolation", freq_hz=60.0, low_hz=20.0, high_hz=450.0, rationale="t"
    )
    config = FilterChainConfig(emg_stages=(stage,), mic_stages=())
    with pytest.raises(FilterDesignError, match="continuous recording"):
        apply_filter_chain(np.zeros(1200), config, FS, modality="emg")


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
