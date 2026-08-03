"""Figures about the *data* a run consumed: inventory, segmentation, filtering, wavelets.

Everything here is derived by calling the production code paths -- the same
:func:`~bruxism.preprocessing.filters.apply_filter_chain`, the same
:class:`~bruxism.data.dataset.RecordingCache`, the same
:class:`~bruxism.data.segments.WindowIndex` and the same
:func:`~bruxism.preprocessing.wavelets.decompose` that training used. A figure here
therefore cannot drift from what the model actually saw; if the filter chain changes, the
figure changes with it.

These plots need the raw data root, so they are produced at the end of a training run (where
the manifest, window index and filtered-recording cache are already in memory) or by
``bruxism-figures`` for a finished run.

Signal amplitudes are always labelled ``arbitrary_adc_units``: the acquisition chain never
documented a physical calibration, so no axis in this project says uV, Pa or dB SPL.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bruxism.data.labels import ClassificationTask, TaskFamily  # noqa: E402
from bruxism.data.manifest import DatasetManifest, RecordingRecord  # noqa: E402
from bruxism.data.schema import EMG_COLUMNS, EMG_MUSCLE_MAP, SIGNAL_UNITS  # noqa: E402
from bruxism.data.segments import (  # noqa: E402
    SegmentationConfig,
    WindowIndex,
    WindowRecord,
    segments_for_recording,
)
from bruxism.preprocessing.augmentation import AugmentationConfig, Augmenter  # noqa: E402
from bruxism.preprocessing.filters import FilterChainConfig, apply_filter_chain  # noqa: E402
from bruxism.preprocessing.wavelets import WaveletConfig, band_frequencies, decompose  # noqa: E402
from bruxism.utils.logging import get_logger  # noqa: E402
from bruxism.visualization.paper_figures import FigureStyle, caveat, save_figure  # noqa: E402

__all__ = [
    "plot_augmentation_examples",
    "plot_class_spectra",
    "plot_dataset_inventory",
    "plot_example_windows",
    "plot_filter_response",
    "plot_preprocessing_stages",
    "plot_segmentation_timeline",
    "plot_wavelet_band_energy",
    "plot_wavelet_bands",
    "plot_window_inventory",
    "sample_windows_by_class",
]

logger = get_logger(__name__)

#: Colour of the microphone trace everywhere, so audio is recognisable at a glance.
MIC_COLOR = "#444444"


# --------------------------------------------------------------------- helpers ---


def emg_channel_labels(channel_indices: Sequence[int]) -> list[str]:
    """Column name plus its tentative muscle site, for axis labels.

    The muscle map is tentative (``Data/README.txt``); the column name is authoritative,
    so both are shown and the caption says which is which.
    """
    labels: list[str] = []
    for index in channel_indices:
        column = EMG_COLUMNS[index]
        labels.append(f"{column}\n({EMG_MUSCLE_MAP.get(column, 'site unconfirmed')})")
    return labels


def sample_windows_by_class(
    window_index: WindowIndex,
    task: ClassificationTask,
    *,
    max_per_class: int = 150,
) -> dict[int, list[WindowRecord]]:
    """Deterministic, evenly spaced sample of windows per class label.

    No RNG is involved: windows are sorted by ``sample_id`` and thinned with
    :func:`numpy.linspace`, so the same window index always yields the same examples and a
    regenerated figure is byte-comparable.
    """
    grouped: dict[int, list[WindowRecord]] = {}
    for window in sorted(window_index.windows, key=lambda w: w.sample_id):
        label = task.label_for_family(TaskFamily(window.task_family))
        if label is None:
            continue  # family excluded from this task
        grouped.setdefault(label, []).append(window)

    sampled: dict[int, list[WindowRecord]] = {}
    for label, windows in sorted(grouped.items()):
        if len(windows) > max_per_class:
            positions = np.linspace(0, len(windows) - 1, max_per_class).astype(int)
            sampled[label] = [windows[i] for i in positions]
        else:
            sampled[label] = list(windows)
    return sampled


def _load_windows(cache: Any, windows: Sequence[WindowRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``(n_windows, n_samples, n_channels)`` EMG and ``(n_windows, n_samples)`` mic."""
    emg_blocks, mic_blocks = [], []
    for window in windows:
        emg, mic = cache.window(window)
        emg_blocks.append(emg)
        mic_blocks.append(mic)
    if not emg_blocks:
        raise ValueError("no windows to load")
    return np.stack(emg_blocks), np.stack(mic_blocks)


def _median_energy_window(
    cache: Any, windows: Sequence[WindowRecord], *, probe: int = 40
) -> WindowRecord:
    """The window whose EMG energy is the median of a probe subset.

    A median-energy example is representative; the loudest window would flatter the class
    and the quietest would misrepresent it. The choice is deterministic.
    """
    subset = list(windows)
    if len(subset) > probe:
        positions = np.linspace(0, len(subset) - 1, probe).astype(int)
        subset = [subset[i] for i in positions]
    energies = []
    for window in subset:
        emg, _ = cache.window(window)
        energies.append(float(np.mean(np.square(emg))))
    order = int(np.argsort(energies)[len(energies) // 2])
    return subset[order]


def _wrap_stage(stage: Any) -> str:
    """A filter stage's description, wrapped so it fits a narrow row label."""
    text = stage.describe()
    head, _, tail = text.partition(" ")
    return f"{head}\n{tail}" if tail else text


def _welch(signal: np.ndarray, sampling_rate: float, *, nperseg: int = 256) -> tuple[Any, Any]:
    from scipy.signal import welch

    return welch(signal, fs=sampling_rate, nperseg=min(nperseg, signal.shape[-1]), axis=-1)


def _class_colors(task: ClassificationTask) -> dict[int, str]:
    return {index: FigureStyle.color(index) for index in range(task.num_classes)}


def _pretty(name: str) -> str:
    return name.replace("_", " ")


# ----------------------------------------------------------------- inventory ---


def _counts_frame(window_index: WindowIndex, task: ClassificationTask) -> pd.DataFrame:
    """Window counts per participant and *task class* (not raw family)."""
    rows: list[dict[str, Any]] = []
    for window in window_index.windows:
        label = task.label_for_family(TaskFamily(window.task_family))
        if label is None:
            continue
        rows.append(
            {
                "subject_id": window.subject_id,
                "class": task.class_names[label],
                "class_index": label,
                "recording_id": window.recording_id,
                "condition": window.condition,
                "segment_source": window.segment_source,
            }
        )
    return pd.DataFrame(rows)


def plot_dataset_inventory(
    window_index: WindowIndex,
    task: ClassificationTask,
    output_dir: Path,
    *,
    stem: str = "01_dataset_inventory",
) -> list[Path]:
    """Analysable windows per participant and class, with the class balance beside it.

    This is the figure that tells a reader what the model was actually trained on: how many
    examples exist, how unevenly they are spread across participants, and how imbalanced the
    classes are before any weighting.
    """
    FigureStyle.apply()
    frame = _counts_frame(window_index, task)
    if frame.empty:
        raise ValueError("the window index contains no windows for this task")

    pivot = (
        frame.groupby(["subject_id", "class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(task.class_names), fill_value=0)
    )
    totals = pivot.sum(axis=0)
    grand_total = int(totals.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), width_ratios=[1.55, 1.0])
    bottom = np.zeros(len(pivot))
    for index, class_name in enumerate(task.class_names):
        values = pivot[class_name].to_numpy(dtype=float)
        axes[0].bar(
            pivot.index,
            values,
            bottom=bottom,
            width=0.62,
            label=_pretty(class_name),
            color=FigureStyle.color(index),
        )
        bottom += values
    for position, total in enumerate(bottom):
        axes[0].text(
            float(position),
            float(total + bottom.max() * 0.015),
            f"{int(total):,}",
            ha="center",
            fontsize=8,
        )
    axes[0].set_ylabel("Analysable windows")
    axes[0].set_xlabel("Participant")
    # Headroom for the legend, so it cannot land on the tallest participant's total.
    axes[0].set_ylim(0, bottom.max() * 1.42)
    axes[0].set_title(f"Windows per participant ({grand_total:,} total)")
    axes[0].legend(fontsize=7, ncols=3, loc="upper center")

    order = list(range(len(task.class_names)))
    values = totals.to_numpy(dtype=float)
    axes[1].barh(
        order,
        values,
        color=[FigureStyle.color(index) for index in order],
        height=0.62,
    )
    for index, value in zip(order, values):
        axes[1].text(
            value + values.max() * 0.02,
            float(index),
            f"{int(value):,}  ({value / max(grand_total, 1):.1%})",
            va="center",
            fontsize=8,
        )
    axes[1].set_yticks(order)
    axes[1].set_yticklabels([_pretty(name) for name in task.class_names])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, values.max() * 1.42)
    axes[1].set_xlabel("Windows")
    axes[1].set_title(f"Class balance ({task.task_id})")

    segmentation = window_index.config
    caveat(
        fig,
        f"Segmentation: {segmentation.policy}, {segmentation.window_seconds:g}s window / "
        f"{segmentation.stride_seconds:g}s stride, {segmentation.guard_seconds:g}s transition "
        f"guard, {segmentation.startup_guard_seconds:g}s startup guard. Adjacent windows "
        f"overlap by {100 * (1 - segmentation.stride_seconds / segmentation.window_seconds):.0f}% "
        "and are not independent observations; five participants supply every window here.",
    )
    return save_figure(fig, output_dir, stem)


def plot_window_inventory(
    window_index: WindowIndex,
    task: ClassificationTask,
    output_dir: Path,
    *,
    stem: str = "02_window_inventory",
) -> list[Path]:
    """Participant x class window-count heatmap, with each participant's row share.

    Reads differently from the stacked bars: it exposes cells where a participant
    contributed very few windows of a class, which is where a leave-one-subject-out fold
    can produce an undefined per-class metric.
    """
    FigureStyle.apply()
    frame = _counts_frame(window_index, task)
    if frame.empty:
        raise ValueError("the window index contains no windows for this task")
    pivot = (
        frame.groupby(["subject_id", "class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(task.class_names), fill_value=0)
    )
    counts = pivot.to_numpy(dtype=float)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.62 * len(pivot) + 3.4))
    for ax, matrix, title, fmt, vmax in (
        (axes[0], counts, "Window count", ",.0f", None),
        (axes[1], shares, "Share of that participant's windows", ".0%", 1.0),
    ):
        image = ax.imshow(
            shares, cmap="Blues", aspect="auto", vmin=0, vmax=vmax or shares.max(), origin="upper"
        )
        ax.set_xticks(range(len(task.class_names)))
        ax.set_xticklabels([_pretty(n) for n in task.class_names], rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_title(title)
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    format(matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if shares[i, j] > 0.45 else "black",
                )
    axes[0].set_ylabel("Participant")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Row share")
    fig.suptitle("Analysable windows per participant and class", y=0.99)
    caveat(
        fig,
        "Both panels are shaded by the row share so the colour means the same thing in "
        "each. A near-empty cell is a warning: that class contributes almost nothing to "
        "the fold in which this participant is held out.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------- segmentation ---


def _windows_per_segment(segment: Any, config: SegmentationConfig, sampling_rate: int) -> int:
    """How many windows the tiling arithmetic emits inside one approved segment."""
    window = config.window_samples(sampling_rate)
    stride = config.stride_samples(sampling_rate)
    if segment.n_samples < window:
        return 0
    return (segment.n_samples - window) // stride + 1


def _timeline_recording(manifest: DatasetManifest, window_index: WindowIndex) -> RecordingRecord:
    """Pick the recording that shows the segmentation policy most clearly, deterministically.

    The figure has to make two things visible at once: guards being applied at several
    trigger transitions, and windows tiling densely enough inside a run to show the 50%
    overlap. Ranking on both -- segments rich enough to host a run of windows, then number
    of trigger runs, then total windows -- avoids picking a recording that is one long run
    (no visible guards) or sixteen one-second runs (no visible tiling).
    """
    produced: dict[str, int] = {}
    for window in window_index.windows:
        produced[window.recording_id] = produced.get(window.recording_id, 0) + 1
    rate = window_index.sampling_rate_hz

    scored: list[tuple[tuple[int, int, int], str, RecordingRecord]] = []
    for record in manifest.included:
        n_windows = produced.get(record.recording_id, 0)
        if n_windows == 0:
            continue
        segments = segments_for_recording(record, window_index.config, sampling_rate=rate)
        rich = sum(
            1
            for segment in segments
            if _windows_per_segment(segment, window_index.config, rate) >= 4
        )
        # Caps keep one extreme from dominating the ranking.
        scored.append(
            ((min(rich, 6), min(record.n_trigger_runs, 12), n_windows), record.recording_id, record)
        )
    if not scored:
        raise ValueError("no included recording produced any window")
    scored.sort(key=lambda item: (tuple(-value for value in item[0]), item[1]))
    return scored[0][2]


def _envelope(signal: np.ndarray, sampling_rate: float, *, seconds: float = 0.05) -> np.ndarray:
    """Moving-RMS envelope across channels; display only, never a model input."""
    magnitude = np.sqrt(np.mean(np.square(np.asarray(signal, dtype=np.float64)), axis=1))
    width = max(int(round(seconds * sampling_rate)), 1)
    kernel = np.ones(width) / width
    return np.convolve(magnitude, kernel, mode="same")


def _draw_timeline(
    ax_signal: plt.Axes,
    ax_windows: plt.Axes,
    *,
    envelope: np.ndarray,
    sampling_rate: float,
    record: RecordingRecord,
    segments: Sequence[Any],
    windows: Sequence[WindowRecord],
    segmentation: SegmentationConfig,
    span: tuple[float, float],
    y_max: float,
    show_xlabel: bool,
) -> None:
    """Draw one time span: trigger runs, guards, the envelope and the emitted windows."""
    start_s, end_s = span
    lo = int(start_s * sampling_rate)
    hi = min(int(end_s * sampling_rate), envelope.shape[0])
    time = np.arange(lo, hi) / sampling_rate

    guard = segmentation.guard_seconds
    for run in record.trigger_run_boundaries:
        run_start = run["start_sample"] / sampling_rate
        run_end = run["end_sample"] / sampling_rate
        ax_signal.axvspan(run_start, run_end, color="#009E73", alpha=0.13, lw=0)
        for edge in (run_start, run_end):
            ax_signal.axvspan(edge - guard, edge + guard, color="#D55E00", alpha=0.16, lw=0)
    if segmentation.startup_guard_seconds > 0:
        ax_signal.axvspan(
            0.0, segmentation.startup_guard_seconds, color="#777777", alpha=0.22, lw=0
        )

    ax_signal.plot(time, envelope[lo:hi], color=FigureStyle.color(0), linewidth=0.8)
    ax_signal.set_xlim(start_s, end_s)
    # A robust ceiling: the amplifier-settling transient at sample 0 is an order of
    # magnitude larger than the task signal and would flatten every other trace.
    ax_signal.set_ylim(0, y_max)
    ax_signal.set_ylabel(f"EMG RMS\n({SIGNAL_UNITS})", fontsize=8)
    ax_signal.set_xticklabels([])

    for index, segment in enumerate(segments):
        ax_windows.axvspan(
            segment.start_sample / sampling_rate,
            segment.end_sample / sampling_rate,
            color="#009E73",
            alpha=0.10,
            lw=0,
            label="approved segment" if index == 0 else None,
        )
    drawn = 0
    for index, window in enumerate(windows):
        if window.end_seconds < start_s or window.start_seconds > end_s:
            continue
        ax_windows.plot(
            [window.start_seconds, window.end_seconds],
            [index % 6, index % 6],
            linewidth=3.4,
            solid_capstyle="butt",
            color=FigureStyle.color(2 + (index % 2)),
            alpha=0.9,
        )
        drawn += 1
    ax_windows.set_ylim(-0.8, 6.0)
    ax_windows.set_yticks([])
    ax_windows.set_xlim(start_s, end_s)
    ax_windows.set_ylabel(f"windows\n({drawn} shown)", fontsize=8)
    ax_windows.grid(False)
    if show_xlabel:
        ax_windows.set_xlabel("Time within recording (s)")


def plot_segmentation_timeline(
    manifest: DatasetManifest,
    window_index: WindowIndex,
    cache: Any,
    output_dir: Path,
    *,
    stem: str = "03_segmentation_timeline",
    recording_id: str | None = None,
) -> list[Path]:
    """How continuous recordings become labelled windows -- the methods figure.

    Shows, on one real recording: the trigger-high runs the participant marked, the guard
    interval excluded on each side of every transition, the startup guard, and the
    overlapping windows the policy actually emitted. The lower pair zooms into a single
    trigger run so the window/stride tiling is legible.
    """
    FigureStyle.apply()
    record = (
        manifest.by_id(recording_id)
        if recording_id
        else _timeline_recording(manifest, window_index)
    )
    segmentation = window_index.config
    rate = float(window_index.sampling_rate_hz)
    segments = segments_for_recording(record, segmentation, sampling_rate=int(rate))
    windows = sorted(
        (w for w in window_index.windows if w.recording_id == record.recording_id),
        key=lambda w: w.start_sample,
    )

    array = cache.get(record.recording_id)
    envelope = _envelope(np.asarray(array[:, :-1], dtype=np.float64), rate)
    duration = record.n_samples / rate
    startup = int(segmentation.startup_guard_seconds * rate)
    settled = envelope[startup:] if envelope.shape[0] > startup else envelope
    y_max = float(np.percentile(settled, 99.8)) * 1.3 or 1.0

    # Zoom on the richest approved segment: the one whose tiling shows the most windows.
    zoom = (0.0, min(duration, 12.0))
    ranked = sorted(
        segments,
        key=lambda segment: (
            -_windows_per_segment(segment, segmentation, int(rate)),
            segment.start_sample,
        ),
    )
    if ranked and _windows_per_segment(ranked[0], segmentation, int(rate)) >= 2:
        segment = ranked[0]
        pad = segmentation.guard_seconds + 0.8
        start = segment.start_sample / rate
        zoom = (
            max(0.0, start - pad),
            min(duration, start + min(segment.n_samples / rate, 7.0) + pad),
        )

    fig = plt.figure(figsize=(12.5, 8.0))
    grid = fig.add_gridspec(4, 1, height_ratios=[2.1, 1.0, 2.1, 1.0], hspace=0.45)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(4)]
    _draw_timeline(
        axes[0],
        axes[1],
        envelope=envelope,
        sampling_rate=rate,
        record=record,
        segments=segments,
        windows=windows,
        segmentation=segmentation,
        span=(0.0, duration),
        y_max=y_max,
        show_xlabel=False,
    )
    _draw_timeline(
        axes[2],
        axes[3],
        envelope=envelope,
        sampling_rate=rate,
        record=record,
        segments=segments,
        windows=windows,
        segmentation=segmentation,
        span=zoom,
        y_max=y_max,
        show_xlabel=True,
    )
    axes[0].set_title(
        f"{record.recording_id}  -- whole recording "
        f"({duration:.0f} s, {record.n_trigger_runs} trigger run(s), "
        f"{len(windows)} window(s) emitted)",
        fontsize=10,
    )
    axes[2].set_title(
        f"Zoom {zoom[0]:.1f}-{zoom[1]:.1f} s: "
        f"{segmentation.window_seconds:g} s windows at a {segmentation.stride_seconds:g} s "
        f"stride, staggered vertically to show the overlap",
        fontsize=10,
    )

    handles = [
        plt.Line2D([], [], color="#009E73", alpha=0.35, linewidth=8, label="trigger active"),
        plt.Line2D([], [], color="#D55E00", alpha=0.35, linewidth=8, label="transition guard"),
        plt.Line2D([], [], color="#777777", alpha=0.4, linewidth=8, label="startup guard"),
        plt.Line2D([], [], color=FigureStyle.color(2), linewidth=3.4, label="emitted window"),
    ]
    axes[0].legend(handles=handles, fontsize=7, ncols=4, loc="upper right")
    caveat(
        fig,
        f"Policy {segmentation.policy}: a window is emitted only when it lies wholly inside "
        f"one trigger-active interval, at least {segmentation.guard_seconds:g} s clear of "
        f"every transition and {segmentation.startup_guard_seconds:g} s clear of the "
        "recording start. Trigger semantics were confirmed by the investigator (2026-07-27). "
        "The envelope is a display-only moving RMS; the model receives the filtered samples.",
    )
    return save_figure(fig, output_dir, stem)


# ----------------------------------------------------------------- filtering ---


def plot_filter_response(
    filter_config: FilterChainConfig,
    sampling_rate: float,
    output_dir: Path,
    *,
    stem: str = "04_filter_response",
) -> list[Path]:
    """Magnitude response of every production filter stage and of the whole chain.

    The dashed curve is the response actually realised: ``sosfiltfilt`` runs the chain
    forwards and backwards, so the effective magnitude is the square of the single-pass
    response and the phase is exactly zero -- which is also why the chain is acausal and
    supports no streaming claim.
    """
    from scipy.signal import sosfreqz

    FigureStyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    nyquist = sampling_rate / 2.0

    for ax, stages, title in (
        (axes[0], filter_config.emg_stages, "EMG chain"),
        (axes[1], filter_config.mic_stages, "Microphone chain"),
    ):
        if not stages:
            ax.text(0.5, 0.5, "no stages configured", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        combined: np.ndarray | None = None
        frequencies = np.array([])
        for index, stage in enumerate(stages):
            sos = stage.design(sampling_rate)
            worN = np.geomspace(0.5, nyquist * 0.999, 2048)
            frequencies, response = sosfreqz(sos, worN=worN, fs=sampling_rate)
            magnitude = np.abs(response)
            combined = magnitude if combined is None else combined * magnitude
            ax.semilogx(
                frequencies,
                20 * np.log10(np.maximum(magnitude, 1e-6)),
                color=FigureStyle.color(index),
                linewidth=1.2,
                label=stage.describe(),
            )
        assert combined is not None
        ax.semilogx(
            frequencies,
            20 * np.log10(np.maximum(combined, 1e-6)),
            color="#000000",
            linewidth=1.6,
            label="chain (single pass)",
        )
        if filter_config.zero_phase:
            ax.semilogx(
                frequencies,
                40 * np.log10(np.maximum(combined, 1e-6)),
                color="#000000",
                linewidth=1.2,
                linestyle="--",
                label="chain as applied (filtfilt, forward+reverse)",
            )
        ax.axhline(-3.0, color="#999999", linewidth=0.8, linestyle=":")
        ax.text(0.55, -3.0, " -3 dB", fontsize=6.5, color="#666666", va="bottom")
        ax.set_xlim(0.5, nyquist)
        ax.set_ylim(-80, 8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title)
        # Lower right: every chain here is a high-pass or band-pass, so that corner is the
        # one region no curve occupies.
        ax.legend(fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("Magnitude (dB)")

    mode = "zero-phase (filtfilt, acausal)" if filter_config.zero_phase else "causal (lfilter)"
    fig.suptitle(f"Production filter chain at {sampling_rate:g} Hz -- {mode}", y=0.99)
    caveat(
        fig,
        "Designed with scipy second-order sections and applied to the continuous recording "
        "before any window is cut. "
        + (
            "Zero-phase filtering reads samples from the future of each output sample: it "
            "cannot support a real-time, streaming or wearable latency claim."
            if filter_config.zero_phase
            else "This chain is causal and could run on a live stream."
        ),
    )
    return save_figure(fig, output_dir, stem)


def plot_preprocessing_stages(
    manifest: DatasetManifest,
    window_index: WindowIndex,
    output_dir: Path,
    *,
    filter_config: FilterChainConfig,
    stem: str = "05_preprocessing_stages",
    recording_id: str | None = None,
    seconds: float = 3.0,
    emg_channel: int = 0,
) -> list[Path]:
    """Raw -> notch -> bandpass on a real excerpt, with the spectrum beside each stage.

    Each stage is produced by re-running the production chain truncated after that stage,
    on the *continuous* recording, then slicing the excerpt -- the same ordering training
    uses. Filtering per window instead would add an edge transient to every example.
    """
    from bruxism.data.schema import MIC_COLUMN, read_recording_csv

    FigureStyle.apply()
    record = (
        manifest.by_id(recording_id)
        if recording_id
        else _timeline_recording(manifest, window_index)
    )
    rate = float(manifest.sampling_rate_hz)
    frame = read_recording_csv(manifest.csv_path(record))
    raw_emg = frame[list(EMG_COLUMNS)].to_numpy(dtype=np.float64)
    raw_mic = frame[MIC_COLUMN].to_numpy(dtype=np.float64)

    # Excerpt: start of the first approved segment, so the trace shows real task activity.
    segments = segments_for_recording(record, window_index.config, sampling_rate=int(rate))
    start = int(segments[0].start_sample) if segments else int(rate)
    stop = min(start + int(seconds * rate), raw_emg.shape[0])
    time = np.arange(start, stop) / rate

    emg_stack: list[tuple[str, np.ndarray]] = [("EMG\nraw", raw_emg[:, emg_channel])]
    for count in range(1, len(filter_config.emg_stages) + 1):
        partial = FilterChainConfig(
            emg_stages=filter_config.emg_stages[:count],
            mic_stages=filter_config.mic_stages,
            zero_phase=filter_config.zero_phase,
            padlen=filter_config.padlen,
        )
        filtered = apply_filter_chain(raw_emg, partial, rate, modality="emg")
        emg_stack.append(
            (f"+ {_wrap_stage(filter_config.emg_stages[count - 1])}", filtered[:, emg_channel])
        )
    mic_stack: list[tuple[str, np.ndarray]] = [("MIC\nraw", raw_mic)]
    if filter_config.mic_stages:
        mic_stack.append(
            (
                f"+ {_wrap_stage(filter_config.mic_stages[-1])}",
                apply_filter_chain(raw_mic, filter_config, rate, modality="mic"),
            )
        )

    n_rows = len(emg_stack) + len(mic_stack)
    fig = plt.figure(figsize=(12.5, 1.35 * n_rows + 1.4))
    grid = fig.add_gridspec(n_rows, 2, width_ratios=[1.6, 1.0], hspace=0.5, wspace=0.22)
    split = len(emg_stack)

    for row, (label, signal) in enumerate([*emg_stack, *mic_stack]):
        ax = fig.add_subplot(grid[row, 0])
        colour = FigureStyle.color(row) if row < split else MIC_COLOR
        ax.plot(time, signal[start:stop], linewidth=0.6, color=colour)
        ax.set_ylabel(label, fontsize=7.5)
        if row != n_rows - 1:
            ax.set_xticklabels([])
    ax.set_xlabel("Time within recording (s)")

    ax_emg = fig.add_subplot(grid[0:split, 1])
    for index, (label, signal) in enumerate(emg_stack):
        freqs, power = _welch(signal[start:stop], rate)
        ax_emg.semilogy(
            freqs,
            np.maximum(power, 1e-12),
            linewidth=1.0,
            label=label,
            color=FigureStyle.color(index),
        )
    ax_emg.axvline(60.0, color="#D55E00", linewidth=0.9, linestyle=":")
    ax_emg.text(62, ax_emg.get_ylim()[1], " 60 Hz", fontsize=6.5, color="#D55E00", va="top")
    ax_emg.set_xlim(0, rate / 2)
    ax_emg.set_title("EMG spectrum per stage", fontsize=9)
    ax_emg.set_ylabel(f"PSD ({SIGNAL_UNITS}$^2$/Hz)", fontsize=8)
    # The frequency axis is shared with the microphone panel below; dropping these labels
    # keeps them from colliding with that panel's title.
    ax_emg.set_xticklabels([])
    ax_emg.legend(fontsize=6.5)

    ax_mic = fig.add_subplot(grid[split:, 1])
    for index, (label, signal) in enumerate(mic_stack):
        freqs, power = _welch(signal[start:stop], rate)
        ax_mic.semilogy(
            freqs,
            np.maximum(power, 1e-12),
            linewidth=1.0,
            label=label,
            color=MIC_COLOR if index else "#B0B0B0",
        )
    ax_mic.set_xlim(0, rate / 2)
    ax_mic.set_title("Microphone spectrum per stage", fontsize=9)
    ax_mic.set_xlabel("Frequency (Hz)")
    ax_mic.set_ylabel(f"PSD ({SIGNAL_UNITS}$^2$/Hz)", fontsize=8)
    ax_mic.legend(fontsize=6.5)

    channel = EMG_COLUMNS[emg_channel]
    fig.suptitle(
        f"Preprocessing stages -- {record.recording_id}, channel {channel} "
        f"({EMG_MUSCLE_MAP.get(channel, 'site unconfirmed')}) and microphone",
        y=0.995,
    )
    caveat(
        fig,
        "Each stage is the production chain truncated after that filter, applied to the "
        "whole recording and then sliced -- never applied window by window. Amplitudes are "
        f"{SIGNAL_UNITS}: the acquisition chain documented no physical calibration, so these "
        "are not uV or Pa. Per-fold z-scoring, fitted on training participants only, follows "
        "this chain and is not shown here.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------------- windows ---


def plot_example_windows(
    window_index: WindowIndex,
    cache: Any,
    task: ClassificationTask,
    output_dir: Path,
    *,
    stem: str = "07_example_windows",
    overlay: int = 25,
) -> list[Path]:
    """One representative model input per class, with faint overlays showing variability.

    Left: every EMG channel of the median-energy window of that class. Right: the
    simultaneous microphone window. The faint traces behind are other windows of the same
    class, so a reader sees the within-class spread rather than one flattering example.
    """
    FigureStyle.apply()
    sampled = sample_windows_by_class(window_index, task, max_per_class=max(overlay * 3, 60))
    if not sampled:
        raise ValueError("no windows available for this task")
    rate = float(window_index.sampling_rate_hz)
    channel_indices = list(cache.emg_channels)

    present = [label for label in range(task.num_classes) if sampled.get(label)]
    fig, axes = plt.subplots(
        len(present),
        2,
        figsize=(12.0, 1.55 * len(present) + 1.2),
        squeeze=False,
        width_ratios=[1.35, 1.0],
    )
    for row, label in enumerate(present):
        windows = sampled[label]
        chosen = _median_energy_window(cache, windows)
        emg, mic = cache.window(chosen)
        time = np.arange(emg.shape[0]) / rate

        # A 5-95 percentile band over other windows of the class, rather than a tangle of
        # individual traces: it shows the amplitude spread without burying the example.
        others = windows[:: max(len(windows) // max(overlay, 1), 1)][:overlay]
        if len(others) >= 3:
            other_emg = np.stack([cache.window(other)[0][:, 0] for other in others])
            other_mic = np.stack([cache.window(other)[1] for other in others])
            for ax, block in ((axes[row][0], other_emg), (axes[row][1], other_mic)):
                low, high = np.percentile(block, [5, 95], axis=0)
                ax.fill_between(time, low, high, color="#C8C8C8", alpha=0.55, linewidth=0, zorder=1)
        for position, channel in enumerate(channel_indices):
            axes[row][0].plot(
                time,
                emg[:, position],
                linewidth=0.7,
                color=FigureStyle.color(position),
                label=EMG_COLUMNS[channel] if row == 0 else None,
                zorder=2,
            )
        axes[row][1].plot(time, mic, linewidth=0.7, color=MIC_COLOR, zorder=2)
        axes[row][0].set_ylabel(_pretty(task.class_names[label]), fontsize=8)
        axes[row][0].set_xlim(0, time[-1])
        axes[row][1].set_xlim(0, time[-1])
        if row != len(present) - 1:
            axes[row][0].set_xticklabels([])
            axes[row][1].set_xticklabels([])
    axes[0][0].set_title(f"EMG ({len(channel_indices)} channels, {SIGNAL_UNITS})")
    axes[0][1].set_title(f"Microphone ({SIGNAL_UNITS})")
    axes[0][0].legend(fontsize=6.5, ncols=len(channel_indices), loc="upper right")
    axes[-1][0].set_xlabel("Time within window (s)")
    axes[-1][1].set_xlabel("Time within window (s)")

    fig.suptitle(
        f"Representative {window_index.config.window_seconds:g}-second windows per class "
        "(median-energy example; grey band = 5-95th percentile of other windows of that class)",
        y=0.995,
    )
    caveat(
        fig,
        "Signals are shown after the production filter chain, which is what the wavelet "
        "stage receives. Per-fold z-scoring is applied on top of this and is deliberately "
        "not shown, because its statistics differ between leave-one-subject-out folds.",
    )
    return save_figure(fig, output_dir, stem)


def plot_augmentation_examples(
    window_index: WindowIndex,
    cache: Any,
    task: ClassificationTask,
    augmentation: AugmentationConfig,
    output_dir: Path,
    *,
    stem: str = "10_augmentation_examples",
    seed: int = 0,
) -> list[Path]:
    """What training-time augmentation actually does to a window.

    Produced by calling the real :class:`~bruxism.preprocessing.augmentation.Augmenter` with
    ``stage="train"``, so the figure documents the transformation that ran rather than a
    re-implementation of it. Augmentation is training-only and minority-class-only by
    default; the augmenter refuses any other stage.
    """
    FigureStyle.apply()
    sampled = sample_windows_by_class(window_index, task, max_per_class=40)
    counts = {label: len(windows) for label, windows in sampled.items()}
    if not counts:
        raise ValueError("no windows available for this task")
    minority = Augmenter.minority_labels_from_counts(
        counts, threshold=augmentation.minority_threshold
    )
    label = min(minority) if minority else min(counts, key=lambda k: counts[k])
    window = _median_energy_window(cache, sampled[label])
    emg, mic = cache.window(window)
    rate = float(window_index.sampling_rate_hz)
    time = np.arange(emg.shape[0]) / rate

    # Force each variant to fire by driving the augmenter with a permissive config, so the
    # figure shows every transformation the real config can apply rather than whichever
    # happened to be drawn for this sample.
    variants: list[tuple[str, np.ndarray, np.ndarray]] = [("original", emg, mic)]
    recipes = {
        "amplitude scale": {"noise_probability": 0.0, "shift_probability": 0.0},
        "additive noise": {"amplitude_scale_probability": 0.0, "shift_probability": 0.0},
        "circular shift": {"amplitude_scale_probability": 0.0, "noise_probability": 0.0},
        "all three": {},
    }
    from dataclasses import replace

    for name, overrides in recipes.items():
        forced = replace(
            augmentation,
            enabled=True,
            probability=1.0,
            minority_only=False,
            amplitude_scale_probability=overrides.get("amplitude_scale_probability", 1.0),
            noise_probability=overrides.get("noise_probability", 1.0),
            shift_probability=overrides.get("shift_probability", 1.0),
        )
        augmenter = Augmenter(forced, run_seed=seed, minority_labels=frozenset({label}))
        aug_emg, aug_mic = augmenter(
            emg, mic, label=label, sample_id=window.sample_id, epoch=1, stage="train"
        )
        variants.append((name, aug_emg, aug_mic))

    fig, axes = plt.subplots(
        len(variants),
        2,
        figsize=(12.0, 1.35 * len(variants) + 1.3),
        squeeze=False,
        width_ratios=[1.35, 1.0],
        sharex=True,
    )
    for row, (name, variant_emg, variant_mic) in enumerate(variants):
        axes[row][0].plot(time, emg[:, 0], color="#BBBBBB", linewidth=0.8, zorder=1)
        axes[row][0].plot(
            time, variant_emg[:, 0], color=FigureStyle.color(row), linewidth=0.7, zorder=2
        )
        axes[row][1].plot(time, mic, color="#BBBBBB", linewidth=0.8, zorder=1)
        axes[row][1].plot(time, variant_mic, color=MIC_COLOR, linewidth=0.7, zorder=2)
        axes[row][0].set_ylabel(name, fontsize=8)
    axes[0][0].set_title(f"EMG channel {EMG_COLUMNS[cache.emg_channels[0]]}")
    axes[0][1].set_title("Microphone")
    axes[-1][0].set_xlabel("Time within window (s)")
    axes[-1][1].set_xlabel("Time within window (s)")

    ranges = augmentation.amplitude_scale_range
    fig.suptitle(
        f"Training-time augmentation of one '{_pretty(task.class_names[label])}' window "
        "(grey = the unaugmented original)",
        y=0.995,
    )
    caveat(
        fig,
        f"Configured: p(augment)={augmentation.probability:g}, amplitude scale "
        f"{ranges[0]:g}-{ranges[1]:g}, Gaussian noise at "
        f"{augmentation.noise_std_fraction:.0%} of the window's own standard deviation, "
        f"circular shift up to +/-{augmentation.max_shift_samples} samples, applied to "
        + ("minority classes only. " if augmentation.minority_only else "every class. ")
        + "Each panel here forces one transformation on for illustration; during training "
        "they fire independently at their own probabilities, and never on validation or "
        "held-out data.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------------ spectral ---


def plot_class_spectra(
    window_index: WindowIndex,
    cache: Any,
    task: ClassificationTask,
    output_dir: Path,
    *,
    stem: str = "06_class_spectra",
    max_per_class: int = 150,
) -> list[Path]:
    """Mean power spectrum of each class, for EMG and for audio.

    This is the engineering rationale for modality-specific processing made visible: if two
    classes separate in the microphone spectrum but not the EMG spectrum, the audio branch
    has something to contribute, and vice versa.
    """
    FigureStyle.apply()
    sampled = sample_windows_by_class(window_index, task, max_per_class=max_per_class)
    rate = float(window_index.sampling_rate_hz)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    counted: dict[str, int] = {}
    # Range over the informative band only: the filter stopband runs many decades lower and
    # would otherwise squash every curve into the top of the axis.
    extremes: dict[int, list[float]] = {0: [], 1: []}
    for label, windows in sorted(sampled.items()):
        if not windows:
            continue
        emg, mic = _load_windows(cache, windows)
        counted[task.class_names[label]] = len(windows)
        freqs, emg_power = _welch(np.moveaxis(emg, 1, 2), rate)  # (n, ch, freq)
        mean_emg = emg_power.mean(axis=(0, 1))
        _, mic_power = _welch(mic, rate)
        mean_mic = mic_power.mean(axis=0)
        colour = FigureStyle.color(label)
        axes[0].loglog(
            freqs[1:],
            mean_emg[1:],
            color=colour,
            linewidth=1.3,
            label=f"{_pretty(task.class_names[label])} (n={len(windows)})",
        )
        axes[1].loglog(
            freqs[1:],
            mean_mic[1:],
            color=colour,
            linewidth=1.3,
            label=_pretty(task.class_names[label]),
        )
        informative = (freqs >= 15) & (freqs <= 500)
        for position, curve in ((0, mean_emg), (1, mean_mic)):
            values = curve[informative]
            values = values[values > 0]
            if values.size:
                extremes[position] += [float(values.min()), float(values.max())]

    for position, (ax, title) in enumerate(
        ((axes[0], "EMG (mean over channels)"), (axes[1], "Microphone"))
    ):
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Mean PSD ({SIGNAL_UNITS}$^2$/Hz)")
        ax.set_title(title)
        ax.set_xlim(1, rate / 2)
        if extremes[position]:
            ax.set_ylim(min(extremes[position]) / 5.0, max(extremes[position]) * 4.0)
        ax.legend(fontsize=7)
    axes[0].axvspan(20, 450, color="#0072B2", alpha=0.06, lw=0)
    axes[0].text(
        21, axes[0].get_ylim()[0] * 1.6, "20-450 Hz passband", fontsize=6.5, color="#0072B2"
    )

    total = sum(counted.values())
    fig.suptitle(
        f"Class-conditional power spectra of the filtered signal ({total:,} windows sampled)",
        y=0.99,
    )
    caveat(
        fig,
        "Welch estimates over a deterministic, evenly spaced sample of windows per class, "
        "computed after the production filter chain -- which is why EMG power falls away "
        "outside 20-450 Hz and the 60 Hz notch is visible. Descriptive: windows are "
        "correlated within participants and recordings.",
    )
    return save_figure(fig, output_dir, stem)


# ------------------------------------------------------------------ wavelets ---


def _band_order(config: WaveletConfig) -> list[str]:
    """Every band of the decomposition, ordered low frequency to high."""
    return [f"A{config.level}", *[f"D{k}" for k in range(config.level, 0, -1)]]


def _band_label(band: str, config: WaveletConfig, sampling_rate: float) -> str:
    low, high = band_frequencies(band, config.level, sampling_rate)
    return f"{band}\n{low:.0f}-{high:.0f} Hz"


def plot_wavelet_bands(
    window_index: WindowIndex,
    cache: Any,
    task: ClassificationTask,
    output_dir: Path,
    *,
    emg_wavelet: WaveletConfig,
    mic_wavelet: WaveletConfig,
    stem: str = "08_wavelet_bands",
) -> list[Path]:
    """The coefficient bands each branch consumes, one representative window per class.

    Columns are the configured bands in the order the branches expect them, annotated with
    their nominal dyadic frequency range. Rows are classes. This is the figure that shows
    *why* the network has one small convolutional stack per band rather than one over the
    raw window.
    """
    FigureStyle.apply()
    sampled = sample_windows_by_class(window_index, task, max_per_class=40)
    present = [label for label in range(task.num_classes) if sampled.get(label)]
    rate = float(window_index.sampling_rate_hz)
    columns = [("emg", band, emg_wavelet) for band in emg_wavelet.bands]
    columns += [("mic", band, mic_wavelet) for band in mic_wavelet.bands]

    fig, axes = plt.subplots(
        len(present),
        len(columns),
        figsize=(2.05 * len(columns) + 1.2, 1.35 * len(present) + 1.6),
        squeeze=False,
    )
    for row, label in enumerate(present):
        window = _median_energy_window(cache, sampled[label])
        emg, mic = cache.window(window)
        emg_bands = decompose(emg[:, 0], emg_wavelet, check_level=False)
        mic_bands = decompose(mic, mic_wavelet, check_level=False)
        for column, (modality, band, config) in enumerate(columns):
            values = (emg_bands if modality == "emg" else mic_bands)[band]
            ax = axes[row][column]
            ax.plot(
                values,
                linewidth=0.7,
                color=FigureStyle.color(label) if modality == "emg" else MIC_COLOR,
            )
            ax.set_xlim(0, len(values) - 1)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(
                    f"{'EMG' if modality == 'emg' else 'MIC'} {_band_label(band, config, rate)}",
                    fontsize=8,
                )
            if column == 0:
                ax.set_ylabel(_pretty(task.class_names[label]), fontsize=8)
            if row == len(present) - 1:
                ax.set_xlabel("coefficient", fontsize=7)
    fig.suptitle(
        f"Wavelet bands fed to the branches -- EMG {emg_wavelet.wavelet} level "
        f"{emg_wavelet.level}, microphone {mic_wavelet.wavelet} level {mic_wavelet.level}",
        y=0.995,
    )
    caveat(
        fig,
        "Bands are addressed by name, never by list position: at level 4 the pywt "
        "coefficient list runs [A4, D4, D3, D2, D1], so indexing position 0 as 'highest "
        "frequency' would silently plot the lowest. Frequency ranges are the nominal dyadic "
        "half-band edges of the filter cascade; real wavelet filters overlap across them.",
    )
    return save_figure(fig, output_dir, stem)


def plot_wavelet_band_energy(
    window_index: WindowIndex,
    cache: Any,
    task: ClassificationTask,
    output_dir: Path,
    *,
    emg_wavelet: WaveletConfig,
    mic_wavelet: WaveletConfig,
    stem: str = "09_wavelet_band_energy",
    max_per_class: int = 120,
) -> list[Path]:
    """Where each class puts its energy across the full decomposition.

    Every band of the decomposition is shown, not only the three each branch consumes, so a
    reader can judge what the band selection discards. Bands the model actually receives are
    marked. Class separation visible here is what the per-band convolutional stacks exploit.
    """
    FigureStyle.apply()
    sampled = sample_windows_by_class(window_index, task, max_per_class=max_per_class)
    present = [label for label in range(task.num_classes) if sampled.get(label)]
    if not present:
        raise ValueError("no windows available for this task")

    def relative_energy(signal: np.ndarray, config: WaveletConfig) -> np.ndarray:
        """Fraction of total coefficient energy in each band, ordered low -> high."""
        full = WaveletConfig(
            wavelet=config.wavelet,
            level=config.level,
            bands=tuple(_band_order(config)),
            mode=config.mode,
        )
        bands = decompose(signal, full, check_level=False)
        energies = np.array([float(np.sum(np.square(bands[band]))) for band in _band_order(config)])
        return energies / max(float(energies.sum()), 1e-12)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.8),
        width_ratios=[emg_wavelet.level + 1, mic_wavelet.level + 1],
    )
    for ax, config, modality in (
        (axes[0], emg_wavelet, "emg"),
        (axes[1], mic_wavelet, "mic"),
    ):
        order = _band_order(config)
        width = 0.8 / len(present)
        for offset, label in enumerate(present):
            per_window = []
            for window in sampled[label]:
                emg, mic = cache.window(window)
                signal = emg[:, 0] if modality == "emg" else mic
                per_window.append(relative_energy(signal, config))
            matrix = np.stack(per_window)  # (n_windows, n_bands)
            positions = np.arange(len(order)) + offset * width - 0.4 + width / 2
            box = ax.boxplot(
                [matrix[:, i] for i in range(len(order))],
                positions=positions,
                widths=width * 0.86,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.9},
                whiskerprops={"linewidth": 0.6},
                capprops={"linewidth": 0.6},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(FigureStyle.color(label))
                patch.set_alpha(0.75)
                patch.set_linewidth(0.5)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [
                _band_label(band, config, float(window_index.sampling_rate_hz))
                + ("\n(used)" if band in config.bands else "")
                for band in order
            ],
            fontsize=7,
        )
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"{'EMG' if modality == 'emg' else 'Microphone'} -- {config.wavelet}, "
            f"level {config.level}"
        )
        ax.set_ylabel("Share of window energy")
        for index, band in enumerate(order):
            if band in config.bands:
                ax.axvspan(index - 0.45, index + 0.45, color="#000000", alpha=0.04, lw=0)

    handles = [
        plt.Line2D(
            [],
            [],
            color=FigureStyle.color(label),
            linewidth=7,
            alpha=0.75,
            label=_pretty(task.class_names[label]),
        )
        for label in present
    ]
    axes[0].legend(handles=handles, fontsize=7, ncols=min(len(present), 3), loc="upper center")
    fig.suptitle("Relative wavelet-band energy per class", y=0.99)
    caveat(
        fig,
        "Boxes span the interquartile range over a deterministic sample of windows per "
        "class; whiskers are 1.5 IQR and outliers are hidden. Shaded columns marked '(used)' "
        "are the bands actually passed to the network -- the unshaded bands are computed by "
        "the cascade and then discarded. Descriptive only: windows within a participant are "
        "correlated.",
    )
    return save_figure(fig, output_dir, stem)
