"""Evidence for the segmentation choices: guard width and window length.

Two measurements, both of which the project previously took on assumption.

**How wide must the transition guard be?** The guard exists so that no window straddles a
trigger onset and mixes two conditions. Its width was set to 0.25 s because 0.5 s starved
some participant x class cells -- a reason about sample counts, not about contamination.
:func:`trigger_onset_alignment` measures the thing the guard is actually for: the lag
between the trigger edge and the moment the EMG envelope actually moves. A guard narrower
than that lag admits genuinely mislabelled samples; a guard wider than it only costs data.

**How long must the window be?** :func:`window_guard_sweep` reports, for each (window,
guard) pair, how many windows survive and how small the smallest participant x class cell
gets. The two pull against each other -- shorter windows fix the starved cells and destroy
the 1-2 Hz burst structure that separates chewing from clenching -- and the table is the
evidence for where to sit between them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from bruxism.data.dataset import RecordingCache
from bruxism.data.manifest import DatasetManifest
from bruxism.data.segments import SegmentationConfig, build_window_index
from bruxism.utils import progress
from bruxism.utils.logging import get_logger

__all__ = [
    "GUARD_SWEEP_SECONDS",
    "WINDOW_SWEEP_SECONDS",
    "trigger_onset_alignment",
    "window_guard_sweep",
]

logger = get_logger(__name__)

#: Guard widths swept. Declared here so the sweep is prespecified rather than chosen after
#: seeing which value wins.
GUARD_SWEEP_SECONDS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.25, 0.375, 0.5)

#: Window lengths swept, in seconds.
WINDOW_SWEEP_SECONDS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


def _envelope(signal: np.ndarray, sampling_rate: float, *, smooth_seconds: float = 0.05):
    """Moving-RMS envelope of the pooled EMG channels."""
    rectified = np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2, axis=1))
    width = max(1, int(round(smooth_seconds * sampling_rate)))
    kernel = np.ones(width) / width
    return np.convolve(rectified, kernel, mode="same")


def trigger_onset_alignment(
    manifest: DatasetManifest,
    cache: RecordingCache,
    *,
    probe_seconds: float = 1.0,
    min_run_seconds: float = 1.0,
    max_onsets: int = 400,
) -> dict[str, Any]:
    """Measure the lag between each trigger onset and the actual EMG envelope transition.

    For every trigger run long enough to have a settled interior, the envelope is read over
    ``+/- probe_seconds`` around the onset. The transition time is where the envelope first
    crosses the midpoint between its pre-onset and post-onset plateaus. A positive lag means
    the muscle activated *after* the trigger was marked, so the guard must cover it or
    windows just inside the trigger will contain rest.

    Returns
    -------
    dict
        Percentiles of the lag distribution plus the guard each percentile implies, and the
        per-participant breakdown -- participants marked the trigger with very different
        granularity and there is no reason to assume they marked it with the same latency.
    """
    rate = float(manifest.sampling_rate_hz)
    probe = int(round(probe_seconds * rate))
    minimum = int(round(min_run_seconds * rate))

    rows: list[dict[str, Any]] = []
    censored = 0
    records = [r for r in manifest.included if r.trigger_run_boundaries]
    for record in progress.track(
        records, "trigger-onset alignment", total=len(records), unit="recording"
    ):
        signal = cache.get(record.recording_id)
        emg = np.asarray(signal[:, :-1], dtype=np.float64)
        envelope = _envelope(emg, rate)
        boundaries = record.trigger_run_boundaries
        for position, boundary in enumerate(boundaries):
            start, end = boundary["start_sample"], boundary["end_sample"]
            if end - start < minimum or start < probe or start + probe >= len(envelope):
                continue
            # The pre-onset baseline must be quiet, or the "activation" being timed is the
            # tail of the previous repetition. S02 marked every repetition separately, so
            # without this the measurement is dominated by back-to-back runs and saturates
            # at the probe edge.
            if position > 0 and start - boundaries[position - 1]["end_sample"] < probe:
                censored += 1
                continue
            before = envelope[start - probe : start]
            after = envelope[start : start + probe]
            low, high = float(np.median(before)), float(np.median(after))
            if high <= low * 1.5:
                # No detectable activation at this onset: either the task is low-amplitude
                # or the mark does not correspond to a movement. Recorded, not silently
                # dropped, because a large share would itself be a finding.
                rows.append(
                    {
                        "subject_id": record.subject_id,
                        "task_family": record.task_family,
                        "lag_seconds": np.nan,
                        "detected": False,
                    }
                )
                continue
            threshold = low + 0.5 * (high - low)
            window = envelope[start - probe : start + probe]
            crossings = np.flatnonzero(window >= threshold)
            if crossings.size == 0:
                continue
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "task_family": record.task_family,
                    "lag_seconds": float((crossings[0] - probe) / rate),
                    "detected": True,
                }
            )
            if len(rows) >= max_onsets:
                break

    frame = pd.DataFrame(rows)
    detected = frame[frame["detected"]] if not frame.empty else frame
    if detected.empty:
        return {"n_onsets": 0, "note": "no trigger onset showed a detectable activation"}

    lags = detected["lag_seconds"].to_numpy(dtype=np.float64)
    percentiles = {f"p{p}": float(np.percentile(np.abs(lags), p)) for p in (50, 75, 90, 95, 99)}
    # A crossing found at the very edge of the probe means the envelope was already high
    # when the probe opened: the lag is at least this large but its true value is unknown.
    at_edge = int(np.isclose(np.abs(lags), probe_seconds, atol=1.0 / rate).sum())
    return {
        "probe_seconds": probe_seconds,
        "n_onsets_examined": int(len(frame)),
        "n_onsets_with_detectable_activation": int(len(detected)),
        "fraction_detectable": float(len(detected) / len(frame)),
        "n_onsets_skipped_adjacent_repetition": censored,
        "n_lags_censored_at_probe_edge": at_edge,
        "fraction_censored": float(at_edge / len(detected)),
        "lag_seconds": {
            "mean": float(np.mean(lags)),
            "median": float(np.median(lags)),
            "std": float(np.std(lags, ddof=1)) if len(lags) > 1 else 0.0,
            "min": float(np.min(lags)),
            "max": float(np.max(lags)),
        },
        "absolute_lag_percentiles": percentiles,
        "guard_implied_by_p95": percentiles["p95"],
        # Direction is what decides how dangerous a narrow guard is. A NEGATIVE lag means
        # the muscle was already active before the trigger was marked, so a window placed
        # just inside the trigger contains task activity -- the label is right and the
        # guard was protecting against nothing. A POSITIVE lag is the harmful case: the
        # window contains rest but carries the task label.
        "fraction_activity_precedes_trigger": float(np.mean(lags < 0)),
        "positive_lag_percentiles": {
            f"p{p}": float(np.percentile(lags[lags > 0], p)) if (lags > 0).any() else 0.0
            for p in (50, 75, 90, 95)
        },
        "n_positive_lags": int((lags > 0).sum()),
        "max_positive_lag_seconds": float(lags[lags > 0].max()) if (lags > 0).any() else 0.0,
        "per_subject_absolute_lag_median": {
            str(subject): float(np.median(np.abs(group["lag_seconds"])))
            for subject, group in detected.groupby("subject_id", observed=True)
        },
        "interpretation": (
            "A guard at least as wide as the p95 absolute lag keeps 95 % of onsets outside "
            "every emitted window. A narrower guard admits samples whose true condition "
            "differs from their label; the sweep table prices what the narrower guard buys. "
            "Percentiles at or near probe_seconds are CENSORED, not measured: read "
            "n_lags_censored_at_probe_edge before quoting them."
        ),
    }


def window_guard_sweep(
    manifest: DatasetManifest,
    *,
    window_seconds: Sequence[float] = WINDOW_SWEEP_SECONDS,
    guard_seconds: Sequence[float] = GUARD_SWEEP_SECONDS,
    startup_guard_seconds: float = 0.5,
    stride_fraction: float = 0.5,
    min_cell: int = 30,
) -> pd.DataFrame:
    """Window count and cell occupancy for every (window, guard) pair.

    ``min_cell`` is the target from the work order: no participant x class cell below about
    30 windows for a class the task uses. The column ``cells_below_target`` counts the
    violations, which is the number the choice has to drive to zero.
    """
    rows: list[dict[str, Any]] = []
    combinations = [(w, g) for w in window_seconds for g in guard_seconds]
    for window, guard in progress.track(
        combinations, "window/guard sweep", total=len(combinations), unit="config"
    ):
        index = build_window_index(
            manifest,
            SegmentationConfig(
                window_seconds=window,
                stride_seconds=window * stride_fraction,
                guard_seconds=guard,
                startup_guard_seconds=startup_guard_seconds,
            ),
        )
        frame = index.frame
        if frame.empty:
            continue
        pivot = frame.pivot_table(
            index="subject_id", columns="task_family", aggfunc="size", fill_value=0
        )
        counts = pivot.to_numpy()
        row: dict[str, Any] = {
            "window_seconds": window,
            "guard_seconds": guard,
            "min_run_seconds_required": window + 2 * guard,
            "total_windows": int(len(frame)),
            "min_subject_class_cell": int(counts.min()),
            "cells_below_target": int((counts < min_cell).sum()),
            "n_cells": int(counts.size),
        }
        for family in sorted(pivot.columns):
            row[f"n_{family}"] = int(pivot[family].sum())
        # The starved cell that motivated the whole question.
        if "S02" in pivot.index and "movement" in pivot.columns:
            row["S02_movement"] = int(pivot.loc["S02", "movement"])
        rows.append(row)
    return pd.DataFrame(rows)
