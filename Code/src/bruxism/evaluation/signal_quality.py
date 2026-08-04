"""Per participant x class signal quality of the windows a pipeline actually emits.

The three acceptance measurements of ``new_prompt.md`` Phase 1.3, computed from the
filtered windows rather than from a filter design:

* **mains-harmonic power fraction** -- must fall from 85-99.8 % to the noise floor;
* **class contrast** -- an activity's RMS divided by that participant's own rest RMS. This
  is the quantity the old chain destroyed: S01's resting EMG was larger than S03's and
  S04's clenching EMG, so no amplitude-driven decision function could separate them;
* **between-participant amplitude spread** -- max/min of the per-participant median RMS.

Everything here reads the same :class:`~bruxism.data.dataset.RecordingCache` the model
reads, so a measurement cannot describe a different signal from the one that was trained
on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from bruxism.data.dataset import RecordingCache
from bruxism.data.segments import WindowIndex, WindowRecord
from bruxism.preprocessing.interference import (
    harmonic_excess_ratio,
    measure_mains_contamination,
)
from bruxism.utils import progress
from bruxism.utils.logging import get_logger

__all__ = ["contrast_table", "spread_summary", "window_quality_table"]

logger = get_logger(__name__)


def window_quality_table(
    window_index: WindowIndex,
    cache: RecordingCache,
    *,
    max_windows_per_cell: int = 40,
    subjects: Sequence[str] | None = None,
    label: str = "signal quality",
) -> pd.DataFrame:
    """One row per (participant, task family) cell with contamination and amplitude.

    Parameters
    ----------
    max_windows_per_cell
        Deterministic even-spaced subsample per cell. Spectral estimates on 40 windows are
        stable to well within the effect sizes being measured, and the full set would read
        every window of every recording for every filter variant.

    Returns
    -------
    pandas.DataFrame
        Columns ``subject_id``, ``task_family``, ``n_windows``, ``median_rms``,
        ``mains_fraction``, ``harmonic_excess_max``.
    """
    rate = float(window_index.sampling_rate_hz)
    wanted = set(subjects) if subjects else None
    cells: dict[tuple[str, str], list[WindowRecord]] = {}
    for window in window_index.windows:
        if wanted is not None and window.subject_id not in wanted:
            continue
        cells.setdefault((window.subject_id, window.task_family), []).append(window)

    rows: list[dict[str, Any]] = []
    ordered = sorted(cells.items())
    for (subject, family), windows in progress.track(
        ordered, label, total=len(ordered), unit="cell"
    ):
        selected = windows
        if len(selected) > max_windows_per_cell:
            step = len(selected) / max_windows_per_cell
            selected = [selected[int(i * step)] for i in range(max_windows_per_cell)]

        blocks = [cache.window(window)[0] for window in selected]
        rms = [float(np.sqrt(np.mean(block**2))) for block in blocks]
        # Concatenating windows would splice discontinuities into the spectrum, so the
        # contamination of a cell is the mean over its windows, not the spectrum of their
        # concatenation.
        fractions, excesses = [], []
        for block in blocks:
            fractions.append(measure_mains_contamination(block, rate).fraction)
            excesses.append(harmonic_excess_ratio(block, rate)["max"])
        rows.append(
            {
                "subject_id": subject,
                "task_family": family,
                "n_windows": len(windows),
                "n_measured": len(selected),
                "median_rms": float(np.median(rms)),
                "mains_fraction": float(np.mean(fractions)),
                "harmonic_excess_max": float(np.mean(excesses)),
            }
        )
    return pd.DataFrame(rows).sort_values(["subject_id", "task_family"], ignore_index=True)


def contrast_table(quality: pd.DataFrame, *, rest_family: str = "rest") -> pd.DataFrame:
    """Each activity's median RMS divided by that participant's own rest median RMS.

    A participant with no rest cell is dropped with a warning rather than compared against
    another participant's rest, which would be the amplitude confound this table exists to
    expose.
    """
    rest = quality[quality["task_family"] == rest_family].set_index("subject_id")["median_rms"]
    rows: list[dict[str, Any]] = []
    for row in quality.itertuples(index=False):
        if row.task_family == rest_family:
            continue
        reference = rest.get(row.subject_id)
        if reference is None or reference <= 0:
            logger.warning("participant %s has no usable rest cell; skipping", row.subject_id)
            continue
        rows.append(
            {
                "subject_id": row.subject_id,
                "task_family": row.task_family,
                "contrast": float(row.median_rms / reference),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.pivot(index="subject_id", columns="task_family", values="contrast").reset_index()


def spread_summary(quality: pd.DataFrame) -> dict[str, Any]:
    """Between-participant amplitude spread, and the worst rest-vs-activity inversion.

    ``inversions`` counts participant pairs where one participant's *rest* amplitude
    exceeds another's *activity* amplitude -- the concrete failure the old chain produced,
    and the reason a model trained on four participants mis-ranked the fifth.
    """
    per_subject = quality.groupby("subject_id", observed=True)["median_rms"].median()
    rest = quality[quality["task_family"] == "rest"].set_index("subject_id")["median_rms"]
    activity = quality[quality["task_family"] != "rest"]

    inversions = 0
    worst: dict[str, Any] | None = None
    for subject, rest_rms in rest.items():
        others = activity[activity["subject_id"] != subject]
        for row in others.itertuples(index=False):
            if rest_rms > row.median_rms:
                inversions += 1
                ratio = float(rest_rms / row.median_rms)
                if worst is None or ratio > worst["ratio"]:
                    worst = {
                        "rest_subject": str(subject),
                        "rest_rms": float(rest_rms),
                        "activity_subject": str(row.subject_id),
                        "activity_family": str(row.task_family),
                        "activity_rms": float(row.median_rms),
                        "ratio": ratio,
                    }
    return {
        "per_subject_median_rms": {str(k): float(v) for k, v in per_subject.items()},
        "spread_ratio": float(per_subject.max() / per_subject.min())
        if per_subject.min() > 0
        else float("inf"),
        "n_rest_above_activity_inversions": inversions,
        "worst_inversion": worst,
    }
