"""The calibration block: which of a participant's own data may set their scale.

Per-participant normalisation needs statistics from the participant being evaluated. Using
their whole labelled session for that is an upper bound, not a procedure -- it answers "how
well could this work if we had everything?" rather than "what would a clinic actually do?".

This module defines the deployable alternative. A **calibration block** is what a fitting
session produces before any diagnosis is attempted:

* the participant's dedicated **rest** recording, which sets their baseline, and
* **one guided repetition of each task family** -- the first trigger run of the first
  recording of that family -- which sets their dynamic range.

The block is excluded from training *and* from evaluation, so no window is ever both the
thing that set a scale and the thing that was scored by it.

What the block does and does not concede:

* it uses the held-out participant's **signal**, which is transductive test-time adaptation
  and must be declared wherever the calibrated number is reported;
* it does **not** use their labels. The class identity of a guided repetition is known by
  construction -- a clinician said "clench now" and recorded it -- but the statistics
  computed from it are a mean and a standard deviation over pooled signal, and
  :meth:`~bruxism.preprocessing.normalization.Normalizer.calibrate` receives no labels to
  use even if it wanted them.

Selection is deterministic: recordings and trigger runs are taken in sorted order, never
sampled, so the same dataset always yields the same block and the same window-index hash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bruxism.data.segments import WindowIndex, WindowRecord
from bruxism.utils.logging import get_logger

__all__ = ["CalibrationBlock", "build_calibration_block"]

logger = get_logger(__name__)

_REST_FAMILY = "rest"


@dataclass(frozen=True)
class CalibrationBlock:
    """The windows each participant contributes to setting their own scale."""

    sample_ids_by_subject: dict[str, tuple[str, ...]]
    detail: dict[str, Any]
    source: str

    @property
    def all_sample_ids(self) -> frozenset[str]:
        return frozenset(
            sample_id for ids in self.sample_ids_by_subject.values() for sample_id in ids
        )

    def for_subject(self, subject: str) -> tuple[str, ...]:
        return self.sample_ids_by_subject.get(subject, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_subjects": len(self.sample_ids_by_subject),
            "n_windows_total": len(self.all_sample_ids),
            "per_subject": self.detail,
            "excluded_from": "training and evaluation",
            "uses_held_out_labels": False,
            "note": (
                "Transductive: the held-out participant's own unlabelled signal sets their "
                "normalisation scale. Report any number derived from this as requiring a "
                "fitting session."
            ),
        }


#: Windows any one family may contribute. At a 1 s window and 0.5 s stride this is about
#: 10 s of signal per family, which is what a fitting session actually collects.
#:
#: The cap is not a nicety. Each participant has exactly ONE dedicated rest recording, so an
#: uncapped block swallows every rest window they have and the rest class disappears from
#: training entirely -- observed directly: "classes [0] are absent from the training
#: labels". A calibration block that consumes a whole class is not a calibration block.
DEFAULT_MAX_WINDOWS_PER_FAMILY: int = 20


def build_calibration_block(
    window_index: WindowIndex,
    *,
    source: str = "rest_plus_one_repetition",
    repetitions_per_family: int = 1,
    max_windows_per_family: int = DEFAULT_MAX_WINDOWS_PER_FAMILY,
    subjects: Sequence[str] | None = None,
) -> CalibrationBlock:
    """Select each participant's calibration windows.

    Parameters
    ----------
    source
        ``"rest_plus_one_repetition"`` is the deployable block described in the module
        docstring. ``"all_windows_upper_bound"`` takes every window the participant has,
        which measures the ceiling rather than a procedure and must be labelled as such.
    repetitions_per_family
        How many trigger runs per task family form the block. One is a fitting session;
        more is a longer one.
    max_windows_per_family
        Cap per family; see :data:`DEFAULT_MAX_WINDOWS_PER_FAMILY`. Windows are taken from
        the start of the selected recording or run, which is where a fitting session would
        take them.

    Raises
    ------
    ValueError
        On an unknown ``source``; if a participant has no rest recording -- their baseline
        would then be undefined, and silently falling back to the pooled statistic would
        make the calibrated arm mean something different for that participant; or if the
        block would leave any participant x family with no windows for training and
        evaluation.
    """
    wanted = set(subjects) if subjects else None
    by_subject: dict[str, list[WindowRecord]] = {}
    for window in window_index.windows:
        if wanted is not None and window.subject_id not in wanted:
            continue
        by_subject.setdefault(window.subject_id, []).append(window)

    if source == "all_windows_upper_bound":
        selected = {
            subject: tuple(sorted(w.sample_id for w in windows))
            for subject, windows in sorted(by_subject.items())
        }
        detail = {
            subject: {"n_windows": len(ids), "recordings": "all", "runs": "all"}
            for subject, ids in selected.items()
        }
        return CalibrationBlock(sample_ids_by_subject=selected, detail=detail, source=source)

    if source != "rest_plus_one_repetition":
        raise ValueError(
            f"unknown calibration source {source!r}; expected 'rest_plus_one_repetition' "
            f"or 'all_windows_upper_bound'"
        )

    selected: dict[str, tuple[str, ...]] = {}
    detail: dict[str, Any] = {}
    for subject, windows in sorted(by_subject.items()):
        chosen: list[WindowRecord] = []
        contributions: dict[str, Any] = {}

        rest_windows = [w for w in windows if w.task_family == _REST_FAMILY]
        if not rest_windows:
            raise ValueError(
                f"participant {subject} has no rest windows, so no calibration baseline "
                f"can be defined; a per-participant scale without a baseline is not the "
                f"same procedure for every participant"
            )
        rest_recording = min(w.recording_id for w in rest_windows)
        rest_selected = sorted(
            (w for w in rest_windows if w.recording_id == rest_recording),
            key=lambda w: w.start_sample,
        )[:max_windows_per_family]
        chosen.extend(rest_selected)
        contributions[_REST_FAMILY] = {
            "recording_id": rest_recording,
            "n_windows": len(rest_selected),
            "runs": f"first {len(rest_selected)} window(s) of the recording",
        }

        families = sorted({w.task_family for w in windows if w.task_family != _REST_FAMILY})
        for family in families:
            family_windows = [w for w in windows if w.task_family == family]
            recording = min(w.recording_id for w in family_windows)
            in_recording = [w for w in family_windows if w.recording_id == recording]
            runs = sorted(
                {w.trigger_run_index for w in in_recording if w.trigger_run_index is not None}
            )
            keep_runs = set(runs[:repetitions_per_family])
            repetition = [w for w in in_recording if w.trigger_run_index in keep_runs]
            if not repetition:
                # No trigger structure in this recording: take the earliest window instead,
                # recorded explicitly rather than silently skipping the family.
                repetition = sorted(in_recording, key=lambda w: w.start_sample)[:1]
                keep_runs = {"first window (no trigger runs)"}  # type: ignore[assignment]
            repetition = sorted(repetition, key=lambda w: w.start_sample)[:max_windows_per_family]
            chosen.extend(repetition)
            contributions[family] = {
                "recording_id": recording,
                "n_windows": len(repetition),
                "runs": sorted(str(r) for r in keep_runs),
            }

        ids = tuple(sorted({w.sample_id for w in chosen}))
        selected[subject] = ids
        detail[subject] = {"n_windows": len(ids), "by_family": contributions}
        logger.info(
            "calibration block for %s: %d window(s) from %d family/families",
            subject,
            len(ids),
            len(contributions),
            extra={"subject_id": subject, "n_calibration_windows": len(ids)},
        )

    block = CalibrationBlock(sample_ids_by_subject=selected, detail=detail, source=source)
    _assert_leaves_every_class_trainable(window_index, block, wanted)
    return block


def _assert_leaves_every_class_trainable(
    window_index: WindowIndex,
    block: CalibrationBlock,
    wanted: set[str] | None,
) -> None:
    """Refuse a block that consumes an entire participant x family.

    Each participant has exactly one dedicated rest recording, so an uncapped block takes
    every rest window they have and the rest class silently vanishes from the training
    labels -- which is exactly what happened the first time this ran. The symptom appeared
    far downstream, as a warning about a class absent from the training labels, so the
    condition is asserted here where it can name its cause.
    """
    withheld = block.all_sample_ids
    remaining: dict[tuple[str, str], int] = {}
    for window in window_index.windows:
        if wanted is not None and window.subject_id not in wanted:
            continue
        key = (window.subject_id, window.task_family)
        remaining.setdefault(key, 0)
        if window.sample_id not in withheld:
            remaining[key] += 1

    starved = sorted(key for key, count in remaining.items() if count == 0)
    if starved:
        raise ValueError(
            f"the calibration block would consume every window of {starved}, leaving that "
            f"class with nothing to train or evaluate on. Reduce max_windows_per_family, "
            f"or collect more than one recording of that condition per participant."
        )
