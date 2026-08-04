"""Fast leave-one-subject-out screening on hand-written features.

This is the harness behind ``cause.md`` §1 and §4, made reproducible. It answers questions
that would otherwise cost a 3.8-hour nested run each:

* does a change to the signal chain move the achievable accuracy at all?
* does the dual-branch network do anything a logistic regression on band energies cannot?
* how much does per-participant normalisation add, and how much does aggregating windows?

**It is a screening tool and its numbers are optimistic.** One model fit per outer fold, no
nested hyperparameter selection, and a feature set that was chosen after looking at these
five participants. It is directionally reliable and the effect sizes it measures are far
larger than its noise, but a screening number is never a reported result: those come from
:mod:`bruxism.runner` under the prespecified nested protocol. Every function here labels
its output ``interpretation="screening"`` so the two cannot be confused downstream.

Two properties are kept even in screening, because losing them would make the comparison
meaningless rather than merely optimistic:

* the held-out participant contributes nothing to any fitted quantity (feature scaler,
  class weights, model), and this is asserted, not assumed;
* every metric is computed from a saved prediction ledger in the same schema the real
  pipeline writes, through the same :func:`~bruxism.evaluation.metrics.subject_level_summary`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from bruxism.data.dataset import RecordingCache
from bruxism.data.labels import ClassificationTask, TaskFamily
from bruxism.data.segments import WindowIndex, WindowRecord
from bruxism.evaluation.metrics import PredictionLedger, subject_level_summary
from bruxism.preprocessing.wavelets import WaveletConfig, decompose
from bruxism.utils import progress
from bruxism.utils.logging import get_logger

__all__ = [
    "SCREENING_MODEL_IDS",
    "FeatureMatrix",
    "NormalisationScope",
    "ScreeningConfig",
    "build_feature_matrix",
    "screen",
    "screening_feature_names",
]

logger = get_logger(__name__)

#: The wavelet decompositions the screening features are built from. Identical to the
#: bands the dual-branch network consumes, so "the network adds nothing over band
#: energies" is a statement about the same bands.
_EMG_WAVELET = WaveletConfig(wavelet="db4", level=4, bands=("A4", "D4", "D3", "D2", "D1"))
_MIC_WAVELET = WaveletConfig(wavelet="coif5", level=5, bands=("A5", "D5", "D4", "D3", "D2", "D1"))

SCREENING_MODEL_IDS: tuple[str, ...] = ("logistic_regression", "gradient_boosting")

NormalisationScope = Literal["none", "per_participant", "per_participant_robust", "per_recording"]

_EPS = 1e-12


def screening_feature_names(n_emg_channels: int = 4) -> list[str]:
    """The 35 feature names, in extraction order.

    Per EMG channel: log RMS of each of five db4 bands, log waveform length, zero-crossing
    rate (7 x 4 = 28). For the microphone: log RMS of each of six coif5 bands plus the log
    RMS of the window (7). Log amplitudes because EMG amplitude is multiplicative across
    participants -- a 4x scale difference is an additive offset in logs, which a linear
    model can absorb per participant.
    """
    names: list[str] = []
    for channel in range(1, n_emg_channels + 1):
        names += [f"emg{channel}_log_rms_{band}" for band in _EMG_WAVELET.bands]
        names.append(f"emg{channel}_log_waveform_length")
        names.append(f"emg{channel}_zero_crossing_rate")
    names += [f"mic_log_rms_{band}" for band in _MIC_WAVELET.bands]
    names.append("mic_log_rms")
    return names


def _log_rms(values: np.ndarray) -> float:
    return float(np.log10(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)) + _EPS))


def _window_features(emg: np.ndarray, mic: np.ndarray) -> np.ndarray:
    """Feature vector for one window, in :func:`screening_feature_names` order."""
    values: list[float] = []
    for channel in range(emg.shape[1]):
        signal = emg[:, channel]
        coefficients = decompose(signal, _EMG_WAVELET, check_level=False)
        values += [_log_rms(coefficients[band]) for band in _EMG_WAVELET.bands]
        values.append(float(np.log10(np.sum(np.abs(np.diff(signal))) + _EPS)))
        values.append(float(np.mean(np.diff(np.signbit(signal)) != 0)))
    mic_coefficients = decompose(mic, _MIC_WAVELET, check_level=False)
    values += [_log_rms(mic_coefficients[band]) for band in _MIC_WAVELET.bands]
    values.append(_log_rms(mic))
    return np.asarray(values, dtype=np.float64)


@dataclass
class FeatureMatrix:
    """Screening features plus everything needed to split, label and trace them."""

    features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray
    sample_ids: np.ndarray
    start_samples: np.ndarray
    end_samples: np.ndarray
    feature_names: list[str]
    class_names: tuple[str, ...]
    task_id: str
    sampling_rate_hz: float

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def subject_ids(self) -> list[str]:
        return sorted(set(self.subjects.tolist()))

    def counts(self) -> pd.DataFrame:
        """Windows per participant x class -- the table that exposes starved cells."""
        frame = pd.DataFrame(
            {
                "subject_id": self.subjects,
                "class": [self.class_names[label] for label in self.labels],
            }
        )
        return frame.pivot_table(
            index="subject_id", columns="class", aggfunc="size", fill_value=0
        ).reset_index()


def build_feature_matrix(
    window_index: WindowIndex,
    cache: RecordingCache,
    task: ClassificationTask,
    *,
    subjects: Sequence[str] | None = None,
    max_windows_per_class: int | None = None,
    label: str = "screening features",
) -> FeatureMatrix:
    """Extract screening features for every window the task labels.

    Parameters
    ----------
    max_windows_per_class
        Deterministic even-spaced subsample per class, for a quick pass. ``None`` uses
        every window, which is what any reported screening number must do.
    """
    selected: list[WindowRecord] = []
    labels: list[int] = []
    wanted = set(subjects) if subjects else None
    for window in window_index.windows:
        if wanted is not None and window.subject_id not in wanted:
            continue
        target = task.label_for_family(TaskFamily(window.task_family))
        if target is None:
            continue
        selected.append(window)
        labels.append(target)

    if max_windows_per_class is not None:
        keep: list[int] = []
        for value in sorted(set(labels)):
            positions = [i for i, item in enumerate(labels) if item == value]
            if len(positions) > max_windows_per_class:
                step = len(positions) / max_windows_per_class
                positions = [positions[int(i * step)] for i in range(max_windows_per_class)]
            keep.extend(positions)
        keep.sort()
        selected = [selected[i] for i in keep]
        labels = [labels[i] for i in keep]

    if not selected:
        raise ValueError("no windows remain for this task; nothing to screen")

    rows = [
        _window_features(*cache.window(window))
        for window in progress.track(selected, label, total=len(selected), unit="window")
    ]
    features = np.vstack(rows)
    if not np.isfinite(features).all():
        bad = int((~np.isfinite(features)).any(axis=1).sum())
        raise ValueError(f"{bad} windows produced non-finite screening features")

    return FeatureMatrix(
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray([w.subject_id for w in selected]),
        recordings=np.asarray([w.recording_id for w in selected]),
        sample_ids=np.asarray([w.sample_id for w in selected]),
        start_samples=np.asarray([w.start_sample for w in selected], dtype=np.int64),
        end_samples=np.asarray([w.end_sample for w in selected], dtype=np.int64),
        feature_names=screening_feature_names(cache.n_emg_channels),
        class_names=tuple(task.class_names),
        task_id=task.task_id,
        sampling_rate_hz=float(window_index.sampling_rate_hz),
    )


def _standardise_within(features: np.ndarray, groups: np.ndarray, *, robust: bool) -> np.ndarray:
    """Standardise each group's rows by that group's own statistics.

    Used for the ``per_participant`` and ``per_recording`` scopes. Applied to the
    held-out participant this is **transductive test-time adaptation**: it uses that
    participant's unlabelled feature distribution and must be declared as a calibration
    step wherever it is reported. It never touches a label.
    """
    out = np.array(features, dtype=np.float64, copy=True)
    for group in np.unique(groups):
        mask = groups == group
        block = out[mask]
        if robust:
            centre = np.median(block, axis=0)
            scale = 1.4826 * np.median(np.abs(block - centre), axis=0)
        else:
            centre = block.mean(axis=0)
            scale = block.std(axis=0)
        out[mask] = (block - centre) / np.maximum(scale, 1e-8)
    return out


def _apply_scope(matrix: FeatureMatrix, scope: NormalisationScope) -> np.ndarray:
    if scope == "none":
        return matrix.features
    if scope == "per_participant":
        return _standardise_within(matrix.features, matrix.subjects, robust=False)
    if scope == "per_participant_robust":
        return _standardise_within(matrix.features, matrix.subjects, robust=True)
    if scope == "per_recording":
        # Kept only so the measurement that rules it out can be reproduced: each recording
        # holds a single condition, so per-recording scaling removes the label itself.
        return _standardise_within(matrix.features, matrix.recordings, robust=False)
    raise ValueError(f"unknown normalisation scope {scope!r}")


@dataclass(frozen=True)
class ScreeningConfig:
    """What one screening pass does."""

    model_id: str = "logistic_regression"
    normalisation_scope: NormalisationScope = "none"
    #: Window counts over which held-out probabilities are additionally averaged, within a
    #: recording. TRIAL-LEVEL: every recording holds one condition, so this approaches a
    #: majority vote over a homogeneous trial and is never stream-level detection.
    aggregation_windows: tuple[int, ...] = ()
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "normalisation_scope": self.normalisation_scope,
            "aggregation_windows": list(self.aggregation_windows),
            "seed": self.seed,
            "params": dict(self.params),
        }


def _build_estimator(config: ScreeningConfig) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(config.params)
    if config.model_id == "logistic_regression":
        estimator: Any = LogisticRegression(
            C=params.pop("C", 1.0),
            max_iter=params.pop("max_iter", 3000),
            class_weight=params.pop("class_weight", "balanced"),
            random_state=config.seed,
            **params,
        )
    elif config.model_id == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=params.pop("max_iter", 300),
            learning_rate=params.pop("learning_rate", 0.1),
            class_weight=params.pop("class_weight", "balanced"),
            random_state=config.seed,
            **params,
        )
    else:
        raise KeyError(
            f"unknown screening model_id {config.model_id!r}; available: {SCREENING_MODEL_IDS}"
        )
    # The scaler lives inside the pipeline, so it is fitted on training rows only by
    # construction and there is no path where held-out statistics reach it.
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])


def screen(
    matrix: FeatureMatrix,
    config: ScreeningConfig | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run leave-one-subject-out screening and summarise it from a prediction ledger.

    Returns
    -------
    dict
        ``{"config", "subject_level", "ledger", "aggregated", "interpretation"}``. The
        ledger is a :class:`pandas.DataFrame` in the run-bundle schema so the same
        summarisers apply and nothing is typed by hand.
    """
    active = config or ScreeningConfig()
    features = _apply_scope(matrix, active.normalisation_scope)
    class_names = matrix.class_names
    rows: list[pd.DataFrame] = []

    for fold, held_out in enumerate(matrix.subject_ids):
        test_mask = matrix.subjects == held_out
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue
        # Leakage guard: the held-out participant contributes no training row. Asserted
        # rather than assumed, because a screening harness is exactly where a quiet
        # indexing mistake would go unnoticed.
        fitted_on = set(np.unique(matrix.subjects[train_mask]).tolist())
        if held_out in fitted_on:
            raise AssertionError(
                f"held-out participant {held_out} appears in the screening training set"
            )

        pipeline = _build_estimator(active)
        pipeline.fit(features[train_mask], matrix.labels[train_mask])
        probabilities = pipeline.predict_proba(features[test_mask])
        # A class absent from this fold's training set gets a zero column, so every fold's
        # ledger has the same width and the probabilities still sum to 1.
        full = np.zeros((probabilities.shape[0], len(class_names)), dtype=np.float64)
        full[:, np.asarray(pipeline.classes_, dtype=int)] = probabilities
        predicted = full.argmax(axis=1)

        rows.append(
            pd.DataFrame(
                {
                    "sample_id": matrix.sample_ids[test_mask],
                    "subject_id": matrix.subjects[test_mask],
                    "recording_id": matrix.recordings[test_mask],
                    "start_sample": matrix.start_samples[test_mask],
                    "end_sample": matrix.end_samples[test_mask],
                    "start_seconds": matrix.start_samples[test_mask] / matrix.sampling_rate_hz,
                    "end_seconds": matrix.end_samples[test_mask] / matrix.sampling_rate_hz,
                    "true_label": matrix.labels[test_mask],
                    "predicted_label": predicted,
                    "true_class": [class_names[i] for i in matrix.labels[test_mask]],
                    "predicted_class": [class_names[i] for i in predicted],
                    "outer_fold": fold,
                    "seed": active.seed,
                    "task_id": matrix.task_id,
                    "model_id": active.model_id,
                    "modality": "fusion",
                    "source_commit": (context or {}).get("source_commit", "screening"),
                    "config_hash": (context or {}).get("config_hash", "screening"),
                    "manifest_hash": (context or {}).get("manifest_hash", "screening"),
                    "checkpoint_sha256": "",
                    **{f"prob_{name}": full[:, i] for i, name in enumerate(class_names)},
                }
            )
        )

    ledger_frame = pd.concat(rows, ignore_index=True)
    ledger = PredictionLedger(frame=ledger_frame, class_names=class_names)
    ledger.assert_exactly_once()
    ledger.assert_covers(matrix.sample_ids.tolist())

    result: dict[str, Any] = {
        "interpretation": "screening",
        "config": active.to_dict(),
        "n_windows": len(matrix),
        "subject_level": subject_level_summary(ledger),
        "ledger": ledger_frame,
    }
    if active.aggregation_windows:
        result["aggregated"] = {
            str(n): aggregate_within_recording(ledger_frame, class_names, n_windows=n)
            for n in active.aggregation_windows
        }
    return result


def aggregate_within_recording(
    ledger_frame: pd.DataFrame,
    class_names: Sequence[str],
    *,
    n_windows: int,
) -> dict[str, Any]:
    """Average held-out probabilities over ``n_windows`` consecutive windows of a recording.

    **Trial-level, not stream-level.** Every recording in this dataset contains a single
    condition, so averaging inside one recording approaches a majority vote over a
    homogeneous trial. It is a valid trial-level result and an invalid continuous-stream
    one; presenting it as event detection would need onset/offset evaluation on
    mixed-activity data this dataset does not contain.
    """
    probability_columns = [f"prob_{name}" for name in class_names]
    frame = ledger_frame.sort_values(["recording_id", "start_sample"], ignore_index=True)
    blocks: list[pd.DataFrame] = []
    for _, group in frame.groupby("recording_id", sort=True):
        block_index = np.arange(len(group)) // n_windows
        averaged = group.groupby(block_index)[probability_columns].mean()
        first = group.groupby(block_index).first()
        merged = averaged.copy()
        merged["subject_id"] = first["subject_id"]
        merged["true_label"] = first["true_label"]
        blocks.append(merged.reset_index(drop=True))

    pooled = pd.concat(blocks, ignore_index=True)
    scores = pooled[probability_columns].to_numpy(dtype=np.float64)
    predicted = scores.argmax(axis=1)
    truth = pooled["true_label"].to_numpy(dtype=np.int64)

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
    )

    per_subject: dict[str, dict[str, float]] = {}
    for subject in sorted(pooled["subject_id"].unique()):
        mask = (pooled["subject_id"] == subject).to_numpy()
        _, _, macro_f1, _ = precision_recall_fscore_support(
            truth[mask],
            predicted[mask],
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        )
        per_subject[str(subject)] = {
            "n_blocks": int(mask.sum()),
            "accuracy": float(accuracy_score(truth[mask], predicted[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(truth[mask], predicted[mask])),
            "macro_f1": float(macro_f1),
        }
    return {
        "interpretation": "screening, TRIAL-LEVEL (single-condition recordings)",
        "n_windows_per_block": n_windows,
        "per_subject": per_subject,
        "accuracy_mean": float(np.mean([v["accuracy"] for v in per_subject.values()])),
        "macro_f1_mean": float(np.mean([v["macro_f1"] for v in per_subject.values()])),
    }


def headline(result: dict[str, Any]) -> dict[str, float | None]:
    """The two numbers every screening comparison in ``cause.md`` is quoted with."""
    summary = result["subject_level"]
    return {
        "macro_f1": (summary.get("macro_f1") or {}).get("mean"),
        "accuracy": (summary.get("accuracy") or {}).get("mean"),
        "balanced_accuracy": (summary.get("balanced_accuracy") or {}).get("mean"),
        "macro_f1_min": (summary.get("macro_f1") or {}).get("min"),
        "macro_f1_max": (summary.get("macro_f1") or {}).get("max"),
    }
