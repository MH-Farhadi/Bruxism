"""Normalisation statistics, fitted on training data only.

A :class:`Normalizer` is *fitted* once per split stage -- inner-training participants when
selecting hyperparameters, outer-training participants when fitting the final model -- and
then applied unchanged to the validation or held-out data. Applying an unfitted normalizer
raises; there is no lazy self-fitting path that could quietly absorb test statistics.

The fitted parameters are serialised into every run bundle so a reviewer can confirm which
participants produced them.

**Participant scope is a protocol change, not a preprocessing tweak.** A single mean and
standard deviation fitted on four people cannot align a fifth whose amplitude scale differs
by 4x, and ``scope="per_participant"`` fixes that by standardising each participant's
windows by that participant's own statistics -- *including the held-out participant's*.
That is transductive test-time adaptation. It uses the held-out participant's **signal**
and never their **labels**, and the two are kept apart by construction:

``fitted_on``
    Participants whose labelled training windows produced the pooled statistics.
    :meth:`Normalizer.assert_not_fitted_on` guards this and its meaning is unchanged: the
    held-out participant must never appear here.
``calibrated_on``
    Participants whose unlabelled calibration block produced per-participant statistics.
    The held-out participant *may* appear here, and when they do the run bundle records
    exactly which windows were used, so a reader can price the concession.

Anything reported from a calibrated normalizer must be labelled as requiring a fitting
session. The strict number -- training-set statistics only -- is the one that supports a
no-calibration deployment claim, and both are reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from bruxism.utils.logging import get_logger

__all__ = ["NormalizationConfig", "Normalizer", "NotFittedError"]

logger = get_logger(__name__)

_EPS = 1e-8


class NotFittedError(RuntimeError):
    """Raised when a normalizer is applied before being fitted on training data."""


@dataclass(frozen=True)
class NormalizationConfig:
    """How normalisation statistics are computed.

    Attributes
    ----------
    scope
        ``"per_channel"`` fits one mean/std per EMG channel across all training samples;
        ``"global"`` pools every channel into a single statistic; ``"per_participant"``
        fits one set per participant from that participant's own calibration block, with
        the pooled training statistics as the fallback for anyone not calibrated.
    method
        ``"zscore"`` uses mean/std. ``"robust"`` uses median and a MAD-derived scale, which
        is markedly less sensitive to the amplifier-settling transients present in these
        recordings. Screening measured robust participant statistics at macro-F1 0.409 vs
        0.527 for mean/std: the per-participant distribution is multimodal and
        chewing-dominated, so the MAD tracks the majority class rather than the spread.
    calibration
        Which of a participant's own data forms the calibration block.
        ``"rest_plus_one_repetition"`` is the deployable procedure: the dedicated rest
        recording plus one guided repetition of each task family, excluded from both
        training and evaluation. ``"all_windows_upper_bound"`` uses the whole session and
        is an upper bound rather than a procedure -- it requires
        ``calibration_approved_by`` so it cannot be selected by accident.
    calibration_approved_by
        Name and date of the approval, recorded in the run bundle.
    """

    scope: Literal["per_channel", "global", "per_participant"] = "per_channel"
    method: Literal["zscore", "robust"] = "zscore"
    calibration: Literal["none", "rest_plus_one_repetition", "all_windows_upper_bound"] = "none"
    calibration_approved_by: str | None = None

    def __post_init__(self) -> None:
        if self.scope == "per_participant" and self.calibration == "none":
            raise ValueError(
                "scope='per_participant' requires a calibration source; set "
                "calibration='rest_plus_one_repetition' (the deployable procedure) and "
                "declare it in the protocol, or leave scope at 'per_channel'"
            )
        if self.scope != "per_participant" and self.calibration != "none":
            raise ValueError(
                f"calibration={self.calibration!r} only applies to "
                f"scope='per_participant', not {self.scope!r}"
            )
        if self.calibration == "all_windows_upper_bound" and not self.calibration_approved_by:
            raise ValueError(
                "calibration='all_windows_upper_bound' uses the participant's entire "
                "session, which is an upper bound and not a deployable procedure; it "
                "requires calibration_approved_by so the concession is recorded"
            )

    @property
    def is_transductive(self) -> bool:
        """Whether applying this requires data from the participant being evaluated."""
        return self.scope == "per_participant"

    def to_dict(self) -> dict[str, Any]:
        """Exactly the fields, and nothing derived.

        ``resolved_config.yaml`` is re-loaded by ``bruxism-figures`` and by every resumed
        run, and :class:`~bruxism.config.ExperimentConfig` rejects unknown keys by design.
        A derived entry here -- ``is_transductive`` was one, briefly -- therefore makes a
        run bundle unreadable by the code that wrote it. Derived facts belong in the run
        bundle's own descriptive files, not in the round-tripped configuration.
        """
        return {
            "scope": self.scope,
            "method": self.method,
            "calibration": self.calibration,
            "calibration_approved_by": self.calibration_approved_by,
        }


@dataclass
class Normalizer:
    """Per-modality normalisation statistics fitted on a specific participant set.

    Attributes
    ----------
    fitted_on
        Participant IDs whose data produced these statistics. Recorded so that a leakage
        test can assert the held-out participant is absent.
    """

    config: NormalizationConfig
    emg_center: np.ndarray | None = None
    emg_scale: np.ndarray | None = None
    mic_center: float | None = None
    mic_scale: float | None = None
    fitted_on: tuple[str, ...] = ()
    n_fit_samples: int = 0
    #: Per-participant statistics from each participant's own calibration block.
    subject_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Participants whose UNLABELLED calibration block produced per-participant
    #: statistics. May legitimately include the held-out participant; see the module
    #: docstring. Deliberately a different attribute from ``fitted_on`` so no leakage
    #: assertion can be satisfied by the wrong one.
    calibrated_on: tuple[str, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self.emg_center is not None and self.mic_center is not None

    def _centre_and_scale(self, values: np.ndarray, axis: int | None) -> tuple[Any, Any]:
        if self.config.method == "robust":
            centre = np.median(values, axis=axis)
            # 1.4826 * MAD is the consistent estimator of sigma for Gaussian data.
            scale = 1.4826 * np.median(np.abs(values - centre), axis=axis)
        else:
            centre = np.mean(values, axis=axis)
            scale = np.std(values, axis=axis)
        scale = np.maximum(scale, _EPS)
        return centre, scale

    def fit(
        self,
        emg: np.ndarray,
        mic: np.ndarray,
        *,
        subjects: tuple[str, ...] = (),
    ) -> Normalizer:
        """Fit statistics from **training** signal only.

        Parameters
        ----------
        emg
            ``(n_samples_total, n_channels)`` concatenation of training EMG.
        mic
            ``(n_samples_total,)`` concatenation of training microphone samples.
        subjects
            The participants these samples came from, recorded for the leakage audit.
        """
        emg = np.asarray(emg, dtype=np.float64)
        mic = np.asarray(mic, dtype=np.float64).ravel()
        if emg.ndim != 2:
            raise ValueError(f"emg must be (n_samples, n_channels), got shape {emg.shape}")
        if emg.size == 0 or mic.size == 0:
            raise ValueError("cannot fit a normalizer on empty training data")

        # Participant scope still fits a pooled fallback per channel: it is what applies to
        # anyone without a calibration block, and it is the "strict" comparison arm.
        axis = None if self.config.scope == "global" else 0
        emg_centre, emg_scale = self._centre_and_scale(emg, axis)
        mic_centre, mic_scale = self._centre_and_scale(mic, None)

        self.emg_center = np.atleast_1d(np.asarray(emg_centre, dtype=np.float64))
        self.emg_scale = np.atleast_1d(np.asarray(emg_scale, dtype=np.float64))
        self.mic_center = float(mic_centre)
        self.mic_scale = float(mic_scale)
        self.fitted_on = tuple(sorted(subjects))
        self.n_fit_samples = int(emg.shape[0])

        logger.debug(
            "normalizer fitted on %s (%d samples)",
            ",".join(self.fitted_on) or "<unspecified>",
            self.n_fit_samples,
            extra={"fitted_on": list(self.fitted_on)},
        )
        return self

    def calibrate(
        self,
        subject: str,
        emg: np.ndarray,
        mic: np.ndarray,
        *,
        sample_ids: tuple[str, ...] = (),
    ) -> Normalizer:
        """Fit one participant's statistics from that participant's calibration block.

        No label is read here, and none is available to this method: it receives signal
        only. The sample ids are recorded so the run bundle can state exactly which of the
        participant's data produced the statistics, which is the disclosure the concession
        requires.

        Raises
        ------
        ValueError
            If the configured scope is not ``"per_participant"``. Calibrating a normalizer
            that will not consult the result would be a silent no-op.
        """
        if self.config.scope != "per_participant":
            raise ValueError(
                f"calibrate() requires scope='per_participant', not {self.config.scope!r}; "
                f"the per-participant statistics would never be applied"
            )
        emg = np.asarray(emg, dtype=np.float64)
        mic = np.asarray(mic, dtype=np.float64).ravel()
        if emg.ndim != 2 or emg.size == 0 or mic.size == 0:
            raise ValueError(
                f"calibration block for {subject} is empty or malformed (emg shape "
                f"{emg.shape}); a participant cannot be calibrated on no data"
            )
        emg_centre, emg_scale = self._centre_and_scale(emg, 0)
        mic_centre, mic_scale = self._centre_and_scale(mic, None)
        self.subject_stats[subject] = {
            "emg_center": np.atleast_1d(np.asarray(emg_centre, dtype=np.float64)),
            "emg_scale": np.atleast_1d(np.asarray(emg_scale, dtype=np.float64)),
            "mic_center": float(mic_centre),
            "mic_scale": float(mic_scale),
            "n_samples": int(emg.shape[0]),
            "n_windows": len(sample_ids),
            "sample_ids": list(sample_ids),
        }
        self.calibrated_on = tuple(sorted({*self.calibrated_on, subject}))
        logger.info(
            "normalizer calibrated for %s on %d window(s) of that participant's own "
            "unlabelled signal (declared calibration step, not label leakage)",
            subject,
            len(sample_ids),
            extra={"subject_id": subject, "n_calibration_windows": len(sample_ids)},
        )
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                "normalizer has not been fitted; fit it on training participants before "
                "transforming any data (never fit on validation or held-out data)"
            )

    def _stats_for(self, subject: str | None) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Per-participant statistics when calibrated, otherwise the pooled fallback."""
        self._require_fitted()
        assert self.emg_center is not None and self.emg_scale is not None
        assert self.mic_center is not None and self.mic_scale is not None
        if subject is not None and subject in self.subject_stats:
            stats = self.subject_stats[subject]
            return (
                stats["emg_center"],
                stats["emg_scale"],
                stats["mic_center"],
                stats["mic_scale"],
            )
        return self.emg_center, self.emg_scale, self.mic_center, self.mic_scale

    def transform_emg(self, emg: np.ndarray, *, subject: str | None = None) -> np.ndarray:
        """Apply fitted EMG statistics to ``(..., n_samples, n_channels)`` data."""
        centre, scale, _, _ = self._stats_for(subject)
        return (np.asarray(emg, dtype=np.float64) - centre) / scale

    def transform_mic(self, mic: np.ndarray, *, subject: str | None = None) -> np.ndarray:
        """Apply fitted microphone statistics."""
        _, _, centre, scale = self._stats_for(subject)
        return (np.asarray(mic, dtype=np.float64) - centre) / scale

    def assert_not_fitted_on(self, forbidden_subjects: tuple[str, ...] | list[str]) -> None:
        """Raise if any forbidden participant contributed to these statistics.

        Called with the outer-test participant before every final evaluation. Checks
        ``fitted_on`` -- the label-bearing training fit -- and deliberately not
        ``calibrated_on``: a declared calibration block is allowed to contain the held-out
        participant's unlabelled signal, and conflating the two would make this assertion
        either uselessly permissive or unable to express the calibrated protocol at all.
        Use :meth:`assert_calibration_disjoint_from` for the calibration guarantee.
        """
        leaked = sorted(set(self.fitted_on) & set(forbidden_subjects))
        if leaked:
            raise AssertionError(
                f"normalizer was fitted on held-out participant(s) {leaked}; "
                f"normalisation statistics must come from training participants only"
            )

    def assert_calibration_disjoint_from(self, evaluated_sample_ids: Sequence[str]) -> None:
        """Raise if any calibration window is also an evaluated example.

        The calibration block is excluded from training *and* evaluation. A window used to
        set a participant's scale and then scored would flatter the calibrated arm for a
        reason that has nothing to do with calibration working.
        """
        calibration_ids = {
            sample_id
            for stats in self.subject_stats.values()
            for sample_id in stats.get("sample_ids", ())
        }
        overlap = calibration_ids & set(evaluated_sample_ids)
        if overlap:
            raise AssertionError(
                f"{len(overlap)} calibration windows are also being evaluated "
                f"(e.g. {sorted(overlap)[:3]}); the calibration block must be excluded "
                f"from both training and evaluation"
            )

    def describe_calibration(self) -> dict[str, Any]:
        """Exactly which of each participant's data produced their statistics."""
        return {
            "scope": self.config.scope,
            "calibration": self.config.calibration,
            "is_transductive": self.config.is_transductive,
            "calibrated_on": list(self.calibrated_on),
            "per_subject": {
                subject: {
                    "n_calibration_windows": stats["n_windows"],
                    "n_samples": stats["n_samples"],
                    "sample_ids": stats["sample_ids"],
                }
                for subject, stats in sorted(self.subject_stats.items())
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable statistics, written into every run bundle."""
        self._require_fitted()
        assert self.emg_center is not None and self.emg_scale is not None
        return {
            "config": self.config.to_dict(),
            "emg_center": self.emg_center.tolist(),
            "emg_scale": self.emg_scale.tolist(),
            "mic_center": self.mic_center,
            "mic_scale": self.mic_scale,
            "fitted_on": list(self.fitted_on),
            "n_fit_samples": self.n_fit_samples,
            "calibrated_on": list(self.calibrated_on),
            "calibration": self.describe_calibration(),
            "subject_stats": {
                subject: {
                    "emg_center": stats["emg_center"].tolist(),
                    "emg_scale": stats["emg_scale"].tolist(),
                    "mic_center": stats["mic_center"],
                    "mic_scale": stats["mic_scale"],
                    "n_samples": stats["n_samples"],
                    "n_windows": stats["n_windows"],
                    "sample_ids": stats["sample_ids"],
                }
                for subject, stats in sorted(self.subject_stats.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Normalizer:
        config_payload = {
            key: value
            for key, value in payload["config"].items()
            if key in {"scope", "method", "calibration", "calibration_approved_by"}
        }
        subject_stats = {
            subject: {
                "emg_center": np.asarray(stats["emg_center"], dtype=np.float64),
                "emg_scale": np.asarray(stats["emg_scale"], dtype=np.float64),
                "mic_center": float(stats["mic_center"]),
                "mic_scale": float(stats["mic_scale"]),
                "n_samples": int(stats["n_samples"]),
                "n_windows": int(stats["n_windows"]),
                "sample_ids": list(stats.get("sample_ids", ())),
            }
            for subject, stats in payload.get("subject_stats", {}).items()
        }
        return cls(
            config=NormalizationConfig(**config_payload),
            emg_center=np.asarray(payload["emg_center"], dtype=np.float64),
            emg_scale=np.asarray(payload["emg_scale"], dtype=np.float64),
            mic_center=float(payload["mic_center"]),
            mic_scale=float(payload["mic_scale"]),
            fitted_on=tuple(payload.get("fitted_on", ())),
            n_fit_samples=int(payload.get("n_fit_samples", 0)),
            subject_stats=subject_stats,
            calibrated_on=tuple(payload.get("calibrated_on", ())),
        )
