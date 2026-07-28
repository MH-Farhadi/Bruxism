"""Normalisation statistics, fitted on training data only.

A :class:`Normalizer` is *fitted* once per split stage -- inner-training participants when
selecting hyperparameters, outer-training participants when fitting the final model -- and
then applied unchanged to the validation or held-out data. Applying an unfitted normalizer
raises; there is no lazy self-fitting path that could quietly absorb test statistics.

The fitted parameters are serialised into every run bundle so a reviewer can confirm which
participants produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        ``"global"`` pools every channel into a single statistic.
    method
        ``"zscore"`` uses mean/std. ``"robust"`` uses median and a MAD-derived scale, which
        is markedly less sensitive to the amplifier-settling transients present in these
        recordings.
    """

    scope: Literal["per_channel", "global"] = "per_channel"
    method: Literal["zscore", "robust"] = "zscore"

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "method": self.method}


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

        axis = 0 if self.config.scope == "per_channel" else None
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

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                "normalizer has not been fitted; fit it on training participants before "
                "transforming any data (never fit on validation or held-out data)"
            )

    def transform_emg(self, emg: np.ndarray) -> np.ndarray:
        """Apply fitted EMG statistics to ``(..., n_samples, n_channels)`` data."""
        self._require_fitted()
        assert self.emg_center is not None and self.emg_scale is not None
        return (np.asarray(emg, dtype=np.float64) - self.emg_center) / self.emg_scale

    def transform_mic(self, mic: np.ndarray) -> np.ndarray:
        """Apply fitted microphone statistics."""
        self._require_fitted()
        assert self.mic_center is not None and self.mic_scale is not None
        return (np.asarray(mic, dtype=np.float64) - self.mic_center) / self.mic_scale

    def assert_not_fitted_on(self, forbidden_subjects: tuple[str, ...] | list[str]) -> None:
        """Raise if any forbidden participant contributed to these statistics.

        Called with the outer-test participant before every final evaluation.
        """
        leaked = sorted(set(self.fitted_on) & set(forbidden_subjects))
        if leaked:
            raise AssertionError(
                f"normalizer was fitted on held-out participant(s) {leaked}; "
                f"normalisation statistics must come from training participants only"
            )

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Normalizer:
        return cls(
            config=NormalizationConfig(**payload["config"]),
            emg_center=np.asarray(payload["emg_center"], dtype=np.float64),
            emg_scale=np.asarray(payload["emg_scale"], dtype=np.float64),
            mic_center=float(payload["mic_center"]),
            mic_scale=float(payload["mic_scale"]),
            fitted_on=tuple(payload.get("fitted_on", ())),
            n_fit_samples=int(payload.get("n_fit_samples", 0)),
        )
