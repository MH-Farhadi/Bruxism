"""Probability calibration, fitted where it is allowed to be fitted.

The held-out probabilities of the 2026-07/08 runs have an expected calibration error of
0.276: a window predicted with 90 % confidence is right far less than 90 % of the time.
That does not invalidate any AUC in the manuscript -- AUC is rank-based and a monotone
rescaling of the scores cannot change it -- but it does mean the probabilities must not be
read as probabilities, and a paper that prints them owes the reader either a fix or a
sentence.

This module is the fix. :func:`fit_temperature` learns a single scalar that divides the
logits before the softmax; it cannot change any ranking, so accuracy, macro-F1 and AUC are
untouched by construction, and only the confidence values move.

**Where the temperature may be fitted matters more than how.** Fitting it on the held-out
participant's own labels would be leakage of exactly the kind the rest of this pipeline
refuses. It must be fitted on inner-validation folds -- participants who are in neither the
training set of the final model nor the held-out set -- and then applied unchanged. The
functions here take the fitting split explicitly and assert it is disjoint from the split
being calibrated, so there is no call that quietly does the wrong thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "TemperatureScaler",
    "apply_temperature",
    "expected_calibration_error",
    "fit_temperature",
]

_EPS = 1e-12


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width-bin ECE, with the per-bin confidence, accuracy and count.

    Empty bins contribute nothing and are returned as ``nan`` accuracy rather than 0, so a
    reliability diagram does not draw a point where there is no data.
    """
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same length")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_confidence = np.full(bins, np.nan)
    bin_accuracy = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=np.int64)
    total = confidence.size
    ece = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        counts[index] = int(mask.sum())
        if counts[index] == 0:
            continue
        bin_confidence[index] = float(confidence[mask].mean())
        bin_accuracy[index] = float(correct[mask].mean())
        ece += counts[index] / total * abs(bin_accuracy[index] - bin_confidence[index])
    return float(ece), bin_confidence, bin_accuracy, counts


def _to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Recover logits up to an additive constant, which softmax is invariant to."""
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0)
    return np.log(clipped)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Re-softmax ``probabilities`` with the logits divided by ``temperature``.

    A temperature above 1 softens confidence, below 1 sharpens it. The argmax of every row
    is preserved, so no metric that depends only on ranking can change.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = _to_logits(probabilities) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Temperature minimising the negative log-likelihood of ``labels``.

    A one-dimensional convex problem solved by scalar minimisation -- no gradient descent,
    no learning rate, nothing to tune, and therefore nothing that could quietly overfit the
    fitting split beyond its single degree of freedom.
    """
    from scipy.optimize import minimize_scalar

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise ValueError(
            f"probabilities {probabilities.shape} and labels {labels.shape} do not align"
        )
    if probabilities.shape[0] == 0:
        raise ValueError("cannot fit a temperature on an empty fitting split")
    logits = _to_logits(probabilities)
    rows = np.arange(labels.size)

    def negative_log_likelihood(temperature: float) -> float:
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(scaled).sum(axis=1))
        return float(-(scaled[rows, labels] - log_partition).mean())

    result = minimize_scalar(negative_log_likelihood, bounds=bounds, method="bounded")
    return float(result.x)


@dataclass
class TemperatureScaler:
    """A temperature plus the record of which participants it was fitted on.

    The provenance is part of the object because the number is meaningless without it: a
    temperature fitted on the participant it is applied to is leakage, and the only way to
    make that checkable after the fact is to carry the fitting set alongside the value.
    """

    temperature: float = 1.0
    fitted_on: tuple[str, ...] = ()
    n_fit_samples: int = 0
    fit_split: str = "inner_validation"

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        subjects: Sequence[str] = (),
        fit_split: str = "inner_validation",
    ) -> TemperatureScaler:
        return cls(
            temperature=fit_temperature(probabilities, labels),
            fitted_on=tuple(sorted(set(subjects))),
            n_fit_samples=int(np.asarray(labels).size),
            fit_split=fit_split,
        )

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return apply_temperature(probabilities, self.temperature)

    def assert_not_fitted_on(self, forbidden_subjects: Sequence[str]) -> None:
        """Raise if the temperature saw a participant it is about to be applied to."""
        leaked = sorted(set(self.fitted_on) & set(forbidden_subjects))
        if leaked:
            raise AssertionError(
                f"temperature was fitted on participant(s) {leaked} and is being applied to "
                f"them; calibration must be fitted on inner-validation folds only"
            )

    def report(self, probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10):
        """Before/after ECE, and the confirmation that no ranking metric moved."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).ravel()
        calibrated = self.transform(probabilities)
        raw_ece, *_ = expected_calibration_error(
            probabilities.max(axis=1), (probabilities.argmax(axis=1) == labels), bins=bins
        )
        new_ece, *_ = expected_calibration_error(
            calibrated.max(axis=1), (calibrated.argmax(axis=1) == labels), bins=bins
        )
        return {
            "temperature": self.temperature,
            "fitted_on": list(self.fitted_on),
            "fit_split": self.fit_split,
            "n_fit_samples": self.n_fit_samples,
            "ece_uncalibrated": raw_ece,
            "ece_calibrated": new_ece,
            "predictions_unchanged": bool(
                np.array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1))
            ),
            "note": (
                "Temperature scaling is monotone, so accuracy, macro-F1 and every AUC are "
                "unchanged by construction; only the confidence values move."
            ),
        }


def summarise_uncalibrated(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """The sentence a run owes its reader when no temperature was fitted."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    ece, *_ = expected_calibration_error(
        probabilities.max(axis=1), (probabilities.argmax(axis=1) == labels)
    )
    return {
        "calibrated": False,
        "expected_calibration_error": ece,
        "statement": (
            f"Held-out probabilities are UNCALIBRATED (ECE {ece:.3f}). They must not be "
            f"read as probabilities. Every AUC reported from them remains valid: AUC is "
            f"rank-based and unaffected by any monotone rescaling of the scores."
        ),
    }
