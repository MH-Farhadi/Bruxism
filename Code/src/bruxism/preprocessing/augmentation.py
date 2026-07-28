"""Training-only signal augmentation with deterministic per-sample seeding.

Two invariants are enforced structurally rather than by convention:

* :meth:`Augmenter.__call__` raises unless ``stage="train"``. Validation and held-out data
  physically cannot be augmented, so the minority-class metrics cannot be inflated.
* Randomness is drawn from a generator seeded by ``(run_seed, epoch, sample_id)``. The
  same sample in the same epoch of the same run always receives the same transformation,
  independent of worker count, batch order or shuffling.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from bruxism.utils.logging import get_logger

__all__ = ["AugmentationConfig", "Augmenter", "AugmentationStageError"]

logger = get_logger(__name__)

Stage = Literal["train", "val", "test"]


class AugmentationStageError(RuntimeError):
    """Raised when augmentation is requested for a non-training stage."""


@dataclass(frozen=True)
class AugmentationConfig:
    """Which augmentations to apply and how strongly.

    Attributes
    ----------
    enabled
        Master switch. When false the augmenter is an identity function even for training.
    probability
        Probability that a given eligible sample is augmented at all.
    minority_only
        Restrict augmentation to classes whose training count is below
        ``minority_threshold`` times the largest class count.
    amplitude_scale_range
        Multiplicative gain range, applied to EMG and microphone jointly so their relative
        levels are preserved.
    noise_std_fraction
        Standard deviation of additive Gaussian noise, as a fraction of the sample's own
        standard deviation.
    max_shift_samples
        Maximum circular time shift. Circular shifting wraps signal from one end to the
        other; with a 1 s window and a shift of at most ~4% of it, the discontinuity is
        small relative to EMG burst structure, but it is still an artifact and is recorded
        as such.
    """

    enabled: bool = True
    probability: float = 0.4
    minority_only: bool = True
    minority_threshold: float = 0.7
    amplitude_scale_range: tuple[float, float] = (0.9, 1.1)
    amplitude_scale_probability: float = 0.5
    noise_std_fraction: float = 0.02
    noise_probability: float = 0.3
    max_shift_samples: int = 50
    shift_probability: float = 0.3

    def __post_init__(self) -> None:
        for name, value in (
            ("probability", self.probability),
            ("amplitude_scale_probability", self.amplitude_scale_probability),
            ("noise_probability", self.noise_probability),
            ("shift_probability", self.shift_probability),
            ("minority_threshold", self.minority_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        low, high = self.amplitude_scale_range
        if low <= 0 or high < low:
            raise ValueError(f"amplitude_scale_range must be 0 < low <= high, got {(low, high)}")
        if self.max_shift_samples < 0:
            raise ValueError("max_shift_samples must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["amplitude_scale_range"] = list(self.amplitude_scale_range)
        return payload


def _sample_rng(run_seed: int, epoch: int, sample_id: str) -> np.random.Generator:
    """Deterministic generator for one ``(run, epoch, sample)`` triple.

    Hashing the sample id rather than using its position means the stream is independent of
    dataset ordering, shuffling and DataLoader worker count.
    """
    digest = hashlib.blake2b(f"{run_seed}|{epoch}|{sample_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


class Augmenter:
    """Applies training-only augmentation to EMG/microphone window pairs."""

    def __init__(
        self,
        config: AugmentationConfig,
        *,
        run_seed: int,
        minority_labels: frozenset[int] | set[int] | None = None,
    ):
        self.config = config
        self.run_seed = int(run_seed)
        self.minority_labels = frozenset(minority_labels or ())
        self._n_applied = 0
        self._n_seen = 0

    @staticmethod
    def minority_labels_from_counts(
        counts: dict[int, int], *, threshold: float = 0.7
    ) -> frozenset[int]:
        """Labels whose count falls below ``threshold`` times the largest class count.

        Computed from **training** counts only; the caller is responsible for passing
        counts that exclude validation and held-out participants.
        """
        if not counts:
            return frozenset()
        largest = max(counts.values())
        return frozenset(label for label, count in counts.items() if count < threshold * largest)

    @property
    def stats(self) -> dict[str, int]:
        """How many samples were seen and augmented, recorded in the run bundle."""
        return {"n_seen": self._n_seen, "n_augmented": self._n_applied}

    def __call__(
        self,
        emg: np.ndarray,
        mic: np.ndarray,
        *,
        label: int,
        sample_id: str,
        epoch: int,
        stage: Stage,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment one example.

        Parameters
        ----------
        emg
            ``(n_samples, n_channels)`` window.
        mic
            ``(n_samples,)`` window.
        stage
            Must be ``"train"``. Any other value raises :class:`AugmentationStageError`.

        Returns
        -------
        (emg, mic)
            Augmented copies, or the inputs unchanged when augmentation does not fire.
        """
        if stage != "train":
            raise AugmentationStageError(
                f"augmentation requested for stage {stage!r}; augmentation may only be "
                f"applied to training samples, never to validation or held-out data"
            )
        self._n_seen += 1
        if not self.config.enabled:
            return emg, mic
        if self.config.minority_only and label not in self.minority_labels:
            return emg, mic

        rng = _sample_rng(self.run_seed, epoch, sample_id)
        if rng.random() > self.config.probability:
            return emg, mic

        emg_out = np.array(emg, dtype=np.float64, copy=True)
        mic_out = np.array(mic, dtype=np.float64, copy=True)

        if rng.random() < self.config.amplitude_scale_probability:
            low, high = self.config.amplitude_scale_range
            # One shared gain, so EMG/audio relative level is not scrambled.
            scale = rng.uniform(low, high)
            emg_out *= scale
            mic_out *= scale

        if rng.random() < self.config.noise_probability:
            emg_std = float(np.std(emg_out))
            mic_std = float(np.std(mic_out))
            if emg_std > 0:
                emg_out += rng.normal(0.0, self.config.noise_std_fraction * emg_std, emg_out.shape)
            if mic_std > 0:
                mic_out += rng.normal(0.0, self.config.noise_std_fraction * mic_std, mic_out.shape)

        if self.config.max_shift_samples > 0 and rng.random() < self.config.shift_probability:
            shift = int(
                rng.integers(-self.config.max_shift_samples, self.config.max_shift_samples + 1)
            )
            if shift:
                # Both modalities shift together: they are time-aligned recordings of the
                # same event and desynchronising them would destroy the fusion signal.
                emg_out = np.roll(emg_out, shift, axis=0)
                mic_out = np.roll(mic_out, shift, axis=0)

        self._n_applied += 1
        return emg_out, mic_out
