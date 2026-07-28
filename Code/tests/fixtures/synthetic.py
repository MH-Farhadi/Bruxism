"""Synthetic dataset generator used by the whole test suite.

Produces files that match the real acquisition layout, schema, filename pattern and
metadata format -- but every sample is generated from a seeded RNG. Nothing here is derived
from, or reversible to, any participant recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bruxism.data.schema import CANONICAL_COLUMNS

__all__ = ["SyntheticDatasetSpec", "make_recording", "write_synthetic_dataset"]


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    """What the generated dataset should contain."""

    n_subjects: int = 5
    sampling_rate_hz: int = 1200
    duration_seconds: float = 20.0
    #: (condition token, number of trigger runs per recording, repeats)
    conditions: tuple[tuple[str, int, int], ...] = (
        ("rest", 0, 1),
        ("open_close", 4, 1),
        ("deviation_left_right", 4, 1),
        ("protrusion_retrusion", 4, 1),
        ("bite_left", 3, 1),
        ("bite_right", 3, 1),
        ("molar_clench", 3, 1),
        ("incisor_clench", 3, 1),
        ("natural_bruxing", 3, 1),
        ("cheese", 2, 1),
        ("carrots", 2, 1),
        ("gum", 2, 1),
    )
    #: Anomalies deliberately injected so the validation gate has something to catch.
    short_recording: tuple[int, str] = (2, "gum")
    metadata_conflict: tuple[int, str, str] = (5, "molar_clench", "incisor_clench")
    secondary_location: tuple[int, str] = (5, "rest")
    startup_transient: tuple[int, str] = (3, "open_close")
    base_date: str = "20250804"
    seed: int = 20260727
    _unused: tuple[str, ...] = field(default_factory=tuple, repr=False)


def _amplitude_for(condition: str) -> float:
    """Class-dependent EMG amplitude so a model can actually learn something."""
    return {
        "rest": 0.3,
        "open_close": 1.0,
        "deviation_left_right": 1.1,
        "protrusion_retrusion": 0.9,
        "bite_left": 3.0,
        "bite_right": 3.1,
        "molar_clench": 3.4,
        "incisor_clench": 3.2,
        "natural_bruxing": 2.6,
        "cheese": 2.0,
        "carrots": 2.2,
        "gum": 1.9,
    }.get(condition, 1.0)


def make_recording(
    condition: str,
    n_runs: int,
    spec: SyntheticDatasetSpec,
    *,
    seed: int,
    n_samples: int | None = None,
    with_startup_transient: bool = False,
) -> pd.DataFrame:
    """Generate one recording with the canonical six-column schema."""
    rng = np.random.default_rng(seed)
    total = (
        n_samples if n_samples is not None else int(spec.duration_seconds * spec.sampling_rate_hz)
    )
    t = np.arange(total) / spec.sampling_rate_hz

    trigger = np.zeros(total, dtype=np.float64)
    if n_runs > 0:
        # Evenly spaced runs with a margin at each end, so guards have something to cut.
        margin = int(1.5 * spec.sampling_rate_hz)
        usable = total - 2 * margin
        block = usable // n_runs
        run_length = int(block * 0.6)
        for i in range(n_runs):
            start = margin + i * block
            trigger[start : start + run_length] = 1.0

    amplitude = _amplitude_for(condition)
    active = trigger > 0
    emg = rng.standard_normal((total, 4)) * 0.4
    # Class-specific band energy injected only where the trigger is high.
    frequency = 40.0 + 25.0 * (hash(condition) % 7)
    burst = amplitude * np.sin(2 * np.pi * frequency * t)[:, None]
    emg[active] += burst[active] * np.array([1.0, 0.8, 0.9, 0.7])
    emg *= 1000.0  # integer-ish ADC-like scale, like the real recordings

    mic = 150.0 + rng.standard_normal(total) * 2.0
    # Chewing-like conditions get a loud audio signature, mirroring the real shortcut.
    if condition in ("cheese", "carrots", "gum"):
        mic[active] += 30.0 * np.abs(np.sin(2 * np.pi * 4.0 * t))[active]
    mic = np.round(mic)

    if with_startup_transient:
        emg[:200] += 6.0e4

    return pd.DataFrame(
        {
            CANONICAL_COLUMNS[0]: emg[:, 0],
            CANONICAL_COLUMNS[1]: emg[:, 1],
            CANONICAL_COLUMNS[2]: emg[:, 2],
            CANONICAL_COLUMNS[3]: emg[:, 3],
            CANONICAL_COLUMNS[4]: trigger,
            CANONICAL_COLUMNS[5]: mic,
        }
    )


def _metadata_text(
    subject: int, condition_key: str, stem: str, n_samples: int, rate: int, target_seconds: float
) -> str:
    pretty = condition_key.replace("_", " ").title()
    return "\n".join(
        [
            f"subject_id: {subject}",
            f"condition: {pretty}",
            f"condition_key: {condition_key}",
            "timestamp_start: 2025-08-04 10:00:00",
            "timestamp_current: 2025-08-04 10:01:00",
            "status: COMPLETED",
            f"sampling_rate: {rate}",
            f"target_duration_seconds: {target_seconds:g}",
            "elapsed_seconds: 61.0",
            f"samples_saved: {n_samples}",
            f"expected_samples: {n_samples + 17}",
            "configuration: Synthetic test fixture",
            "bandpass_filter: Index 143",
            "notch_filter: Index 9",
            f"csv_file: {stem}.csv",
            f"npy_file: {stem}.npy",
            f"video_file: {stem}.avi",
            "",
        ]
    )


def write_synthetic_dataset(root: Path, spec: SyntheticDatasetSpec) -> Path:
    """Write a complete synthetic dataset tree under ``root``."""
    root = Path(root)
    seed = spec.seed
    for subject in range(1, spec.n_subjects + 1):
        primary = root / f"Subject_{subject}"
        primary.mkdir(parents=True, exist_ok=True)
        for index, (condition, n_runs, repeats) in enumerate(spec.conditions):
            for repeat in range(repeats):
                seed += 1
                clock = f"{10 + index:02d}{30 + repeat:02d}{subject:02d}"
                stem = f"{condition}_{subject}_{spec.base_date}_{clock}"

                n_samples = None
                if spec.short_recording == (subject, condition):
                    n_samples = int(0.55 * spec.duration_seconds * spec.sampling_rate_hz)
                frame = make_recording(
                    condition,
                    n_runs,
                    spec,
                    seed=seed,
                    n_samples=n_samples,
                    with_startup_transient=spec.startup_transient == (subject, condition),
                )

                target_dir = primary
                if spec.secondary_location == (subject, condition):
                    target_dir = root / "More Data" / "Data" / f"Subject_{subject}"
                    target_dir.mkdir(parents=True, exist_ok=True)

                frame.to_csv(target_dir / f"{stem}.csv", index=False)
                # A placeholder video: the manifest records its presence, never its content.
                (target_dir / f"{stem}.avi").write_bytes(b"RIFF____AVI synthetic placeholder")

                condition_key = condition
                if spec.metadata_conflict[:2] == (subject, condition):
                    condition_key = spec.metadata_conflict[2]
                (target_dir / f"{stem}_metadata.txt").write_text(
                    _metadata_text(
                        subject,
                        condition_key,
                        stem,
                        len(frame),
                        spec.sampling_rate_hz,
                        spec.duration_seconds,
                    ),
                    encoding="utf-8",
                )
    (root / "README.txt").write_text(
        "EMG1_2: left masseter\nEMG3_4: left temporalis\n"
        "EMG5_6: right masseter\nEMG7_8: right temporalis\n",
        encoding="utf-8",
    )
    return root
