"""The recording file schema, and strict validation against it.

The acquisition software wrote every recording as a six-column CSV. Nothing here guesses:
a file whose header, dtypes, trigger alphabet or sampling rate departs from the declared
schema raises :class:`SchemaError` rather than being coerced.

Channel semantics (confirmed by the investigator, 2026-07-27; see
``docs/open_questions.md`` Q2 for what remains open):
the four EMG columns are four differential signals recorded from two bilateral bipolar
pairs -- one pair per side of the head. Physical units are **unknown** and are recorded as
``arbitrary_adc_units`` everywhere. Do not label them uV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "EMG_COLUMNS",
    "EMG_MUSCLE_MAP",
    "FILENAME_PATTERN",
    "MIC_COLUMN",
    "NOMINAL_SAMPLING_RATE_HZ",
    "SIGNAL_UNITS",
    "TRIGGER_COLUMN",
    "TRIGGER_VALUES",
    "RecordingKey",
    "SchemaError",
    "parse_recording_filename",
    "read_recording_csv",
    "validate_columns",
]


class SchemaError(ValueError):
    """Raised when a file does not match the declared recording schema."""


EMG_COLUMNS: Final[tuple[str, ...]] = (
    "EMG1_1-2",
    "EMG2_3-4",
    "EMG3_5-6",
    "EMG4_7-8",
)
TRIGGER_COLUMN: Final[str] = "Trigger"
MIC_COLUMN: Final[str] = "Mic"
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (*EMG_COLUMNS, TRIGGER_COLUMN, MIC_COLUMN)

#: Tentative electrode map from ``Data/README.txt``. Laterality, electrode type, amplifier
#: and gain remain unconfirmed -- see ``docs/open_questions.md`` Q3.
EMG_MUSCLE_MAP: Final[dict[str, str]] = {
    "EMG1_1-2": "left_masseter",
    "EMG2_3-4": "left_temporalis",
    "EMG3_5-6": "right_masseter",
    "EMG4_7-8": "right_temporalis",
}

#: Physical units were never documented by the acquisition chain. Every artifact reports
#: this string rather than inventing uV / Pa / dB.
SIGNAL_UNITS: Final[str] = "arbitrary_adc_units"

NOMINAL_SAMPLING_RATE_HZ: Final[int] = 1200

#: The only trigger values the acquisition software is known to emit.
TRIGGER_VALUES: Final[tuple[float, ...]] = (0.0, 1.0)

#: ``<condition_token>_<subject>_<yyyymmdd>_<hhmmss>``; the condition token itself may
#: contain underscores (``deviation_left_right``), so it is captured non-greedily.
FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<condition>.+?)_(?P<subject>\d+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)


@dataclass(frozen=True, order=True)
class RecordingKey:
    """Identity of one recording, parsed from its filename stem.

    Attributes
    ----------
    subject_id
        Canonical participant identifier, e.g. ``"S01"``. Never a participant name.
    condition_token
        The raw token from the filename, preserved verbatim (``natural_bruxing`` stays
        ``natural_bruxing`` here; scientific renaming happens in :mod:`bruxism.data.labels`).
    date, time
        Acquisition timestamp tokens, used as the repetition discriminator.
    """

    subject_id: str
    condition_token: str
    date: str
    time: str

    @property
    def subject_number(self) -> int:
        return int(self.subject_id.removeprefix("S"))

    @property
    def repetition_token(self) -> str:
        """The timestamp token that distinguishes repeats of the same condition."""
        return f"{self.date}_{self.time}"

    @property
    def stem(self) -> str:
        return f"{self.condition_token}_{self.subject_number}_{self.date}_{self.time}"

    @property
    def recording_id(self) -> str:
        """Stable, identifier-free ID used in manifests and prediction ledgers."""
        return f"{self.subject_id}_{self.condition_token}_{self.date}T{self.time}"


def parse_recording_filename(name: str) -> RecordingKey:
    """Parse ``<condition>_<subject>_<date>_<time>`` from a file name or stem.

    Raises
    ------
    SchemaError
        If the name does not match :data:`FILENAME_PATTERN`.
    """
    stem = Path(name).stem
    stem = stem.removesuffix("_metadata")
    match = FILENAME_PATTERN.match(stem)
    if match is None:
        raise SchemaError(
            f"filename {name!r} does not match the expected "
            f"'<condition>_<subject>_<yyyymmdd>_<hhmmss>' pattern"
        )
    return RecordingKey(
        subject_id=f"S{int(match['subject']):02d}",
        condition_token=match["condition"],
        date=match["date"],
        time=match["time"],
    )


def validate_columns(columns: list[str] | tuple[str, ...] | pd.Index) -> None:
    """Assert that ``columns`` is exactly the canonical schema, in order.

    Reordering is rejected as well as renaming: downstream code addresses EMG channels
    positionally, so a silently reordered file would swap muscles without any error.
    """
    actual = tuple(str(c) for c in columns)
    if actual == CANONICAL_COLUMNS:
        return
    missing = [c for c in CANONICAL_COLUMNS if c not in actual]
    unexpected = [c for c in actual if c not in CANONICAL_COLUMNS]
    detail = []
    if missing:
        detail.append(f"missing={missing}")
    if unexpected:
        detail.append(f"unexpected={unexpected}")
    if not detail:
        detail.append("columns are reordered relative to the canonical schema")
    raise SchemaError(
        f"column schema mismatch: expected {list(CANONICAL_COLUMNS)}, got {list(actual)} "
        f"({'; '.join(detail)})"
    )


def read_recording_csv(path: Path | str, *, validate: bool = True) -> pd.DataFrame:
    """Read one recording CSV and validate it against the declared schema.

    Parameters
    ----------
    path
        Path to a ``*.csv`` recording.
    validate
        When true (default) the column schema, numeric dtypes, finiteness and trigger
        alphabet are all checked. Turning it off is only for the legacy-diagnostic path.

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`CANONICAL_COLUMNS`; EMG/Trigger as ``float64``, ``Mic`` as
        ``float64`` (it is integer-valued on disk but kept float for uniform maths).

    Raises
    ------
    SchemaError
        On any schema, dtype, finiteness or trigger-alphabet violation.
    """
    resolved = Path(path)
    try:
        frame = pd.read_csv(resolved)
    except Exception as exc:  # pragma: no cover - pandas raises many parser types
        raise SchemaError(f"failed to parse {resolved}: {exc}") from exc

    if validate:
        validate_columns(frame.columns)

    for column in CANONICAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaError(
                f"{resolved.name}: column {column!r} is {frame[column].dtype}, expected numeric"
            )

    frame = frame.astype(dict.fromkeys(CANONICAL_COLUMNS, "float64"))

    if validate:
        values = frame[list(CANONICAL_COLUMNS)].to_numpy()
        if not np.isfinite(values).all():
            bad = {
                column: int((~np.isfinite(frame[column].to_numpy())).sum())
                for column in CANONICAL_COLUMNS
                if not np.isfinite(frame[column].to_numpy()).all()
            }
            raise SchemaError(f"{resolved.name}: non-finite values present: {bad}")

        observed = np.unique(frame[TRIGGER_COLUMN].to_numpy())
        unexpected = sorted(set(observed.tolist()) - set(TRIGGER_VALUES))
        if unexpected:
            raise SchemaError(
                f"{resolved.name}: trigger contains values outside "
                f"{list(TRIGGER_VALUES)}: {unexpected}"
            )
    return frame
