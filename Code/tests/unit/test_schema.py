"""Schema parsing and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from bruxism.data.schema import (
    CANONICAL_COLUMNS,
    SchemaError,
    parse_recording_filename,
    read_recording_csv,
    validate_columns,
)


def test_accepts_the_exact_six_column_schema(synthetic_root):
    csv = next(synthetic_root.rglob("*.csv"))
    frame = read_recording_csv(csv)
    assert tuple(frame.columns) == CANONICAL_COLUMNS
    assert all(str(frame[c].dtype) == "float64" for c in CANONICAL_COLUMNS)


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(CANONICAL_COLUMNS[:-1], id="missing_mic"),
        pytest.param((*CANONICAL_COLUMNS, "Extra"), id="unexpected_column"),
        pytest.param(("Mic", *CANONICAL_COLUMNS[:-1]), id="reordered"),
        pytest.param(tuple(c.lower() for c in CANONICAL_COLUMNS), id="renamed"),
    ],
)
def test_rejects_non_canonical_columns(columns):
    with pytest.raises(SchemaError):
        validate_columns(list(columns))


def test_rejects_non_numeric_column(tmp_path):
    frame = pd.DataFrame({c: [0.0, 1.0] for c in CANONICAL_COLUMNS})
    frame["Mic"] = ["a", "b"]
    path = tmp_path / "rest_1_20250804_100000.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="expected numeric"):
        read_recording_csv(path)


def test_rejects_non_finite_values(tmp_path):
    frame = pd.DataFrame({c: [0.0, 1.0] for c in CANONICAL_COLUMNS})
    frame.loc[0, "EMG1_1-2"] = float("nan")
    path = tmp_path / "rest_1_20250804_100000.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="non-finite"):
        read_recording_csv(path)


def test_rejects_unexpected_trigger_values(tmp_path):
    frame = pd.DataFrame({c: [0.0, 1.0] for c in CANONICAL_COLUMNS})
    frame.loc[1, "Trigger"] = 2.0
    path = tmp_path / "rest_1_20250804_100000.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="trigger contains values outside"):
        read_recording_csv(path)


@pytest.mark.parametrize(
    ("name", "condition", "subject", "recording_id"),
    [
        ("rest_1_20250804_102808.csv", "rest", "S01", "S01_rest_20250804T102808"),
        (
            "deviation_left_right_5_20250807_145256.csv",
            "deviation_left_right",
            "S05",
            "S05_deviation_left_right_20250807T145256",
        ),
        (
            "molar_clench_5_20250807_145916_metadata.txt",
            "molar_clench",
            "S05",
            "S05_molar_clench_20250807T145916",
        ),
    ],
)
def test_parses_filenames_including_multiword_conditions(name, condition, subject, recording_id):
    key = parse_recording_filename(name)
    assert key.condition_token == condition
    assert key.subject_id == subject
    assert key.recording_id == recording_id


def test_rejects_unparseable_filename():
    with pytest.raises(SchemaError, match="does not match"):
        parse_recording_filename("not_a_recording.csv")
