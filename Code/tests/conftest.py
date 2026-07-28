"""Shared fixtures.

Every fixture is **synthetic**. No participant recording, video, survey, photograph or
administrative file is ever read by the test suite, so the whole suite runs on a machine
that has no access to the private data root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.synthetic import SyntheticDatasetSpec, write_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic dataset with the same layout and schema as the real one.

    Includes, deliberately:

    * five participants, so nested LOSO has the same shape as the real experiment;
    * a dedicated rest recording per participant with a flat-zero trigger;
    * one short recording;
    * one recording whose metadata condition contradicts its filename;
    * one participant whose files live in a secondary ``More Data`` directory;
    * one recording with a startup transient.
    """
    root = tmp_path_factory.mktemp("synthetic_data")
    write_synthetic_dataset(root, SyntheticDatasetSpec())
    return root


@pytest.fixture(scope="session")
def synthetic_manifest(synthetic_root: Path):
    from bruxism.data.manifest import build_manifest

    return build_manifest(synthetic_root, probe_video=False)


@pytest.fixture(scope="session")
def synthetic_window_index(synthetic_manifest):
    from bruxism.data.segments import SegmentationConfig, build_window_index

    return build_window_index(
        synthetic_manifest,
        SegmentationConfig(guard_seconds=0.25, startup_guard_seconds=0.5),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260727)
