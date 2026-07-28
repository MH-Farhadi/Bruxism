"""Leakage guards. These are the tests that protect the scientific claim."""

from __future__ import annotations

import numpy as np
import pytest

from bruxism.data.segments import SegmentationConfig, WindowIndex, WindowRecord
from bruxism.data.splits import (
    NestedLOSOSplitter,
    OuterFoldSealError,
    assert_group_disjoint,
)
from bruxism.preprocessing.augmentation import (
    AugmentationConfig,
    AugmentationStageError,
    Augmenter,
)
from bruxism.preprocessing.normalization import NormalizationConfig, Normalizer, NotFittedError


def _window(subject: str, index: int) -> WindowRecord:
    return WindowRecord(
        sample_id=f"{subject}#{index:04d}",
        subject_id=subject,
        recording_id=f"{subject}_rec",
        condition="rest",
        condition_token="rest",
        task_family="rest",
        repetition_token="t",
        start_sample=index * 600,
        end_sample=index * 600 + 1200,
        start_seconds=0.0,
        end_seconds=1.0,
        trigger_run_index=None,
        segment_source="dedicated_rest",
        segmentation_policy="trigger_constrained",
        safe_for_inference=True,
    )


@pytest.fixture
def five_subject_index() -> WindowIndex:
    return WindowIndex(
        windows=[_window(f"S0{s}", i) for s in range(1, 6) for i in range(6)],
        config=SegmentationConfig(),
        sampling_rate_hz=1200,
        manifest_hash="h",
    )


# ------------------------------------------------------------------- splits ---


def test_inner_loso_has_four_folds_not_five(five_subject_index):
    splitter = NestedLOSOSplitter(five_subject_index)
    assert splitter.n_outer_folds == 5
    assert splitter.n_inner_folds() == 4
    for fold in splitter.outer_folds():
        assert fold.n_inner_folds == 4, (
            "with four training participants a participant-grouped inner LOSO has four "
            "folds; five is arithmetically impossible"
        )


def test_every_participant_is_held_out_exactly_once(five_subject_index):
    held = [f.test_subject for f in NestedLOSOSplitter(five_subject_index).outer_folds()]
    assert sorted(held) == ["S01", "S02", "S03", "S04", "S05"]
    assert len(held) == len(set(held))


def test_outer_subject_never_appears_in_any_inner_split(five_subject_index):
    for fold in NestedLOSOSplitter(five_subject_index).outer_folds():
        for inner in fold.inner_folds:
            assert fold.test_subject not in inner.train_subjects
            assert fold.test_subject != inner.val_subject
            assert not set(inner.train_sample_ids) & set(fold._test_sample_ids)
            assert not set(inner.val_sample_ids) & set(fold._test_sample_ids)


@pytest.mark.parametrize(
    "purpose",
    ["hyperparameter_search", "early_stopping", "checkpoint_selection", "threshold_tuning", ""],
)
def test_outer_test_ids_are_sealed_against_selection(five_subject_index, purpose):
    fold = NestedLOSOSplitter(five_subject_index).outer_folds()[0]
    with pytest.raises(OuterFoldSealError):
        fold.release_test_ids(purpose=purpose)


def test_outer_test_ids_release_once_for_final_evaluation(five_subject_index):
    fold = NestedLOSOSplitter(five_subject_index).outer_folds()[0]
    ids = fold.release_test_ids(purpose="final_evaluation")
    assert len(ids) == 6
    with pytest.raises(OuterFoldSealError, match="already been released"):
        fold.release_test_ids(purpose="final_evaluation")


def test_train_and_test_sample_ids_are_disjoint(five_subject_index):
    for fold in NestedLOSOSplitter(five_subject_index).outer_folds():
        ids = fold.release_test_ids(purpose="final_evaluation")
        assert not set(fold.train_sample_ids) & set(ids)


def test_assert_group_disjoint_detects_overlap():
    with pytest.raises(AssertionError, match="leakage"):
        assert_group_disjoint(train=["a", "b"], test=["b", "c"])
    assert_group_disjoint(train=["a"], val=["b"], test=["c"])


def test_random_window_kfold_is_not_available():
    """A window-level K-fold splitter must not exist in the production API."""
    import bruxism.data.splits as splits

    exported = set(splits.__all__)
    for forbidden in ("KFold", "random_split", "window_kfold", "StratifiedKFold"):
        assert forbidden not in exported
    assert not any("kfold" in name.lower() for name in dir(splits))


def test_too_few_participants_is_rejected():
    index = WindowIndex(
        windows=[_window(f"S0{s}", i) for s in range(1, 3) for i in range(3)],
        config=SegmentationConfig(),
        sampling_rate_hz=1200,
        manifest_hash="h",
    )
    with pytest.raises(ValueError, match="at least 3 participants"):
        NestedLOSOSplitter(index)


# ------------------------------------------------------------ normalisation ---


def test_normalizer_must_be_fitted_before_use():
    normalizer = Normalizer(NormalizationConfig())
    with pytest.raises(NotFittedError):
        normalizer.transform_emg(np.zeros((10, 4)))


def test_normalizer_records_and_guards_its_training_participants(rng):
    normalizer = Normalizer(NormalizationConfig()).fit(
        rng.standard_normal((500, 4)), rng.standard_normal(500), subjects=("S02", "S03")
    )
    assert normalizer.fitted_on == ("S02", "S03")
    normalizer.assert_not_fitted_on(["S01"])
    with pytest.raises(AssertionError, match="fitted on held-out participant"):
        normalizer.assert_not_fitted_on(["S02"])


def test_normalizer_changes_with_training_data_not_with_held_out_data(rng):
    train_a = rng.standard_normal((800, 4))
    train_b = rng.standard_normal((800, 4)) * 5.0 + 3.0
    mic = rng.standard_normal(800)

    a = Normalizer(NormalizationConfig()).fit(train_a, mic, subjects=("S02",))
    a_again = Normalizer(NormalizationConfig()).fit(train_a, mic, subjects=("S02",))
    b = Normalizer(NormalizationConfig()).fit(train_b, mic, subjects=("S03",))

    # Same training data -> identical statistics, regardless of what is held out.
    assert np.allclose(a.emg_center, a_again.emg_center)
    assert np.allclose(a.emg_scale, a_again.emg_scale)
    # Different training data -> different statistics.
    assert not np.allclose(a.emg_center, b.emg_center)


def test_normalizer_roundtrips_through_serialisation(rng):
    original = Normalizer(NormalizationConfig(method="robust")).fit(
        rng.standard_normal((400, 4)), rng.standard_normal(400), subjects=("S01",)
    )
    restored = Normalizer.from_dict(original.to_dict())
    probe = rng.standard_normal((20, 4))
    assert np.allclose(original.transform_emg(probe), restored.transform_emg(probe))


# ------------------------------------------------------------- augmentation ---


@pytest.mark.parametrize("stage", ["val", "test"])
def test_augmentation_refuses_non_training_stages(stage, rng):
    augmenter = Augmenter(AugmentationConfig(), run_seed=0, minority_labels={0})
    with pytest.raises(AugmentationStageError, match="only be applied to training"):
        augmenter(
            rng.standard_normal((1200, 4)),
            rng.standard_normal(1200),
            label=0,
            sample_id="x",
            epoch=0,
            stage=stage,
        )


def test_augmentation_is_deterministic_per_run_epoch_and_sample(rng):
    emg, mic = rng.standard_normal((1200, 4)), rng.standard_normal(1200)
    kwargs = {"label": 0, "sample_id": "S01#0001", "epoch": 3, "stage": "train"}
    a = Augmenter(AugmentationConfig(probability=1.0), run_seed=7, minority_labels={0})
    b = Augmenter(AugmentationConfig(probability=1.0), run_seed=7, minority_labels={0})
    out_a = a(emg, mic, **kwargs)
    out_b = b(emg, mic, **kwargs)
    assert np.allclose(out_a[0], out_b[0]) and np.allclose(out_a[1], out_b[1])

    # A different epoch or a different sample gives a different transformation.
    other = a(emg, mic, **{**kwargs, "epoch": 4})
    assert not np.allclose(out_a[0], other[0])


def test_augmentation_skips_non_minority_classes_when_configured(rng):
    emg, mic = rng.standard_normal((1200, 4)), rng.standard_normal(1200)
    augmenter = Augmenter(
        AugmentationConfig(probability=1.0, minority_only=True),
        run_seed=0,
        minority_labels={1},
    )
    untouched = augmenter(emg, mic, label=0, sample_id="a", epoch=0, stage="train")
    assert untouched[0] is emg and untouched[1] is mic


def test_minority_labels_derive_from_training_counts_only():
    labels = Augmenter.minority_labels_from_counts({0: 100, 1: 60, 2: 95}, threshold=0.7)
    assert labels == frozenset({1})
