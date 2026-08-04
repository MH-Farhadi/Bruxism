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
from bruxism.preprocessing.calibration import build_calibration_block
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


# --------------------------------------- per-participant calibration (Phase 3) ---


def _mixed_index() -> WindowIndex:
    """Five participants, each with a rest recording and two active families.

    30 rest windows per participant, so the default 20-window calibration cap still leaves
    the rest class trainable -- the condition ``_assert_leaves_every_class_trainable``
    exists to enforce.
    """
    windows: list[WindowRecord] = []
    for subject in (f"S0{s}" for s in range(1, 6)):
        for index in range(30):
            windows.append(_window(subject, index))
        for family, offset in (("clench", 100), ("chewing", 200)):
            for run in range(3):
                for index in range(2):
                    position = offset + run * 10 + index
                    window = _window(subject, position)
                    windows.append(
                        WindowRecord(
                            **{
                                **window.to_row(),
                                "recording_id": f"{subject}_{family}",
                                "condition": family,
                                "condition_token": family,
                                "task_family": family,
                                "trigger_run_index": run,
                                "segment_source": "trigger_active",
                            }
                        )
                    )
    return WindowIndex(
        windows=windows,
        config=SegmentationConfig(),
        sampling_rate_hz=1200,
        manifest_hash="h",
    )


def test_participant_scope_requires_a_declared_calibration_source():
    """The transductive scope cannot be selected without saying what calibrates it."""
    with pytest.raises(ValueError, match="requires a calibration source"):
        NormalizationConfig(scope="per_participant")
    with pytest.raises(ValueError, match="only applies to"):
        NormalizationConfig(scope="per_channel", calibration="rest_plus_one_repetition")
    with pytest.raises(ValueError, match="upper bound"):
        NormalizationConfig(scope="per_participant", calibration="all_windows_upper_bound")


def test_calibration_block_is_rest_plus_one_repetition_per_family():
    block = build_calibration_block(_mixed_index())
    detail = block.detail["S01"]
    assert set(detail["by_family"]) == {"rest", "clench", "chewing"}
    # Exactly one trigger run per active family -- a fitting session, not a whole session.
    assert detail["by_family"]["clench"]["runs"] == ["0"]
    assert detail["by_family"]["chewing"]["runs"] == ["0"]
    assert block.to_dict()["uses_held_out_labels"] is False
    # Capped: a fitting session records a little rest, not the participant's only rest
    # recording in its entirety.
    assert detail["by_family"]["rest"]["n_windows"] == 20


def test_a_calibration_block_that_would_consume_a_whole_class_is_refused():
    """The rest class has one recording per participant; an uncapped block eats it."""
    index = _mixed_index()
    with pytest.raises(ValueError, match="would consume every window"):
        build_calibration_block(index, max_windows_per_family=1000)


def test_calibration_windows_are_withheld_from_every_split():
    """A window that set a participant's scale is neither trained on nor scored."""
    index = _mixed_index()
    block = build_calibration_block(index)
    splitter = NestedLOSOSplitter(index, exclude_sample_ids=block.all_sample_ids)
    for fold in splitter.outer_folds():
        assert not set(fold.train_sample_ids) & block.all_sample_ids
        released = fold.release_test_ids(purpose="final_evaluation")
        assert not set(released) & block.all_sample_ids
        for inner in fold.inner_folds:
            assert not set(inner.train_sample_ids) & block.all_sample_ids
            assert not set(inner.val_sample_ids) & block.all_sample_ids


def test_participant_calibration_never_touches_held_out_labels(rng):
    """The calibrated normalizer sees signal only, and says so in its own bookkeeping."""
    config = NormalizationConfig(scope="per_participant", calibration="rest_plus_one_repetition")
    normalizer = Normalizer(config).fit(
        rng.standard_normal((500, 4)), rng.standard_normal(500), subjects=("S01", "S02")
    )
    normalizer.calibrate(
        "S05",
        7.0 + 3.0 * rng.standard_normal((200, 4)),
        rng.standard_normal(200),
        sample_ids=("S05#0001",),
    )

    # The held-out participant is calibrated but NOT fitted on: the distinction the whole
    # protocol rests on.
    assert normalizer.calibrated_on == ("S05",)
    assert "S05" not in normalizer.fitted_on
    normalizer.assert_not_fitted_on(["S05"])  # must not raise

    # `calibrate` has no label parameter to misuse.
    import inspect

    assert "label" not in inspect.signature(Normalizer.calibrate).parameters

    # A calibration window that is also evaluated is refused.
    with pytest.raises(AssertionError, match="calibration windows are also being evaluated"):
        normalizer.assert_calibration_disjoint_from(["S05#0001", "S05#0002"])
    normalizer.assert_calibration_disjoint_from(["S05#0002"])


def test_calibrated_statistics_are_applied_per_participant(rng):
    """A calibrated participant is standardised by their own scale, not the pooled one."""
    config = NormalizationConfig(scope="per_participant", calibration="rest_plus_one_repetition")
    normalizer = Normalizer(config).fit(
        rng.standard_normal((500, 4)), rng.standard_normal(500), subjects=("S01",)
    )
    # S05 records at 10x the amplitude of everyone else.
    block = 10.0 * rng.standard_normal((400, 4))
    normalizer.calibrate("S05", block, rng.standard_normal(400), sample_ids=("S05#0001",))

    window = 10.0 * rng.standard_normal((300, 4))
    calibrated = normalizer.transform_emg(window, subject="S05")
    pooled = normalizer.transform_emg(window, subject="S01")
    assert abs(calibrated.std() - 1.0) < 0.2, "calibrated participant should be unit scale"
    assert pooled.std() > 5.0, "the pooled statistic cannot align a 10x participant"


def test_calibrate_is_refused_when_the_scope_would_ignore_it():
    normalizer = Normalizer(NormalizationConfig(scope="per_channel"))
    with pytest.raises(ValueError, match="requires scope='per_participant'"):
        normalizer.calibrate("S01", np.ones((10, 4)), np.ones(10))


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


def test_every_config_to_dict_round_trips_through_the_loader(tmp_path):
    """A run bundle must be readable by the code that wrote it.

    ``resolved_config.yaml`` is re-loaded by ``bruxism-figures`` and by every resumed run,
    and ExperimentConfig rejects unknown keys by design -- so any derived entry in a
    ``to_dict()`` makes the bundle unloadable. ``is_transductive`` was exactly that bug,
    and it only surfaced in an integration test 20 minutes deep.
    """
    from bruxism.config import ExperimentConfig, load_experiment_config
    from bruxism.utils.io import write_yaml

    for scope, calibration in (
        ("per_channel", "none"),
        ("global", "none"),
        ("per_participant", "rest_plus_one_repetition"),
    ):
        config = ExperimentConfig(
            name="roundtrip",
            normalization=NormalizationConfig(scope=scope, calibration=calibration),
        )
        path = tmp_path / f"{scope}.yaml"
        write_yaml(path, config.to_dict())
        reloaded = load_experiment_config(path)
        assert reloaded.to_dict() == config.to_dict()
        assert reloaded.config_hash == config.config_hash
