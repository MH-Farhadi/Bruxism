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


# ------------------------------------------------------- channel identity ---
#
# Everything below this heading exists because of one omission. Every other test in this
# file checks that *indices and statistics* stay on the right side of the split -- which
# fold a sample_id is in, which participants a normalizer saw, which stage may augment.
# None of them ever asked whether the held-out participant's SIGNAL was also in the
# training set. In the real dataset it was: 83 of 100 recordings carry a microphone
# waveform that is bit-identical, after a circular rotation, to another participant's
# recording of the same condition (``audio.md`` 1.1). A leakage suite that cannot see that
# is checking the bookkeeping and not the science.


def test_no_measured_channel_waveform_is_shared_across_subjects(synthetic_manifest):
    """No measured channel of any recording may reappear under a different participant.

    This is the test that would have caught the microphone defect a year early. It runs on
    the clean synthetic dataset, where it must pass; the companion test below injects the
    defect and requires it to fail.

    "Measured" means the four EMG columns and the microphone. The trigger is deliberately
    excluded: it is binary and protocol-driven, so a fingerprint taken over sorted samples
    collides whenever two recordings merely share a duty cycle -- every all-zero rest
    trigger matches every other one. That collision carries no information about copied
    data, and asserting on it would make this test fire on a correct dataset, which is the
    fastest way to get a leakage test disabled.
    """
    from bruxism.preprocessing.mic_integrity import duplicate_groups

    records = synthetic_manifest.records
    subject_of = {record.recording_id: record.subject_id for record in records}

    channels: dict[str, dict[str, str]] = {
        "mic": {r.recording_id: r.mic_sorted_sha256 for r in records}
    }
    for index in range(len(records[0].emg_sorted_sha256)):
        channels[f"emg{index + 1}"] = {r.recording_id: r.emg_sorted_sha256[index] for r in records}

    offences = {
        channel: groups
        for channel, fingerprints in channels.items()
        if (
            groups := duplicate_groups(fingerprints, subject_of=subject_of, cross_subject_only=True)
        )
    }
    assert not offences, (
        f"measured channel waveforms are shared across participants: {offences}. A "
        f"held-out participant whose signal is already in the training fold is not held out."
    )
    # The trigger is still reported, and still labelled as what it is.
    assert synthetic_manifest.duplication["trigger"]["informational"] is True


def test_the_channel_identity_test_actually_detects_a_planted_duplicate(tmp_path):
    """The guard above must fail on a dataset that has the defect.

    A leakage test that cannot fail is decoration. This writes a dataset in which every
    participant's ``gum`` recording carries the same microphone waveform, rotated -- the
    real failure mode, not a byte-for-byte copy -- and requires the manifest to notice.
    """
    from bruxism.data.manifest import build_manifest
    from bruxism.data.quality import QualityFlag
    from tests.fixtures.synthetic import SyntheticDatasetSpec, write_synthetic_dataset

    root = tmp_path / "planted"
    write_synthetic_dataset(root, SyntheticDatasetSpec(duplicate_mic_condition="gum"))
    manifest = build_manifest(root, probe_video=False)

    flagged = [
        record.recording_id
        for record in manifest.records
        if QualityFlag.MIC_WAVEFORM_DUPLICATED.value in record.quality_flags
    ]
    # Four, not five: S02's gum recording is the fixture's deliberately short one, so it
    # cannot carry a rotation of the others' longer waveform and joins no group. That the
    # count is four rather than five is itself the detector behaving correctly -- it groups
    # on identical content, not on a shared condition label.
    assert len(flagged) == 4, f"expected the four full-length gum recordings, got {flagged}"
    assert all("gum" in recording_id for recording_id in flagged)
    assert not any("S02" in recording_id for recording_id in flagged)
    # And the EMG must stay clean, or the detector is flagging on something else.
    assert manifest.duplication["emg1"]["n_cross_subject_groups"] == 0


def test_planted_duplicates_are_exact_rotations_not_merely_similar(tmp_path):
    """A shared fingerprint is confirmed by exact rotation, never by correlation alone."""
    import pandas as pd

    from bruxism.preprocessing.mic_integrity import is_circular_rotation
    from tests.fixtures.synthetic import SyntheticDatasetSpec, write_synthetic_dataset

    root = tmp_path / "planted"
    write_synthetic_dataset(root, SyntheticDatasetSpec(duplicate_mic_condition="gum"))
    columns = {
        path.name: pd.read_csv(path, usecols=["Mic"])["Mic"].to_numpy(float)
        for path in sorted(root.rglob("gum_*.csv"))
    }
    # Only the full-length recordings share a waveform; the short one has its own.
    longest = max(len(values) for values in columns.values())
    replayed = {name: v for name, v in columns.items() if len(v) == longest}
    assert len(replayed) == 4

    offsets = set()
    reference = next(iter(replayed.values()))
    for name, candidate in replayed.items():
        rotated, offset = is_circular_rotation(reference, candidate)
        assert rotated, f"{name} shares a fingerprint but is not a rotation"
        offsets.add(offset)
    # Distinct rotations, so the test is not passing on trivially identical copies -- which
    # is what makes it a faithful miniature of the real defect.
    assert len(offsets) == len(replayed)


def test_audio_run_is_refused_on_flagged_data(tmp_path):
    """A fusion run must not start on data whose microphone is flagged, unsigned."""
    from bruxism.config import ExperimentConfig
    from bruxism.data.manifest import build_manifest
    from bruxism.runner import DataDefectError, assert_modality_is_supported_by_data
    from tests.fixtures.synthetic import SyntheticDatasetSpec, write_synthetic_dataset

    root = tmp_path / "planted"
    write_synthetic_dataset(root, SyntheticDatasetSpec(duplicate_mic_condition="gum"))
    manifest = build_manifest(root, probe_video=False)

    fusion = ExperimentConfig(name="t", modality="fusion")
    with pytest.raises(DataDefectError, match="mic_waveform_duplicated"):
        assert_modality_is_supported_by_data(fusion, manifest)

    # EMG-only is unaffected: the defect is in a channel it never reads.
    assert_modality_is_supported_by_data(ExperimentConfig(name="t", modality="emg_only"), manifest)

    # And the concession can be taken, in writing.
    signed = ExperimentConfig(
        name="t", modality="fusion", mic_defect_acknowledged_by="tester, 2026-08-12: reason"
    )
    assert_modality_is_supported_by_data(signed, manifest)


def test_acknowledgements_do_not_change_the_configuration_hash():
    """Who signed for a run cannot change what the run computes, so it cannot change its id.

    The published bundles (``2b6fb5ac``, ``cead62e4``) must keep their hashes after the
    acknowledgements are added to their configurations, or the manuscript's provenance
    section stops being true.
    """
    from bruxism.config import ExperimentConfig

    plain = ExperimentConfig(name="t")
    signed = ExperimentConfig(
        name="t",
        mic_defect_acknowledged_by="tester, 2026-08-12: reason",
        stopband_bands_acknowledged_by="tester, 2026-08-12: reason",
    )
    assert plain.config_hash == signed.config_hash
    # ...but they are still recorded, or the audit trail would be lost.
    assert signed.to_dict()["mic_defect_acknowledged_by"] is not None


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
