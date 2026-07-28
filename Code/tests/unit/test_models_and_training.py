"""Model shapes, devices, losses, checkpoints and determinism."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from bruxism.data.labels import get_task
from bruxism.models.baselines import NEURAL_MODEL_IDS, build_neural_model
from bruxism.models.dwt import WaveletDecompose1d, symmetric_pad
from bruxism.preprocessing.wavelets import WaveletConfig, decompose
from bruxism.training.losses import FocalLoss, build_loss, compute_class_weights
from bruxism.training.selection import (
    EpochRecord,
    SelectionConfig,
    TrialResult,
    select_best_epoch,
    select_best_trial,
    select_epoch_budget,
)

# ----------------------------------------------------------------------- DWT ---


@pytest.mark.parametrize(
    ("wavelet", "level", "n", "channels"),
    [("db4", 4, 1200, 4), ("coif5", 5, 1200, 1), ("sym8", 3, 997, 2), ("db2", 6, 4096, 3)],
)
def test_differentiable_dwt_reproduces_pywt_exactly(wavelet, level, n, channels):
    bands = tuple([f"A{level}"] + [f"D{k}" for k in range(1, level + 1)])
    config = WaveletConfig(wavelet, level, bands)
    x = torch.randn(3, channels, n, dtype=torch.float64)
    produced = WaveletDecompose1d(config, channels)(x)
    reference = decompose(x.numpy(), config, check_level=False)
    for band in bands:
        assert produced[band].shape[-1] == reference[band].shape[-1]
        assert np.abs(produced[band].numpy() - reference[band]).max() < 1e-12


def test_symmetric_pad_matches_numpy():
    x = torch.arange(7, dtype=torch.float64).reshape(1, 1, 7)
    produced = symmetric_pad(x, 3)[0, 0].numpy()
    assert np.array_equal(produced, np.pad(np.arange(7.0), 3, mode="symmetric"))


def test_dwt_is_differentiable_and_has_no_learnable_parameters():
    module = WaveletDecompose1d(WaveletConfig("db4", 4, ("A4", "D1")), 4)
    assert sum(p.numel() for p in module.parameters()) == 0
    x = torch.randn(2, 4, 1200, requires_grad=True)
    sum(v.pow(2).sum() for v in module(x).values()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# --------------------------------------------------------------------- shapes ---


@pytest.mark.parametrize("model_id", NEURAL_MODEL_IDS)
@pytest.mark.parametrize(
    "task_id", ["five_class", "no_chewing_four_class", "ternary", "binary_tooth_contact"]
)
def test_model_shapes_for_every_class_count(model_id, task_id):
    task = get_task(task_id)
    model = build_neural_model(model_id, num_classes=task.num_classes, emg_channels=4).eval()
    emg, mic = torch.randn(5, 4, 1200), torch.randn(5, 1, 1200)
    with torch.no_grad():
        logits = model(emg, mic)
    assert logits.shape == (5, task.num_classes)
    # Logits, not probabilities: rows must not already sum to one.
    assert not torch.allclose(logits.sum(dim=1), torch.ones(5), atol=1e-3)


@pytest.mark.parametrize("model_id", NEURAL_MODEL_IDS)
@pytest.mark.parametrize("modality", ["fusion", "emg_only", "audio_only"])
def test_every_modality_path_runs(model_id, modality):
    model = build_neural_model(model_id, num_classes=5, modality=modality, emg_channels=4).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 4, 1200), torch.randn(4, 1, 1200))
    assert out.shape == (4, 5)


def test_single_modality_models_really_drop_the_other_branch():
    counts = {
        m: build_neural_model(
            "dual_branch_wavelet_cnn", num_classes=5, modality=m
        ).parameter_counts()
        for m in ("fusion", "emg_only", "audio_only")
    }
    assert counts["emg_only"]["mic_branch"] == 0
    assert counts["audio_only"]["emg_branch"] == 0
    assert counts["fusion"]["total"] > counts["emg_only"]["total"]


def test_parameter_counts_are_computed_not_quoted():
    model = build_neural_model("dual_branch_wavelet_cnn", num_classes=5, emg_channels=4)
    counts = model.parameter_counts()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["trainable"] == counts["total"]
    assert counts["total"] == sum(
        counts[k] for k in ("emg_branch", "mic_branch", "fusion", "classifier")
    )


def test_embeddings_are_exposed_for_the_tsne_figure():
    model = build_neural_model("dual_branch_wavelet_cnn", num_classes=5, emg_channels=4).eval()
    with torch.no_grad():
        embedding = model.embed(torch.randn(6, 4, 1200), torch.randn(6, 1, 1200))
    assert embedding.shape == (6, model.embedding_dim)


def test_architecture_record_is_serialisable_and_complete():
    model = build_neural_model("dual_branch_wavelet_cnn", num_classes=5, emg_channels=4)
    record = model.architecture_record(1200.0)
    assert record["parameter_counts"]["total"] > 0
    assert record["band_frequencies_hz"]["emg"]["A4"] == [0.0, 37.5]
    import json

    json.loads(json.dumps(record))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cpu_and_cuda_agree():
    torch.manual_seed(0)
    model = build_neural_model("dual_branch_wavelet_cnn", num_classes=5, emg_channels=4).eval()
    emg, mic = torch.randn(4, 4, 1200), torch.randn(4, 1, 1200)
    with torch.no_grad():
        on_cpu = model(emg, mic)
        on_gpu = model.cuda()(emg.cuda(), mic.cuda()).cpu()
    assert torch.allclose(on_cpu, on_gpu, atol=1e-4)


# ---------------------------------------------------------------------- loss ---


def test_focal_loss_matches_a_hand_computed_value():
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    target = torch.tensor([0])
    gamma = 2.0

    probabilities = torch.softmax(logits, dim=1)
    pt = float(probabilities[0, 0])
    expected = ((1 - pt) ** gamma) * (-np.log(pt))

    produced = FocalLoss(gamma=gamma, reduction="none")(logits, target)
    assert float(produced) == pytest.approx(expected, rel=1e-6)


def test_focal_loss_applies_alpha_separately_from_the_focusing_term():
    """The prototype derived pt from the WEIGHTED cross-entropy; that is the bug."""
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    target = torch.tensor([0])
    alpha = torch.tensor([3.0, 1.0, 1.0])
    gamma = 2.0

    pt = float(torch.softmax(logits, dim=1)[0, 0])
    correct = 3.0 * ((1 - pt) ** gamma) * (-np.log(pt))
    produced = float(FocalLoss(alpha=alpha, gamma=gamma, reduction="none")(logits, target))
    assert produced == pytest.approx(correct, rel=1e-6)

    # The prototype's formulation: pt = exp(-weighted_ce). It gives a different number.
    weighted_ce = 3.0 * (-np.log(pt))
    prototype = ((1 - np.exp(-weighted_ce)) ** gamma) * weighted_ce
    assert abs(prototype - correct) > 1e-3


def test_focal_loss_with_gamma_zero_reduces_to_weighted_cross_entropy(rng):
    logits = torch.tensor(rng.standard_normal((16, 5)), dtype=torch.float32)
    targets = torch.tensor(rng.integers(0, 5, 16))
    alpha = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])

    focal = FocalLoss(alpha=alpha, gamma=0.0, reduction="none")(logits, targets)
    reference = torch.nn.functional.cross_entropy(logits, targets, weight=alpha, reduction="none")
    assert torch.allclose(focal, reference, atol=1e-6)


def test_focal_loss_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError):
        FocalLoss(label_smoothing=1.5)
    with pytest.raises(ValueError):
        FocalLoss(reduction="median")


def test_class_weights_use_training_labels_and_handle_absent_classes():
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2)
    assert weights[1] > weights[0]
    # An absent class gets weight 1.0, never an infinite weight.
    absent = compute_class_weights([0, 0, 1], num_classes=4)
    assert np.isfinite(absent).all() and absent[2] == 1.0 and absent[3] == 1.0
    assert np.allclose(compute_class_weights([0, 1], 2, scheme="none"), [1.0, 1.0])


def test_build_loss_rejects_unknown_ids():
    with pytest.raises(KeyError, match="unknown loss_id"):
        build_loss("hinge")


# --------------------------------------------------------------- checkpoints ---


def test_deep_copied_best_weights_survive_further_optimizer_steps():
    """A shallow state_dict().copy() shares storage; the snapshot must be a deep copy."""
    torch.manual_seed(0)
    model = build_neural_model("dual_branch_wavelet_cnn", num_classes=5, emg_channels=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    shallow = model.state_dict().copy()
    deep = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    before = {k: v.clone() for k, v in deep.items()}

    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(4, 4, 1200), torch.randn(4, 1, 1200)).sum().backward()
        optimizer.step()

    key = next(k for k, v in deep.items() if v.dtype.is_floating_point and v.numel() > 1)
    assert torch.equal(deep[key], before[key]), "deep copy must not track later updates"
    assert not torch.equal(shallow[key], before[key]), (
        "the shallow copy DOES track later updates -- this was the prototype's bug"
    )


# ---------------------------------------------------------------- selection ---


def _epoch(n, f1, loss=1.0):
    return EpochRecord(
        epoch=n,
        train_loss=loss,
        val_loss=loss,
        val_accuracy=f1,
        val_balanced_accuracy=f1,
        val_macro_f1=f1,
    )


def test_best_epoch_uses_the_declared_objective_and_tiebreak():
    history = [_epoch(1, 0.5), _epoch(2, 0.9), _epoch(3, 0.9, loss=0.5), _epoch(4, 0.7)]
    result = select_best_epoch(history, SelectionConfig(objective="macro_f1"))
    assert result.best_epoch == 3  # tie on F1, lower val_loss wins
    assert result.tie_broken_by == "lowest_val_loss"

    tied = [_epoch(2, 0.9), _epoch(5, 0.9)]
    assert select_best_epoch(tied).best_epoch == 2  # earliest epoch wins
    assert select_best_epoch(tied).tie_broken_by == "earliest_epoch"


def test_epoch_budget_uses_the_median_rule_and_clamps():
    selections = [select_best_epoch([_epoch(n, 0.9)]) for n in (4, 8, 10, 30)]
    assert select_epoch_budget(selections, SelectionConfig(min_epochs=1, max_epochs=60)) == 9
    assert select_epoch_budget(selections, SelectionConfig(min_epochs=20, max_epochs=60)) == 20
    assert select_epoch_budget(selections, SelectionConfig(min_epochs=1, max_epochs=5)) == 5


def test_trial_selection_prefers_the_higher_mean_then_the_lower_spread():
    def trial(name, values):
        return TrialResult(
            trial_id=name,
            hyperparameters={},
            inner_selections=[select_best_epoch([_epoch(1, v)]) for v in values],
        )

    # Higher mean wins outright.
    assert select_best_trial([trial("a", [0.5, 0.5]), trial("b", [0.8, 0.8])]).trial_id == "b"
    # Equal means (both 0.7): the more stable configuration -- lower across-fold spread -- wins.
    assert select_best_trial([trial("a", [0.9, 0.5]), trial("b", [0.7, 0.7])]).trial_id == "b"
    # Equal mean AND equal spread: the lexicographically smallest trial id wins, so the
    # outcome never depends on iteration order.
    assert select_best_trial([trial("z", [0.7, 0.7]), trial("a", [0.7, 0.7])]).trial_id == "a"


def test_all_trials_failing_raises_rather_than_falling_back():
    failed = [TrialResult(trial_id="a", hyperparameters={}, failed=True, failure_reason="boom")]
    with pytest.raises(ValueError, match="every hyperparameter trial failed"):
        select_best_trial(failed)
