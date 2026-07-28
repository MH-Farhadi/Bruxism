"""Loss functions, including a corrected focal loss.

The research prototype computed::

    ce   = cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt   = exp(-ce)
    loss = (1 - pt) ** gamma * ce

When ``alpha`` is supplied this is wrong: ``ce`` is then the *weighted* cross-entropy, so
``exp(-ce)`` is ``alpha_t`` -th-power-scaled and is not the model's probability of the true
class. The focusing term therefore depends on the class weights, which is not the focal
loss of Lin et al. (2017) and makes gamma and alpha interact in an uncontrolled way.

:class:`FocalLoss` computes the focusing term from the **unweighted** true-class
probability and applies alpha as a separate multiplicative factor::

    p_t  = softmax(logits)[target]
    loss = alpha_t * (1 - p_t) ** gamma * (-log p_t)

``tests/unit/test_losses.py`` checks this against hand-computed values with and without
alpha, and confirms that ``gamma=0`` with alpha reduces exactly to weighted cross-entropy.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch
from torch import nn

from bruxism.utils.logging import get_logger

__all__ = ["FocalLoss", "build_loss", "compute_class_weights"]

logger = get_logger(__name__)

Reduction = Literal["mean", "sum", "none"]


class FocalLoss(nn.Module):
    """Multi-class focal loss with correctly separated focusing and alpha terms.

    Parameters
    ----------
    alpha
        Per-class weights, shape ``(num_classes,)``. Must be derived from **training**
        data only. ``None`` disables class weighting.
    gamma
        Focusing exponent. ``gamma=0`` reduces to (weighted) cross-entropy exactly.
    reduction
        ``"mean"`` averages over the batch; with alpha this is a plain mean, not an
        alpha-normalised mean, so it stays comparable to ``CrossEntropyLoss(reduction="mean")``
        only when alpha averages to one.
    label_smoothing
        Smooths the target distribution before the log-likelihood term. Applied to the
        cross-entropy component only; the focusing term still uses the true-class
        probability, which is what "hard example" means.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: Reduction = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.gamma = float(gamma)
        self.reduction: Reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits
            ``(batch, num_classes)`` raw scores -- **not** softmax probabilities.
        targets
            ``(batch,)`` int64 class indices.
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be (batch, num_classes), got {tuple(logits.shape)}")
        log_probs = torch.log_softmax(logits, dim=1)

        # Cross-entropy term, optionally label-smoothed. No class weights here.
        if self.label_smoothing > 0:
            n_classes = logits.shape[1]
            smooth = self.label_smoothing / n_classes
            target_dist = torch.full_like(log_probs, smooth)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
            ce = -(target_dist * log_probs).sum(dim=1)
        else:
            ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focusing term from the UNWEIGHTED true-class probability. This is the whole
        # correction: p_t comes from softmax(logits), never from exp(-weighted_ce).
        pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        focal = (1.0 - pt).clamp_min(0.0) ** self.gamma

        loss = focal * ce
        if self.alpha is not None:
            alpha = cast(torch.Tensor, self.alpha)
            loss = alpha.to(loss.device, loss.dtype).gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    labels: np.ndarray | list[int],
    num_classes: int,
    *,
    scheme: Literal["balanced", "none", "inverse_sqrt"] = "balanced",
) -> np.ndarray:
    """Class weights derived from **training** labels only.

    Parameters
    ----------
    labels
        Training labels. Passing validation or held-out labels here is a leakage bug; the
        caller records which participants produced them.
    num_classes
        Total classes in the task. Classes absent from ``labels`` receive weight 1.0 and a
        warning rather than an infinite weight.
    scheme
        ``"balanced"`` uses ``n_samples / (n_classes * count)``. ``"inverse_sqrt"`` is a
        gentler alternative for the extreme imbalance the trigger-constrained protocol
        produces. ``"none"`` returns ones.

    Returns
    -------
    numpy.ndarray
        Shape ``(num_classes,)``, dtype float64.
    """
    values = np.asarray(labels, dtype=np.int64).ravel()
    if values.size == 0:
        raise ValueError("cannot compute class weights from an empty label array")
    counts = np.bincount(values, minlength=num_classes).astype(np.float64)

    if scheme == "none":
        return np.ones(num_classes, dtype=np.float64)

    missing = np.flatnonzero(counts == 0)
    if missing.size:
        logger.warning(
            "classes %s are absent from the training labels; assigning weight 1.0 rather "
            "than an infinite weight",
            missing.tolist(),
            extra={"missing_classes": missing.tolist()},
        )
    safe = np.where(counts > 0, counts, np.nan)
    if scheme == "balanced":
        weights = values.size / (num_classes * safe)
    else:
        weights = np.sqrt(values.size / (num_classes * safe))
    return np.nan_to_num(weights, nan=1.0)


def build_loss(
    loss_id: Literal["cross_entropy", "focal"],
    *,
    class_weights: np.ndarray | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Construct a loss by id.

    Whether focal loss is retained is a hyperparameter chosen on inner folds, not an
    assumption -- ``cross_entropy`` is a first-class option, not a fallback.
    """
    weights = (
        torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    if loss_id == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if loss_id == "focal":
        return FocalLoss(alpha=weights, gamma=gamma, label_smoothing=label_smoothing)
    raise KeyError(f"unknown loss_id {loss_id!r}; available: 'cross_entropy', 'focal'")
