"""Baseline models, all receiving the same inputs as the dual-branch network.

Reviewer-facing constraint (``Temp.md`` item I): an architecture comparison is confounded
if the proposed model gets EMG + audio and the baselines get EMG only. Every neural
baseline here takes the identical ``(emg, mic)`` pair, the identical windows, the identical
splits and the identical training loop, so the only thing that varies is the architecture.
The classical baselines consume features derived from **both** modalities.

The registry is keyed by ``model_id``, which is recorded on every prediction row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from bruxism.models.dual_branch import (
    BranchConfig,
    DualBranchConfig,
    Modality,
    build_model,
)
from bruxism.preprocessing.wavelets import WaveletConfig

__all__ = [
    "NEURAL_MODEL_IDS",
    "SKLEARN_MODEL_IDS",
    "TEMPORAL_DUAL_BRANCH_DEFAULTS",
    "BiLSTMBaseline",
    "BiLSTMConfig",
    "EarlyFusionCNN",
    "EarlyFusionConfig",
    "SklearnBaselineConfig",
    "build_neural_model",
    "build_sklearn_model",
]

NEURAL_MODEL_IDS: tuple[str, ...] = (
    "dual_branch_wavelet_cnn",
    "dual_branch_temporal_cnn",
    "early_fusion_cnn",
    "bilstm",
)
SKLEARN_MODEL_IDS: tuple[str, ...] = ("random_forest", "mlp", "logistic_regression")

#: ``dual_branch_temporal_cnn`` is ``dual_branch_wavelet_cnn`` with the time axis kept.
#:
#: The original pooled each band to a single mean over the whole window, which is why a
#: logistic regression on band energies matched it exactly (``cause.md`` §4): the network
#: was a band-energy calculator with extra steps. This variant adds exactly the two
#: capabilities the diagnosis says are missing, and nothing else:
#:
#: * ``pooling="stats"`` -- mean, standard deviation and max per band instead of the mean
#:   alone, so *how much a band varies* is representable at all;
#: * ``modulation=True`` -- a modulation spectrum of each band's envelope, so the 1-2 Hz
#:   burst-relax rhythm that separates chewing and grinding from clenching is measured
#:   directly.
#:
#: Capacity is deliberately not grown further. Four-segment pooling was tried first and
#: reached 98,957 parameters against roughly 6,000 training windows, almost all of it in
#: one dense fusion layer; this configuration is 3x the original rather than 13x, and every
#: added feature answers a named deficiency. It is a separate ``model_id`` rather than a
#: changed default so the superseded model stays reproducible and the two appear side by
#: side in one table.
TEMPORAL_DUAL_BRANCH_DEFAULTS: dict[str, Any] = {
    "emg": BranchConfig(
        in_channels=4,
        wavelet=WaveletConfig(wavelet="db4", level=4, bands=("A4", "D3", "D1")),
        hidden_channels=(8, 16),
        pooling="stats",
        modulation=True,
        modulation_bins=6,
    ),
    "mic": BranchConfig(
        in_channels=1,
        wavelet=WaveletConfig(wavelet="coif5", level=5, bands=("A5", "D3", "D1")),
        hidden_channels=(4, 8),
        pooling="stats",
        modulation=True,
        modulation_bins=6,
    ),
    "fusion_hidden": (64, 32),
    "dropout": 0.5,
}


def _stack_modalities(
    emg: torch.Tensor, mic: torch.Tensor, *, uses_emg: bool, uses_mic: bool
) -> torch.Tensor:
    """Concatenate the modalities along the channel axis, honouring the ablation."""
    if mic.dim() == 2:
        mic = mic.unsqueeze(1)
    parts: list[torch.Tensor] = []
    if uses_emg:
        parts.append(emg)
    if uses_mic:
        parts.append(mic)
    if not parts:
        raise ValueError("at least one modality must be enabled")
    return torch.cat(parts, dim=1)


@dataclass(frozen=True)
class EarlyFusionConfig:
    """Raw-signal CNN over EMG and microphone concatenated as input channels."""

    num_classes: int
    emg_channels: int = 4
    mic_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64)
    kernel_size: int = 7
    pool_size: int = 4
    dropout: float = 0.5
    modality: Modality = "fusion"

    def in_channels(self) -> int:
        total = 0
        if self.modality in ("fusion", "emg_only"):
            total += self.emg_channels
        if self.modality in ("fusion", "audio_only"):
            total += self.mic_channels
        return total

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload


class EarlyFusionCNN(nn.Module):
    """1-D CNN on the raw (filtered, normalised) signal, no wavelet stage.

    Shapes mirror :class:`~bruxism.models.dual_branch.DualBranchWaveletCNN`: it takes
    ``(emg, mic)`` and returns ``(batch, num_classes)`` logits.
    """

    def __init__(self, config: EarlyFusionConfig):
        super().__init__()
        self.config = config
        self.uses_emg = config.modality in ("fusion", "emg_only")
        self.uses_mic = config.modality in ("fusion", "audio_only")

        layers: list[nn.Module] = []
        width = config.in_channels()
        for out_channels in config.channels:
            layers += [
                nn.Conv1d(
                    width,
                    out_channels,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(config.pool_size),
            ]
            width = out_channels
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)
        self.embedding_dim = width
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(width, config.num_classes))

    def embed(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        stacked = _stack_modalities(emg, mic, uses_emg=self.uses_emg, uses_mic=self.uses_mic)
        return self.features(stacked).squeeze(-1)

    def forward(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(emg, mic))

    def parameter_counts(self) -> dict[str, int]:
        return {
            "features": sum(p.numel() for p in self.features.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "buffers": sum(b.numel() for b in self.buffers()),
        }

    def architecture_record(self, sampling_rate: float = 1200.0) -> dict[str, Any]:  # noqa: ARG002
        """Architecture record. ``sampling_rate`` is accepted for interface parity with
        the dual-branch model, which uses it to report wavelet band edges."""
        return {
            "class": type(self).__name__,
            "config": self.config.to_dict(),
            "parameter_counts": self.parameter_counts(),
            "embedding_dim": self.embedding_dim,
        }


@dataclass(frozen=True)
class BiLSTMConfig:
    """Bidirectional LSTM sequence baseline.

    The raw 1200-sample window is decimated by ``downsample`` before the recurrence: an
    LSTM over 1200 timesteps is dominated by sequence length rather than by modelling
    capacity, and decimating keeps the comparison about architecture.
    """

    num_classes: int
    emg_channels: int = 4
    mic_channels: int = 1
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.5
    downsample: int = 8
    modality: Modality = "fusion"

    def in_channels(self) -> int:
        total = 0
        if self.modality in ("fusion", "emg_only"):
            total += self.emg_channels
        if self.modality in ("fusion", "audio_only"):
            total += self.mic_channels
        return total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM over the decimated multi-channel window."""

    def __init__(self, config: BiLSTMConfig):
        super().__init__()
        self.config = config
        self.uses_emg = config.modality in ("fusion", "emg_only")
        self.uses_mic = config.modality in ("fusion", "audio_only")
        self.pool = nn.AvgPool1d(config.downsample) if config.downsample > 1 else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=config.in_channels(),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.embedding_dim = 2 * config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(self.embedding_dim, config.num_classes)
        )

    def embed(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        stacked = _stack_modalities(emg, mic, uses_emg=self.uses_emg, uses_mic=self.uses_mic)
        sequence = self.pool(stacked).transpose(1, 2)  # (batch, time, channels)
        output, _ = self.lstm(sequence)
        # Mean over time is order-insensitive and avoids depending on the final timestep.
        return output.mean(dim=1)

    def forward(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(emg, mic))

    def parameter_counts(self) -> dict[str, int]:
        return {
            "lstm": sum(p.numel() for p in self.lstm.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "buffers": sum(b.numel() for b in self.buffers()),
        }

    def architecture_record(self, sampling_rate: float = 1200.0) -> dict[str, Any]:  # noqa: ARG002
        """Architecture record. ``sampling_rate`` is accepted for interface parity with
        the dual-branch model, which uses it to report wavelet band edges."""
        return {
            "class": type(self).__name__,
            "config": self.config.to_dict(),
            "parameter_counts": self.parameter_counts(),
            "embedding_dim": self.embedding_dim,
        }


def build_neural_model(
    model_id: str,
    *,
    num_classes: int,
    modality: Modality = "fusion",
    emg_channels: int = 4,
    window_samples: int = 1200,
    overrides: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a neural model by id, with matched interface and inputs.

    Raises
    ------
    KeyError
        On an unknown ``model_id``. Unsupported options fail loudly rather than falling
        back to a default architecture.
    """
    extra = dict(overrides or {})
    if model_id in ("dual_branch_wavelet_cnn", "dual_branch_temporal_cnn"):
        from dataclasses import replace as dataclass_replace

        if model_id == "dual_branch_temporal_cnn":
            extra = {**TEMPORAL_DUAL_BRANCH_DEFAULTS, **extra}
        config = DualBranchConfig(
            num_classes=num_classes,
            modality=modality,
            window_samples=window_samples,
            **extra,
        )
        if config.emg.in_channels != emg_channels:
            config = dataclass_replace(
                config, emg=dataclass_replace(config.emg, in_channels=emg_channels)
            )
        return build_model(config)
    if model_id == "early_fusion_cnn":
        return EarlyFusionCNN(
            EarlyFusionConfig(
                num_classes=num_classes,
                emg_channels=emg_channels,
                modality=modality,
                **extra,
            )
        )
    if model_id == "bilstm":
        return BiLSTMBaseline(
            BiLSTMConfig(
                num_classes=num_classes,
                emg_channels=emg_channels,
                modality=modality,
                **extra,
            )
        )
    raise KeyError(f"unknown neural model_id {model_id!r}; available: {NEURAL_MODEL_IDS}")


@dataclass(frozen=True)
class SklearnBaselineConfig:
    """Classical baseline on explicitly defined features from both modalities."""

    model_id: Literal["random_forest", "mlp", "logistic_regression"] = "random_forest"
    modality: Modality = "fusion"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "modality": self.modality,
            "params": dict(self.params),
        }


def build_sklearn_model(config: SklearnBaselineConfig, *, seed: int) -> Any:
    """Instantiate a scikit-learn baseline inside a standardising pipeline.

    The scaler is part of the pipeline, so it is fitted on training folds only by
    construction -- there is no path where test statistics reach it.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(config.params)
    if config.model_id == "random_forest":
        estimator: Any = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 400),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 2),
            class_weight=params.pop("class_weight", "balanced"),
            n_jobs=params.pop("n_jobs", -1),
            random_state=seed,
            **params,
        )
    elif config.model_id == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(params.pop("hidden_layer_sizes", (64, 32))),
            alpha=params.pop("alpha", 1e-3),
            max_iter=params.pop("max_iter", 500),
            early_stopping=params.pop("early_stopping", False),
            random_state=seed,
            **params,
        )
    elif config.model_id == "logistic_regression":
        estimator = LogisticRegression(
            C=params.pop("C", 1.0),
            max_iter=params.pop("max_iter", 2000),
            class_weight=params.pop("class_weight", "balanced"),
            random_state=seed,
            **params,
        )
    else:
        raise KeyError(
            f"unknown sklearn model_id {config.model_id!r}; available: {SKLEARN_MODEL_IDS}"
        )
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])


def mask_features_by_modality(
    features: np.ndarray, feature_names: list[str], modality: Modality
) -> np.ndarray:
    """Drop the other modality's feature columns for a modality ablation.

    Columns are identified by their ``emg``/``mic`` name prefix; cross-modality
    correlation features are dropped in either single-modality condition because they are
    not computable from one modality alone.
    """
    if modality == "fusion":
        return features
    keep: list[int] = []
    for index, name in enumerate(feature_names):
        is_cross = name.startswith("corr_") and "mic" in name
        if (
            modality == "emg_only"
            and name.startswith("emg")
            and not is_cross
            or modality == "audio_only"
            and name.startswith("mic")
        ):
            keep.append(index)
    if not keep:
        raise ValueError(f"no features remain for modality {modality!r}")
    return features[:, keep]
