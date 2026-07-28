"""The configurable dual-branch wavelet CNN.

One EMG branch and one microphone branch each decompose their input into named wavelet
bands and run a small convolutional stack per band; the pooled band features are
concatenated and passed through a fusion head to a classifier.

Differences from the research prototype, all of which change either correctness or
measurability:

* **Bands are named, not positional.** The prototype indexed ``details[0]`` as the highest
  frequency (it is the *lowest*) and ``details[2]`` as "D3" (at level 4 it is D2). Here a
  branch is configured with e.g. ``("A4", "D3", "D1")`` and
  :mod:`bruxism.preprocessing.wavelets` resolves the index.
* **The decomposition is a differentiable conv cascade** (:class:`~bruxism.models.dwt.WaveletDecompose1d`)
  that stays on-device, instead of a per-batch GPU -> NumPy -> GPU round trip inside a
  Python double loop.
* **Logits are returned**, never softmax probabilities, so standard PyTorch losses apply.
* **Parameter counts are computed programmatically** by :meth:`DualBranchWaveletCNN.parameter_counts`;
  no count is ever quoted from prose.
* **Embeddings are exposed** through :meth:`DualBranchWaveletCNN.embed` for the exploratory
  t-SNE figure, so that figure comes from the evaluated checkpoint rather than a re-run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from bruxism.models.dwt import WaveletDecompose1d
from bruxism.preprocessing.wavelets import WaveletConfig, band_frequencies

__all__ = [
    "BranchConfig",
    "DualBranchConfig",
    "DualBranchWaveletCNN",
    "build_model",
]

Modality = Literal["fusion", "emg_only", "audio_only"]


@dataclass(frozen=True)
class BranchConfig:
    """One modality branch: its wavelet decomposition and per-band convolutional widths."""

    in_channels: int = 4
    wavelet: WaveletConfig = WaveletConfig(wavelet="db4", level=4, bands=("A4", "D3", "D1"))
    hidden_channels: tuple[int, int] = (8, 16)
    kernel_size: int = 3
    pool_size: int = 2

    @property
    def n_bands(self) -> int:
        return len(self.wavelet.bands)

    @property
    def out_features(self) -> int:
        """Feature width contributed by this branch: one pooled vector per band."""
        return self.n_bands * self.hidden_channels[-1]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["wavelet"] = self.wavelet.to_dict()
        payload["hidden_channels"] = list(self.hidden_channels)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BranchConfig:
        data = dict(payload)
        wavelet = data.pop("wavelet", None)
        if wavelet is not None:
            data["wavelet"] = WaveletConfig(
                wavelet=wavelet.get("wavelet", "db4"),
                level=int(wavelet.get("level", 4)),
                bands=tuple(wavelet.get("bands", ("A4", "D3", "D1"))),
                mode=wavelet.get("mode", "symmetric"),
            )
        if "hidden_channels" in data:
            data["hidden_channels"] = tuple(data["hidden_channels"])
        return cls(**data)


@dataclass(frozen=True)
class DualBranchConfig:
    """Complete architecture specification, saved alongside every checkpoint."""

    num_classes: int
    emg: BranchConfig = field(default_factory=lambda: BranchConfig(in_channels=4))
    mic: BranchConfig = field(
        default_factory=lambda: BranchConfig(
            in_channels=1,
            wavelet=WaveletConfig(wavelet="coif5", level=5, bands=("A5", "D3", "D1")),
            hidden_channels=(4, 8),
        )
    )
    fusion_hidden: tuple[int, ...] = (48, 32)
    dropout: float = 0.5
    modality: Modality = "fusion"
    window_samples: int = 1200

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.modality not in ("fusion", "emg_only", "audio_only"):
            raise ValueError(f"unknown modality {self.modality!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "emg": self.emg.to_dict(),
            "mic": self.mic.to_dict(),
            "fusion_hidden": list(self.fusion_hidden),
            "dropout": self.dropout,
            "modality": self.modality,
            "window_samples": self.window_samples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DualBranchConfig:
        data = dict(payload)
        if "emg" in data:
            data["emg"] = BranchConfig.from_dict(data["emg"])
        if "mic" in data:
            data["mic"] = BranchConfig.from_dict(data["mic"])
        if "fusion_hidden" in data:
            data["fusion_hidden"] = tuple(data["fusion_hidden"])
        return cls(**data)


class _BandBranch(nn.Module):
    """Convolutional stack applied to one wavelet band, pooled to a fixed-width vector."""

    def __init__(self, in_channels: int, hidden: tuple[int, int], kernel_size: int, pool: int):
        super().__init__()
        first, second = hidden
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, first, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
            nn.Conv1d(first, second, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(second),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _ModalityBranch(nn.Module):
    """Wavelet decomposition plus one :class:`_BandBranch` per configured band."""

    def __init__(self, config: BranchConfig):
        super().__init__()
        self.config = config
        self.decompose = WaveletDecompose1d(config.wavelet, config.in_channels)
        self.bands = nn.ModuleDict(
            {
                band: _BandBranch(
                    config.in_channels,
                    config.hidden_channels,
                    config.kernel_size,
                    config.pool_size,
                )
                for band in config.wavelet.bands
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficients = self.decompose(x)
        return torch.cat(
            [self.bands[band](coefficients[band]) for band in self.config.wavelet.bands],
            dim=1,
        )


class DualBranchWaveletCNN(nn.Module):
    """Dual-branch wavelet CNN over EMG and microphone windows.

    Shapes
    ------
    ``emg``
        ``(batch, emg_channels, n_samples)``
    ``mic``
        ``(batch, 1, n_samples)`` (a ``(batch, n_samples)`` tensor is accepted and
        unsqueezed)
    returns
        ``(batch, num_classes)`` **logits**

    In ``emg_only`` / ``audio_only`` mode the unused branch is not constructed at all, so
    the ablation genuinely removes its parameters instead of feeding it zeros.
    """

    def __init__(self, config: DualBranchConfig):
        super().__init__()
        self.config = config
        self.uses_emg = config.modality in ("fusion", "emg_only")
        self.uses_mic = config.modality in ("fusion", "audio_only")

        fusion_in = 0
        if self.uses_emg:
            self.emg_branch = _ModalityBranch(config.emg)
            fusion_in += config.emg.out_features
        if self.uses_mic:
            self.mic_branch = _ModalityBranch(config.mic)
            fusion_in += config.mic.out_features
        self.fusion_in_features = fusion_in

        layers: list[nn.Module] = []
        width = fusion_in
        for hidden in config.fusion_hidden:
            layers += [
                nn.Linear(width, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ]
            width = hidden
        self.fusion = nn.Sequential(*layers)
        self.embedding_dim = width
        self.classifier = nn.Linear(width, config.num_classes)

    @staticmethod
    def _as_3d(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"{name} must be (batch, channels, time), got shape {tuple(x.shape)}")

    def embed(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        """Penultimate representation ``(batch, embedding_dim)``, for the t-SNE figure."""
        features: list[torch.Tensor] = []
        if self.uses_emg:
            features.append(self.emg_branch(self._as_3d(emg, "emg")))
        if self.uses_mic:
            features.append(self.mic_branch(self._as_3d(mic, "mic")))
        return self.fusion(torch.cat(features, dim=1))

    def forward(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        """Return **logits**; apply softmax outside if probabilities are needed."""
        return self.classifier(self.embed(emg, mic))

    def parameter_counts(self) -> dict[str, int]:
        """Programmatic parameter counts, per component and in total.

        Never quote a parameter count from prose -- read it from here.
        """
        counts: dict[str, int] = {}
        for name, module in (
            ("emg_branch", getattr(self, "emg_branch", None)),
            ("mic_branch", getattr(self, "mic_branch", None)),
            ("fusion", self.fusion),
            ("classifier", self.classifier),
        ):
            counts[name] = sum(p.numel() for p in module.parameters()) if module is not None else 0
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["buffers"] = sum(b.numel() for b in self.buffers())
        return counts

    def size_bytes(self, *, dtype_bytes: int = 4) -> int:
        """Approximate serialised size of the trainable weights."""
        return self.parameter_counts()["trainable"] * dtype_bytes

    def band_frequency_table(self, sampling_rate: float) -> dict[str, dict[str, list[float]]]:
        """Nominal Hz range of every band each branch consumes, for the Methods table."""
        table: dict[str, dict[str, list[float]]] = {}
        if self.uses_emg:
            cfg = self.config.emg.wavelet
            table["emg"] = {
                band: list(band_frequencies(band, cfg.level, sampling_rate)) for band in cfg.bands
            }
        if self.uses_mic:
            cfg = self.config.mic.wavelet
            table["mic"] = {
                band: list(band_frequencies(band, cfg.level, sampling_rate)) for band in cfg.bands
            }
        return table

    def architecture_record(self, sampling_rate: float = 1200.0) -> dict[str, Any]:
        """Everything needed to rebuild and describe this model, saved with the checkpoint."""
        return {
            "class": type(self).__name__,
            "config": self.config.to_dict(),
            "parameter_counts": self.parameter_counts(),
            "fusion_in_features": self.fusion_in_features,
            "embedding_dim": self.embedding_dim,
            "band_frequencies_hz": self.band_frequency_table(sampling_rate),
        }


def build_model(config: DualBranchConfig) -> DualBranchWaveletCNN:
    """Construct a :class:`DualBranchWaveletCNN` from a validated configuration."""
    return DualBranchWaveletCNN(config)
