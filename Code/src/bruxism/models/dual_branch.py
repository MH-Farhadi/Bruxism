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


TemporalPooling = Literal["avg", "stats", "segments", "stats_segments"]


@dataclass(frozen=True)
class BranchConfig:
    """One modality branch: its wavelet decomposition and per-band convolutional widths.

    Attributes
    ----------
    pooling
        How each band's convolutional feature map is reduced to a fixed-width vector.

        ``"avg"``
            ``AdaptiveAvgPool1d(1)``: the mean over the whole window. This is what the
            original model did, and it is why a logistic regression on hand-computed band
            energies matched it exactly (``cause.md`` §4) -- averaging a ~7-tap detector
            over a second leaves a per-band mean rectified amplitude and nothing else.
        ``"stats"``
            Mean, standard deviation and max. Three numbers instead of one; the standard
            deviation is the cheapest possible measure of whether a band is bursting or
            steady, which is the clench/chew distinction.
        ``"segments"``
            ``AdaptiveAvgPool1d(n_segments)``: keeps a coarse temporal layout, so onset
            shape survives.
        ``"stats_segments"``
            Both.
    n_segments
        Segments kept by the ``segments`` poolings.
    modulation
        Add a modulation-spectrum head over each band's envelope. Chewing and grinding are
        distinguished from clenching by 1-2 Hz burst-relax rhythm, which every pooling
        above still discards; this measures it directly. See :class:`_ModulationPool`.
    modulation_bins
        Number of modulation-frequency bins retained, from ``modulation_low_hz`` upwards.
    """

    in_channels: int = 4
    wavelet: WaveletConfig = WaveletConfig(wavelet="db4", level=4, bands=("A4", "D3", "D1"))
    hidden_channels: tuple[int, int] = (8, 16)
    kernel_size: int = 3
    pool_size: int = 2
    pooling: TemporalPooling = "avg"
    n_segments: int = 4
    modulation: bool = False
    modulation_bins: int = 6
    modulation_frames: int = 64

    def __post_init__(self) -> None:
        if self.pooling not in ("avg", "stats", "segments", "stats_segments"):
            raise ValueError(f"unknown pooling {self.pooling!r}")
        if self.n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")
        if self.modulation and self.modulation_bins < 1:
            raise ValueError("modulation_bins must be >= 1 when modulation is enabled")

    @property
    def n_bands(self) -> int:
        return len(self.wavelet.bands)

    @property
    def pool_multiplier(self) -> int:
        """How many values the temporal pooling emits per feature channel."""
        segments = self.n_segments if "segments" in self.pooling else 1
        statistics = 3 if "stats" in self.pooling else 1
        return segments * statistics

    @property
    def out_features(self) -> int:
        """Feature width contributed by this branch, across all bands."""
        per_band = self.hidden_channels[-1] * self.pool_multiplier
        if self.modulation:
            per_band += self.in_channels * self.modulation_bins
        return self.n_bands * per_band

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


class _TemporalPool(nn.Module):
    """Reduce ``(batch, channels, time)`` to a fixed-width vector, keeping time or not.

    ``AdaptiveAvgPool1d(1)`` -- the original behaviour -- destroys every temporal fact
    about the window. The other modes keep some: ``stats`` keeps how much the band varies
    and how high it peaks, ``segments`` keeps a coarse layout.
    """

    def __init__(self, mode: TemporalPooling, n_segments: int):
        super().__init__()
        self.mode = mode
        self.n_segments = n_segments if "segments" in mode else 1
        self.pool = nn.AdaptiveAvgPool1d(self.n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if "stats" not in self.mode:
            return self.pool(x).flatten(1)
        if self.n_segments == 1:
            blocks = [x]
        else:
            # Split into contiguous, near-equal segments and take statistics within each.
            blocks = list(torch.chunk(x, self.n_segments, dim=-1))
        parts: list[torch.Tensor] = []
        for block in blocks:
            parts.append(block.mean(dim=-1))
            # unbiased=False keeps this defined for a single-sample segment.
            parts.append(block.std(dim=-1, unbiased=False))
            parts.append(block.amax(dim=-1))
        return torch.cat(parts, dim=1)


class _ModulationPool(nn.Module):
    """Modulation spectrum of a band's envelope, in the 1-2 Hz burst-rhythm range.

    Chewing and grinding differ from clenching by *rhythm*: burst-relax cycles at roughly
    1-2 Hz. Convolutional features pooled over the window cannot express that, which is a
    structural reason -- not a capacity one -- why the original model could not separate
    those classes. This computes it explicitly and differentiably:

    ``|x|`` -> average-pool to ``n_frames`` -> remove the mean -> ``rfft`` -> magnitudes of
    the first ``n_bins`` modulation frequencies above DC.

    Over a 1 s window with 64 frames the bin spacing is 1 Hz, so six bins cover 1-6 Hz.
    """

    def __init__(self, n_frames: int, n_bins: int):
        super().__init__()
        self.n_frames = n_frames
        self.n_bins = n_bins
        self.envelope = nn.AdaptiveAvgPool1d(n_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.envelope(x.abs())
        frames = frames - frames.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(frames, dim=-1).abs()
        # Bin 0 is DC and was just removed; take the bins above it.
        available = spectrum.shape[-1] - 1
        taken = spectrum[..., 1 : 1 + min(self.n_bins, available)]
        if taken.shape[-1] < self.n_bins:
            taken = nn.functional.pad(taken, (0, self.n_bins - taken.shape[-1]))
        # Log-compress: envelope magnitudes are multiplicative across participants.
        return torch.log1p(taken).flatten(1)


class _BandBranch(nn.Module):
    """Convolutional stack applied to one wavelet band, pooled to a fixed-width vector."""

    def __init__(
        self,
        in_channels: int,
        hidden: tuple[int, int],
        kernel_size: int,
        pool: int,
        *,
        pooling: TemporalPooling = "avg",
        n_segments: int = 4,
    ):
        super().__init__()
        first, second = hidden
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, first, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
            nn.Conv1d(first, second, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(second),
            nn.ReLU(inplace=True),
        )
        self.pool = _TemporalPool(pooling, n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x))


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
                    pooling=config.pooling,
                    n_segments=config.n_segments,
                )
                for band in config.wavelet.bands
            }
        )
        self.modulation = (
            _ModulationPool(config.modulation_frames, config.modulation_bins)
            if config.modulation
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficients = self.decompose(x)
        parts: list[torch.Tensor] = []
        for band in self.config.wavelet.bands:
            coefficient = coefficients[band]
            parts.append(self.bands[band](coefficient))
            if self.modulation is not None:
                # Rhythm is read from the raw band envelope, not from the convolutional
                # features: the conv stack has already pooled away the time axis by a
                # factor of `pool_size` and its output is not an envelope.
                parts.append(self.modulation(coefficient))
        return torch.cat(parts, dim=1)


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
