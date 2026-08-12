"""Typed experiment configuration.

An :class:`ExperimentConfig` is the complete, resolved description of a run: data root,
segmentation policy, preprocessing, model, training, selection and output settings. It is
loaded from YAML, validated on construction, and serialised verbatim into every run bundle
as ``resolved_config.yaml``. Its hash appears on every prediction row.

Unknown keys raise. A typo in a config file must not silently fall back to a default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

from bruxism.data.labels import get_task
from bruxism.data.quality import ExclusionPolicy
from bruxism.data.segments import SegmentationConfig, SegmentationPolicy
from bruxism.models.dual_branch import Modality
from bruxism.preprocessing.augmentation import AugmentationConfig
from bruxism.preprocessing.filters import FilterChainConfig
from bruxism.preprocessing.normalization import NormalizationConfig
from bruxism.training.selection import SelectionConfig
from bruxism.utils.io import hash_mapping, read_yaml

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "OutputConfig",
    "TrainingConfig",
    "load_experiment_config",
]

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for an invalid or unrecognised configuration key or value."""


def _build(cls: type[T], payload: dict[str, Any] | None, *, path: str) -> T:
    """Construct a dataclass from a mapping, rejecting unknown keys."""
    data = dict(payload or {})
    if not is_dataclass(cls):  # pragma: no cover - programming error
        raise TypeError(f"{cls} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {unknown} under '{path}'; valid keys are {sorted(known)}"
        )
    return cls(**data)  # type: ignore[return-value]


@dataclass(frozen=True)
class DataConfig:
    """Where the data is and how it becomes examples."""

    data_root: str = "../Data"
    sampling_rate_hz: int = 1200
    segmentation_policy: str = str(SegmentationPolicy.TRIGGER_CONSTRAINED)
    window_seconds: float = 1.0
    stride_seconds: float = 0.5
    guard_seconds: float = 0.25
    startup_guard_seconds: float = 0.5
    shutdown_guard_seconds: float = 0.0
    allow_trigger_off_as_rest: bool = False
    trigger_off_rest_approved_by: str | None = None
    emg_channels: tuple[int, ...] | None = None
    subjects: tuple[str, ...] | None = None
    cache_dir: str | None = "outputs/cache"

    def segmentation(self) -> SegmentationConfig:
        return SegmentationConfig(
            policy=SegmentationPolicy(self.segmentation_policy),
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
            guard_seconds=self.guard_seconds,
            startup_guard_seconds=self.startup_guard_seconds,
            shutdown_guard_seconds=self.shutdown_guard_seconds,
            allow_trigger_off_as_rest=self.allow_trigger_off_as_rest,
            trigger_off_rest_approved_by=self.trigger_off_rest_approved_by,
        )

    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sampling_rate_hz))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["emg_channels"] = list(self.emg_channels) if self.emg_channels else None
        payload["subjects"] = list(self.subjects) if self.subjects else None
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser, loss and loader settings."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    loss: Literal["cross_entropy", "focal"] = "focal"
    focal_gamma: float = 1.5
    label_smoothing: float = 0.0
    class_weight_scheme: Literal["balanced", "none", "inverse_sqrt"] = "balanced"
    grad_clip_norm: float | None = 1.0
    num_workers: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool = True
    seeds: tuple[int, ...] = (0,)
    #: Hyperparameter grid searched on inner folds. Keys must be TrainingConfig fields.
    search_space: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if not self.seeds:
            raise ConfigError("at least one seed is required")
        valid = {f.name for f in fields(self)} - {"search_space", "seeds"}
        unknown = sorted(set(self.search_space) - valid)
        if unknown:
            raise ConfigError(
                f"search_space contains keys that are not TrainingConfig fields: {unknown}"
            )

    def replace(self, **overrides: Any) -> TrainingConfig:
        payload = asdict(self)
        payload["seeds"] = tuple(self.seeds)
        payload["search_space"] = dict(self.search_space)
        payload.update(overrides)
        return TrainingConfig(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        return payload


@dataclass(frozen=True)
class OutputConfig:
    """Where a run writes and what it writes."""

    runs_root: str = "outputs/runs"
    run_id: str | None = None
    save_checkpoints: bool = True
    save_embeddings: bool = True
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete, resolved experiment specification."""

    name: str
    task_id: str = "five_class"
    model_id: str = "dual_branch_wavelet_cnn"
    modality: Modality = "fusion"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    filters: FilterChainConfig = field(default_factory=FilterChainConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model_overrides: dict[str, Any] = field(default_factory=dict)
    #: Extra tasks/modalities to sweep. Used by the ablation and baseline runners.
    sweep_task_ids: tuple[str, ...] = ()
    sweep_modalities: tuple[Modality, ...] = ()
    sweep_model_ids: tuple[str, ...] = ()
    #: ``"<name>, <date>: <reason>"``. Required before any run that reads the microphone
    #: may start on data carrying a microphone-integrity flag. See ``audio.md`` and rule
    #: ``R4_mic_channel_is_not_analysable_audio``: 83 of the 100 recordings in the 2025-08
    #: collection carry another participant's audio, so a fusion or audio-only run on them
    #: is not evaluating leave-one-subject-out on that channel. Same idiom as
    #: ``trigger_off_rest_approved_by`` and ``calibration_approved_by`` -- a concession that
    #: has to be signed for rather than defaulted into.
    mic_defect_acknowledged_by: str | None = None
    #: ``"<name>, <date>: <reason>"``. Required before a branch may read a wavelet band that
    #: its own filter chain has already removed. The published runs read the mic ``A5``
    #: band (0-18.75 Hz) behind a 20 Hz high-pass and must keep reproducing, so they declare
    #: it here; a new configuration that wants the same thing has to say so in writing.
    stopband_bands_acknowledged_by: str | None = None

    def __post_init__(self) -> None:
        get_task(self.task_id)  # raises on an unknown task id
        for task_id in self.sweep_task_ids:
            get_task(task_id)
        if self.modality not in ("fusion", "emg_only", "audio_only"):
            raise ConfigError(f"unknown modality {self.modality!r}")

    @property
    def reads_microphone(self) -> bool:
        """Whether any condition this run will execute consumes the microphone channel."""
        modalities = set(self.sweep_modalities) or {self.modality}
        return bool(modalities & {"fusion", "audio_only"})

    @property
    def task(self):  # noqa: ANN201 - returns ClassificationTask
        return get_task(self.task_id)

    def exclusion_policy(self) -> ExclusionPolicy:
        return ExclusionPolicy()

    def to_dict(self) -> dict[str, Any]:
        """Fully resolved configuration, written as ``resolved_config.yaml``."""
        return {
            "name": self.name,
            "description": self.description,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "modality": self.modality,
            "data": self.data.to_dict(),
            "filters": self.filters.to_dict(),
            "normalization": self.normalization.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "training": self.training.to_dict(),
            "selection": self.selection.to_dict(),
            "output": self.output.to_dict(),
            "model_overrides": dict(self.model_overrides),
            "sweep_task_ids": list(self.sweep_task_ids),
            "sweep_modalities": list(self.sweep_modalities),
            "sweep_model_ids": list(self.sweep_model_ids),
            "mic_defect_acknowledged_by": self.mic_defect_acknowledged_by,
            "stopband_bands_acknowledged_by": self.stopband_bands_acknowledged_by,
        }

    #: Keys that are recorded in ``resolved_config.yaml`` but excluded from
    #: :attr:`config_hash`. An acknowledgement says who authorised a run and why; it does
    #: not change a single number the run computes. Two runs that differ only in who signed
    #: for them must produce identical results, and therefore identical identity -- and the
    #: published bundles (``2b6fb5ac``, ``cead62e4``) must keep their hashes after the
    #: acknowledgements required by ``audio.md`` are added to their configurations.
    _UNHASHED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"mic_defect_acknowledged_by", "stopband_bands_acknowledged_by"}
    )

    def hashed_dict(self) -> dict[str, Any]:
        """The part of the configuration that determines the numbers, and only that."""
        return {
            key: value for key, value in self.to_dict().items() if key not in self._UNHASHED_KEYS
        }

    @property
    def config_hash(self) -> str:
        """Identity of this configuration, recorded on every prediction row.

        Computed from :meth:`hashed_dict`, not :meth:`to_dict`: see :attr:`_UNHASHED_KEYS`.
        """
        return hash_mapping(self.hashed_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        data = dict(payload)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"unknown top-level config key(s) {unknown}; valid keys are {sorted(known)}"
            )
        if "name" not in data:
            raise ConfigError("configuration must declare a 'name'")

        nested: dict[str, Any] = {}
        nested["data"] = _build(DataConfig, data.pop("data", None), path="data")
        nested["normalization"] = _build(
            NormalizationConfig, data.pop("normalization", None), path="normalization"
        )
        nested["augmentation"] = _build(
            AugmentationConfig, data.pop("augmentation", None), path="augmentation"
        )
        nested["training"] = _build(TrainingConfig, data.pop("training", None), path="training")
        nested["selection"] = _build(SelectionConfig, data.pop("selection", None), path="selection")
        nested["output"] = _build(OutputConfig, data.pop("output", None), path="output")
        nested["filters"] = FilterChainConfig.from_dict(data.pop("filters", None) or {})

        for key in ("sweep_task_ids", "sweep_modalities", "sweep_model_ids"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data, **nested)


def _coerce_tuples(payload: dict[str, Any]) -> dict[str, Any]:
    """YAML lists become tuples where the dataclasses expect immutability."""
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("emg_channels", "subjects"):
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
    training = payload.get("training")
    if isinstance(training, dict) and isinstance(training.get("seeds"), list):
        training["seeds"] = tuple(training["seeds"])
    return payload


def load_experiment_config(path: Path | str, *, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration.

    Parameters
    ----------
    path
        Path to the YAML file.
    overrides
        ``dotted.key=value`` strings applied after loading, for CLI overrides such as
        ``--set training.batch_size=32``. Values are parsed as YAML scalars.

    Raises
    ------
    ConfigError
        On an unknown key, an invalid value, or an override that names a missing key.
    """
    import yaml

    payload = read_yaml(path)
    if not isinstance(payload, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override {override!r} must be of the form key.path=value")
        dotted, _, raw = override.partition("=")
        cursor: Any = payload
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(
                    f"override {dotted!r} refers to '{part}', which is not present in {path}"
                )
            cursor = cursor[part]
        cursor[parts[-1]] = yaml.safe_load(raw)

    return ExperimentConfig.from_dict(_coerce_tuples(payload))
