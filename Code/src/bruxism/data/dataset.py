"""Lazy, cached access to windowed recordings.

Design constraints this module exists to satisfy:

* **Filter the continuous recording, then cut windows.** :class:`RecordingCache` filters a
  whole recording once and memory-maps the result; windows are slices of that array. The
  prototype's alternative -- filtering each one-second window -- injects a filter transient
  at both edges of every example.
* **Never hold every overlapping window in RAM.** The prototype materialised ~12,000
  windows of ``(1200, 5)`` floats up front. Here windows are ``numpy`` views into a
  memory-mapped per-recording array, so resident memory is bounded by the recordings a
  worker touches, not by the window count.
* **Cache keys include everything that changes the bytes.** A cache entry is keyed by the
  source CSV's SHA-256, the resolved filter configuration and the channel selection. A
  changed filter cutoff produces a different key rather than a stale hit.
* **Wavelets are computed here, not inside ``forward()``.** The network receives ready
  tensors, so no GPU batch is round-tripped through NumPy per step.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from bruxism.data.labels import ClassificationTask, TaskFamily
from bruxism.data.manifest import DatasetManifest
from bruxism.data.schema import EMG_COLUMNS, MIC_COLUMN, read_recording_csv
from bruxism.data.segments import WindowIndex, WindowRecord
from bruxism.preprocessing.augmentation import Augmenter
from bruxism.preprocessing.filters import FilterChainConfig, apply_filter_chain
from bruxism.preprocessing.normalization import Normalizer
from bruxism.utils.io import hash_mapping
from bruxism.utils.logging import get_logger

__all__ = [
    "Modality",
    "RecordingCache",
    "WindowDataset",
    "collate_windows",
]

logger = get_logger(__name__)

Modality = Literal["fusion", "emg_only", "audio_only"]
Stage = Literal["train", "val", "test"]

_CACHE_FORMAT_VERSION = "1"


class RecordingCache:
    """Filters whole recordings once and serves memory-mapped views of the result.

    Parameters
    ----------
    manifest
        Provides the data root and the per-recording CSV checksums used as cache keys.
    filter_config
        Applied to the continuous recording before any window is cut.
    cache_dir
        Where ``.npy`` caches live. ``None`` disables disk caching and keeps filtered
        recordings in an in-process dictionary instead (used by tests and smoke runs).
    emg_channels
        Which EMG columns to keep, by position. Defaults to all four. Exposed because the
        exact montage interpretation is an open investigator question; the default keeps
        every recorded channel rather than discarding data on an assumption.
    max_resident
        Upper bound on simultaneously open memory maps, evicted least-recently-used.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        filter_config: FilterChainConfig,
        *,
        cache_dir: Path | str | None = None,
        emg_channels: Sequence[int] | None = None,
        max_resident: int = 24,
    ):
        self.manifest = manifest
        self.filter_config = filter_config
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.emg_channels = tuple(
            emg_channels if emg_channels is not None else range(len(EMG_COLUMNS))
        )
        if not self.emg_channels or max(self.emg_channels) >= len(EMG_COLUMNS):
            raise ValueError(
                f"emg_channels {self.emg_channels} out of range for {len(EMG_COLUMNS)} columns"
            )
        self.max_resident = max_resident
        self._resident: dict[str, np.ndarray] = {}
        self._order: list[str] = []
        self._computed = 0  # cache misses filtered so far, reported as progress

    @property
    def n_emg_channels(self) -> int:
        return len(self.emg_channels)

    def _cache_key(self, recording_id: str) -> str:
        record = self.manifest.by_id(recording_id)
        return hash_mapping(
            {
                "format": _CACHE_FORMAT_VERSION,
                "csv_sha256": record.csv_sha256,
                "filter": self.filter_config.to_dict(),
                "emg_channels": list(self.emg_channels),
                "sampling_rate": self.manifest.sampling_rate_hz,
            },
            length=24,
        )

    def _compute(self, recording_id: str) -> np.ndarray:
        """Filter a whole recording. Returns ``(n_samples, n_emg + 1)`` float32."""
        record = self.manifest.by_id(recording_id)
        frame = read_recording_csv(self.manifest.csv_path(record))
        columns = [EMG_COLUMNS[i] for i in self.emg_channels]
        emg = frame[columns].to_numpy(dtype=np.float64)
        mic = frame[MIC_COLUMN].to_numpy(dtype=np.float64)

        rate = self.manifest.sampling_rate_hz
        emg = apply_filter_chain(emg, self.filter_config, rate, modality="emg")
        mic = apply_filter_chain(mic, self.filter_config, rate, modality="mic")
        return np.ascontiguousarray(np.concatenate([emg, mic[:, None]], axis=1), dtype=np.float32)

    def get(self, recording_id: str) -> np.ndarray:
        """Filtered recording as ``(n_samples, n_emg_channels + 1)``; last column is mic."""
        if recording_id in self._resident:
            self._order.remove(recording_id)
            self._order.append(recording_id)
            return self._resident[recording_id]

        if self.cache_dir is None:
            array = self._compute(recording_id)
        else:
            path = self.cache_dir / f"{recording_id}.{self._cache_key(recording_id)}.npy"
            if not path.is_file():
                # A cache miss filters a whole recording, which takes long enough to look
                # like a hang on a cold cache. Misses only happen once per (recording,
                # filter chain), so reporting each one at INFO is not noise on a warm run.
                started = time.perf_counter()
                path.parent.mkdir(parents=True, exist_ok=True)
                computed = self._compute(recording_id)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, computed)
                tmp.replace(path)
                self._computed += 1
                logger.info(
                    "filter cache miss: built %s in %.1fs (%d of at most %d recording(s))",
                    recording_id,
                    time.perf_counter() - started,
                    self._computed,
                    len(self.manifest.included),
                    extra={"recording_id": recording_id, "n_computed": self._computed},
                )
            array = np.load(path, mmap_mode="r")

        self._resident[recording_id] = array
        self._order.append(recording_id)
        while len(self._order) > self.max_resident:
            evicted = self._order.pop(0)
            self._resident.pop(evicted, None)
        return array

    def window(self, window: WindowRecord) -> tuple[np.ndarray, np.ndarray]:
        """``(emg, mic)`` slices for one window, as ``float64`` copies.

        Copies are returned rather than views so that in-place normalisation and
        augmentation cannot corrupt the shared memory map.
        """
        array = self.get(window.recording_id)
        block = np.asarray(array[window.start_sample : window.end_sample], dtype=np.float64)
        expected = window.end_sample - window.start_sample
        if block.shape[0] != expected:
            raise ValueError(
                f"window {window.sample_id} requests samples "
                f"[{window.start_sample}, {window.end_sample}) but the recording only has "
                f"{array.shape[0]}"
            )
        return block[:, :-1], block[:, -1]

    def training_statistics_source(
        self, windows: Sequence[WindowRecord], *, max_windows: int | None = 2000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated signal for fitting a :class:`Normalizer`.

        Parameters
        ----------
        windows
            **Training** windows only. The caller is responsible for excluding held-out
            participants; :meth:`Normalizer.assert_not_fitted_on` verifies it afterwards.
        max_windows
            Subsample deterministically (evenly spaced, no RNG) above this count. Mean and
            standard deviation converge long before 2000 one-second windows, and the
            subsampling is reproducible.
        """
        selected = list(windows)
        if max_windows is not None and len(selected) > max_windows:
            step = len(selected) / max_windows
            selected = [selected[int(i * step)] for i in range(max_windows)]
        emg_blocks, mic_blocks = [], []
        for window in selected:
            emg, mic = self.window(window)
            emg_blocks.append(emg)
            mic_blocks.append(mic)
        if not emg_blocks:
            raise ValueError("no training windows supplied for normalization fitting")
        return np.concatenate(emg_blocks, axis=0), np.concatenate(mic_blocks, axis=0)


@dataclass
class _Example:
    window: WindowRecord
    label: int


class WindowDataset(Dataset):
    """Torch dataset over a subset of a :class:`WindowIndex`, for one classification task.

    Every item is ``(emg, mic, label, sample_index)``:

    ``emg``
        ``float32`` tensor ``(n_emg_channels, n_samples)`` -- channels first, as Conv1d wants.
    ``mic``
        ``float32`` tensor ``(1, n_samples)``.
    ``label``
        ``int64`` scalar tensor.
    ``sample_index``
        ``int64`` position in :attr:`sample_ids`, so predictions can be joined back to the
        ledger without shipping strings through the collate function.

    Parameters
    ----------
    stage
        ``"train"`` is the only stage for which ``augmenter`` may fire; the dataset raises
        if an augmenter is supplied for any other stage.
    modality
        ``"emg_only"`` zeroes the microphone tensor and ``"audio_only"`` zeroes the EMG
        tensor **after** normalisation, so all three modality conditions see identical
        windows, splits and preprocessing -- the ablation isolates the modality and nothing
        else.
    """

    def __init__(
        self,
        window_index: WindowIndex,
        sample_ids: Sequence[str],
        task: ClassificationTask,
        cache: RecordingCache,
        *,
        stage: Stage,
        normalizer: Normalizer | None = None,
        augmenter: Augmenter | None = None,
        modality: Modality = "fusion",
    ):
        self.task = task
        self.cache = cache
        self.stage = stage
        self.normalizer = normalizer
        self.modality = modality
        self._epoch = 0

        if augmenter is not None and stage != "train":
            raise ValueError(
                f"an augmenter was supplied for stage {stage!r}; augmentation is a "
                f"training-only transformation"
            )
        self.augmenter = augmenter

        wanted = set(sample_ids)
        examples: list[_Example] = []
        for window in window_index.windows:
            if window.sample_id not in wanted:
                continue
            label = task.label_for_family(TaskFamily(window.task_family))
            if label is None:
                continue  # family excluded from this task (e.g. chewing in no-chewing)
            examples.append(_Example(window=window, label=label))
        examples.sort(key=lambda example: example.window.sample_id)
        self.examples = examples
        self.sample_ids: list[str] = [example.window.sample_id for example in examples]

        missing = wanted - set(self.sample_ids)
        excluded_by_task = {
            window.sample_id
            for window in window_index.windows
            if window.sample_id in wanted
            and task.label_for_family(TaskFamily(window.task_family)) is None
        }
        unexplained = missing - excluded_by_task
        if unexplained:
            raise ValueError(
                f"{len(unexplained)} requested sample ids are not in the window index "
                f"(e.g. {sorted(unexplained)[:3]})"
            )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed augmentation. Called once per training epoch."""
        self._epoch = int(epoch)

    @property
    def labels(self) -> np.ndarray:
        return np.array([example.label for example in self.examples], dtype=np.int64)

    @property
    def subjects(self) -> list[str]:
        return [example.window.subject_id for example in self.examples]

    def label_counts(self) -> dict[int, int]:
        """Per-class example counts, used for class weights and minority detection."""
        counts = dict.fromkeys(range(self.task.num_classes), 0)
        for example in self.examples:
            counts[example.label] += 1
        return counts

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        example = self.examples[index]
        emg, mic = self.cache.window(example.window)

        if self.augmenter is not None:
            emg, mic = self.augmenter(
                emg,
                mic,
                label=example.label,
                sample_id=example.window.sample_id,
                epoch=self._epoch,
                stage=self.stage,
            )

        if self.normalizer is not None:
            # The subject is passed so a per-participant normalizer can apply that
            # participant's own statistics; every other scope ignores it.
            subject = example.window.subject_id
            emg = self.normalizer.transform_emg(emg, subject=subject)
            mic = self.normalizer.transform_mic(mic, subject=subject)

        emg_tensor = torch.from_numpy(np.ascontiguousarray(emg.T, dtype=np.float32))
        mic_tensor = torch.from_numpy(np.ascontiguousarray(mic[None, :], dtype=np.float32))

        if self.modality == "emg_only":
            mic_tensor = torch.zeros_like(mic_tensor)
        elif self.modality == "audio_only":
            emg_tensor = torch.zeros_like(emg_tensor)

        return emg_tensor, mic_tensor, torch.tensor(example.label, dtype=torch.long), index

    def describe(self) -> dict[str, Any]:
        """Summary written into the run bundle."""
        return {
            "task_id": self.task.task_id,
            "stage": self.stage,
            "modality": self.modality,
            "n_examples": len(self),
            "subjects": sorted(set(self.subjects)),
            "label_counts": {
                self.task.class_names[label]: count
                for label, count in sorted(self.label_counts().items())
            },
            "augmentation": self.augmenter.config.to_dict() if self.augmenter else None,
            "normalizer_fitted_on": (list(self.normalizer.fitted_on) if self.normalizer else None),
        }


def collate_windows(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a batch of ``(emg, mic, label, index)`` tuples."""
    emg = torch.stack([item[0] for item in batch])
    mic = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    indices = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return emg, mic, labels, indices
