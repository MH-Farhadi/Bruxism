"""Honest runtime benchmarking.

Three latencies are measured and reported **separately**, because collapsing them is the
overclaim this project exists to avoid:

``input_context_latency_seconds``
    The observation window. A decision about time *t* cannot be made before ``t + 1.0 s``,
    because the model needs a full one-second window. This dominates everything else.
``decision_update_interval_seconds``
    The stride. A new decision is produced every 0.5 s.
``processing_latency_seconds``
    Compute time: filtering, wavelet/feature transform and the forward pass.

A sub-millisecond forward pass does not create a low-latency detector. The reported
end-to-end responsiveness is bounded below by the window length.

Timing method: a warm-up phase is excluded, CUDA is synchronised around each timed region,
and median plus p95 are reported over repeated trials rather than a single mean.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from bruxism.models import BruxismModel
from bruxism.preprocessing.filters import FilterChainConfig, apply_filter_chain
from bruxism.utils.logging import get_logger
from bruxism.utils.reproducibility import collect_environment

__all__ = ["BenchmarkConfig", "LatencyBudget", "benchmark_model", "time_callable"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark parameters."""

    batch_sizes: tuple[int, ...] = (1, 32)
    n_warmup: int = 20
    n_trials: int = 200
    window_samples: int = 1200
    sampling_rate_hz: int = 1200
    emg_channels: int = 4
    device: str = "cpu"
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_sizes": list(self.batch_sizes),
            "n_warmup": self.n_warmup,
            "n_trials": self.n_trials,
            "window_samples": self.window_samples,
            "sampling_rate_hz": self.sampling_rate_hz,
            "emg_channels": self.emg_channels,
            "device": self.device,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class LatencyBudget:
    """The three latencies, kept apart by construction."""

    input_context_latency_seconds: float
    decision_update_interval_seconds: float
    processing_latency_seconds: float

    @property
    def earliest_decision_seconds(self) -> float:
        """Earliest wall-clock time a decision about event onset can exist."""
        return self.input_context_latency_seconds + self.processing_latency_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_context_latency_seconds": self.input_context_latency_seconds,
            "decision_update_interval_seconds": self.decision_update_interval_seconds,
            "processing_latency_seconds": self.processing_latency_seconds,
            "earliest_decision_seconds": self.earliest_decision_seconds,
            "note": (
                "Processing latency is compute time only. The system cannot report an "
                "event sooner than one full observation window after its onset, and "
                "produces a new decision once per stride. Do not describe the compute "
                "time alone as detection latency."
            ),
            "realtime_claim_supported": False,
            "realtime_claim_note": (
                "No streaming implementation exists and the production filter chain is "
                "zero-phase (acausal). Wearable, embedded or clinical real-time readiness "
                "is not demonstrated by these numbers."
            ),
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_callable(
    fn: Callable[[], Any],
    *,
    n_warmup: int,
    n_trials: int,
    device: torch.device,
) -> dict[str, float]:
    """Time ``fn`` with warm-up excluded, returning median / p95 / mean / min / max."""
    for _ in range(n_warmup):
        fn()
    _synchronize(device)

    samples: list[float] = []
    for _ in range(n_trials):
        _synchronize(device)
        started = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append(time.perf_counter() - started)

    ordered = sorted(samples)
    return {
        "median_seconds": statistics.median(ordered),
        "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "mean_seconds": statistics.fmean(ordered),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "n_trials": len(ordered),
    }


def benchmark_model(
    model: torch.nn.Module,
    config: BenchmarkConfig | None = None,
    *,
    filter_config: FilterChainConfig | None = None,
    window_seconds: float = 1.0,
    stride_seconds: float = 0.5,
) -> dict[str, Any]:
    """Benchmark preprocessing, the forward pass and the two combined.

    Returns a dict containing per-batch-size timings, the programmatic parameter count, the
    hardware/software record and the :class:`LatencyBudget`.
    """
    active = config or BenchmarkConfig()
    device = torch.device(active.device)
    dtype = getattr(torch, active.dtype)
    model = model.to(device=device, dtype=dtype).eval()

    chain = filter_config or FilterChainConfig()
    rng = np.random.default_rng(0)
    # Filtering is benchmarked on a full 60 s recording, because that is how the pipeline
    # actually applies it -- once per recording, not once per window.
    recording_samples = active.sampling_rate_hz * 60
    raw_emg = rng.standard_normal((recording_samples, active.emg_channels))
    raw_mic = rng.standard_normal(recording_samples)

    results: dict[str, Any] = {
        "config": active.to_dict(),
        "filter_chain": chain.describe(),
        "parameter_counts": (
            cast(BruxismModel, model).parameter_counts()
            if hasattr(model, "parameter_counts")
            else {"total": sum(p.numel() for p in model.parameters())}
        ),
        "model_size_bytes_fp32": sum(p.numel() for p in model.parameters()) * 4,
        "hardware": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "environment": collect_environment(),
        "batch_sizes": {},
    }

    # --- preprocessing, measured once per recording and normalised per window ---
    def filter_recording() -> None:
        apply_filter_chain(raw_emg, chain, active.sampling_rate_hz, modality="emg")
        apply_filter_chain(raw_mic, chain, active.sampling_rate_hz, modality="mic")

    filter_timing = time_callable(
        filter_recording,
        n_warmup=max(2, active.n_warmup // 10),
        n_trials=max(5, active.n_trials // 20),
        device=torch.device("cpu"),
    )
    windows_per_recording = int((60.0 - window_seconds) / stride_seconds) + 1
    results["preprocessing"] = {
        "scope": "one 60 s recording, filtered once before windowing",
        **filter_timing,
        "windows_per_recording": windows_per_recording,
        "amortized_per_window_seconds": (filter_timing["median_seconds"] / windows_per_recording),
    }

    for batch_size in active.batch_sizes:
        emg = torch.randn(
            batch_size, active.emg_channels, active.window_samples, device=device, dtype=dtype
        )
        mic = torch.randn(batch_size, 1, active.window_samples, device=device, dtype=dtype)

        runnable = cast(BruxismModel, model)

        @torch.no_grad()
        def forward() -> None:
            runnable(emg, mic)  # noqa: B023 - closes over the current batch

        forward_timing = time_callable(
            forward, n_warmup=active.n_warmup, n_trials=active.n_trials, device=device
        )
        results["batch_sizes"][str(batch_size)] = {
            "forward_only": forward_timing,
            "forward_per_window_seconds": forward_timing["median_seconds"] / batch_size,
            "end_to_end_per_window_seconds": (
                forward_timing["median_seconds"] / batch_size
                + results["preprocessing"]["amortized_per_window_seconds"]
            ),
        }

    per_window_processing = results["batch_sizes"][str(active.batch_sizes[0])][
        "end_to_end_per_window_seconds"
    ]
    results["latency_budget"] = LatencyBudget(
        input_context_latency_seconds=window_seconds,
        decision_update_interval_seconds=stride_seconds,
        processing_latency_seconds=per_window_processing,
    ).to_dict()
    logger.info(
        "benchmark: %.3f ms compute per window vs %.0f ms observation window",
        per_window_processing * 1e3,
        window_seconds * 1e3,
    )
    return results
