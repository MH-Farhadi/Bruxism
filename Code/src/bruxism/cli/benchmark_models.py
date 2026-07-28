"""``bruxism-benchmark`` -- measure parameter counts and the three latencies.

Reports input/context latency, decision update interval and processing latency as three
separate quantities. Compute time alone is never presented as detection latency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bruxism.cli._common import add_common_arguments, make_parser, parse_and_run
from bruxism.data.labels import get_task
from bruxism.evaluation.benchmark import BenchmarkConfig, benchmark_model
from bruxism.models.baselines import NEURAL_MODEL_IDS, build_neural_model
from bruxism.utils.io import write_csv, write_json
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-benchmark", __doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--models", nargs="+", default=list(NEURAL_MODEL_IDS), choices=list(NEURAL_MODEL_IDS)
    )
    parser.add_argument("--task", default="five_class", help="Task id (sets the output width).")
    parser.add_argument(
        "--modality", default="fusion", choices=["fusion", "emg_only", "audio_only"]
    )
    parser.add_argument("--emg-channels", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--sampling-rate", type=int, default=1200)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="CPU is the honest default: it is the deployment target implied by a low-cost sensor.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks"), help="Output directory."
    )
    return parser


def main_impl(args: argparse.Namespace) -> int:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but no CUDA device is available")
    task = get_task(args.task)
    window_samples = int(round(args.window_seconds * args.sampling_rate))

    config = BenchmarkConfig(
        batch_sizes=tuple(args.batch_sizes),
        n_warmup=args.n_warmup,
        n_trials=args.n_trials,
        window_samples=window_samples,
        sampling_rate_hz=args.sampling_rate,
        emg_channels=args.emg_channels,
        device=args.device,
    )

    results: dict[str, object] = {}
    rows = []
    for model_id in args.models:
        model = build_neural_model(
            model_id,
            num_classes=task.num_classes,
            modality=args.modality,
            emg_channels=args.emg_channels,
            window_samples=window_samples,
        )
        payload = benchmark_model(
            model,
            config,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        payload["model_id"] = model_id
        results[model_id] = payload
        budget = payload["latency_budget"]
        first = payload["batch_sizes"][str(args.batch_sizes[0])]
        rows.append(
            {
                "model_id": model_id,
                "task_id": task.task_id,
                "modality": args.modality,
                "trainable_parameters": payload["parameter_counts"].get("trainable"),
                "model_size_kib_fp32": payload["model_size_bytes_fp32"] / 1024,
                "forward_ms_batch1": first["forward_only"]["median_seconds"] * 1e3,
                "forward_p95_ms_batch1": first["forward_only"]["p95_seconds"] * 1e3,
                "preprocessing_ms_per_window": (
                    payload["preprocessing"]["amortized_per_window_seconds"] * 1e3
                ),
                "processing_latency_ms": budget["processing_latency_seconds"] * 1e3,
                "input_context_latency_ms": budget["input_context_latency_seconds"] * 1e3,
                "decision_update_interval_ms": budget["decision_update_interval_seconds"] * 1e3,
                "earliest_decision_ms": budget["earliest_decision_seconds"] * 1e3,
                "device": args.device,
            }
        )
        print(
            f"{model_id:26s} params={payload['parameter_counts'].get('trainable'):>8,}  "
            f"forward={first['forward_only']['median_seconds'] * 1e3:7.3f} ms  "
            f"processing={budget['processing_latency_seconds'] * 1e3:7.3f} ms  "
            f"context={budget['input_context_latency_seconds'] * 1e3:.0f} ms"
        )

    import pandas as pd

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "benchmark.json", results)
    write_csv(args.output / "benchmark.csv", pd.DataFrame(rows))
    print(f"\nwritten to {args.output}")
    print(
        "\nNOTE: processing latency is compute time only. No decision about an event can "
        f"exist sooner than {args.window_seconds:g} s after its onset, and decisions update "
        f"every {args.stride_seconds:g} s. The production filter chain is zero-phase "
        "(acausal), so these numbers do not support a real-time or wearable claim."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
