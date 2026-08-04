"""``bruxism-screen`` -- compare signal chains and preprocessing choices cheaply.

Answers the questions that would otherwise cost a 3.8-hour nested run each, and produces
the evidence ``new_prompt.md`` Phase 1.3 asks for:

    outputs/screening/<stamp>/
    |-- screening.json          all measurements, machine-readable
    |-- screening.md            the comparison tables
    |-- variants.csv            one row per filter variant
    |-- quality_<variant>.csv   per participant x class contamination and amplitude
    `-- ledger_<variant>.parquet

**Every number this writes is a screening estimate.** One model fit per outer fold, no
nested selection, a feature set chosen after looking at these five participants. It decides
*which* change is worth a confirmatory run; it never reports a result. See
``bruxism.evaluation.screening``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from bruxism.cli._common import (
    add_common_arguments,
    add_data_root_argument,
    make_parser,
    parse_and_run,
    resolve_data_root,
)
from bruxism.data.dataset import RecordingCache
from bruxism.data.labels import get_task
from bruxism.data.manifest import build_manifest
from bruxism.data.segments import SegmentationConfig, build_window_index
from bruxism.evaluation.screening import (
    ScreeningConfig,
    build_feature_matrix,
    headline,
    screen,
)
from bruxism.evaluation.signal_quality import (
    contrast_table,
    spread_summary,
    window_quality_table,
)
from bruxism.preprocessing.filters import (
    DEFAULT_EMG_BAND_HZ,
    DEFAULT_MAINS_HZ,
    FilterChainConfig,
    FilterStage,
    emg_stages,
)
from bruxism.utils.io import write_csv, write_json, write_parquet
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)

_LOW, _HIGH = DEFAULT_EMG_BAND_HZ

_BANDPASS = FilterStage(
    kind="bandpass",
    low_hz=_LOW,
    high_hz=_HIGH,
    order=4,
    rationale="Standard surface-EMG band, below the 600 Hz Nyquist frequency.",
)


def _variants() -> dict[str, tuple[FilterChainConfig, str]]:
    """The candidate EMG chains, each with the claim it is being tested for."""
    return {
        "superseded_60hz_only": (
            FilterChainConfig(emg_stages=tuple(emg_stages(notch_harmonics=False))),
            "The chain in every run before 2026-08-03: notch the fundamental the hardware "
            "had already removed, pass 180/300/420 Hz untouched.",
        ),
        "bandpass_only": (
            FilterChainConfig(emg_stages=(_BANDPASS,)),
            "Control: no mains removal at all. Isolates how much of any change is the "
            "notching rather than some other difference.",
        ),
        "notch_bank_constant_q": (
            FilterChainConfig(emg_stages=tuple(emg_stages(notch_bandwidth_hz=None, quality=30.0))),
            "Seven IIR notches at a constant Q=30: 2 Hz wide at 60 Hz, 14 Hz at 420 Hz. "
            "Costs 13 % of the band, and is too narrow at the fundamental where the "
            "hardware notch left a residue.",
        ),
        "notch_bank_bw4": (
            FilterChainConfig(emg_stages=tuple(emg_stages(notch_bandwidth_hz=4.0))),
            "Constant 4 Hz notches. Costs only 6.5 % of the band.",
        ),
        "notch_bank": (
            FilterChainConfig(emg_stages=tuple(emg_stages())),
            "PRODUCTION: seven constant-width 8 Hz notches, one per mains multiple in the "
            "passband. Same 13 % band cost as constant Q=30, 6.7x less residue.",
        ),
        "comb": (
            FilterChainConfig(
                emg_stages=(
                    FilterStage(
                        kind="comb",
                        freq_hz=DEFAULT_MAINS_HZ,
                        quality=30.0,
                        rationale="Single comb, nulls at every mains multiple. Cheaper.",
                    ),
                    _BANDPASS,
                )
            ),
            "One comb instead of seven notches: cheaper and causal-friendly, but notches "
            "every multiple up to Nyquist and needs an integer rate/mains ratio.",
        ),
        "spectral_interpolation": (
            FilterChainConfig(
                emg_stages=(
                    FilterStage(
                        kind="spectral_interpolation",
                        freq_hz=DEFAULT_MAINS_HZ,
                        low_hz=_LOW,
                        high_hz=_HIGH,
                        half_width_hz=1.5,
                        rationale=(
                            "Replace the magnitude of each harmonic's bins with the "
                            "surrounding noise floor, keeping phase. Preserves bandwidth."
                        ),
                    ),
                    _BANDPASS,
                )
            ),
            "Estimate and remove the sinusoidal components instead of deleting the band. "
            "Preserves bandwidth; inherently acausal.",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-screen", __doc__.splitlines()[0])
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument("--task-id", default="five_class", help="Classification task to screen.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Filter variants to compare. Default: all of them.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/screening"), help="Bundle directory."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--guard-seconds", type=float, default=0.25)
    parser.add_argument("--startup-guard-seconds", type=float, default=0.5)
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Subsample for a quick pass. Omit for the full set, which any quoted number needs.",
    )
    parser.add_argument(
        "--normalisation-scopes",
        nargs="*",
        default=["none", "per_participant"],
        help="Feature normalisation scopes to compare on the winning variant.",
    )
    parser.add_argument(
        "--aggregation-windows",
        nargs="*",
        type=int,
        default=[1, 2, 4, 6, 10, 16],
        help="Consecutive-window block sizes for the TRIAL-LEVEL secondary analysis.",
    )
    parser.add_argument("--skip-quality", action="store_true", help="Skip the spectral tables.")
    return parser


def _screen_variant(
    name: str,
    config: FilterChainConfig,
    *,
    manifest: Any,
    window_index: Any,
    task: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    cache = RecordingCache(manifest, config, cache_dir=args.cache_dir)
    payload: dict[str, Any] = {
        "filters": config.to_dict(),
        "describe": config.describe(),
        "supports_realtime_claim": config.realtime_claim_supported,
    }

    if not args.skip_quality:
        quality = window_quality_table(window_index, cache, label=f"{name}: signal quality")
        write_csv(output_dir / f"quality_{name}.csv", quality)
        payload["quality"] = {
            "per_cell": quality.to_dict("records"),
            "mains_fraction_max": float(quality["mains_fraction"].max()),
            "mains_fraction_mean": float(quality["mains_fraction"].mean()),
            "mains_fraction_rest_max": float(
                quality.loc[quality["task_family"] == "rest", "mains_fraction"].max()
            ),
            "harmonic_excess_max": float(quality["harmonic_excess_max"].max()),
            "contrast": contrast_table(quality).to_dict("records"),
            "spread": spread_summary(quality),
        }

    matrix = build_feature_matrix(
        window_index,
        cache,
        task,
        max_windows_per_class=args.max_windows_per_class,
        label=f"{name}: features",
    )
    payload["n_windows"] = len(matrix)
    payload["window_counts"] = matrix.counts().to_dict("records")

    models: dict[str, Any] = {}
    for model_id in ("logistic_regression", "gradient_boosting"):
        result = screen(matrix, ScreeningConfig(model_id=model_id))
        models[model_id] = {
            "headline": headline(result),
            "per_subject": {
                subject: {
                    key: entry[key]
                    for key in ("accuracy", "macro_f1", "balanced_accuracy", "n_samples")
                }
                for subject, entry in result["subject_level"]["per_subject"].items()
            },
        }
        if model_id == "logistic_regression":
            write_parquet(output_dir / f"ledger_{name}.parquet", result["ledger"])
    payload["models"] = models
    return payload


def main_impl(args: argparse.Namespace) -> int:
    data_root = resolve_data_root(args.data_root)
    assert data_root is not None

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = Path(args.output_root) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(data_root, probe_video=False)
    segmentation = SegmentationConfig(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        guard_seconds=args.guard_seconds,
        startup_guard_seconds=args.startup_guard_seconds,
    )
    window_index = build_window_index(manifest, segmentation)
    task = get_task(args.task_id)

    available = _variants()
    names = args.variants or list(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise KeyError(f"unknown variant(s) {unknown}; available: {sorted(available)}")

    report: dict[str, Any] = {
        "interpretation": (
            "SCREENING ONLY. One model fit per outer fold, no nested hyperparameter "
            "selection, feature set chosen after seeing these five participants. "
            "Directionally reliable, optimistic in level. Confirmatory numbers come from "
            "bruxism-train."
        ),
        "manifest_hash": manifest.manifest_hash,
        "window_index_hash": window_index.index_hash,
        "task_id": args.task_id,
        "segmentation": segmentation.to_dict(),
        "n_windows_total": len(window_index.windows),
        "variant_rationale": {name: available[name][1] for name in names},
        "variants": {},
    }

    for name in names:
        logger.info("screening filter variant %s", name)
        started = time.perf_counter()
        report["variants"][name] = _screen_variant(
            name,
            available[name][0],
            manifest=manifest,
            window_index=window_index,
            task=task,
            args=args,
            output_dir=output_dir,
        )
        report["variants"][name]["seconds"] = time.perf_counter() - started
        head = report["variants"][name]["models"]["logistic_regression"]["headline"]
        logger.info(
            "%s: LR macro-F1 %.3f accuracy %.3f",
            name,
            head["macro_f1"] or float("nan"),
            head["accuracy"] or float("nan"),
        )

    # The winner by screening macro-F1 gets the normalisation and aggregation sweeps.
    winner = max(
        names,
        key=lambda name: (
            report["variants"][name]["models"]["logistic_regression"]["headline"]["macro_f1"] or 0.0
        ),
    )
    report["winner_by_screening_macro_f1"] = winner

    cache = RecordingCache(manifest, available[winner][0], cache_dir=args.cache_dir)
    matrix = build_feature_matrix(
        window_index,
        cache,
        task,
        max_windows_per_class=args.max_windows_per_class,
        label=f"{winner}: features (sweeps)",
    )
    scopes: dict[str, Any] = {}
    for scope in args.normalisation_scopes:
        result = screen(matrix, ScreeningConfig(normalisation_scope=scope))
        scopes[scope] = headline(result)
    report["normalisation_scopes"] = {
        "measured_on": winner,
        "results": scopes,
        "caveat": (
            "per_participant uses the held-out participant's own UNLABELLED feature "
            "distribution. That is transductive test-time adaptation, not label leakage, "
            "and it must be reported as a declared calibration step."
        ),
    }

    aggregated = screen(
        matrix,
        ScreeningConfig(
            normalisation_scope="per_participant",
            aggregation_windows=tuple(args.aggregation_windows),
        ),
    )
    report["aggregation"] = {
        "measured_on": f"{winner} + per_participant normalisation",
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "results": {
            n: {
                "seconds_of_context": args.window_seconds + (int(n) - 1) * args.stride_seconds,
                "macro_f1_mean": payload["macro_f1_mean"],
                "accuracy_mean": payload["accuracy_mean"],
            }
            for n, payload in aggregated["aggregated"].items()
        },
        "caveat": (
            "TRIAL-LEVEL. Every recording holds one condition, so averaging inside a "
            "recording approaches a majority vote over a homogeneous trial. Valid as a "
            "labelled secondary analysis; invalid as continuous-stream detection."
        ),
    }

    write_json(output_dir / "screening.json", report)
    write_csv(output_dir / "variants.csv", _variants_frame(report))
    (output_dir / "screening.md").write_text(_render(report), encoding="utf-8")

    logger.info("screening bundle written to %s", output_dir)
    print(_render(report))
    return 0


def _variants_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, payload in report["variants"].items():
        row: dict[str, Any] = {"variant": name, "n_windows": payload["n_windows"]}
        quality = payload.get("quality")
        if quality:
            row["mains_fraction_mean"] = round(quality["mains_fraction_mean"], 4)
            row["mains_fraction_rest_max"] = round(quality["mains_fraction_rest_max"], 4)
            row["harmonic_excess_max"] = round(quality["harmonic_excess_max"], 3)
            row["amplitude_spread"] = round(quality["spread"]["spread_ratio"], 3)
            row["rest_above_activity_inversions"] = quality["spread"][
                "n_rest_above_activity_inversions"
            ]
        for model_id, entry in payload["models"].items():
            prefix = "lr" if model_id == "logistic_regression" else "gbm"
            row[f"{prefix}_macro_f1"] = _round(entry["headline"]["macro_f1"])
            row[f"{prefix}_accuracy"] = _round(entry["headline"]["accuracy"])
            row[f"{prefix}_macro_f1_min"] = _round(entry["headline"]["macro_f1_min"])
        rows.append(row)
    return pd.DataFrame(rows)


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _table(frame: pd.DataFrame) -> str:
    """Markdown table via the project's renderer, which needs no optional dependency."""
    from bruxism.reporting import _table as render

    return render(frame) + "\n"


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# Filter-chain and preprocessing screening",
        "",
        f"> **{report['interpretation']}**",
        "",
        f"Manifest `{report['manifest_hash']}`, window index "
        f"`{report['window_index_hash']}`, task `{report['task_id']}`, "
        f"{report['n_windows_total']:,} windows.",
        "",
        "## Filter variants",
        "",
        _table(_variants_frame(report)),
        "Rationale for each variant:",
        "",
    ]
    for name, why in report["variant_rationale"].items():
        lines.append(f"- **`{name}`** -- {why}")
    lines += [
        "",
        f"Winner by screening macro-F1: **`{report['winner_by_screening_macro_f1']}`**",
        "",
        "## Per-participant macro-F1 (logistic regression)",
        "",
    ]
    per_subject = pd.DataFrame(
        {
            name: {
                subject: round(entry["macro_f1"], 3)
                for subject, entry in payload["models"]["logistic_regression"][
                    "per_subject"
                ].items()
            }
            for name, payload in report["variants"].items()
        }
    )
    lines.append(_table(per_subject.reset_index(names="subject_id")))

    lines += ["## Feature normalisation scope", ""]
    scope_frame = pd.DataFrame(
        [
            {"scope": scope, **{k: _round(v) for k, v in values.items()}}
            for scope, values in report["normalisation_scopes"]["results"].items()
        ]
    )
    lines += [
        _table(scope_frame),
        f"> {report['normalisation_scopes']['caveat']}",
        "",
        "## Temporal aggregation (trial-level secondary analysis)",
        "",
    ]
    aggregation_frame = pd.DataFrame(
        [
            {
                "windows": int(n),
                "seconds_of_context": payload["seconds_of_context"],
                "macro_f1": round(payload["macro_f1_mean"], 4),
                "accuracy": round(payload["accuracy_mean"], 4),
            }
            for n, payload in report["aggregation"]["results"].items()
        ]
    )
    lines += [_table(aggregation_frame), f"> {report['aggregation']['caveat']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
