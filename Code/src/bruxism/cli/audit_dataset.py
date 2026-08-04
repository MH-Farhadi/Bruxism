"""``bruxism-audit`` -- inspect the raw dataset without modifying it.

Produces the machine-readable and human-readable audit bundle described in the
implementation brief:

    outputs/data_audit/<manifest_hash>/
    |-- manifest.parquet
    |-- manifest.csv
    |-- data_audit.json
    |-- data_audit.md
    |-- trigger_summary.csv
    |-- window_counts.csv
    |-- guard_sensitivity.csv
    `-- quality_figures/

The raw data is opened read-only. No participant name, absolute path, survey, photograph,
receipt or reimbursement file is referenced in any generated artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bruxism.cli._common import (
    add_common_arguments,
    add_data_root_argument,
    make_parser,
    parse_and_run,
    resolve_data_root,
)
from bruxism.data.dataset import RecordingCache
from bruxism.data.manifest import build_manifest
from bruxism.data.quality import CONFLICT_RESOLUTIONS, QualityFlag, describe_flag
from bruxism.data.schema import (
    CANONICAL_COLUMNS,
    EMG_MUSCLE_MAP,
    NOMINAL_SAMPLING_RATE_HZ,
    SIGNAL_UNITS,
)
from bruxism.data.segments import SegmentationConfig, SegmentationPolicy, build_window_index
from bruxism.evaluation.segmentation import trigger_onset_alignment, window_guard_sweep
from bruxism.preprocessing.filters import FilterChainConfig
from bruxism.preprocessing.interference import (
    MAINS_CONTAMINATION_THRESHOLD,
    MAINS_HALF_WIDTH_HZ,
)
from bruxism.reporting import render_audit_markdown
from bruxism.utils.io import write_csv, write_json, write_parquet
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)

#: Guard widths swept for the sensitivity table. The approved value must be chosen by an
#: investigator; this table is the evidence for that choice.
GUARD_SWEEP_SECONDS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-audit", __doc__.splitlines()[0])
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/data_audit"),
        help="Directory that will receive a <manifest_hash> subdirectory.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=NOMINAL_SAMPLING_RATE_HZ,
        help="Configured nominal sampling rate; recordings that disagree are flagged.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip probing .avi files (faster).")
    parser.add_argument(
        "--hash-video",
        action="store_true",
        help="SHA-256 each .avi as well (slow: the videos total roughly 1.5 GB).",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip the quality figures.")
    parser.add_argument(
        "--no-onset-alignment",
        action="store_true",
        help="Skip the trigger-onset alignment measurement (it filters every recording).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache"),
        help="Filtered-recording cache used by the onset-alignment measurement.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Scan and report to the console; write nothing.",
    )
    return parser


def _guard_sensitivity(manifest: Any) -> pd.DataFrame:
    """How the analysable window count depends on the transition guard width."""
    rows: list[dict[str, Any]] = []
    for guard in GUARD_SWEEP_SECONDS:
        index = build_window_index(
            manifest,
            SegmentationConfig(guard_seconds=guard, startup_guard_seconds=0.5),
        )
        frame = index.frame
        if frame.empty:
            continue
        pivot = frame.pivot_table(
            index="subject_id", columns="task_family", aggfunc="size", fill_value=0
        )
        row: dict[str, Any] = {
            "guard_seconds": guard,
            "total_windows": int(len(frame)),
            "min_subject_class_cell": int(pivot.to_numpy().min()),
            "n_cells_below_10": int((pivot.to_numpy() < 10).sum()),
        }
        for family in sorted(pivot.columns):
            row[f"n_{family}"] = int(pivot[family].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _mains_contamination(frame: pd.DataFrame) -> dict[str, Any]:
    """Per-recording mains contamination of the RAW signal, worst offenders first.

    This is the measurement that was missing when the defect in ``cause.md`` went
    undetected through three reproducible runs: nothing ever reported how much of the
    signal was interference. It describes the acquisition, so the corrected filter chain
    does not change it -- and must not, or the audit would stop being able to say that a
    channel arrived contaminated.
    """
    ordered = frame.sort_values("mains_harmonic_power_fraction", ascending=False)
    by_subject = (
        frame.groupby("subject_id", observed=True)["mains_harmonic_power_fraction"]
        .agg(["mean", "min", "max"])
        .round(4)
        .reset_index()
    )
    by_condition = (
        frame.groupby("condition", observed=True)["mains_harmonic_power_fraction"]
        .agg(["mean", "min", "max"])
        .round(4)
        .reset_index()
    )
    flagged = frame["quality_flags"].str.contains(QualityFlag.MAINS_CONTAMINATION.value, na=False)
    return {
        "threshold": MAINS_CONTAMINATION_THRESHOLD,
        "half_width_hz": MAINS_HALF_WIDTH_HZ,
        "band_hz": [20.0, 450.0],
        "measured_on": "raw signal, before any offline filtering",
        "n_flagged": int(flagged.sum()),
        "n_recordings": int(len(frame)),
        "median_fraction": float(frame["mains_harmonic_power_fraction"].median()),
        "max_fraction": float(frame["mains_harmonic_power_fraction"].max()),
        "by_subject": by_subject.to_dict("records"),
        "by_condition": by_condition.to_dict("records"),
        "worst_recordings": ordered.head(15)[
            [
                "recording_id",
                "subject_id",
                "condition",
                "mains_harmonic_power_fraction",
                "mains_harmonic_breakdown",
            ]
        ].to_dict("records"),
        "policy": describe_flag(QualityFlag.MAINS_CONTAMINATION),
    }


def _trigger_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("condition", observed=True).agg(
        n_recordings=("recording_id", "size"),
        mean_active_fraction=("trigger_active_fraction", "mean"),
        min_active_fraction=("trigger_active_fraction", "min"),
        max_active_fraction=("trigger_active_fraction", "max"),
        mean_n_runs=("n_trigger_runs", "mean"),
        min_n_runs=("n_trigger_runs", "min"),
        max_n_runs=("n_trigger_runs", "max"),
    )
    return grouped.round(4).reset_index()


def _run_duration_frame(manifest: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in manifest.records:
        for boundary in record.trigger_run_boundaries:
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "condition": record.condition,
                    "task_family": record.task_family,
                    "duration_seconds": (boundary["end_sample"] - boundary["start_sample"])
                    / manifest.sampling_rate_hz,
                }
            )
    return pd.DataFrame(rows)


def _quality_figures(manifest: Any, runs: pd.DataFrame, output_dir: Path) -> list[str]:
    """Trigger-run duration distributions and startup-transient summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bruxism.visualization.paper_figures import FigureStyle, save_figure

    FigureStyle.apply()
    written: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs.empty:
        families = sorted(runs["task_family"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        data = [
            runs.loc[runs.task_family == family, "duration_seconds"].to_numpy()
            for family in families
        ]
        parts = ax.boxplot(data, tick_labels=families, showfliers=False, patch_artist=True)
        for index, patch in enumerate(parts["boxes"]):
            patch.set_facecolor(FigureStyle.color(index))
            patch.set_alpha(0.65)
        ax.axhline(
            2.0,
            color="#CC0000",
            linestyle="--",
            linewidth=1.1,
            label="2.0 s: shortest run yielding one window at a 0.5 s guard",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Trigger-run duration (s, log scale)")
        ax.set_title("Trigger-run durations by task family")
        ax.legend(fontsize=7)
        for path in save_figure(fig, output_dir, "trigger_run_durations"):
            written.append(path.name)

    frame = manifest.frame
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].hist(frame["startup_transient_seconds"], bins=24, color=FigureStyle.color(0))
    axes[0].set_xlabel("Settling time (s)")
    axes[0].set_ylabel("Recordings")
    axes[0].set_title("Startup transient duration")
    axes[1].hist(
        np.log10(frame["startup_transient_peak_ratio"].clip(lower=1)),
        bins=24,
        color=FigureStyle.color(1),
    )
    axes[1].set_xlabel("log10(peak excursion / robust scale)")
    axes[1].set_title("Startup transient magnitude")
    for path in save_figure(fig, output_dir, "startup_transients"):
        written.append(path.name)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    counts = frame.groupby("condition", observed=True)["trigger_active_fraction"].mean()
    ax.barh(list(counts.index), counts.to_numpy(), color=FigureStyle.color(2))
    ax.set_xlabel("Mean trigger-active fraction")
    ax.set_title("Trigger-active fraction by condition")
    for path in save_figure(fig, output_dir, "trigger_active_fraction"):
        written.append(path.name)
    return written


def main_impl(args: argparse.Namespace) -> int:
    data_root = resolve_data_root(args.data_root)
    assert data_root is not None

    manifest = build_manifest(
        data_root,
        sampling_rate_hz=args.sampling_rate,
        probe_video=not args.no_video,
        hash_video=args.hash_video,
    )
    frame = manifest.frame
    runs = _run_duration_frame(manifest)

    flag_counts = Counter(flag for record in manifest.records for flag in record.quality_flags)
    legacy_index = build_window_index(
        manifest,
        SegmentationConfig(
            policy=SegmentationPolicy.WHOLE_RECORDING_LEGACY, startup_guard_seconds=0.0
        ),
    )
    approved_index = build_window_index(
        manifest, SegmentationConfig(guard_seconds=0.25, startup_guard_seconds=0.5)
    )
    sensitivity = _guard_sensitivity(manifest)
    sweep = window_guard_sweep(manifest)
    onset_alignment: dict[str, Any] = {"skipped": "requires --data-root signal access"}
    if not args.no_onset_alignment:
        # The guard's job is to keep windows clear of a condition change. Measuring where
        # the change actually happens is the only way to price the guard from data rather
        # than from a round number. Uses the production filter chain.
        cache = RecordingCache(manifest, FilterChainConfig(), cache_dir=args.cache_dir)
        onset_alignment = trigger_onset_alignment(manifest, cache)

    audit: dict[str, Any] = {
        "manifest_hash": manifest.manifest_hash,
        "quality_policy_version": manifest.policy.policy_version,
        "data_root_name": data_root.name,
        "sampling_rate_hz": args.sampling_rate,
        "signal_units": SIGNAL_UNITS,
        "channel_map_tentative": EMG_MUSCLE_MAP,
        "schema": list(CANONICAL_COLUMNS),
        "totals": {
            "n_csv": len(manifest.records),
            "n_avi": int(frame["avi_relpath"].notna().sum()),
            "n_metadata": int(frame["metadata_relpath"].notna().sum()),
            "n_npy_present": int(frame["npy_relpath"].notna().sum()),
            "n_npy_claimed": int(frame["npy_claimed"].sum()),
            "n_included": len(manifest.included),
            "n_excluded": len(manifest.records) - len(manifest.included),
            "total_samples": int(frame["n_samples"].sum()),
            "total_duration_hours": float(frame["duration_seconds"].sum() / 3600),
        },
        "subjects": {
            subject: int((frame["subject_id"] == subject).sum()) for subject in manifest.subject_ids
        },
        "quality_flags": {
            flag: {"count": count, "policy": describe_flag(QualityFlag(flag))}
            for flag, count in sorted(flag_counts.items())
        },
        "conflict_resolutions": {
            rule_id: asdict(resolution) for rule_id, resolution in CONFLICT_RESOLUTIONS.items()
        },
        "excluded_recordings": [
            {"recording_id": r.recording_id, "reason": r.exclusion_reason}
            for r in manifest.records
            if r.excluded
        ],
        "short_recordings": frame.loc[
            frame["quality_flags"].str.contains("short_recording", na=False),
            ["recording_id", "n_samples", "duration_seconds"],
        ].to_dict("records"),
        "secondary_location_recordings": frame.loc[
            frame["quality_flags"].str.contains("secondary_location", na=False),
            ["recording_id", "csv_relpath"],
        ].to_dict("records"),
        "metadata_conflicts": frame.loc[
            frame["quality_flags"].str.contains("metadata_condition_conflict", na=False),
            ["recording_id", "condition_token", "metadata_condition_key", "conflict_rules_applied"],
        ].to_dict("records"),
        "npy_findings": frame.loc[
            frame["npy_relpath"].notna(),
            ["recording_id", "npy_relpath", "npy_agrees_with_csv", "npy_disagreement"],
        ].to_dict("records"),
        "trigger_run_stats": (
            runs.groupby("task_family", observed=True)["duration_seconds"]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .round(3)
            .reset_index()
            .to_dict("records")
            if not runs.empty
            else []
        ),
        "video": {
            "n_readable": int(frame["video_readable"].sum()),
            "fps_values": sorted({float(v) for v in frame["video_fps"].dropna()}),
            "resolutions": sorted(
                {
                    f"{int(w)}x{int(h)}"
                    for w, h in zip(frame["video_width"].dropna(), frame["video_height"].dropna())
                }
            ),
            "codecs": sorted({str(c) for c in frame["video_codec"].dropna()}),
        },
        "window_counts": {
            "whole_recording_legacy": {
                "total": len(legacy_index.windows),
                "by_family": legacy_index.counts_by("task_family").to_dict("records"),
                "safe_for_inference": legacy_index.safe_for_inference,
            },
            "trigger_constrained_guard_0.25s": {
                "total": len(approved_index.windows),
                "by_family": approved_index.counts_by("task_family").to_dict("records"),
                "by_subject_family": approved_index.counts_by("subject_id", "task_family").to_dict(
                    "records"
                ),
                "safe_for_inference": approved_index.safe_for_inference,
            },
        },
        "guard_sensitivity": sensitivity.to_dict("records"),
        "window_guard_sweep": sweep.to_dict("records"),
        "trigger_onset_alignment": onset_alignment,
        "mains_contamination": _mains_contamination(frame),
        "historical_confusion_matrix_check": _historical_check(legacy_index),
    }

    if args.validate_only:
        import json

        print(
            json.dumps(
                {k: audit[k] for k in ("manifest_hash", "totals", "quality_flags")},
                indent=2,
                default=str,
            )
        )
        return 0

    output_dir = Path(args.output_root) / manifest.manifest_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(output_dir / "manifest.parquet", frame)
    write_csv(output_dir / "manifest.csv", frame)
    write_csv(output_dir / "trigger_summary.csv", _trigger_summary(frame))
    write_csv(output_dir / "guard_sensitivity.csv", sensitivity)
    write_csv(output_dir / "window_guard_sweep.csv", sweep)
    write_csv(
        output_dir / "mains_contamination.csv",
        frame[
            [
                "recording_id",
                "subject_id",
                "condition",
                "mains_harmonic_power_fraction",
                "mains_harmonic_power_fraction_per_channel",
                "mains_harmonic_breakdown",
            ]
        ].sort_values("mains_harmonic_power_fraction", ascending=False),
    )
    write_csv(
        output_dir / "window_counts.csv",
        approved_index.counts_by("subject_id", "task_family", "condition"),
    )
    if not runs.empty:
        write_csv(output_dir / "trigger_runs.csv", runs)
    if not args.no_figures:
        audit["quality_figures"] = _quality_figures(manifest, runs, output_dir / "quality_figures")
    write_json(output_dir / "data_audit.json", audit)
    (output_dir / "data_audit.md").write_text(
        render_audit_markdown(audit, manifest), encoding="utf-8"
    )

    _assert_no_identifiers(output_dir)
    logger.info("audit written to %s", output_dir)
    print(f"manifest hash : {manifest.manifest_hash}")
    print(f"recordings    : {len(manifest.records)} ({len(manifest.included)} included)")
    print(f"audit bundle  : {output_dir}")
    return 0


def _historical_check(legacy_index: Any) -> dict[str, Any]:
    """Compare the reproducible legacy window counts with the original paper's matrix."""
    counts = {
        row["task_family"]: int(row["n_windows"])
        for row in legacy_index.counts_by("task_family").to_dict("records")
    }
    published = {"movement": 1877, "clench": 2503, "instructed_grinding": 1861, "chewing": 5604}
    reproduced_total = sum(counts.values())
    return {
        "reproduced_whole_recording_counts": counts,
        "reproduced_total_all_five_families": reproduced_total,
        "published_four_class_matrix_supports": published,
        "published_matrix_total": sum(published.values()),
        "totals_match": reproduced_total == sum(published.values()),
        "per_class_supports_match": {
            family: counts.get(family) == value for family, value in published.items()
        },
        "finding": (
            "The published four-class confusion matrix totals 11,845 windows, which equals "
            "the whole-recording count across ALL FIVE families including the 595 dedicated "
            "rest windows -- even though the published matrix has no rest class. The "
            "per-class supports do not match the reproducible whole-recording counts for "
            "any of the four classes. The published matrix is therefore not reproducible "
            "from these recordings under whole-recording labelling, and must not be reused."
        ),
    }


def _assert_no_identifiers(output_dir: Path) -> None:
    """Fail if a generated artifact leaks a private path or administrative filename."""
    forbidden = ("Survey", "Receipt", "Reconciliation", "IMG_", ".HEIC", "/home/", "C:\\")
    offenders: list[str] = []
    for path in output_dir.rglob("*"):
        if path.suffix.lower() not in {".json", ".csv", ".md", ".txt", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.name}: contains {token!r}")
    if offenders:
        raise AssertionError(
            "generated audit artifacts reference private or machine-specific material:\n  "
            + "\n  ".join(offenders)
        )


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
