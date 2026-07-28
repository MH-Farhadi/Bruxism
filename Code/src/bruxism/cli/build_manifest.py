"""``bruxism-build-manifest`` -- build the versioned analysis manifest and window index.

Unlike ``bruxism-audit`` (which characterises the raw data), this command applies an
approved segmentation configuration and writes the exact example set an experiment will
consume, together with its hash. Training refuses to resume across a different hash.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bruxism.cli._common import (
    add_common_arguments,
    add_data_root_argument,
    make_parser,
    parse_and_run,
    resolve_data_root,
)
from bruxism.config import load_experiment_config
from bruxism.data.manifest import build_manifest
from bruxism.data.segments import build_window_index
from bruxism.data.splits import NestedLOSOSplitter
from bruxism.utils.io import write_csv, write_json, write_parquet
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-build-manifest", __doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment or data YAML declaring the approved segmentation policy.",
    )
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/manifests"),
        help="Directory that will receive a <window_index_hash> subdirectory.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. --set data.guard_seconds=0.5",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Report the counts without writing anything.",
    )
    return parser


def main_impl(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config, overrides=args.overrides)
    data_root = resolve_data_root(args.data_root, config_value=config.data.data_root)
    assert data_root is not None

    manifest = build_manifest(
        data_root,
        sampling_rate_hz=config.data.sampling_rate_hz,
        policy=config.exclusion_policy(),
        probe_video=False,
    )
    segmentation = config.data.segmentation()
    index = build_window_index(manifest, segmentation)

    print(f"manifest hash      : {manifest.manifest_hash}")
    print(f"window index hash  : {index.index_hash}")
    print(f"segmentation policy: {segmentation.policy}")
    print(f"safe for inference : {index.safe_for_inference}")
    print(f"windows            : {len(index.windows):,}")
    print()
    print(
        index.counts_by("subject_id", "task_family")
        .pivot(index="subject_id", columns="task_family", values="n_windows")
        .fillna(0)
        .astype(int)
        .to_string()
    )
    if not index.safe_for_inference:
        print(
            "\nWARNING: this segmentation policy is DIAGNOSTIC ONLY. Results produced from "
            "it must not be reported as scientific findings."
        )

    splitter = NestedLOSOSplitter(index, subjects=config.data.subjects)
    print(
        f"\nnested LOSO: {splitter.n_outer_folds} outer folds, "
        f"{splitter.n_inner_folds()} participant-grouped inner folds each"
    )

    if args.validate_only:
        return 0

    output_dir = Path(args.output_root) / index.index_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(output_dir / "windows.parquet", index.frame)
    write_csv(output_dir / "window_counts.csv", index.counts_by("subject_id", "task_family"))
    write_parquet(output_dir / "recordings.parquet", manifest.frame)
    write_json(
        output_dir / "index.json",
        {
            "manifest_hash": manifest.manifest_hash,
            "window_index_hash": index.index_hash,
            "segmentation": segmentation.to_dict(),
            "quality_policy_version": manifest.policy.policy_version,
            "n_windows": len(index.windows),
            "n_recordings_included": len(manifest.included),
            "subjects": index.subject_ids,
            "safe_for_inference": index.safe_for_inference,
            "dropped": index.dropped,
            "config_name": config.name,
            "config_hash": config.config_hash,
        },
    )
    write_json(output_dir / "folds.json", splitter.to_dict())
    print(f"\nwritten to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
