"""``bruxism-train`` -- run a leakage-free nested leave-one-subject-out experiment.

The same entry point serves the smoke run, the primary five-class experiment and any
single-condition run; what differs is only the configuration file.

    bruxism-train --config configs/experiments/smoke.yaml
    bruxism-train --config configs/experiments/five_class_nested_loso.yaml

Runs are resumable: completed folds are reused when the configuration, data manifest and
window index all hash identically, and refused otherwise.
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
from bruxism.runner import run_experiment
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-train", __doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML.")
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. --set training.batch_size=32",
    )
    parser.add_argument("--run-id", default=None, help="Override the generated run id.")
    parser.add_argument("--runs-root", type=Path, default=None, help="Override output.runs_root.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of reusing completed folds.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build the data, write the fold plan and stop before training.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help=(
            "Stop after this many (condition, seed, fold) triples. The resulting run is "
            "INCOMPLETE and is labelled as such."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help=(
            "Skip the figure set written into <run_dir>/figures when the run finishes. "
            "Figures are display artifacts only: they change no recorded number and are "
            "not part of the configuration hash."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override training.seeds.",
    )
    return parser


def main_impl(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config, overrides=args.overrides)
    if args.run_id or args.runs_root:
        from dataclasses import replace

        config = replace(
            config,
            output=replace(
                config.output,
                run_id=args.run_id or config.output.run_id,
                runs_root=str(args.runs_root) if args.runs_root else config.output.runs_root,
            ),
        )
    if args.seeds:
        from dataclasses import replace

        config = replace(config, training=config.training.replace(seeds=tuple(args.seeds)))

    data_root = resolve_data_root(args.data_root, config_value=config.data.data_root)
    bundle = run_experiment(
        config,
        data_root=data_root,
        resume=not args.no_resume,
        dry_run=args.validate_only,
        max_folds=args.max_folds,
        figures=not args.no_figures,
    )
    print(f"run id        : {bundle.run_id}")
    print(f"run directory : {bundle.run_dir}")
    print(f"config hash   : {bundle.config_hash}")
    print(f"manifest hash : {bundle.manifest_hash}")
    if not args.validate_only:
        print(f"predictions   : {bundle.predictions_path}")
        print(f"metrics       : {bundle.metrics_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
