"""``bruxism-ablations`` -- matched modality and no-chewing ablations.

Thin wrapper over the same runner as ``bruxism-train``. The only difference is that the
configuration declares ``sweep_task_ids`` and ``sweep_modalities``, so the runner produces
every (task x modality) condition with identical windows, folds, seeds and selection
budget. That matching is what makes the fusion-minus-EMG-only difference interpretable.
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
from bruxism.models.ablations import MODALITY_CONDITIONS
from bruxism.runner import run_experiment
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-ablations", __doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Ablation experiment YAML.")
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        choices=list(MODALITY_CONDITIONS),
        help="Override sweep_modalities.",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Override sweep_task_ids.")
    parser.add_argument("--run-id", default=None, help="Override the generated run id.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore completed folds.")
    parser.add_argument("--max-folds", type=int, default=None, help="Cap executed folds.")
    parser.add_argument("--validate-only", action="store_true", help="Write the plan and stop.")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help=(
            "Skip the figure set written into <run_dir>/figures when the run finishes. "
            "Figures are display artifacts only: they change no recorded number and are "
            "not part of the configuration hash."
        ),
    )
    return parser


def main_impl(args: argparse.Namespace) -> int:
    from dataclasses import replace

    config = load_experiment_config(args.config, overrides=args.overrides)
    config = replace(
        config,
        sweep_modalities=(tuple(args.modalities) if args.modalities else config.sweep_modalities),
        sweep_task_ids=tuple(args.tasks) if args.tasks else config.sweep_task_ids,
        output=(replace(config.output, run_id=args.run_id) if args.run_id else config.output),
    )

    if not config.sweep_modalities and not config.sweep_task_ids:
        raise ValueError(
            "an ablation config must declare sweep_modalities and/or sweep_task_ids; "
            "otherwise use bruxism-train"
        )

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
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
