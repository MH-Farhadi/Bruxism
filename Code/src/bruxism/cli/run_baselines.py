"""``bruxism-baselines`` -- architecture comparison with matched inputs.

Every model in the comparison receives the identical ``(emg, mic)`` pair, the identical
windows and the identical folds and seeds, so the table measures architecture rather than
"who got more modalities". If a model cannot accept equivalent inputs, that must be stated
in the manuscript as a modality comparison, not an architecture comparison.
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
from bruxism.models.baselines import NEURAL_MODEL_IDS
from bruxism.runner import run_experiment
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-baselines", __doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Baseline experiment YAML.")
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
        "--models",
        nargs="+",
        default=None,
        choices=list(NEURAL_MODEL_IDS),
        help="Override sweep_model_ids.",
    )
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
        sweep_model_ids=tuple(args.models) if args.models else config.sweep_model_ids,
        output=(replace(config.output, run_id=args.run_id) if args.run_id else config.output),
    )

    models = config.sweep_model_ids or (config.model_id,)
    modalities = config.sweep_modalities or (config.modality,)
    if len(set(modalities)) > 1:
        logger.warning(
            "this baseline sweep varies modality as well as architecture (%s); the "
            "resulting table is NOT a clean architecture comparison",
            sorted(set(modalities)),
        )
    logger.info("comparing %s, all receiving modality=%s", list(models), modalities[0])

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
