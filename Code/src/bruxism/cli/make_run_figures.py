"""``bruxism-figures`` -- (re)build the figure set of a finished run, without retraining.

A training run writes these figures itself when it finishes. This command exists for the
cases where that is not enough: a run that finished before the figures existed, a figure
whose code has since been fixed, or a run trained on a machine that had no access to the
raw data root.

    bruxism-figures --run-dir outputs/runs/<run_id> --data-root ../Data

Everything is regenerated from the run bundle -- the prediction ledger, the metrics summary,
the fold outcomes and the saved checkpoints -- plus the raw recordings for the signal-side
figures. No model is retrained and no number is re-derived by hand, so a regenerated figure
always matches the run it sits in.
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
from bruxism.utils.logging import get_logger
from bruxism.visualization.run_figures import generate_run_figures

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-figures", __doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="Run bundle directory, e.g. outputs/runs/<run_id>. Repeatable.",
    )
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=150,
        help="Deterministic cap on windows sampled for the spectral and wavelet figures.",
    )
    parser.add_argument(
        "--tsne-max-samples",
        type=int,
        default=3000,
        help="Deterministic cap on held-out embeddings entering the t-SNE projection.",
    )
    parser.add_argument("--tsne-seed", type=int, default=0, help="Fixed seed for the t-SNE.")
    parser.add_argument(
        "--no-signal-figures",
        action="store_true",
        help="Skip everything that reads raw recordings (inventory, filtering, wavelets).",
    )
    parser.add_argument("--no-tsne", action="store_true", help="Skip the embedding projection.")
    return parser


def main_impl(args: argparse.Namespace) -> int:
    failures = 0
    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(f"run directory not found: {run_path}")
        config_path = run_path / "resolved_config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{run_path} is not a run bundle: no resolved_config.yaml. Point --run-dir at "
                f"outputs/runs/<run_id>, not at outputs/runs."
            )
        if not (run_path / "predictions.parquet").is_file():
            raise FileNotFoundError(
                f"{run_path} has no predictions.parquet, so it never finished a fold. "
                f"Finish or rerun the training first: bruxism-train --config "
                f"{run_path / 'resolved_config.yaml'} --resume"
            )
        config = load_experiment_config(config_path)
        # A run bundle records the data root of the machine that trained it, which may not
        # exist here. That is not an error for this command: the ledger-derived figures are
        # still produced and the rest are skipped with a recorded reason.
        stored = Path(config.data.data_root).expanduser()
        data_root = resolve_data_root(
            args.data_root,
            config_value=config.data.data_root if stored.is_dir() else None,
            required=False,
        )
        if data_root is None:
            logger.warning(
                "no usable data root (--data-root not given and %s is not a directory); "
                "the figures that read raw recordings will be skipped",
                stored,
            )

        summary = generate_run_figures(
            run_path,
            config=config,
            data_root=data_root,
            max_windows_per_class=args.max_windows_per_class,
            tsne_max_samples=args.tsne_max_samples,
            tsne_seed=args.tsne_seed,
            include_signal_figures=not args.no_signal_figures,
            include_tsne=not args.no_tsne,
        )
        failures += int(summary["failed"])
        print(f"\nrun          : {run_path}")
        print(f"figures      : {summary['figures_dir']}")
        print(
            f"produced     : {summary['written']} written, "
            f"{summary['skipped']} skipped, {summary['failed']} failed"
        )
        for record in summary["index"]["figures"]:
            if record["status"] != "written":
                print(f"  - {record['stem']}: {record['status']} -- {record['reason']}")
        print(f"index        : {summary['figures_dir']}/figure_index.json")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
