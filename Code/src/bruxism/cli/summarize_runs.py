"""``bruxism-summarize`` -- recompute metrics from saved prediction ledgers.

Reads one or more run bundles, concatenates their ledgers, verifies that every held-out
example appears exactly once per configuration, and recomputes every metric from scratch.
This is the check that the numbers in ``metrics.json`` really do follow from
``predictions.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bruxism.cli._common import add_common_arguments, make_parser, parse_and_run
from bruxism.evaluation.aggregation import condition_table, summarise_ledger
from bruxism.utils.io import write_csv, write_json
from bruxism.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-summarize", __doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument(
        "--runs-root", type=Path, default=Path("outputs/runs"), help="Directory of run bundles."
    )
    parser.add_argument(
        "--run-id", action="append", default=[], help="Restrict to these run ids (repeatable)."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Where to write the summary (default: stdout)."
    )
    return parser


def collect_predictions(runs_root: Path, run_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate the prediction ledgers of the selected run bundles."""
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root {runs_root} does not exist")
    frames: list[pd.DataFrame] = []
    found: list[str] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_ids and run_dir.name not in run_ids:
            continue
        ledger = run_dir / "predictions.parquet"
        if not ledger.is_file():
            logger.warning("run %s has no predictions.parquet; skipping", run_dir.name)
            continue
        frame = pd.read_parquet(ledger)
        frame["run_id"] = run_dir.name
        frames.append(frame)
        found.append(run_dir.name)
    if not frames:
        raise FileNotFoundError(
            f"no prediction ledgers found under {runs_root}"
            + (f" for run ids {run_ids}" if run_ids else "")
        )
    return pd.concat(frames, ignore_index=True), found


def main_impl(args: argparse.Namespace) -> int:
    predictions, run_ids = collect_predictions(args.runs_root, args.run_id)
    logger.info("loaded %d predictions from %d run(s)", len(predictions), len(run_ids))

    metrics = summarise_ledger(predictions)
    table = condition_table(metrics)

    print(f"runs        : {', '.join(run_ids)}")
    print(f"predictions : {len(predictions):,}")
    print(f"conditions  : {len(metrics['conditions'])}")
    print()
    if not table.empty:
        columns = [
            c
            for c in (
                "task_id",
                "model_id",
                "modality",
                "seed",
                "accuracy_mean",
                "macro_f1_mean",
                "macro_f1_std",
                "macro_roc_auc_mean",
                "n_windows_descriptive",
            )
            if c in table.columns
        ]
        print(table[columns].to_string(index=False))

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "metrics.json", metrics)
        write_csv(args.output / "condition_table.csv", table)
        print(f"\nwritten to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
