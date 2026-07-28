"""Shared argument parsing and startup for every ``bruxism-*`` command.

All commands are non-interactive, support ``--help``, take the data root from the command
line or an environment variable (never a hard-coded path), and fail with an actionable
message rather than a traceback for user-facing errors.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from bruxism import __version__
from bruxism.utils.logging import setup_logging

__all__ = [
    "DATA_ROOT_ENV",
    "add_common_arguments",
    "add_data_root_argument",
    "make_parser",
    "resolve_data_root",
    "run_cli",
]

#: Environment variable consulted when ``--data-root`` is omitted.
DATA_ROOT_ENV = "BRUXISM_DATA_ROOT"


def make_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bruxism {__version__}")
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console verbosity. The JSON-lines run log always records at DEBUG.",
    )
    parser.add_argument("--quiet", action="store_true", help="Equivalent to --log-level WARNING.")
    return parser


def add_data_root_argument(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        required=required,
        help=(
            "Path to the authorized raw data directory containing Subject_<n>/ folders. "
            f"Falls back to ${DATA_ROOT_ENV}, then to the value in the config file. "
            "Raw data is never copied into this repository."
        ),
    )


def resolve_data_root(
    explicit: Path | None, *, config_value: str | None = None, required: bool = True
) -> Path | None:
    """Resolve the data root from CLI flag, environment, then config, in that order."""
    for candidate in (explicit, os.environ.get(DATA_ROOT_ENV), config_value):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_dir():
                raise SystemExit(
                    f"error: data root {path} does not exist or is not a directory.\n"
                    f"       Pass --data-root /path/to/Data or set {DATA_ROOT_ENV}."
                )
            return path.resolve()
    if required:
        raise SystemExit(
            "error: no data root supplied.\n"
            f"       Pass --data-root /path/to/Data or set {DATA_ROOT_ENV}."
        )
    return None


def run_cli(
    main_fn: Callable[[argparse.Namespace], int],
    args: argparse.Namespace,
) -> int:
    """Run a command body, turning expected failures into actionable messages."""
    level = logging.WARNING if getattr(args, "quiet", False) else getattr(args, "log_level", "INFO")
    setup_logging(level)
    try:
        return main_fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, AssertionError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def parse_and_run(
    build_parser: Callable[[], argparse.ArgumentParser],
    main_fn: Callable[[argparse.Namespace], int],
    argv: Sequence[str] | None = None,
) -> int:
    return run_cli(main_fn, build_parser().parse_args(argv))
