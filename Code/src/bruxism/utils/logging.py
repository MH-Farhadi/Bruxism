"""Structured logging shared by the library and the command-line entry points.

Console output stays human readable; when a run directory is supplied every record is
additionally appended as one JSON object per line so a completed run carries its own log
in a machine-parsable form.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["JsonLinesHandler", "get_logger", "log_context", "setup_logging"]

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Console verbosity chosen by the command line, remembered across re-configuration.
#: ``setup_logging`` is called a second time once a run directory exists, and that call
#: must not silently discard ``--log-level DEBUG`` or ``--quiet``.
_console_level: int | str = logging.INFO


class JsonLinesHandler(logging.FileHandler):
    """Emit one JSON object per record, including any ``extra`` fields."""

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatter.formatTime(record, _DATE_FORMAT) if self.formatter else None,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # formatException lives on Formatter, not Handler. Falling back to the module
            # default keeps exception logging working even without an attached formatter.
            formatter = self.formatter or logging.Formatter()
            payload["exception"] = formatter.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def setup_logging(
    level: int | str | None = None,
    *,
    log_dir: Path | str | None = None,
    filename: str = "run.log.jsonl",
) -> logging.Logger:
    """Configure the ``bruxism`` logger tree.

    Parameters
    ----------
    level
        Console verbosity. ``None`` keeps whatever the last explicit call selected, so
        attaching the run-directory log later does not reset the requested verbosity.
        The JSON-lines file always records at DEBUG.
    log_dir
        When given, a ``filename`` JSON-lines log is written inside it.
    """
    # Imported here rather than at module scope: progress reporting needs `get_logger`.
    from bruxism.utils.progress import console_stream

    global _console_level
    if level is not None:
        _console_level = level

    root = logging.getLogger("bruxism")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    # Writes go through a proxy that defers to `tqdm.write` while a progress bar is live,
    # so a log record cannot smear a redrawing bar.
    console = logging.StreamHandler(stream=console_stream())
    console.setLevel(_console_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    if log_dir is not None:
        target = Path(log_dir)
        target.mkdir(parents=True, exist_ok=True)
        jsonl = JsonLinesHandler(target / filename, encoding="utf-8")
        jsonl.setLevel(logging.DEBUG)
        jsonl.setFormatter(logging.Formatter(datefmt=_DATE_FORMAT))
        root.addHandler(jsonl)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``bruxism`` logger (``name`` may be a dotted module path)."""
    suffix = name.removeprefix("bruxism.").removeprefix("bruxism")
    return logging.getLogger(f"bruxism.{suffix}" if suffix else "bruxism")


def log_context(**fields: Any) -> dict[str, Any]:
    """Package ``fields`` for the ``extra=`` argument of a logging call."""
    return {"extra": fields}
