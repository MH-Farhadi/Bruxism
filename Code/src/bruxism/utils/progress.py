"""Live progress for runs that take hours.

Design constraints this module exists to satisfy:

* **A long run must be observable.** One outer fold of the primary experiment trains
  ``n_trials x n_inner_folds`` models before its refit; with only a per-fold log line the
  command looks hung for twenty minutes at a time.
* **Display only.** Nothing here is ever read back into a result. Progress never enters
  the configuration hash, the run bundle or a recorded metric, so ``--progress none`` and
  ``--progress bar`` produce byte-identical artifacts.
* **A detached run still has a pulse.** Bars need a terminal. When stderr is redirected
  (``nohup``, ``tee``, CI) the same :class:`ProgressTask` emits a throttled ``INFO`` log
  line instead of drawing, so ``tail -f`` still shows movement.
* **Progress never corrupts the log.** While a bar is live, console log records go through
  ``tqdm.write``, which clears and redraws every bar around the message.
* **tqdm is optional.** It is a declared dependency, but a stripped environment degrades to
  log lines rather than failing at import time.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sys
import time
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Any, Literal, TextIO, cast

from bruxism.utils.logging import get_logger

__all__ = [
    "DEFAULT_LOG_INTERVAL_SECONDS",
    "PROGRESS_MODES",
    "ProgressMode",
    "ProgressTask",
    "bars_enabled",
    "configure",
    "console_stream",
    "format_duration",
    "task",
    "track",
]

logger = get_logger(__name__)

_tqdm: Any
try:  # pragma: no cover - the except branch needs tqdm uninstalled
    from tqdm.auto import tqdm as _imported_tqdm

    _tqdm = _imported_tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

ProgressMode = Literal["auto", "bar", "plain", "none"]

#: Accepted ``--progress`` values, in the order they are offered on the command line.
PROGRESS_MODES: tuple[ProgressMode, ...] = ("auto", "bar", "plain", "none")

#: Seconds between progress log lines when bars are not drawn.
DEFAULT_LOG_INTERVAL_SECONDS = 30.0


class _State:
    """Process-wide progress settings, set once from the command line."""

    mode: ProgressMode = "auto"
    interval_seconds: float = DEFAULT_LOG_INTERVAL_SECONDS
    #: Bars currently drawn. Doubles as the tqdm ``position`` of the next one, so nested
    #: tasks stack downward as long as they are closed in reverse order of creation.
    live: int = 0


_state = _State()


def configure(
    mode: ProgressMode | str = "auto",
    *,
    interval_seconds: float = DEFAULT_LOG_INTERVAL_SECONDS,
) -> None:
    """Set the process-wide progress mode.

    Parameters
    ----------
    mode
        ``"auto"`` draws bars on a terminal and logs periodic lines otherwise; ``"bar"``
        and ``"plain"`` force one backend; ``"none"`` disables progress entirely.
    interval_seconds
        Minimum wall-clock gap between progress log lines in the non-drawing backends.
    """
    if mode not in PROGRESS_MODES:
        raise ValueError(f"unknown progress mode {mode!r}; expected one of {list(PROGRESS_MODES)}")
    _state.mode = mode
    _state.interval_seconds = max(float(interval_seconds), 0.0)


def bars_enabled() -> bool:
    """Whether :class:`ProgressTask` draws a bar rather than logging lines."""
    if _tqdm is None or _state.mode in ("plain", "none"):
        return False
    if _state.mode == "bar":
        return True
    try:  # a redirected, detached or closed stderr is not a terminal
        return bool(sys.stderr is not None and sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


class _TqdmSafeStream:
    """A stderr proxy that routes writes through ``tqdm.write`` while a bar is live.

    ``sys.stderr`` is looked up per call rather than captured, so the handler installed at
    startup still follows a stream that pytest (or the caller) replaces later.
    """

    def write(self, message: str) -> int:
        if _state.live and _tqdm is not None:
            _tqdm.write(message, file=sys.stderr, end="")
            return len(message)
        return sys.stderr.write(message)

    def flush(self) -> None:
        with contextlib.suppress(ValueError):  # stream closed during interpreter shutdown
            sys.stderr.flush()

    def isatty(self) -> bool:
        return bool(getattr(sys.stderr, "isatty", lambda: False)())


_CONSOLE_STREAM = _TqdmSafeStream()


def console_stream() -> TextIO:
    """The stream the console log handler should write to."""
    return cast("TextIO", _CONSOLE_STREAM)


def format_duration(seconds: float | None) -> str:
    """Human-readable duration: ``45s``, ``3m12s``, ``1h04m``. ``"?"`` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_field(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class ProgressTask:
    """One unit of trackable work: a bar on a terminal, throttled log lines otherwise.

    The two backends share one interface so call sites never branch on the display mode::

        with task("epochs", total=60) as epochs:
            for epoch in range(1, 61):
                ...
                epochs.update(1, loss=train_loss, macro_f1=value)

    Tasks nest. Each live bar takes the next tqdm ``position``, so close them in reverse
    order of creation (a ``with`` block does this automatically).

    ``bar_description`` exists because the two backends want different text. A drawn bar
    sits directly under its parent bar and can be short (``"  epochs"``); a log line stands
    alone and must carry its whole context (``"fold 3/15 S03 seed1 trial01 ... epochs"``).
    """

    def __init__(
        self,
        description: str,
        total: int | None = None,
        *,
        unit: str = "it",
        leave: bool = False,
        level: int = logging.INFO,
        bar_description: str | None = None,
    ):
        self.description = description
        self.bar_description = bar_description if bar_description is not None else description
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.level = level
        self.count = 0
        self._fields: dict[str, Any] = {}
        self._started = time.monotonic()
        self._last_log = self._started
        self._closed = False
        self._bar: Any = None
        if bars_enabled():
            self._bar = _tqdm(
                total=self.total,
                desc=self.bar_description,
                unit=unit,
                leave=leave,
                position=_state.live,
                dynamic_ncols=True,
                file=sys.stderr,
                mininterval=0.25,
                smoothing=0.15,
            )
            _state.live += 1

    # ------------------------------------------------------------------ reporting ---

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def eta_seconds(self) -> float | None:
        """Remaining seconds extrapolated from the mean rate so far, if knowable."""
        if not self.total or self.count <= 0 or self.count >= self.total:
            return None
        return self.elapsed / self.count * (self.total - self.count)

    def render(self) -> str:
        """The one-line plain-text form used by the logging backend."""
        parts = [self.description]
        if self.total:
            share = 100.0 * self.count / self.total
            parts.append(f"{self.count}/{self.total} {self.unit} ({share:.0f}%)")
        else:
            parts.append(f"{self.count} {self.unit}")
        parts.append(f"{format_duration(self.elapsed)} elapsed")
        eta = self.eta_seconds()
        if eta is not None:
            parts.append(f"~{format_duration(eta)} left")
        if self._fields:
            parts.append(" ".join(f"{k}={_format_field(v)}" for k, v in self._fields.items()))
        return " | ".join(parts)

    # --------------------------------------------------------------------- driving ---

    def update(self, n: int = 1, **fields: Any) -> None:
        """Advance by ``n`` units and attach ``fields`` (loss, metrics) to the display."""
        self.count += int(n)
        if fields:
            self._fields.update({k: v for k, v in fields.items() if v is not None})
        if self._bar is not None:
            if self._fields:
                self._bar.set_postfix(self._fields, refresh=False)
            self._bar.update(int(n))
            return
        if _state.mode == "none":
            return
        now = time.monotonic()
        if now - self._last_log >= _state.interval_seconds:
            self._last_log = now
            logger.log(self.level, "%s", self.render())

    def set_description(self, description: str, *, bar_description: str | None = None) -> None:
        """Relabel the task; the next :meth:`update` redraws it."""
        self.description = description
        self.bar_description = bar_description if bar_description is not None else description
        if self._bar is not None:
            self._bar.set_description_str(self.bar_description, refresh=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            self._bar.close()
            _state.live = max(_state.live - 1, 0)

    def __enter__(self) -> ProgressTask:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def task(
    description: str,
    total: int | None = None,
    *,
    unit: str = "it",
    leave: bool = False,
    level: int = logging.INFO,
    bar_description: str | None = None,
) -> ProgressTask:
    """Create a :class:`ProgressTask`. Use as a context manager."""
    return ProgressTask(
        description,
        total,
        unit=unit,
        leave=leave,
        level=level,
        bar_description=bar_description,
    )


def track(
    iterable: Iterable[Any],
    description: str,
    *,
    total: int | None = None,
    unit: str = "it",
    leave: bool = False,
    level: int = logging.INFO,
) -> Iterator[Any]:
    """Yield from ``iterable``, reporting progress after each item."""
    if total is None:
        with contextlib.suppress(TypeError):
            total = len(cast("Any", iterable))
    with ProgressTask(description, total, unit=unit, leave=leave, level=level) as reporter:
        for item in iterable:
            yield item
            reporter.update(1)
