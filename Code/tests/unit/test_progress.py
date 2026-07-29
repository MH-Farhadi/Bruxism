"""Progress reporting: both backends, the nesting invariant and the logging interop.

Progress is display only, so the property that matters most is that it can never change
what a run computes. These tests pin the observable behaviour of the two backends and the
bookkeeping that keeps bars and log records from corrupting each other.
"""

from __future__ import annotations

import logging

import pytest

from bruxism.utils import progress
from bruxism.utils.logging import setup_logging


@pytest.fixture(autouse=True)
def _restore_progress_state():
    """Progress mode is process-wide; put it back the way the test found it."""
    previous = (progress._state.mode, progress._state.interval_seconds, progress._state.live)
    yield
    progress._state.mode, progress._state.interval_seconds, progress._state.live = previous


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (9.4, "9s"),
        (59.6, "1m00s"),
        (75, "1m15s"),
        (3600, "1h00m"),
        (13_140, "3h39m"),
        (None, "?"),
        (float("inf"), "?"),
        (-1, "?"),
    ],
)
def test_format_duration_is_readable_at_every_scale(seconds, expected):
    assert progress.format_duration(seconds) == expected


def test_configure_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown progress mode"):
        progress.configure("verbose")


@pytest.mark.parametrize("mode", ["plain", "none"])
def test_bars_are_disabled_in_the_text_modes(mode):
    progress.configure(mode)
    assert progress.bars_enabled() is False


def test_none_mode_emits_nothing(caplog):
    progress.configure("none", interval_seconds=0.0)
    with caplog.at_level(logging.DEBUG, logger="bruxism"), progress.task("fit", total=3) as task:
        for _ in range(3):
            task.update(1, loss=0.5)
    assert caplog.records == []


def test_plain_mode_logs_counts_progress_and_fields(caplog):
    progress.configure("plain", interval_seconds=0.0)
    with (
        caplog.at_level(logging.INFO, logger="bruxism"),
        progress.task("fold 2/15 epochs", total=4, unit="epoch") as task,
    ):
        for _ in range(4):
            task.update(1, loss=0.25)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 4
    assert "fold 2/15 epochs" in messages[0]
    assert "1/4 epoch (25%)" in messages[0]
    assert "4/4 epoch (100%)" in messages[-1]
    assert "loss=0.25" in messages[-1]
    # An estimate is offered while work remains and withheld once it is done.
    assert "left" in messages[0]
    assert "left" not in messages[-1]


def test_plain_mode_throttles_by_wall_clock(caplog):
    progress.configure("plain", interval_seconds=3600.0)
    with caplog.at_level(logging.INFO, logger="bruxism"), progress.task("fit", total=100) as task:
        for _ in range(100):
            task.update(1)
    assert caplog.records == []


def test_log_lines_use_the_full_description_and_bars_the_short_one(caplog):
    progress.configure("plain", interval_seconds=0.0)
    with (
        caplog.at_level(logging.INFO, logger="bruxism"),
        progress.task(
            "fold 2/15 S03 seed1 trial00 epochs", total=1, bar_description="    epochs"
        ) as task,
    ):
        task.update(1)
    assert "fold 2/15 S03 seed1 trial00 epochs" in caplog.records[0].getMessage()


def test_nested_tasks_release_their_bar_positions():
    progress.configure("bar")
    if not progress.bars_enabled():  # pragma: no cover - only when tqdm is absent
        pytest.skip("tqdm is not installed")
    assert progress._state.live == 0
    with progress.task("outer", total=2) as outer:
        assert progress._state.live == 1
        with progress.task("inner", total=2) as inner:
            assert progress._state.live == 2
            inner.update(1)
        assert progress._state.live == 1
        outer.update(1)
    assert progress._state.live == 0


def test_closing_twice_does_not_corrupt_the_position_counter():
    progress.configure("bar")
    if not progress.bars_enabled():  # pragma: no cover - only when tqdm is absent
        pytest.skip("tqdm is not installed")
    task = progress.task("once", total=1)
    task.close()
    task.close()
    assert progress._state.live == 0


def test_track_yields_every_item_and_reports_each(caplog):
    progress.configure("plain", interval_seconds=0.0)
    with caplog.at_level(logging.INFO, logger="bruxism"):
        collected = list(progress.track(iter("abcd"), "reading", total=4, unit="file"))
    assert collected == ["a", "b", "c", "d"]
    assert "4/4 file (100%)" in caplog.records[-1].getMessage()


def test_console_stream_follows_a_replaced_stderr(capsys):
    progress.configure("none")
    progress.console_stream().write("hello\n")
    assert capsys.readouterr().err == "hello\n"


def test_attaching_the_run_log_keeps_the_requested_console_level(tmp_path):
    """Regression: the second setup_logging call must not undo ``--log-level``."""
    setup_logging("DEBUG")
    setup_logging(log_dir=tmp_path)  # what runner.run_experiment does once a run dir exists
    console = next(
        handler
        for handler in logging.getLogger("bruxism").handlers
        if not isinstance(handler, logging.FileHandler)
    )
    assert console.level == logging.DEBUG
    setup_logging(logging.INFO)  # restore the suite's default verbosity
