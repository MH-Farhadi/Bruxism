"""``bruxism-plot-preprocessing`` -- raw versus production-filtered example traces.

Plots are produced by calling the **exact production filter chain**
(:func:`bruxism.preprocessing.filters.apply_filter_chain`), so the figure cannot drift from
what training actually sees. The chain is applied to the continuous recording before any
window is cut, which is also how the pipeline uses it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bruxism.cli._common import (  # noqa: E402
    add_common_arguments,
    add_data_root_argument,
    make_parser,
    parse_and_run,
    resolve_data_root,
)
from bruxism.data.manifest import build_manifest  # noqa: E402
from bruxism.data.schema import (  # noqa: E402
    EMG_COLUMNS,
    MIC_COLUMN,
    SIGNAL_UNITS,
    read_recording_csv,
)
from bruxism.preprocessing.filters import FilterChainConfig, apply_filter_chain  # noqa: E402
from bruxism.utils.io import write_json  # noqa: E402
from bruxism.utils.logging import get_logger  # noqa: E402
from bruxism.visualization.paper_figures import FigureStyle, save_figure  # noqa: E402

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("bruxism-plot-preprocessing", __doc__.splitlines()[0])
    add_data_root_argument(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help="Recording ids to plot (repeatable). Default: one per task family.",
    )
    parser.add_argument("--start-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocessing_examples"))
    return parser


def main_impl(args: argparse.Namespace) -> int:
    data_root = resolve_data_root(args.data_root)
    assert data_root is not None
    manifest = build_manifest(data_root, probe_video=False)
    chain = FilterChainConfig()

    if args.recording_id:
        selected = [manifest.by_id(rid) for rid in args.recording_id]
    else:
        seen: set[str] = set()
        selected = []
        for record in sorted(manifest.included, key=lambda r: r.recording_id):
            if record.task_family not in seen:
                seen.add(record.task_family)
                selected.append(record)

    FigureStyle.apply()
    args.output.mkdir(parents=True, exist_ok=True)
    rate = manifest.sampling_rate_hz
    start = int(args.start_seconds * rate)
    stop = start + int(args.duration_seconds * rate)

    for record in selected:
        frame = read_recording_csv(manifest.csv_path(record))
        raw_emg = frame[list(EMG_COLUMNS)].to_numpy(dtype=np.float64)
        raw_mic = frame[MIC_COLUMN].to_numpy(dtype=np.float64)
        # Filter the CONTINUOUS recording, exactly as the pipeline does.
        filtered_emg = apply_filter_chain(raw_emg, chain, rate, modality="emg")
        filtered_mic = apply_filter_chain(raw_mic, chain, rate, modality="mic")

        stop_clipped = min(stop, raw_emg.shape[0])
        t = np.arange(start, stop_clipped) / rate
        n_rows = len(EMG_COLUMNS) + 1
        fig, axes = plt.subplots(n_rows, 2, figsize=(11.0, 1.6 * n_rows), sharex=True)
        for row, column in enumerate(EMG_COLUMNS):
            axes[row, 0].plot(
                t, raw_emg[start:stop_clipped, row], linewidth=0.6, color=FigureStyle.color(row)
            )
            axes[row, 1].plot(
                t,
                filtered_emg[start:stop_clipped, row],
                linewidth=0.6,
                color=FigureStyle.color(row),
            )
            axes[row, 0].set_ylabel(column, fontsize=7)
        axes[-1, 0].plot(t, raw_mic[start:stop_clipped], linewidth=0.6, color="#333333")
        axes[-1, 1].plot(t, filtered_mic[start:stop_clipped], linewidth=0.6, color="#333333")
        axes[-1, 0].set_ylabel(MIC_COLUMN, fontsize=7)
        axes[0, 0].set_title(f"Raw ({SIGNAL_UNITS})")
        axes[0, 1].set_title("Production filter chain")
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{record.recording_id}  ({record.condition}, family {record.task_family})",
            fontsize=10,
        )
        fig.text(
            0.5,
            -0.01,
            "Filtering is applied to the continuous recording before windowing. The chain is "
            "zero-phase (acausal) and cannot support a streaming or real-time claim. "
            f"Units are {SIGNAL_UNITS}: physical calibration was never documented.",
            ha="center",
            va="top",
            fontsize=6.5,
            style="italic",
            color="#555555",
            wrap=True,
        )
        save_figure(fig, args.output, f"preprocessing_{record.recording_id}")
        print(f"  {record.recording_id}")

    write_json(args.output / "filter_chain.json", chain.describe())
    print(f"\nwritten to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return parse_and_run(build_parser, main_impl, argv)


if __name__ == "__main__":
    raise SystemExit(main())
