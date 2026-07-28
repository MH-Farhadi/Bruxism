"""Narrative report generation: audit markdown and the manuscript-facing results document.

``paper_results.md`` is the hand-off artifact. Every number it states carries the path of
the artifact it came from, so a manuscript author can trace any figure in the text back to
a file. Where an analysis is blocked by an unresolved human decision, the document says
**BLOCKED** and names the decision rather than quoting a placeholder number.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bruxism.data.labels import TASK_DEFINITIONS
from bruxism.data.quality import CONFLICT_RESOLUTIONS
from bruxism.data.schema import EMG_MUSCLE_MAP, SIGNAL_UNITS

__all__ = [
    "SCOPE_STATEMENT",
    "render_audit_markdown",
    "render_paper_results",
]

SCOPE_STATEMENT = (
    "This work classifies **instructed, awake jaw and tooth-contact tasks** recorded in a "
    "single controlled laboratory session from five participants. It is not a clinical "
    "bruxism detection study, it does not address sleep bruxism, and it has not been "
    "validated on spontaneous or naturalistic behaviour. The raw filename token "
    "`natural_bruxing` is an acquisition label; the corresponding class is reported as "
    "**instructed grinding**."
)


def _table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame.empty:
        return "_(no rows)_\n"
    shown = frame.head(max_rows) if max_rows else frame
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    rows = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    out = "\n".join([header, rule, *rows])
    if max_rows and len(frame) > max_rows:
        out += f"\n\n_({len(frame) - max_rows} further rows omitted; see the CSV.)_"
    return out + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        return f"{value:.4g}"
    return str(value).replace("|", r"\|")


# ------------------------------------------------------------------- audit ---


def render_audit_markdown(audit: dict[str, Any], manifest: Any) -> str:  # noqa: ARG001
    """Human-readable companion to ``data_audit.json``."""
    totals = audit["totals"]
    lines: list[str] = [
        "# Raw dataset audit",
        "",
        f"**Manifest hash:** `{audit['manifest_hash']}`  ",
        f"**Quality policy version:** `{audit['quality_policy_version']}`  ",
        f"**Configured sampling rate:** {audit['sampling_rate_hz']} Hz  ",
        f"**Signal units:** `{audit['signal_units']}` "
        "(physical units were never documented by the acquisition chain -- do not label "
        "these values uV, Pa or dB)",
        "",
        "> This audit reads the raw data read-only. No participant name, absolute path, "
        "survey, photograph, receipt or reimbursement file is referenced in any generated "
        "artifact.",
        "",
        "## 1. Inventory",
        "",
        f"- Recording CSVs: **{totals['n_csv']}**",
        f"- Videos (.avi): **{totals['n_avi']}**",
        f"- Metadata sidecars: **{totals['n_metadata']}**",
        f"- `.npy` companions present: **{totals['n_npy_present']}** "
        f"(claimed by metadata: {totals['n_npy_claimed']})",
        f"- Included after the exclusion policy: **{totals['n_included']}** "
        f"(excluded: {totals['n_excluded']})",
        f"- Total samples: **{totals['total_samples']:,}** "
        f"({totals['total_duration_hours']:.2f} h)",
        "",
        "Recordings per participant:",
        "",
        _table(
            pd.DataFrame(sorted(audit["subjects"].items()), columns=["subject_id", "n_recordings"])
        ),
        "## 2. Channel schema",
        "",
        f"Columns: `{'`, `'.join(audit['schema'])}`",
        "",
        "Tentative EMG channel map (from `Data/README.txt`; laterality, electrode type, "
        "amplifier, gain and ADC range remain unconfirmed -- see `docs/open_questions.md`):",
        "",
        _table(
            pd.DataFrame(
                sorted(EMG_MUSCLE_MAP.items()), columns=["column", "tentative_muscle_site"]
            )
        ),
        "## 3. Quality flags",
        "",
    ]

    flags = pd.DataFrame(
        [
            {"flag": name, "count": entry["count"], "policy": entry["policy"]}
            for name, entry in audit["quality_flags"].items()
        ]
    )
    lines += [_table(flags), "## 4. Named conflict resolutions", ""]
    for rule_id, resolution in CONFLICT_RESOLUTIONS.items():
        lines += [
            f"### `{rule_id}`",
            "",
            f"- **Applies to:** {resolution.applies_to}",
            f"- **Conflict:** {resolution.conflict}",
            f"- **Resolution:** {resolution.resolution}",
            f"- **Rationale:** {resolution.rationale}",
            f"- **Approved by:** {resolution.approved_by} on {resolution.approved_on}",
            "",
        ]

    lines += ["## 5. Specific anomalies", "", "### Short recordings", ""]
    lines.append(_table(pd.DataFrame(audit["short_recordings"])))
    lines += ["### Files in the secondary location", ""]
    lines.append(_table(pd.DataFrame(audit["secondary_location_recordings"])))
    lines += ["### Filename / metadata condition conflicts", ""]
    lines.append(_table(pd.DataFrame(audit["metadata_conflicts"])))
    lines += ["### `.npy` companions", ""]
    lines.append(_table(pd.DataFrame(audit["npy_findings"])))
    lines += [
        "",
        "The metadata of every recording names an `.npy` companion, but only three exist. "
        "In those three the sixth column is all zeros and does not reproduce the CSV "
        "microphone channel. They are stale acquisition-time caches and are never read as "
        "input (rule `R2_csv_is_authoritative_over_npy`).",
        "",
        "## 6. Trigger structure",
        "",
        "Trigger semantics were confirmed by the investigator on 2026-07-27: `Trigger == 1` "
        "marks intervals during which the participant was actively performing the named "
        "task. Dedicated rest recordings have a trigger that is zero throughout.",
        "",
        "Trigger-run durations by task family (seconds):",
        "",
        _table(pd.DataFrame(audit["trigger_run_stats"])),
        "> **Finding.** Trigger granularity differs markedly between participants. Some "
        "marked one long run per bout; others marked each repetition separately. Because a "
        "one-second window with a symmetric guard needs a run of at least "
        "`1.0 + 2 x guard` seconds, the guard width materially changes how many examples "
        "each participant contributes. The sensitivity table below is the evidence for "
        "choosing it, and the choice needs investigator sign-off.",
        "",
        "## 7. Segmentation guard sensitivity",
        "",
        _table(pd.DataFrame(audit["guard_sensitivity"])),
        "## 8. Window counts",
        "",
        f"- `whole_recording_legacy` (**diagnostic only, unsafe for inference**): "
        f"**{audit['window_counts']['whole_recording_legacy']['total']:,}** windows",
        f"- `trigger_constrained` at a 0.25 s guard: "
        f"**{audit['window_counts']['trigger_constrained_guard_0.25s']['total']:,}** windows",
        "",
        _table(
            pd.DataFrame(audit["window_counts"]["trigger_constrained_guard_0.25s"]["by_family"])
        ),
        "## 9. Historical confusion-matrix provenance",
        "",
    ]

    check = audit["historical_confusion_matrix_check"]
    comparison = pd.DataFrame(
        [
            {
                "class": family,
                "published_support": published,
                "reproduced_whole_recording": check["reproduced_whole_recording_counts"].get(
                    family
                ),
                "match": check["per_class_supports_match"].get(family),
            }
            for family, published in check["published_four_class_matrix_supports"].items()
        ]
    )
    lines += [
        _table(comparison),
        "",
        f"- Published matrix total: **{check['published_matrix_total']:,}**",
        f"- Reproduced whole-recording total across all five families: "
        f"**{check['reproduced_total_all_five_families']:,}**",
        f"- Totals match: **{check['totals_match']}**",
        "",
        f"**Conclusion.** {check['finding']}",
        "",
        "## 10. Video",
        "",
        f"- Readable: {audit['video']['n_readable']} / {totals['n_csv']}",
        f"- Frame rates: {audit['video']['fps_values']}",
        f"- Resolutions: {audit['video']['resolutions']}",
        f"- Codecs: {audit['video']['codecs']}",
        "",
        "Video is recorded and inventoried but is **not** an input to any model in this "
        "pipeline. It would be the natural adjudication source if trigger-off intervals "
        "were ever to be used as rest.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------- paper results ---


def render_paper_results(
    metrics: dict[str, Any],
    *,
    run_ids: Sequence[str],
    artifact_index: dict[str, str],
    manifest_hash: str,
    source_commit: str | None,
    blockers: Sequence[dict[str, str]] = (),
    benchmark: dict[str, Any] | None = None,
    sample_counts: pd.DataFrame | None = None,
    provenance: dict[str, Any] | None = None,
) -> str:
    """The manuscript-facing narrative. Every number points at an artifact."""
    lines: list[str] = [
        "# Results bundle for `Main_2.tex`",
        "",
        "> **Scope.** " + SCOPE_STATEMENT,
        "",
        "Every value below was computed from a saved prediction ledger "
        "(`predictions.parquet`) by `bruxism-report`. No number in this document was typed "
        "by hand. Regenerate the whole bundle with:",
        "",
        "```bash",
        "bruxism-report --runs-root outputs/runs --output-root outputs/paper_bundle",
        "```",
        "",
        "## Provenance",
        "",
        f"- Source commit: `{source_commit or 'unknown'}`",
        f"- Data manifest hash: `{manifest_hash}`",
        f"- Run IDs: {', '.join(f'`{r}`' for r in run_ids) or '_(none)_'}",
        f"- Signal units: `{SIGNAL_UNITS}` (undocumented -- never write uV/Pa/dB)",
        "",
    ]

    if provenance:
        lines += [
            "| item | value |",
            "|---|---|",
            *[f"| {k} | `{v}` |" for k, v in sorted(provenance.items())],
            "",
        ]

    lines += [
        "## Multi-seed rule",
        "",
        metrics.get("multiseed_rule", "_(not recorded)_"),
        "",
        "## Task definitions",
        "",
    ]
    task_frame = pd.DataFrame(
        [
            {
                "task_id": task.task_id,
                "classes": ", ".join(task.class_names),
                "primary_endpoint": task.primary,
                "description": task.description,
            }
            for task in TASK_DEFINITIONS.values()
        ]
    )
    lines += [_table(task_frame), "## Headline results (participant-level, primary)", ""]

    from bruxism.evaluation.aggregation import condition_table

    table = condition_table(metrics)
    if table.empty:
        lines += ["**BLOCKED** -- no predictions were produced.", ""]
    else:
        headline = table[
            [
                c
                for c in (
                    "task_id",
                    "model_id",
                    "modality",
                    "seed",
                    "accuracy_mean",
                    "accuracy_std",
                    "balanced_accuracy_mean",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "macro_roc_auc_mean",
                    "n_windows_descriptive",
                    "safe_for_inference",
                )
                if c in table.columns
            ]
        ]
        lines += [
            _table(headline),
            "",
            "Each row is the **mean across the five held-out participants** of a metric "
            "computed within each participant. Participants, not windows, are the unit of "
            "generalisation. With five participants these summaries describe the sample and "
            "support no population-level claim.",
            "",
        ]

    if sample_counts is not None and not sample_counts.empty:
        lines += ["## Analysable sample counts", "", _table(sample_counts), ""]

    lines += ["## Modality contrast (fusion vs EMG-only)", ""]
    contrast = metrics.get("modality_contrast", {})
    rows: list[dict[str, Any]] = []
    for _key, entry in sorted(contrast.get("contrasts", {}).items()):
        rows.append(
            {
                "task_id": entry["task_id"],
                "metric": entry["metric"],
                "seed": entry["seed"],
                "mean_difference": entry["mean_difference"],
                "min_difference": entry["min_difference"],
                "max_difference": entry["max_difference"],
                "n_subjects_improved": f"{entry['n_subjects_improved']}/{entry['n_subjects']}",
            }
        )
    if rows:
        lines += [_table(pd.DataFrame(rows)), "", contrast.get("interpretation", ""), ""]
    else:
        lines += [
            "**BLOCKED** -- the fusion and EMG-only conditions were not both run. "
            "Run `bruxism-ablations --config configs/experiments/modality_and_no_chewing.yaml`.",
            "",
        ]

    lines += ["## Does the audio benefit survive removing chewing?", ""]
    chewing = metrics.get("chewing_contrast", {})
    if chewing.get("available"):
        lines += [chewing.get("interpretation", ""), ""]
        pairs = pd.DataFrame(
            [
                {
                    "task_id": entry["task_id"],
                    "metric": entry["metric"],
                    "mean_fusion_minus_emg": entry["mean_difference"],
                    "n_subjects_improved": (
                        f"{entry['n_subjects_improved']}/{entry['n_subjects']}"
                    ),
                }
                for entry in chewing.get("contrasts", {}).values()
            ]
        )
        lines += [_table(pairs), ""]
    else:
        lines += [
            "**BLOCKED** -- "
            + chewing.get(
                "reason",
                "needs both the five-class and no-chewing tasks in the same ledger.",
            ),
            "",
        ]

    if benchmark:
        budget = benchmark.get("latency_budget", {})
        lines += [
            "## Latency and model size",
            "",
            "The three latencies below are reported separately and must stay separate in "
            "the manuscript.",
            "",
            "| quantity | value | meaning |",
            "|---|---|---|",
            f"| Input/context latency | {budget.get('input_context_latency_seconds', 0) * 1e3:.0f} ms "
            "| the observation window; no decision about an event can exist sooner |",
            f"| Decision update interval | {budget.get('decision_update_interval_seconds', 0) * 1e3:.0f} ms "
            "| the stride; how often a new decision is produced |",
            f"| Processing latency | {budget.get('processing_latency_seconds', 0) * 1e3:.3f} ms "
            "| compute only: filtering, transform and forward pass |",
            f"| Earliest decision | {budget.get('earliest_decision_seconds', 0) * 1e3:.0f} ms "
            "| context + processing |",
            "",
            f"Trainable parameters (computed programmatically): "
            f"**{benchmark.get('parameter_counts', {}).get('trainable', 'n/a'):,}**  ",
            f"Model size at fp32: **{benchmark.get('model_size_bytes_fp32', 0) / 1024:.1f} KiB**",
            "",
            "> The compute time alone is **not** detection latency. No streaming "
            "implementation exists and the production filter chain is zero-phase "
            "(acausal), so no wearable, embedded or clinical real-time claim is supported.",
            "",
        ]

    if blockers:
        lines += [
            "## Blocked statements",
            "",
            "The following manuscript statements cannot be supplied from these artifacts "
            "and need a human decision first.",
            "",
            _table(pd.DataFrame(list(blockers))),
        ]

    lines += ["## Artifact index", "", "| artifact | path |", "|---|---|"]
    lines += [f"| {name} | `{path}` |" for name, path in sorted(artifact_index.items())]
    lines += [""]
    return "\n".join(lines)


def write_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
