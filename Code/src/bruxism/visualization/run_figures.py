"""Assemble the complete figure set for one run into ``<run_dir>/figures/``.

A finished run writes its own figures. The folder is self-contained: opening it should be
enough to understand what data the run consumed, how that data was turned into examples,
how the model was selected, how well it did on held-out participants and where it failed --
without opening a notebook or re-running anything.

Two entry paths, one implementation:

* :func:`generate_run_figures` is called at the end of :func:`bruxism.runner.run_experiment`,
  where the manifest, window index and filtered-recording cache are already in memory;
* ``bruxism-figures --run-dir outputs/runs/<id> --data-root ../Data`` calls the same
  function for a run that has already finished, rebuilding what it needs from the run
  bundle. Figures are therefore never stale relative to the code that draws them.

Nothing here can fail a run. Every figure is attempted independently; a failure is logged,
recorded in ``figures/figure_index.json`` with its reason, and the remaining figures still
get produced. A missing data root simply skips the figures that need raw signal.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bruxism.data.labels import ClassificationTask, get_task
from bruxism.evaluation.metrics import PredictionLedger
from bruxism.utils.io import read_json, write_json
from bruxism.utils.logging import get_logger

__all__ = ["FIGURE_CATALOGUE", "generate_run_figures"]

logger = get_logger(__name__)

#: Filename written by ``bruxism-train`` alongside the figures, describing every entry.
INDEX_NAME = "figure_index.json"

#: How :class:`~bruxism.training.engine.NestedLOSOTrainer` names a checkpoint. Kept as a
#: template so this module builds the name instead of parsing it back apart.
_CHECKPOINT_NAME = "{task_id}_{model_id}_{modality}_fold{fold}_seed{seed}.pt"


@dataclass
class FigureRecord:
    """One catalogue entry plus what happened when it was attempted."""

    stem: str
    title: str
    shows: str
    manuscript_slot: str
    status: str = "pending"
    reason: str = ""
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "title": self.title,
            "what_it_shows": self.shows,
            "manuscript_slot": self.manuscript_slot,
            "status": self.status,
            "reason": self.reason,
            "files": sorted(self.files),
        }


#: Declared once, so the index, the README and the drawing code cannot disagree about what
#: a figure is for. ``manuscript_slot`` names the place in ``Main_2.tex`` a figure can fill,
#: or says that it is supporting material rather than a manuscript figure.
FIGURE_CATALOGUE: dict[str, tuple[str, str, str]] = {
    "00_run_scorecard": (
        "Run scorecard",
        "Identity, provenance, protocol and the headline participant-level results of this "
        "run on one page.",
        "supporting: cover page for a supplement or slide deck",
    ),
    "01_dataset_inventory": (
        "Dataset inventory",
        "Analysable windows per participant and per class, and the class balance the loss "
        "had to cope with.",
        "Table 1 (dataset characteristics); the rest / total window counts marked TBD",
    ),
    "02_window_inventory": (
        "Window inventory heatmap",
        "Participant x class window counts and each participant's class shares, exposing "
        "cells too thin to support a per-class metric in that fold.",
        "supporting: supplement to Table 1",
    ),
    "03_segmentation_timeline": (
        "Segmentation timeline",
        "How a continuous recording becomes windows: trigger runs, transition guards, "
        "startup guard, and the overlapping windows actually emitted.",
        "Methods, 'Sensors and dataset' - the 1.0 s / 0.5 s / guard description",
    ),
    "04_filter_response": (
        "Filter response",
        "Magnitude response of every production filter stage and of the chain as applied, "
        "including the forward+reverse zero-phase pass.",
        "Methods, 'Preprocessing and wavelet decomposition'; answers the causal / "
        "zero-phase reproducibility note",
    ),
    "05_preprocessing_stages": (
        "Preprocessing stages",
        "Raw -> notch -> bandpass on a real excerpt with the spectrum at each stage, plus "
        "the microphone chain.",
        "Fig. 2 (fig:emg_preprocessing)",
    ),
    "06_class_spectra": (
        "Class-conditional spectra",
        "Mean EMG and microphone power spectra per class - the evidence for "
        "modality-specific processing.",
        "Discussion, 'Engineering interpretation'; supports Fig. 7 (fig:signal_comparison)",
    ),
    "07_example_windows": (
        "Example windows",
        "A representative one-second model input per class, over a percentile band showing "
        "the within-class spread.",
        "Fig. 7 (fig:signal_comparison), five-class version",
    ),
    "08_wavelet_bands": (
        "Wavelet bands",
        "The named coefficient bands each branch consumes, per class, annotated with their "
        "nominal frequency ranges.",
        "Methods, 'Preprocessing and wavelet decomposition'; supports Fig. 3 (fig:architecture)",
    ),
    "09_wavelet_band_energy": (
        "Wavelet band energy",
        "Share of window energy in every band of the decomposition per class, marking the "
        "bands the network actually receives.",
        "Methods / Discussion: why these bands and what the selection discards",
    ),
    "10_augmentation_examples": (
        "Augmentation examples",
        "What amplitude scaling, noise injection and circular shift do to a real minority "
        "class window, produced by the training augmenter itself.",
        "Methods, 'Training and evaluation'; answers the augmentation TBD",
    ),
    "11_training_curves": (
        "Training curves",
        "Per-epoch loss and objective of every final refit, one column per seed, with each "
        "fold's epoch budget marked.",
        "Fig. 9 (fig:loss_curves)",
    ),
    "12_hyperparameter_selection": (
        "Hyperparameter selection",
        "What the inner search scored per trial and outer fold, which trial the "
        "prespecified rule selected, and the resulting epoch budget.",
        "Table 2 (tab:hyperparameters); the learning-rate / epoch TBDs",
    ),
    "13_confusion_matrix": (
        "Confusion matrices",
        "Held-out counts and row-normalised recall, pooled over leave-one-subject-out folds.",
        "Fig. 6 (fig:confusion)",
    ),
    "14_roc_curves": (
        "ROC curves",
        "One-vs-rest ROC per class with AUC, computed from held-out probabilities.",
        "Tables 3-5 AUC columns; the reviewer's AUC request",
    ),
    "15_pr_curves": (
        "Precision-recall curves",
        "One-vs-rest precision-recall per class with average precision - the more "
        "informative view under this class imbalance.",
        "supporting: supplement to the AUC tables",
    ),
    "16_per_participant": (
        "Per-participant results",
        "One bar per held-out participant with the sample mean marked - the primary "
        "evidence, since participants are the unit of generalisation.",
        "Table 6 (tab:persubject)",
    ),
    "17_per_class_performance": (
        "Per-class performance",
        "Precision, recall, F1, one-vs-rest AUC and average precision for every class, with "
        "support.",
        "Table 5 (tab:perclass)",
    ),
    "18_participant_class_recall": (
        "Participant x class recall",
        "Recall in every (held-out participant, class) cell - which participant failed, on "
        "which class, and with how much support.",
        "Results, 'Class-level diagnostics' and 'Participant-level results'",
    ),
    "19_calibration": (
        "Probability calibration",
        "Reliability diagram with expected calibration error, and the confidence "
        "distribution split by correctness.",
        "Methods note: whether probabilities were calibrated before AUC was computed",
    ),
    "20_seed_stability": (
        "Seed stability",
        "How much each participant's result and the headline metrics move when only the "
        "random seed changes.",
        "supporting: evidence for the multi-seed reporting rule",
    ),
    "21_error_timeline": (
        "Error timeline",
        "Held-out predictions in recording time for a representative participant, showing "
        "whether errors cluster at trial onsets or spread uniformly.",
        "Results, 'Class-level diagnostics'; supports the onset-ambiguity discussion",
    ),
    "22_embedding_tsne": (
        "Embedding t-SNE",
        "One t-SNE projection of held-out embeddings coloured twice: by class, and by participant.",
        "Fig. 5 (fig:tsne)",
    ),
    "23_modality_comparison": (
        "Modality comparison",
        "Participant-level mean per modality and task, for the fusion / EMG-only / "
        "audio-only ablation.",
        "Table 4 (tab:modality_ablation)",
    ),
}


class _Builder:
    """Runs each figure in isolation and records the outcome."""

    def __init__(self, figures_dir: Path):
        self.figures_dir = figures_dir
        self.records: list[FigureRecord] = []

    def run(self, stem: str, draw: Callable[[], Sequence[Path]]) -> None:
        title, shows, slot = FIGURE_CATALOGUE.get(stem, (stem, "", ""))
        record = FigureRecord(stem=stem, title=title, shows=shows, manuscript_slot=slot)
        self.records.append(record)
        try:
            written = draw()
        except Exception as exc:  # noqa: BLE001 - one figure must never fail a whole run
            record.status = "failed"
            record.reason = f"{type(exc).__name__}: {exc}"
            logger.warning("figure %s failed: %s", stem, record.reason)
            logger.debug("figure %s traceback:\n%s", stem, traceback.format_exc())
            return
        record.status = "written"
        record.files = [Path(path).name for path in written]
        logger.debug("figure %s written", stem)

    def skip(self, stem: str, reason: str) -> None:
        title, shows, slot = FIGURE_CATALOGUE.get(stem, (stem, "", ""))
        self.records.append(
            FigureRecord(
                stem=stem,
                title=title,
                shows=shows,
                manuscript_slot=slot,
                status="skipped",
                reason=reason,
            )
        )
        logger.info("figure %s skipped: %s", stem, reason)

    @property
    def counts(self) -> dict[str, int]:
        counts = {"written": 0, "skipped": 0, "failed": 0}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts


# ------------------------------------------------------------------ embeddings ---


def _rebuild_model(payload: dict[str, Any]) -> Any:
    """Rebuild the exact evaluated architecture from a checkpoint's architecture record.

    Sets ``eval()`` but deliberately does **not** touch the global autograd flag: the
    caller wraps its forward passes in ``torch.no_grad()``, and flipping a process-wide
    switch here would silently disable gradients for anything that runs afterwards.
    """
    architecture = payload["architecture"]
    name = architecture["class"]
    config = dict(architecture["config"])
    model: Any
    if name == "DualBranchWaveletCNN":
        from bruxism.models.dual_branch import DualBranchConfig, build_model

        model = build_model(DualBranchConfig.from_dict(config))
    elif name == "EarlyFusionCNN":
        from bruxism.models.baselines import EarlyFusionCNN, EarlyFusionConfig

        config["channels"] = tuple(config.get("channels", (16, 32, 64)))
        model = EarlyFusionCNN(EarlyFusionConfig(**config))
    elif name == "BiLSTMBaseline":
        from bruxism.models.baselines import BiLSTMBaseline, BiLSTMConfig

        model = BiLSTMBaseline(BiLSTMConfig(**config))
    else:  # pragma: no cover - a new architecture must be added here deliberately
        raise ValueError(f"cannot rebuild architecture {name!r} for the embedding figure")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _capture_input(store: dict[str, Any]) -> Callable[[Any, tuple[Any, ...]], None]:
    """Forward pre-hook that stashes a module's input tensor in ``store``.

    Used to read the *pre-fusion* concatenation of the modality branches -- the feature
    vector the manuscript describes -- without modifying the model to return it.
    """

    def _capture(_module: Any, args: tuple[Any, ...]) -> None:
        store["value"] = args[0].detach()

    return _capture


def compute_held_out_embeddings(
    run_dir: Path,
    *,
    window_index: Any,
    cache: Any,
    task: ClassificationTask,
    predictions: pd.DataFrame,
    model_id: str,
    modality: str,
    seed: int,
    max_samples: int = 3000,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Recompute embeddings of the held-out windows from this run's own checkpoints.

    Returns ``(embeddings, labels, subject_ids, checkpoint_ids, source_name)``. Only the
    outer-test participant of each checkpoint's fold is passed through it, so the projection
    shows genuinely held-out data. Where the architecture concatenates modality branches
    before a fusion head, the *pre-fusion* concatenation is captured -- that is the feature
    vector the manuscript describes -- otherwise the penultimate representation is used.
    """
    import torch

    from bruxism.data.dataset import WindowDataset
    from bruxism.preprocessing.normalization import Normalizer

    # The name is built the same way the trainer builds it, rather than parsed: task ids and
    # model ids both contain underscores, so any split of the stem would be ambiguous.
    pattern = _CHECKPOINT_NAME.format(
        task_id=task.task_id, model_id=model_id, modality=modality, fold="*", seed=seed
    )
    wanted = sorted((run_dir / "checkpoints").glob(pattern))
    if not wanted:
        available = sorted(path.name for path in (run_dir / "checkpoints").glob("*.pt"))
        raise ValueError(
            f"no checkpoint matches {pattern!r} in {run_dir / 'checkpoints'}; "
            f"found {available[:4]}{'...' if len(available) > 4 else ''}. Training must run "
            f"with output.save_checkpoints: true for the embedding figure."
        )

    subset = predictions[
        (predictions["task_id"] == task.task_id)
        & (predictions["model_id"] == model_id)
        & (predictions["modality"] == modality)
        & (predictions["seed"] == seed)
    ]
    all_subjects = set(subset["subject_id"].unique())

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    used: list[str] = []
    source_name = "fusion embedding"

    for path in wanted:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = _rebuild_model(payload)
        normalizer = Normalizer.from_dict(payload["normalizer"])
        held_out = sorted(all_subjects - set(normalizer.fitted_on))
        rows = subset[subset["subject_id"].isin(held_out)]
        sample_ids = list(rows["sample_id"].unique())
        if not sample_ids:
            continue

        dataset = WindowDataset(
            window_index,
            sample_ids,
            task,
            cache,
            stage="test",
            normalizer=normalizer,
            modality=payload["architecture"]["config"].get("modality", modality),
        )
        captured: dict[str, Any] = {}
        handle = None
        fusion = getattr(model, "fusion", None)
        if fusion is not None:
            source_name = "concatenated branch feature"
            handle = fusion.register_forward_pre_hook(_capture_input(captured))
        try:
            with torch.no_grad():
                for start in range(0, len(dataset), batch_size):
                    batch = [
                        dataset[index]
                        for index in range(start, min(start + batch_size, len(dataset)))
                    ]
                    emg = torch.stack([item[0] for item in batch])
                    mic = torch.stack([item[1] for item in batch])
                    embedded = model.embed(emg, mic)
                    vector = captured.get("value", embedded)
                    features.append(np.asarray(vector.numpy(), dtype=np.float32))
                    labels.append(np.array([int(item[2]) for item in batch]))
                    subjects.append(
                        np.array(
                            [
                                dataset.examples[index].window.subject_id
                                for index in range(start, min(start + batch_size, len(dataset)))
                            ]
                        )
                    )
        finally:
            if handle is not None:
                handle.remove()
        used.append(path.name)

    if not features:
        raise ValueError("no held-out window produced an embedding")
    stacked = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    subject_array = np.concatenate(subjects, axis=0)
    if stacked.shape[0] > max_samples:
        # Deterministic, evenly spaced thinning; no RNG, so the figure is reproducible.
        positions = np.linspace(0, stacked.shape[0] - 1, max_samples).astype(int)
        stacked = stacked[positions]
        label_array = label_array[positions]
        subject_array = subject_array[positions]
    return stacked, label_array, subject_array, ", ".join(used), source_name


# ------------------------------------------------------------------- assembly ---


def _primary_condition(
    predictions: pd.DataFrame, config_task: str, config_model: str
) -> tuple[str, str, str, int]:
    """Choose one condition for the results figures by a fixed rule, not by which looked best.

    Preference order: the configured task, then the configured model, then fusion, then the
    lowest seed. Any preference that is absent from the ledger falls back to the
    alphabetically first value present, so the choice is deterministic either way.
    """
    tasks = sorted(predictions["task_id"].unique())
    task_id = config_task if config_task in tasks else tasks[0]
    subset = predictions[predictions["task_id"] == task_id]
    models = sorted(subset["model_id"].unique())
    model_id = config_model if config_model in models else models[0]
    subset = subset[subset["model_id"] == model_id]
    modalities = sorted(subset["modality"].unique())
    modality = "fusion" if "fusion" in modalities else modalities[0]
    subset = subset[subset["modality"] == modality]
    seed = int(subset["seed"].min())
    return task_id, model_id, modality, seed


def _condition_entry(
    metrics: dict[str, Any], task_id: str, model_id: str, modality: str, seed: int
) -> dict[str, Any] | None:
    for entry in metrics.get("conditions", {}).values():
        if (
            entry.get("task_id") == task_id
            and entry.get("model_id") == model_id
            and entry.get("modality") == modality
            and int(entry.get("seed", -1)) == seed
        ):
            return entry
    return None


def _write_readme(
    figures_dir: Path,
    *,
    run_id: str,
    run_dir: Path,
    records: Sequence[FigureRecord],
    condition: dict[str, Any],
) -> Path:
    """A README beside the figures: what each one shows and where it belongs in the paper."""
    lines = [
        f"# Figures for run `{run_id}`",
        "",
        "Generated automatically at the end of the run by "
        "`bruxism.visualization.run_figures`. Regenerate at any time without retraining:",
        "",
        "```bash",
        f"bruxism-figures --run-dir {run_dir.as_posix()} --data-root /path/to/Data",
        "```",
        "",
        "Every figure is written as **PNG (300 dpi)** for drafts and **PDF** (vector) for "
        "submission. Machine-readable status for each entry is in "
        f"`{INDEX_NAME}`.",
        "",
        "**Scope.** This run classifies instructed, awake jaw and tooth-contact tasks "
        "recorded in a single laboratory session from five participants. No figure here "
        "supports a clinical bruxism-detection, sleep-bruxism or free-living claim, and the "
        "acquisition token `natural_bruxing` is reported throughout as instructed grinding.",
        "",
        "## Results condition shown",
        "",
        "| field | value |",
        "|---|---|",
        *[f"| {key} | `{value}` |" for key, value in condition.items()],
        "",
        "Chosen by a fixed rule (configured task, configured model, fusion, lowest seed), "
        "not by which condition scored best.",
        "",
        "## Figures",
        "",
        "| figure | what it shows | manuscript slot | status |",
        "|---|---|---|---|",
    ]
    for record in records:
        status = (
            record.status if record.status == "written" else f"{record.status} - {record.reason}"
        )
        lines.append(f"| `{record.stem}` | {record.shows} | {record.manuscript_slot} | {status} |")
    lines += [
        "",
        "## Reading order",
        "",
        "1. `00_run_scorecard` - what this run was and what it produced.",
        "2. `01`-`03` - the data and how it became examples.",
        "3. `04`-`10` - what the signal looks like at every processing stage.",
        "4. `11`-`12` - how the model and its epoch budget were selected, on inner folds only.",
        "5. `13`-`21` - held-out performance, and where and when it fails.",
        "6. `22`-`23` - exploratory representation and the modality ablation.",
        "",
    ]
    path = figures_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_run_figures(
    run_dir: Path | str,
    *,
    config: Any | None = None,
    manifest: Any | None = None,
    window_index: Any | None = None,
    cache: Any | None = None,
    predictions: pd.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    fold_outcomes: Sequence[dict[str, Any]] | None = None,
    data_root: Path | str | None = None,
    max_windows_per_class: int = 150,
    tsne_max_samples: int = 3000,
    tsne_seed: int = 0,
    include_signal_figures: bool = True,
    include_tsne: bool = True,
) -> dict[str, Any]:
    """Write the full figure set for one run and return a summary of what was produced.

    Parameters
    ----------
    run_dir
        A run bundle directory. Anything not passed explicitly is loaded from it.
    config, manifest, window_index, cache, predictions, metrics, fold_outcomes
        Supplied by the training run, which already holds them; loaded from ``run_dir``
        (and rebuilt from ``data_root``) otherwise.
    data_root
        Needed only to rebuild the manifest/window index/cache when they are not supplied.
        Without it, the figures that read raw signal are skipped with a recorded reason.
    max_windows_per_class
        Deterministic cap on the windows sampled for the spectral and wavelet figures.
    include_signal_figures, include_tsne
        Escape hatches for the expensive groups. Both default on.

    Returns
    -------
    dict
        ``{"figures_dir", "written", "skipped", "failed", "index"}``.
    """
    from bruxism.visualization import diagnostics, paper_figures, signal_figures

    run_path = Path(run_dir)
    figures_dir = run_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    builder = _Builder(figures_dir)

    # ---- load whatever the caller did not supply -------------------------------------
    if config is None:
        from bruxism.config import load_experiment_config

        config = load_experiment_config(run_path / "resolved_config.yaml")
    if predictions is None:
        predictions = pd.read_parquet(run_path / "predictions.parquet")
    if metrics is None:
        metrics = read_json(run_path / "metrics.json")
    if fold_outcomes is None:
        outcomes_path = run_path / "selection" / "fold_outcomes.json"
        fold_outcomes = read_json(outcomes_path) if outcomes_path.is_file() else []
    bundle = (
        read_json(run_path / "run_bundle.json") if (run_path / "run_bundle.json").is_file() else {}
    )
    source_state = (
        read_json(run_path / "source_state.json")
        if (run_path / "source_state.json").is_file()
        else {}
    )
    environment = (
        read_json(run_path / "environment.json")
        if (run_path / "environment.json").is_file()
        else {}
    )

    if predictions.empty:
        raise ValueError(f"{run_path} contains no predictions; nothing to plot")

    task_id, model_id, modality, seed = _primary_condition(
        predictions, config.task_id, config.model_id
    )
    task = get_task(task_id)
    chosen = predictions[
        (predictions["task_id"] == task_id)
        & (predictions["model_id"] == model_id)
        & (predictions["modality"] == modality)
        & (predictions["seed"] == seed)
    ].reset_index(drop=True)
    ledger = PredictionLedger(frame=chosen, class_names=task.class_names)
    entry = _condition_entry(metrics, task_id, model_id, modality, seed)
    task_outcomes = [
        outcome
        for outcome in fold_outcomes
        if (outcome.get("condition") or {}).get("task_id", task_id) == task_id
    ]

    # ---- data-side figures (need the raw recordings) ----------------------------------
    signal_skip = ""
    if not include_signal_figures:
        signal_skip = "disabled by include_signal_figures=False"
    elif window_index is None or cache is None or manifest is None:
        if data_root is None:
            signal_skip = (
                "needs the raw data root to rebuild the window index and filtered-recording "
                "cache; pass --data-root"
            )
        else:
            try:
                from bruxism.runner import prepare_data

                manifest, window_index, cache = prepare_data(config, data_root=data_root)
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                signal_skip = f"could not rebuild the dataset: {type(exc).__name__}: {exc}"

    signal_stems = [
        "01_dataset_inventory",
        "02_window_inventory",
        "03_segmentation_timeline",
        "04_filter_response",
        "05_preprocessing_stages",
        "06_class_spectra",
        "07_example_windows",
        "08_wavelet_bands",
        "09_wavelet_band_energy",
        "10_augmentation_examples",
    ]
    if signal_skip:
        for stem in signal_stems:
            builder.skip(stem, signal_skip)
    else:
        assert manifest is not None and window_index is not None and cache is not None
        builder.run(
            "01_dataset_inventory",
            lambda: signal_figures.plot_dataset_inventory(window_index, task, figures_dir),
        )
        builder.run(
            "02_window_inventory",
            lambda: signal_figures.plot_window_inventory(window_index, task, figures_dir),
        )
        builder.run(
            "03_segmentation_timeline",
            lambda: signal_figures.plot_segmentation_timeline(
                manifest, window_index, cache, figures_dir
            ),
        )
        builder.run(
            "04_filter_response",
            lambda: signal_figures.plot_filter_response(
                config.filters, float(config.data.sampling_rate_hz), figures_dir
            ),
        )
        builder.run(
            "05_preprocessing_stages",
            lambda: signal_figures.plot_preprocessing_stages(
                manifest, window_index, figures_dir, filter_config=config.filters
            ),
        )
        builder.run(
            "06_class_spectra",
            lambda: signal_figures.plot_class_spectra(
                window_index, cache, task, figures_dir, max_per_class=max_windows_per_class
            ),
        )
        builder.run(
            "07_example_windows",
            lambda: signal_figures.plot_example_windows(window_index, cache, task, figures_dir),
        )

        wavelets = _wavelet_configs(config, task)
        if wavelets is None:
            reason = (
                f"model {config.model_id!r} has no wavelet branches; the decomposition "
                "figures apply to the dual-branch wavelet CNN only"
            )
            builder.skip("08_wavelet_bands", reason)
            builder.skip("09_wavelet_band_energy", reason)
        else:
            emg_wavelet, mic_wavelet = wavelets
            builder.run(
                "08_wavelet_bands",
                lambda: signal_figures.plot_wavelet_bands(
                    window_index,
                    cache,
                    task,
                    figures_dir,
                    emg_wavelet=emg_wavelet,
                    mic_wavelet=mic_wavelet,
                ),
            )
            builder.run(
                "09_wavelet_band_energy",
                lambda: signal_figures.plot_wavelet_band_energy(
                    window_index,
                    cache,
                    task,
                    figures_dir,
                    emg_wavelet=emg_wavelet,
                    mic_wavelet=mic_wavelet,
                    max_per_class=min(max_windows_per_class, 120),
                ),
            )
        builder.run(
            "10_augmentation_examples",
            lambda: signal_figures.plot_augmentation_examples(
                window_index,
                cache,
                task,
                config.augmentation,
                figures_dir,
                seed=int(config.training.seeds[0]),
            ),
        )

    # ---- selection and training --------------------------------------------------------
    if task_outcomes:
        builder.run(
            "11_training_curves",
            lambda: diagnostics.plot_training_curves_by_seed(task_outcomes, figures_dir),
        )
        builder.run(
            "12_hyperparameter_selection",
            lambda: diagnostics.plot_hyperparameter_selection(task_outcomes, figures_dir),
        )
    else:
        reason = "this run recorded no fold outcomes (selection/fold_outcomes.json is absent)"
        builder.skip("11_training_curves", reason)
        builder.skip("12_hyperparameter_selection", reason)

    # ---- results ------------------------------------------------------------------------
    builder.run(
        "13_confusion_matrix",
        lambda: paper_figures.plot_confusion_matrices(
            ledger, figures_dir, stem="13_confusion_matrix"
        ),
    )
    curves = (entry or {}).get("curves") or {}
    if curves:
        builder.run(
            "14_roc_curves",
            lambda: paper_figures.plot_roc_curves(
                curves,
                figures_dir,
                stem="14_roc_curves",
                title=f"{task_id}: one-vs-rest ROC (held-out participants)",
            ),
        )
        builder.run(
            "15_pr_curves",
            lambda: paper_figures.plot_pr_curves(
                curves,
                figures_dir,
                stem="15_pr_curves",
                title=f"{task_id}: one-vs-rest precision-recall (held-out participants)",
            ),
        )
    else:
        reason = "the metrics summary contains no curve points for this condition"
        builder.skip("14_roc_curves", reason)
        builder.skip("15_pr_curves", reason)

    if entry:
        builder.run(
            "16_per_participant",
            lambda: paper_figures.plot_per_participant(
                entry, figures_dir, stem="16_per_participant"
            ),
        )
        builder.run(
            "17_per_class_performance",
            lambda: diagnostics.plot_per_class_performance(entry, figures_dir),
        )
    else:
        reason = "no metrics entry matched the selected condition"
        builder.skip("16_per_participant", reason)
        builder.skip("17_per_class_performance", reason)

    builder.run(
        "18_participant_class_recall",
        lambda: diagnostics.plot_participant_class_recall(ledger, figures_dir),
    )
    builder.run("19_calibration", lambda: diagnostics.plot_calibration(ledger, figures_dir))

    n_seeds = predictions[predictions["task_id"] == task_id]["seed"].nunique()
    if n_seeds > 1:
        builder.run(
            "20_seed_stability",
            lambda: diagnostics.plot_seed_stability(metrics, task_id, figures_dir),
        )
    else:
        builder.skip("20_seed_stability", "the run used a single seed; there is nothing to compare")

    builder.run("21_error_timeline", lambda: diagnostics.plot_error_timeline(ledger, figures_dir))

    # ---- exploratory embedding ----------------------------------------------------------
    if not include_tsne:
        builder.skip("22_embedding_tsne", "disabled by include_tsne=False")
    elif window_index is None or cache is None:
        builder.skip("22_embedding_tsne", signal_skip or "needs the raw data root")
    else:

        def _tsne() -> Sequence[Path]:
            assert window_index is not None and cache is not None and predictions is not None
            embeddings, labels, subjects, checkpoint_id, source = compute_held_out_embeddings(
                run_path,
                window_index=window_index,
                cache=cache,
                task=task,
                predictions=predictions,
                model_id=model_id,
                modality=modality,
                seed=seed,
                max_samples=tsne_max_samples,
            )
            names = checkpoint_id.split(", ")
            paths, settings = diagnostics.plot_embedding_projection(
                embeddings,
                labels,
                subjects,
                task.class_names,
                figures_dir,
                seed=tsne_seed,
                # The caption gets a compact summary; the full list lives in the settings
                # JSON written beside the figure.
                checkpoint_id=(
                    f"{len(names)} fold checkpoint(s) of {task.task_id}/{model_id}/"
                    f"{modality} seed {seed}"
                ),
                embedding_source=source,
            )
            settings["source_checkpoints"] = names
            write_json(figures_dir / "22_embedding_tsne_settings.json", settings)
            return paths

        builder.run("22_embedding_tsne", _tsne)

    # ---- modality ablation (only meaningful when the run swept modalities) --------------
    from bruxism.evaluation.aggregation import condition_table

    table = condition_table(metrics)
    if not table.empty and table["modality"].nunique() > 1:
        builder.run(
            "23_modality_comparison",
            lambda: paper_figures.plot_modality_comparison(
                table, figures_dir, stem="23_modality_comparison"
            ),
        )
    else:
        builder.skip(
            "23_modality_comparison",
            "this run contains a single modality; run bruxism-ablations with "
            "configs/experiments/modality_and_no_chewing.yaml for the contrast",
        )

    # ---- scorecard last, so it can state how many figures were produced -----------------
    builder.run(
        "00_run_scorecard",
        lambda: diagnostics.plot_run_scorecard(
            run_id=bundle.get("run_id", run_path.name),
            config=config.to_dict(),
            bundle=bundle,
            source_state=source_state,
            environment=environment,
            metrics=metrics,
            task_id=task_id,
            n_folds=len(task_outcomes) or int(predictions["outer_fold"].nunique()),
            output_dir=figures_dir,
        ),
    )
    builder.records.sort(key=lambda record: record.stem)

    condition = {
        "task_id": task_id,
        "model_id": model_id,
        "modality": modality,
        "seed": seed,
        "held_out_windows": int(len(chosen)),
        "participants": ", ".join(sorted(chosen["subject_id"].unique())),
    }
    index = {
        "run_id": bundle.get("run_id", run_path.name),
        "config_hash": bundle.get("config_hash"),
        "manifest_hash": bundle.get("manifest_hash"),
        "source_commit": source_state.get("commit"),
        "results_condition": condition,
        "tasks_present": sorted(predictions["task_id"].unique()),
        "counts": builder.counts,
        "figures": [record.to_dict() for record in builder.records],
    }
    write_json(figures_dir / INDEX_NAME, index)
    _write_readme(
        figures_dir,
        run_id=str(index["run_id"]),
        run_dir=run_path,
        records=builder.records,
        condition=condition,
    )

    counts = builder.counts
    logger.info(
        "figures: %d written, %d skipped, %d failed -> %s",
        counts.get("written", 0),
        counts.get("skipped", 0),
        counts.get("failed", 0),
        figures_dir,
        extra={"figures_written": counts.get("written", 0)},
    )
    return {
        "figures_dir": figures_dir,
        "written": counts.get("written", 0),
        "skipped": counts.get("skipped", 0),
        "failed": counts.get("failed", 0),
        "index": index,
    }


def _wavelet_configs(config: Any, task: ClassificationTask) -> tuple[Any, Any] | None:
    """The EMG and microphone wavelet settings this run's model uses, if it has any."""
    if config.model_id != "dual_branch_wavelet_cnn":
        return None
    from bruxism.models.dual_branch import DualBranchConfig

    model_config = DualBranchConfig(
        num_classes=task.num_classes,
        window_samples=config.data.window_samples(),
        **dict(config.model_overrides),
    )
    return model_config.emg.wavelet, model_config.mic.wavelet
