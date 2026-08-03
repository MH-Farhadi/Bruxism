# Bruxism: classification of instructed awake jaw and tooth-contact tasks

Reproducible analysis pipeline for a five-participant laboratory study combining surface EMG
and a microphone.

---

## ⚠ Read this first: what this project is, and is not

This software **classifies instructed, awake jaw and tooth-contact tasks** recorded from
**five participants** in a **single controlled laboratory session**.

It is **not**:

- a clinical bruxism detection system,
- a sleep-bruxism study,
- a study of spontaneous or naturalistic bruxism,
- a validated wearable, ambulatory or real-time system.

Specific things you must not infer from anything this code produces:

| Do not say | Say instead |
|---|---|
| "bruxism detection" | classification of instructed awake tooth-contact tasks |
| "natural bruxing" | instructed grinding (the raw token `natural_bruxing` is a filename label) |
| "generalises to unseen individuals" | within this controlled five-participant dataset |
| "real-time / low-latency detection" | 1 s observation window, 0.5 s decision interval, *N* ms compute |
| "µV" | arbitrary ADC units — physical calibration was never documented |

Five participants cannot characterise population variability. The production filter chain is
zero-phase (acausal) and no streaming implementation exists.

Several statements the manuscript needs are **blocked** on human decisions that no amount of
code can settle — hardware and units, IRB identifier, recruitment evidence, protocol timing.
See **[`docs/open_questions.md`](docs/open_questions.md)**.

---

## Quick start

```bash
# 1. Install (editable, with dev and video extras)
python -m pip install -e ".[dev,video]"

# 2. Quality gates -- no private data needed, everything runs on synthetic fixtures
python -m ruff check src scripts tests
python -m mypy
python -m pytest -m "not slow"          # fast unit tests (~30 s)
python -m pytest                         # + end-to-end integration (236 tests, ~10 min)

# 3. Point the software at an authorized data root (never copied into this repo)
export BRUXISM_DATA_ROOT=../Data         # or pass --data-root everywhere

# 4. Audit the raw data (read-only)
bruxism-audit --data-root "$BRUXISM_DATA_ROOT" --output-root outputs/data_audit

# 5. Fast one-fold smoke run (~35 s of training, then ~20 s of figures)
bruxism-train --config configs/experiments/smoke.yaml --max-folds 1
open outputs/runs/smoke/figures/README.md     # 24 figures explaining that run

# 6. Regenerate every figure and table
bruxism-benchmark --output outputs/benchmarks
bruxism-report --runs-root outputs/runs --output-root outputs/paper_bundle \
               --data-root "$BRUXISM_DATA_ROOT"
```

Every command supports `--help`, `--data-root`, `--log-level`, `--progress` and, where
meaningful, `--validate-only`.

---

## Which script do I run?

Console scripts are installed by `pip install -e .`. Each is equivalently
`python -m bruxism.cli.<module>` or `python scripts/<area>/<name>.py`.

### Inspect the data

| Goal | Command |
|---|---|
| Audit the raw dataset, build the manifest, list anomalies | `bruxism-audit --data-root ../Data --output-root outputs/data_audit` |
| Check the manifest without writing anything | `bruxism-audit --data-root ../Data --validate-only` |
| Build the versioned analysis window index | `bruxism-build-manifest --config configs/data/trigger_constrained.yaml --data-root ../Data` |
| Plot raw vs production-filtered traces | `bruxism-plot-preprocessing --data-root ../Data` |

### Train

| Goal | Command | Cost |
|---|---|---|
| **Smoke test** — exercise every stage | `bruxism-train --config configs/experiments/smoke.yaml --max-folds 1` | ~35 s |
| **Primary five-class endpoint** | `bruxism-train --config configs/experiments/five_class_nested_loso.yaml` | hours |
| **Modality + no-chewing ablations** (the critical audio analysis) | `bruxism-ablations --config configs/experiments/modality_and_no_chewing.yaml` | hours |
| **Architecture baselines**, matched inputs | `bruxism-baselines --config configs/experiments/baselines.yaml` | hours |
| **Binary / ternary / legacy** endpoints | `bruxism-ablations --config configs/experiments/secondary_tasks.yaml` | hours |
| Check a config without training | add `--validate-only` | seconds |
| Cap the work | add `--max-folds N` | — |
| Change a setting without editing YAML | add `--set training.batch_size=32` | — |
| Skip the figures a finished run draws | add `--no-figures` | saves ~20 s–2 min |

Runs are **resumable**: rerunning the same command reuses completed folds, and refuses to
resume across a changed configuration, manifest or window index.

Every finished run also writes its own figure set — see
[**What a finished run draws**](#what-a-finished-run-draws) below.

### Watching a run

A training command states the size of the job before it starts, then reports every stage as
it happens: the plan, each hyperparameter trial and inner fold, per-epoch loss and
objective, the refit, the held-out score for each participant, and how much wall clock is
left.

```
plan: 1 condition(s) x 3 seed(s) x 5 outer fold(s) = 15 fold run(s); 9 model fit(s) per
      fold (2 trial(s) x 4 inner fold(s), then 1 refit); device=cuda
fold 3/15 five_class S03:  13%|█▎        | 2/15 [38:24<4:09:36, 19.2min/fold]
  inner search trial01 fold 2/4 val=S04:  38%|███▊      | 6/16 [09:12<15:20, 92.0s/fit]
    epochs:  42%|████▏     | 25/60 [00:33<00:47, 1.35s/epoch, loss=0.612, val_loss=0.88,
                                                              val_macro_f1=0.487]
```

| Situation | Flag |
|---|---|
| Terminal | nothing — bars are drawn automatically |
| `nohup`, `tee`, CI, a log file | nothing — the same progress arrives as periodic log lines |
| Force one or the other | `--progress bar` / `--progress plain` |
| Silence it | `--progress none` (or `--quiet`) |
| Change the log-line cadence | `--progress-interval 120` (default 30 s) |
| Per-epoch detail on the console | `--log-level DEBUG` |

`--progress` is display only and is not part of the configuration hash: the same run
produces the same artifacts with bars, with log lines or with nothing. Every progress
milestone is also written to `outputs/runs/<run_id>/logs/run.log.jsonl` regardless, at
DEBUG, one JSON object per line.

### Evaluate and produce figures

| Goal | Command |
|---|---|
| Recompute all metrics from saved predictions | `bruxism-summarize --runs-root outputs/runs --output outputs/summary` |
| Measure parameter counts and the three latencies | `bruxism-benchmark --output outputs/benchmarks` |
| **Rebuild one run's figures** without retraining | `bruxism-figures --run-dir outputs/runs/<run_id> --data-root ../Data` |
| **Regenerate every figure, table, LaTeX macro and the results narrative** | `bruxism-report --runs-root outputs/runs --output-root outputs/paper_bundle --data-root ../Data` |

---

## What a finished run draws

When the last fold completes, the run writes **24 figures** into
`outputs/runs/<run_id>/figures/` — every one as **PNG (300 dpi)** for drafts and **PDF**
(vector) for submission — plus a `README.md` naming what each shows and which manuscript slot
it can fill, and a machine-readable `figure_index.json`. The folder is meant to be
self-contained: opening it should be enough to understand what the run consumed, how the data
became examples, how the model was selected, how it scored on held-out participants and where
it failed.

Nothing here can fail a run. Each figure is attempted independently; a failure is logged and
recorded in the index with its reason, and the rest are still produced.

| # | Figure | What it shows |
|---|---|---|
| 00 | `run_scorecard` | Identity, provenance, protocol and headline participant-level results on one page |
| 01 | `dataset_inventory` | Analysable windows per participant and class, and the class balance |
| 02 | `window_inventory` | Participant × class counts and row shares — which folds are thin |
| 03 | `segmentation_timeline` | Trigger runs, transition guards, startup guard and the emitted overlapping windows, on a real recording |
| 04 | `filter_response` | Every filter stage and the chain as applied, including the forward+reverse zero-phase pass |
| 05 | `preprocessing_stages` | Raw → notch → bandpass on a real excerpt, with the spectrum at each stage |
| 06 | `class_spectra` | Mean EMG and microphone spectra per class |
| 07 | `example_windows` | A representative 1 s model input per class over a percentile band |
| 08 | `wavelet_bands` | The named coefficient bands each branch consumes, per class |
| 09 | `wavelet_band_energy` | Share of window energy per band and class, marking the bands the network receives |
| 10 | `augmentation_examples` | What the training augmenter actually does to a minority-class window |
| 11 | `training_curves` | Per-epoch refit loss and objective, one column per seed |
| 12 | `hyperparameter_selection` | What the inner search scored, what it selected, and the epoch budget |
| 13–16 | `confusion_matrix`, `roc_curves`, `pr_curves`, `per_participant` | Held-out performance |
| 17 | `per_class_performance` | Precision / recall / F1 / AUC / average precision per class |
| 18 | `participant_class_recall` | Recall in every (participant, class) cell — who failed, on what |
| 19 | `calibration` | Reliability diagram and expected calibration error |
| 20 | `seed_stability` | How much each result moves when only the seed changes |
| 21 | `error_timeline` | Held-out predictions in recording time — do errors cluster at onsets? |
| 22 | `embedding_tsne` | One t-SNE projection coloured twice: by class, **and by participant** |
| 23 | `modality_comparison` | Fusion vs EMG-only vs audio-only (ablation runs only) |

Figures 01–10 and 22 need the raw data root; without it they are skipped with a recorded
reason and the ledger-derived figures are still produced. Figures 20 and 23 are skipped for a
single-seed or single-modality run, which is why a typical five-class run reports 23 written
and 1 skipped.

**Customising what a run draws.** The trainer takes `--no-figures`; everything else is a knob
on `bruxism-figures`, which rebuilds the same set from a finished run bundle without
retraining a thing:

```bash
# rebuild one run's figures (also the way to add figures to a run trained before they existed)
bruxism-figures --run-dir outputs/runs/<run_id> --data-root ../Data

# several runs at once
bruxism-figures --run-dir outputs/runs/<a> --run-dir outputs/runs/<b> --data-root ../Data

# cheaper / faster variants
bruxism-figures --run-dir outputs/runs/<id> --no-tsne             # skip the projection
bruxism-figures --run-dir outputs/runs/<id> --no-signal-figures   # ledger-derived only
bruxism-figures --run-dir outputs/runs/<id> --data-root ../Data \
                --max-windows-per-class 400 --tsne-max-samples 6000 --tsne-seed 1
```

| Knob | Effect | Default |
|---|---|---|
| `--max-windows-per-class` | Windows sampled for the spectral and wavelet figures | 150 |
| `--tsne-max-samples` | Held-out embeddings entering the projection | 3000 |
| `--tsne-seed` | Fixed seed for the projection | 0 |
| `--no-signal-figures` | Skip everything that reads raw recordings | off |
| `--no-tsne` | Skip the embedding projection | off |

Sampling is deterministic (evenly spaced, no RNG), so regenerating a figure from the same run
bundle reproduces it.

Figure generation is deliberately **not part of the configuration hash**: it produces no
number a result depends on, so turning it on or off cannot change what a run computes, and
adding it to an existing project does not invalidate a resumable run.

The manuscript bundle (`bruxism-report`) is a different artifact with a different job: it
pools *across* runs to build the paper's tables, LaTeX macros and narrative. Use the run
folder to understand one run; use the bundle to write the paper.

---

## Regenerating the manuscript bundle

For one run's own figures see [What a finished run draws](#what-a-finished-run-draws) above;
this section is about the cross-run bundle the manuscript is written from.

`bruxism-report` rebuilds the **entire** manuscript bundle from saved artifacts. It never
re-runs a model and never re-reads raw data (except for the t-SNE, which must recompute
held-out embeddings from a checkpoint and is skipped with a recorded reason if
`--data-root` is absent).

```bash
bruxism-report --runs-root outputs/runs --output-root outputs/paper_bundle \
               --data-root ../Data
```

produces:

```
outputs/paper_bundle/
├── figures/
│   ├── <task>_confusion_matrix.png/.pdf      raw + row-normalised
│   ├── <task>_roc_curves.png/.pdf            one-vs-rest ROC, per-class AUC
│   ├── <task>_pr_curves.png/.pdf             precision-recall, average precision
│   ├── <task>_per_participant.png/.pdf       PRIMARY figure: one bar per participant
│   ├── <task>_tsne.png + _settings.json      exploratory, held-out embeddings only
│   ├── training_curves.png/.pdf              loss + macro-F1; excludes the outer subject
│   ├── modality_comparison.png/.pdf          fusion vs EMG-only vs audio-only
│   └── sample_flow.png/.pdf                  windows per participant and class
├── tables/
│   ├── condition_comparison.csv/.tex
│   ├── <task>_per_class.csv/.tex
│   ├── selection_summary.csv/.tex
│   ├── benchmark.csv
│   └── macros.tex                            \newcommand per reported value
├── predictions.parquet                       the concatenated ledger
├── metrics.json                              recomputed from the ledger
├── manuscript_asset_map.json                 Main_2.tex placeholder -> artifact
└── paper_results.md                          narrative; every number cites its artifact
```

`tables/macros.tex` lets the manuscript cite `\bruxMacroFOneMean` instead of a literal
number, so the text cannot drift from the artifacts.

Useful flags: `--run-id <id>` (repeatable) to select runs, `--primary-task <id>` to change
which task drives the headline figures, `--no-tsne` to skip the slow projection,
`--tsne-seed` to fix its seed.

**Selection rule for figures:** the fusion condition of `dual_branch_wavelet_cnn` at the
lowest seed — a fixed rule applied before looking at results, not a choice of the best.

---

## Repository layout

```
Code/
├── pyproject.toml                 packaging, dependencies, ruff/mypy/pytest config
├── configs/
│   ├── data/                      segmentation policies
│   ├── experiments/               smoke, five-class, ablations, baselines, secondary
│   └── models/                    architecture defaults
├── src/bruxism/
│   ├── config.py                  typed configuration; unknown keys raise
│   ├── runner.py                  config in -> immutable run bundle out
│   ├── reporting.py               audit and results narratives
│   ├── data/                      schema, manifest, labels, segments, dataset, splits, quality
│   ├── preprocessing/             filters, normalization, wavelets, augmentation
│   ├── features/                  time_frequency
│   ├── models/                    dual_branch, dwt, ablations, baselines
│   ├── training/                  engine, losses, selection
│   ├── evaluation/                metrics, aggregation, benchmark
│   ├── visualization/             paper_figures, signal_figures, diagnostics, run_figures
│   ├── utils/                     reproducibility, io, logging
│   └── cli/                       one module per console script
├── scripts/{data,train,evaluate}/ thin entry points
├── tests/{unit,integration,fixtures}/  synthetic data only
├── docs/                          protocol, data dictionary, reproducibility, crosswalk, open questions
├── legacy/                        the preserved prototype (do not use for results)
└── outputs/                       generated; git-ignored
```

Raw data stays in the sibling `Data/` directory and is supplied through `--data-root`.

---

## How the evaluation avoids leaking

The primary evaluation is **outer leave-one-subject-out** over all five participants. Within
each outer fold, participant-grouped **inner LOSO with four folds** (four training
participants remain — five is arithmetically impossible) selects hyperparameters and the
epoch budget.

The held-out participant is sealed structurally, not by convention:

```python
fold.release_test_ids(purpose="hyperparameter_search")   # OuterFoldSealError
fold.release_test_ids(purpose="early_stopping")          # OuterFoldSealError
fold.release_test_ids(purpose="final_evaluation")        # ok, once
fold.release_test_ids(purpose="final_evaluation")        # OuterFoldSealError (already used)
```

Also enforced: normalisation, class weights and the augmentation minority set are fitted on
training participants only and asserted against the held-out participant before evaluation;
augmentation raises for any non-training stage; no window-level K-fold splitter exists
anywhere in the API.

Full protocol: **[`docs/experiment_protocol.md`](docs/experiment_protocol.md)**.

---

## Privacy

`Data/` holds identifiable video, participant photographs, health-related surveys, receipts
and reimbursement spreadsheets.

- Raw data is **never** copied into this repository or into any generated artifact.
- Videos are inventoried (duration, frame rate, resolution, codec) but never decoded.
- Surveys, photographs, receipts and reimbursement files are never opened; their directories
  are excluded from scanning.
- Manifests use canonical IDs (`S01`) and data-root-relative paths — never names, never
  absolute paths.
- `bruxism-audit` asserts no private token leaks into its outputs and fails if one does; the
  same check runs in the test suite.
- The MIT licence covers the **software only**. It does not license the participant data.

---

## Known limitations

- **Five participants.** No population-level claim is supportable.
- **Instructed, not spontaneous.** Voluntary on-command grinding is plausibly easier to
  detect than spontaneous behaviour — a bias in the optimistic direction.
- **Rest is small.** ~117 windows per participant, from one dedicated recording each, versus
  real deployment where rest dominates.
- **Chewing dominates.** Under the trigger-constrained protocol chewing is ~59 % of windows,
  because chewing bouts are long while clench and movement repetitions are short. This is why
  the no-chewing task exists and why the audio contribution must be reported both with and
  without chewing.
- **Uneven trigger granularity.** Participants marked the trigger very differently, so
  per-participant class counts are unbalanced; the guard width that controls this needs
  investigator sign-off (`open_questions.md` Q1b).
- **Units unknown.** Everything is `arbitrary_adc_units`.
- **Acausal filtering.** No real-time claim is supported.
- **The historical 85 % result is not reproducible.** See below.

---

## The historical 85 % result

The whole-recording legacy policy reproduces the prototype's window counts exactly — 11,845
across all five families. But the published four-class confusion matrix *also* totals 11,845
while having **no rest class**, and none of its per-class supports matches:

| Class | Reproduced | Published | Match |
|---|---:|---:|:--:|
| movement | 1,785 | 1,877 | ✗ |
| clench | 2,380 | 2,503 | ✗ |
| instructed grinding | 1,769 | 1,861 | ✗ |
| chewing | 5,316 | 5,604 | ✗ |
| rest | 595 | *(absent)* | — |
| **total** | **11,845** | **11,845** | ✓ |

The published matrix is **not reproducible** from these recordings under any labelling
policy implemented here. It is documented as irreproducible and must not be reused.
Details: **[`docs/legacy_crosswalk.md`](docs/legacy_crosswalk.md)** §5.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/experiment_protocol.md`](docs/experiment_protocol.md) | The prespecification: labelling, splits, selection rules, metrics |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Schema, units, taxonomy, manifest columns, privacy classification |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Run bundles, hashes, seeding, and what still limits exactness |
| [`docs/legacy_crosswalk.md`](docs/legacy_crosswalk.md) | Every prototype symbol → replacement; 15 corrected defects |
| [`docs/open_questions.md`](docs/open_questions.md) | What is blocked, and on whose decision |

The original prototype README and flowchart are preserved at
`legacy/README_prototype.md` and `legacy/flowchart_prototype.png`. **The flowchart does not
depict the current dual-branch system** and must be redrawn before reuse.

---

## Licence

MIT (see `LICENSE`) — **software only**. The participant data is not licensed by this
repository.
