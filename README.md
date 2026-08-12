# Audio–EMG classification of instructed jaw activities

Proof-of-concept study and reproducible analysis pipeline for classifying **instructed, awake** jaw and tooth-contact tasks from surface EMG and a near-TMJ microphone. Five adults with a prior clinical diagnosis of bruxism completed a single controlled laboratory session.

Manuscript working title: *Audio–EMG Fusion with Dual-Branch Wavelet CNNs for Classifying Instructed Jaw Activities Relevant to Tooth-Contact Bruxism*.

> **Scope.** This is not a clinical bruxism detector, a sleep study, or a validated ambulatory/real-time system. Labels describe experimental task conditions (rest, jaw movement, clench, grinding, chewing), not independently adjudicated bruxism episodes. See [`Code/README.md`](Code/README.md) for the full “what this is / is not” table.

---

## Repository layout

```text
Bruxism/
├── Code/          Reproducible Python package, configs, tests, and docs
├── Data/          Raw recordings (local only — not in Git)
├── Paper/         Manuscript sources, figures, and review materials
├── REPORT.md      Implementation report for the Code/ rebuild
├── sol.md         Original implementation brief that drove the rebuild
└── README.md      This file
```

| Path | What it contains |
|---|---|
| **[`Code/`](Code/)** | Installable `bruxism` package: data manifest & labelling, preprocessing, dual-branch wavelet CNN and baselines, nested LOSO training, evaluation, figure/table generation, CLI, and ~196 tests on synthetic fixtures. Detailed quick start lives in [`Code/README.md`](Code/README.md). |
| **`Data/`** | Per-participant CSV/AVI/metadata recordings, surveys, and admin files. Kept on disk for local analysis; **excluded from Git** (see root `.gitignore`). Point the software at it with `BRUXISM_DATA_ROOT` or `--data-root`. |
| **[`Paper/`](Paper/)** | LaTeX manuscript (`K_Farhadi_Paper_Bruxism/`, primary draft `Main_2.tex`), figures, author photos, and `Reviews/` correspondence. |
| **[`REPORT.md`](REPORT.md)** | Engineering status of the rebuild: what was fixed, what the smoke tests cover, and which scientific claims remain blocked. |
| **[`sol.md`](sol.md)** | Full implementation brief (protocol, defects in the prototype, required analyses). Historical context for why `Code/` looks the way it does. |

A small nested `Bruxism/` folder holds only the GitHub template `LICENSE` / `README` stubs; the real software licence is [`Code/LICENSE`](Code/LICENSE) (MIT, **software only**).

---

## Study at a glance

| | |
|---|---|
| **Participants** | 5 adults (prior clinical bruxism diagnosis) |
| **Setting** | Awake, instructed laboratory tasks — not spontaneous or sleep recordings |
| **Modalities** | Bipolar surface EMG (masseter / temporalis) + a near-TMJ microphone channel (1200 Hz) — **the microphone channel of this collection is defective, see [`audio.md`](audio.md)** |
| **Primary task** | Five-class: quiet rest, jaw movement, clench, grinding, chewing |
| **Evaluation** | Outer leave-one-subject-out; participant-grouped inner folds for selection |
| **Model** | Dual-branch wavelet CNN (EMG + audio fusion), plus modality and architecture baselines |

Secondary analyses (modality ablations, no-chewing sensitivity, binary/ternary endpoints) are configured under `Code/configs/experiments/`.

---

## Getting started (software)

All runnable work lives under `Code/`. From the repo root:

```bash
cd Code
python -m pip install -e ".[dev,video]"

# Quality gates — no private data required
python -m pytest -m "not slow"

# Point at your local data root (never committed)
export BRUXISM_DATA_ROOT=../Data

bruxism-audit --data-root "$BRUXISM_DATA_ROOT" --output-root outputs/data_audit
bruxism-train --config configs/experiments/smoke.yaml --max-folds 1
```

`bruxism-audit` also reports the signal-integrity checks — mains interference in the EMG and the microphone-channel defect. See [**Auditing the signals**](#auditing-the-signals) for what it produces and how to read it.

Full command tables, figure regeneration, leakage controls, and limitations: **[`Code/README.md`](Code/README.md)**.

Protocol and reproducibility docs:

| Document | Role |
|---|---|
| [`Code/docs/experiment_protocol.md`](Code/docs/experiment_protocol.md) | Prespecified labelling, splits, selection, metrics |
| [`Code/docs/data_dictionary.md`](Code/docs/data_dictionary.md) | Schema, taxonomy, privacy classification |
| [`Code/docs/reproducibility.md`](Code/docs/reproducibility.md) | Run bundles, hashes, seeding |
| [`Code/docs/open_questions.md`](Code/docs/open_questions.md) | Claims blocked on human decisions (IRB, hardware, etc.) |
| [`Code/docs/legacy_crosswalk.md`](Code/docs/legacy_crosswalk.md) | Prototype → current mapping; irreproducible historical 85% result |
| [`Code/docs/audio_collection_spec.md`](Code/docs/audio_collection_spec.md) | Normative audio requirements for the next collection, each traced to a specific failure |
| [`audio.md`](audio.md) / [`audio_report.md`](audio_report.md) | The microphone-channel audit, and what was changed in response |
| [`cause.md`](cause.md) | The mains-harmonic defect, its correction, and §11 on the same error in the other modality |

---

## Reproducing the manuscript

Every number in `Paper/K_Farhadi_Paper_Bruxism/Main_2.tex` comes from one of two run bundles:

| | Primary five-class run | RQ2 modality ablation |
|---|---|---|
| **Run** | `five_class_nested_loso_20260807T211827_2b6fb5ac` | `modality_and_no_chewing_20260810T020642_cead62e4` |
| **Protocol** | 3 seeds × 5 outer folds, 18,519 held-out predictions | 6 conditions × 3 seeds × 5 outer folds, 78,399 held-out predictions |
| **Config hash** | `2b6fb5ac` | `cead62e4` |
| **Supplies** | headline five-class, per-class, per-participant, secondary analyses | audio-only / EMG-only / fusion on the five-class and no-chewing tasks |

Both share manifest hash `7ebbcc8d` and window-index hash `b2cf690c`, so every condition in the paper was evaluated on the identical set of windows. The ablation fixes the loss and learning rate instead of selecting them per fold, so only the modality varies between its conditions; that is why its fused condition scores slightly below the primary run's and why the paper computes every modality difference inside the ablation.

### Regenerate the manuscript figures

Seven of the manuscript's twelve figures depend on a training run — six on the primary run and one on the ablation. Rebuild them whenever the run they depict changes; the other five depend only on the data and the filter chain and are left alone.

```bash
cd Code

# seven figures from the primary run
python scripts/evaluate/make_manuscript_figures.py \
    --run-dir outputs/runs/five_class_nested_loso_20260807T211827_2b6fb5ac \
    --data-root ../Data

# the modality-ablation figure (Fig. 7); no --data-root needed
python scripts/evaluate/make_ablation_figure.py \
    --run-dir outputs/runs/modality_and_no_chewing_20260810T020642_cead62e4
```

Everything but the t-SNE is read from `predictions.parquet` and `selection/fold_outcomes.json`, so a figure cannot disagree with the ledger it depicts. The t-SNE recomputes held-out embeddings from the saved fold checkpoints, which is the only reason `--data-root` is needed.

| Flag | Default | Use it to |
|---|---|---|
| `--output-dir` | `../Paper/K_Farhadi_Paper_Bruxism/Figures` | Write elsewhere (e.g. a scratch dir to preview before overwriting) |
| `--figure-seed` | lowest seed in the run | Change which single model the confusion / ROC / PR panels depict |
| `--no-tsne` | off | Skip the slow figure; everything else still runs without `--data-root` |
| `--tsne-max-samples` | `3000` | Denser or sparser projection |
| `--tsne-perplexity` | `30` | Retune the projection |
| `--task-id` | `five_class` | Point at another task in the ledger |

A `manuscript_figures_provenance.json` is written beside the figures recording the run id, hashes, seeds and which seed each panel used.

### Confirmatory experiments: what is done and what is left

Both were previously measured through the defective filter chain and were withdrawn rather than restated. `scripts/train/run_pending_experiments.sh` re-runs them on the corrected chain.

| RQ | Question | Status |
|---|---|---|
| **RQ2** | Does audio help, and only for chewing? | **Unanswerable from this data** — the run completed (`modality_and_no_chewing_20260810T020642_cead62e4`, `Main_2.tex` §4.3 and Table 4) and its numbers stand as a record of the experiment, but a later audit found the microphone channel is not per-participant audio: 37 distinct waveforms across 100 recordings, 83 of them another participant's, rotated. LOSO never held out the audio. See [`audio.md`](audio.md) and `Main_2.tex` §4.2. The EMG results are unaffected. |
| **RQ3** | Does the dual-branch CNN beat the alternatives? | **Pending** — the architecture sweep has not been run. Table 3 carries explicit `pending re-measurement` rows. |

```bash
cd Code
./scripts/train/run_pending_experiments.sh check       # validate both configs, ~15 s, trains nothing
./scripts/train/run_pending_experiments.sh baselines   # RQ3 — the one still outstanding
./scripts/train/run_pending_experiments.sh ablations   # RQ2 — starts a NEW run, see below
./scripts/train/run_pending_experiments.sh             # both, sequentially (default)
```

Roughly 6–10 h on a single modern GPU, so detach it:

```bash
mkdir -p outputs/logs
nohup ./scripts/train/run_pending_experiments.sh baselines \
    > outputs/logs/baselines_nohup.log 2>&1 &
tail -f outputs/logs/baselines_nohup.log
```

Runs are **resumable**. Re-running the same command picks up at the first fold with no saved prediction file; nothing already computed is repeated. That works because the script pins the run ids rather than letting them be timestamped, so pin `RUN_TAG` if a job might cross midnight.

| Variable | Default | Effect |
|---|---|---|
| `DATA_ROOT` | `../Data` | Where the recordings live |
| `RUN_TAG` | today, `YYYYMMDD` | Suffix of the run ids — `baselines_<tag>`, `modality_and_no_chewing_<tag>`. Keep it identical across attempts to resume. |

```bash
DATA_ROOT=/mnt/recordings RUN_TAG=20260811 \
    ./scripts/train/run_pending_experiments.sh baselines
```

> The completed RQ2 run was launched through `bruxism-ablations` directly, so its id carries the auto-generated `<timestamp>_<config hash>` suffix rather than a `RUN_TAG`. Re-running `ablations` through the script starts a **new** run rather than resuming that one.

After training, the script writes one paper bundle per run under `outputs/paper_bundle/<run id>/`; read `paper_results.md` there for that run's conditions, recomputed from its saved ledger. Bundles are per-run rather than pooled because several runs legitimately hold the same `five_class::dual_branch_wavelet_cnn::fusion` condition, and the ledger asserts each held-out window is predicted exactly once per configuration.

For a subset or a smoke test, call the CLIs directly — the script is a thin wrapper:

```bash
bruxism-baselines --config configs/experiments/baselines.yaml \
    --data-root ../Data --run-id baselines_20260811 --progress plain

bruxism-ablations --config configs/experiments/modality_and_no_chewing.yaml \
    --data-root ../Data --run-id modality_and_no_chewing_20260811 --progress plain
```

| Flag | Applies to | Use it to |
|---|---|---|
| `--modalities fusion emg_only audio_only` | ablations | Run a subset of the modality conditions |
| `--tasks five_class no_chewing_four_class` | ablations | Run a subset of the tasks |
| `--models bilstm early_fusion_cnn …` | baselines | Run a subset of the architectures |
| `--max-folds N` | both | Stop after N folds — the fastest end-to-end check |
| `--validate-only` | both | Build the fold plan and exit without training |
| `--no-resume` | both | Recompute folds that already have predictions |
| `--no-figures` | both | Skip the per-run figure set |
| `--set KEY=VALUE` | both | Override any resolved-config field, e.g. `--set training.seeds=[0]` |

> **These are confirmatory runs.** Do not re-run one after seeing its score, and do not edit a config between attempts. A changed config is a new experiment with a new name, and the superseded result is reported as superseded.

---

## Auditing the signals

Two measurement-chain defects have been found in this dataset, both after results had already been reported from the affected channel. The first was mains harmonics in the EMG ([`cause.md`](cause.md)); the second is the microphone channel ([`audio.md`](audio.md)). Both are now measured on every manifest build rather than assumed absent, so the checks below are the ones that would have caught them.

### The one command

```bash
cd Code
bruxism-audit --data-root ../Data --no-video --no-figures --no-onset-alignment
```

**~11 s.** Writes `outputs/data_audit/<manifest_hash>/`. Two artifacts carry the microphone audit:

| Artifact | What it tells you |
|---|---|
| `data_audit.md` §7c *Microphone integrity* | The whole finding on one page: distinct waveforms per channel, recordings affected per participant, quantisation step, sub-10 Hz power share, retained variance, EMG alignment, and which flags fired on how many recordings. |
| `mic_integrity.csv` | One row per recording. The load-bearing column is `mic_duplicate_group`: it names **which other recordings share this one's audio**. |

The channel-identity table is the part to read first, because it carries its own control — the EMG columns are what a clean channel looks like:

| channel | distinct waveforms | of recordings | cross-subject groups |
|---|---|---|---|
| **mic** | **37** | 100 | **20** |
| emg1–emg4 | 100 each | 100 | 0 |
| trigger | 95 | 100 | 1 (informational — see below) |

83 of 100 recordings carry a microphone waveform that is bit-identical, after a circular rotation, to **another participant's** recording of the same condition. Leave-one-subject-out therefore never held out the audio. The trigger row is reported but never flagged: a binary channel's fingerprint collides on duty cycle alone, so a shared all-zero rest trigger means nothing.

### Guards that run themselves

You do not invoke these — they fire inside `bruxism-train` / `bruxism-ablations`, before the run directory is created, so a refused run leaves no half-written bundle.

| Guard | Refuses | Override |
|---|---|---|
| `assert_modality_is_supported_by_data` | a `fusion` / `audio_only` run on flagged microphone data | `mic_defect_acknowledged_by` in the experiment config |
| `assert_bands_are_inside_their_passband` | a branch reading a wavelet band its own filter chain deleted (mic `A5` retains 2.95 % behind the 20 Hz high-pass) | `stopband_bands_acknowledged_by` |

All seven shipped configs declare both, so the published runs still reproduce; a **new** config will not, which is the point. Neither key enters the configuration hash — an acknowledgement records who authorised a run, not what it computes — so `2b6fb5ac` and `cead62e4` are unchanged. To watch a guard fire:

```bash
python -c "
from bruxism.config import ExperimentConfig
from bruxism.data.manifest import build_manifest
from bruxism.runner import assert_modality_is_supported_by_data
assert_modality_is_supported_by_data(ExperimentConfig(name='x', modality='fusion'),
                                     build_manifest('../Data', probe_video=False))"
```

### The regression test

```bash
pytest tests/unit/test_leakage.py tests/unit/test_signal_processing.py -q   # ~30 s
```

`test_no_measured_channel_waveform_is_shared_across_subjects` asserts that no EMG or microphone waveform appears under two participants. It runs on synthetic data, so it needs no data root and belongs in CI. A companion test plants a rotated duplicate and requires the check to **fail** — a leakage test that cannot fail is decoration.

### The cheapest diagnostic

Each activity's RMS divided by *that participant's own* rest RMS, for both modalities on identical windows:

```bash
python - <<'PY'
from bruxism.data.manifest import build_manifest
from bruxism.data.dataset import RecordingCache
from bruxism.data.segments import build_window_index, SegmentationConfig
from bruxism.preprocessing.filters import FilterChainConfig
from bruxism.evaluation.signal_quality import both_modalities_quality_table, contrast_table
m = build_manifest("../Data", probe_video=False)
cache = RecordingCache(m, FilterChainConfig(), cache_dir="outputs/cache")
q = both_modalities_quality_table(build_window_index(m, SegmentationConfig()), cache,
                                  max_windows_per_cell=20)
for mod in ("emg", "mic"):
    print(f"\n{mod.upper()}"); print(contrast_table(q, modality=mod).round(2).to_string(index=False))
PY
```

EMG puts clenching at 9.6–27.9× that participant's rest. The microphone puts it at **0.99–1.36×** — S01's clenching is *quieter* than S01's rest. An acoustic channel cannot do that, and the two lines side by side are the fastest way to see it.

> **Known gap.** The EMG-only screening comparison cited in the manuscript (67.9 % → 63.9 % macro-F1 once the seven microphone features are dropped) was run ad hoc. Its output is saved at `Code/outputs/screening/emg_only_contrast.json`, but no committed script regenerates it.

Full write-up and reproduction snippets: [`audio.md`](audio.md). What changed and why: [`audio_report.md`](audio_report.md). Requirements for a future collection: [`Code/docs/audio_collection_spec.md`](Code/docs/audio_collection_spec.md).

---

## Data and privacy

`Data/` may contain identifiable video, photographs, health-related surveys, and reimbursement material.

- It is **not** tracked in this repository.
- Analysis code reads it only via `--data-root` / `$BRUXISM_DATA_ROOT`.
- Manifests use canonical IDs (`S01`…) and data-root-relative paths.
- The MIT licence covers the software only; it does **not** license participant data.

---

## Licence

Software under `Code/` is MIT. Participant data and paper review materials are not covered by that licence.
