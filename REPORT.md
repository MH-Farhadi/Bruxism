# Implementation report — `sol.md` rebuild

**Date:** 2026-07-27 / 28
**Scope worked:** `Code/` only. `Data/` was read read-only; `Paper/` was read but **not edited**.
**Starting state:** `Code/` clean at commit `a857f9b`, tagged **`pre-refactor-audit`**.
**Working branch:** `refactor/reproducible-pipeline`, HEAD **`1f08209`**, 13 commits, clean tree.

---

## 0. Headline

The engineering is complete and validated end to end. **The definitive scientific
experiments were deliberately not run** — you asked for a smoke test and for runnable
training scripts, and that is what was delivered. Every required analysis is one command
away.

Three things worth reading even if you read nothing else:

1. **The published 85 % result is not reproducible.** The legacy policy reproduces the
   11,845-window total *exactly*, but the published four-class matrix has that same total
   while containing no rest class, and **none** of its four per-class supports matches.
   (§5)
2. **The prototype's held-out evaluation was invalid.** Subject 5 was used as the
   per-epoch validation set for early stopping and checkpoint selection, and then reported
   as the test result. (§4.1)
3. **The trigger-constrained protocol makes chewing *more* dominant, not less** — 59 % of
   windows versus 45 % under whole-recording labelling — because chewing bouts are long
   while clench and movement repetitions are short. This runs against the revision's goal
   and makes the no-chewing analysis more important, not less. (§3.3)

---

## 1. Change summary

### Data

Nothing in `Data/` was modified, moved, copied or deleted. A deterministic scanner now
accounts for **all 100 CSV + 100 AVI + 100 metadata files**, including the two in
`Data/More Data/Data/Subject_5/`. Companions are resolved *across* directories, so S05's
protrusion metadata — which sits in `More Data` while its CSV is in the primary folder — is
found without moving anything.

| Anomaly | Count | Resolution |
|---|---:|---|
| Startup transient | 66 | Measured, not guessed: all settle within **0.40 s**, peaks to **12,580×** the robust scale. Justifies a 0.5 s guard, replacing the prototype's unexplained 3 s skip. |
| Metadata claims a missing `.npy` | 97 | Harmless; CSV is the only input. |
| Stale `.npy` present | 3 | All S01; sixth column is zeros, not the microphone. Never read (rule `R2`). |
| Short recording | 2 | S02 grinding 52.4 s, S05 cheese 40.6 s, against a declared 60 s target. Retained and reported. |
| Secondary location | 2 | S05 rest triple and S05 protrusion metadata. Paths recorded; files never moved. |
| Filename/metadata conflict | 1 | S05 molar-clench metadata says `incisor_clench`. **Filename wins** (rule `R1`) — S05 has a *separate* incisor recording whose metadata says the same thing, so trusting metadata would give two incisor recordings and no molar. |

**New finding not in the brief:** the microphone is very coarsely quantised — integer-valued
with a step of 1 and only **15–145 distinct levels per recording** (≈ 6.7 effective bits,
global range 50–227). This bounds what the audio branch can contribute and belongs in the
Methods.

### Code architecture

`Code/` went from 11 flat scripts (5,869 lines, all preserved) to an installable package:
**11,087 lines** across `src/bruxism/` in nine sub-packages, **2,025 lines** of tests,
**216 lines** of thin script wrappers, **1,457 lines** of documentation.

Nine console commands, all with `--help`, `--data-root`/`$BRUXISM_DATA_ROOT`, and
`--validate-only` where meaningful. **No source file contains a machine-specific path**
(the prototype hard-coded `C:\Users\mhfar\...` in two files).

### Scientific protocol

Prespecified in `Code/docs/experiment_protocol.md` and enforced in code:

- Outer LOSO over all five participants; **four** participant-grouped inner folds (five
  raises — it is arithmetically impossible with four training participants).
- The held-out participant is sealed *structurally*: `release_test_ids()` raises unless
  called with `purpose="final_evaluation"`, and raises again on a second call.
- Normalisation, class weights and the augmentation minority set are fitted on training
  participants only, recorded, and asserted against the held-out participant.
- Augmentation raises for any stage other than `"train"`.
- Selection objective (macro-F1), tie-break, epoch-budget rule and multi-seed rule all
  declared before the run. No seed is ever selected as best.
- Participant-level metrics primary; pooled-window metrics carry
  `interpretation="descriptive_only"`.

### Tests

**196 tests**, all on synthetic fixtures — the suite needs **no access to `Data/`**.
Several tests demonstrate the original defects rather than only asserting the fix.

### Documentation

`experiment_protocol.md`, `data_dictionary.md`, `reproducibility.md`,
`legacy_crosswalk.md`, `open_questions.md`, `claim_to_evidence.md`, plus a rewritten
`README.md`. The old README (which claimed real-time natural-bruxism detection and a
parameter count wrong by 2×) is preserved at `legacy/README_prototype.md`.

---

## 2. Final directory tree

```
Code/
├── pyproject.toml  requirements.lock.txt  README.md  LICENSE  .gitignore
├── .github/workflows/ci.yml
├── configs/
│   ├── data/         trigger_constrained.yaml · whole_recording_legacy.yaml
│   ├── experiments/  smoke · five_class_nested_loso · modality_and_no_chewing
│   │                 · baselines · secondary_tasks
│   └── models/       dual_branch.yaml
├── src/bruxism/
│   ├── __init__.py  config.py  runner.py  reporting.py  py.typed
│   ├── data/         schema · manifest · labels · segments · dataset · splits · quality
│   ├── preprocessing/ filters · normalization · wavelets · augmentation
│   ├── features/     time_frequency
│   ├── models/       __init__ (BruxismModel Protocol) · dual_branch · dwt · ablations · baselines
│   ├── training/     engine · losses · selection
│   ├── evaluation/   metrics · aggregation · benchmark
│   ├── visualization/ paper_figures
│   ├── utils/        reproducibility · io · logging
│   └── cli/          _common · audit_dataset · build_manifest · plot_preprocessing
│                     · run_nested_loso · run_ablations · run_baselines
│                     · summarize_runs · benchmark_models · make_paper_artifacts
├── scripts/{data,train,evaluate}/    9 thin entry points
├── tests/
│   ├── fixtures/synthetic.py         generates a 5-participant synthetic dataset
│   ├── unit/      schema · manifest · segments_and_labels · leakage
│   │              · signal_processing · models_and_training · metrics
│   └── integration/test_end_to_end.py
├── docs/          experiment_protocol · data_dictionary · reproducibility
│                  · legacy_crosswalk · open_questions · claim_to_evidence
├── legacy/        all 11 prototype modules + old README + flowchart + both requirements files
└── outputs/       git-ignored; generated bundles
```

---

## 3. Commands executed and outcomes

| Command | Outcome |
|---|---|
| `ruff format --check src scripts tests` | **pass** — 72 files |
| `ruff check src scripts tests` | **pass** |
| `mypy` | **pass** — 49 source files, 0 issues |
| `pytest tests/unit` | **184 passed** (11 s) |
| `pytest tests/integration` | **12 passed** |
| `pytest` (full suite) | **196 passed** (4 m 59 s) |
| `bruxism-audit --data-root ../Data` | 100 recordings, 100 included, manifest `46aae2d6394de668` |
| `bruxism-build-manifest --config configs/data/trigger_constrained.yaml` | 6,173 windows, index `54593b12d44ffabe` |
| `bruxism-plot-preprocessing` | 5 raw-vs-filtered figures, one per task family |
| `bruxism-train --config configs/experiments/smoke.yaml` | **5/5 folds, 6,173 predictions, 2 m 11 s** |
| `bruxism-benchmark` | 3 models benchmarked |
| `bruxism-report` | 7 figures, 5 CSV + 4 TeX tables, `paper_results.md` |

### 3.1 Window counts under the approved protocol

`trigger_constrained`, 1.0 s window / 0.5 s stride / 0.25 s guard / 0.5 s startup guard:

| Participant | chewing | clench | grinding | movement | rest | total |
|---|---:|---:|---:|---:|---:|---:|
| S01 | 787 | 376 | 316 | 95 | 118 | 1,692 |
| S02 | 754 | 53 | 71 | 11 | 118 | 1,007 |
| S03 | 741 | 57 | 122 | 95 | 118 | 1,133 |
| S04 | 743 | 138 | 97 | 71 | 118 | 1,167 |
| S05 | 610 | 175 | 210 | 61 | 118 | 1,174 |
| **all** | **3,635** | **799** | **816** | **333** | **590** | **6,173** |

### 3.2 The guard-width decision (needs your sign-off)

`sol.md` suggested starting at 0.5 s "only if approved". It is not approvable as-is:

| guard (s) | total | smallest participant×class cell | cells < 10 |
|---:|---:|---:|---:|
| 0.000 | 6,779 | 50 | 0 |
| 0.125 | 6,458 | 34 | 0 |
| **0.250** | **6,173** | **11** | **0** |
| 0.375 | 5,863 | 2 | 1 |
| 0.500 | 5,610 | **1** | 2 |

At 0.5 s, S02 contributes **1** movement window and **3** clench windows — its held-out fold
cannot evaluate those classes at all. Root cause: participants used the trigger at very
different granularities (only 5 % of S02's clench runs reach 2.0 s, versus 98 % of S04's).

**Shipped default: 0.25 s** — the widest guard leaving no degenerate cell. One line in
`configs/data/trigger_constrained.yaml`. Logged as `open_questions.md` **Q1b**.

### 3.3 Chewing dominance moves the wrong way

| Policy | chewing share |
|---|---:|
| `whole_recording_legacy` (historical) | 44.9 % |
| `trigger_constrained` (approved) | **58.9 %** |

Chewing bouts run 8–15 s; clench and movement repetitions run ~2 s. Constraining to
trigger-active intervals therefore *increases* chewing's share. This works against the
revision's goal of de-emphasising chewing, and makes the no-chewing task and the
chewing-excluded audio contrast the load-bearing analyses.

---

## 4. Defects found and corrected

Fifteen, catalogued with evidence in `Code/docs/legacy_crosswalk.md` §3. The five that
change conclusions:

### 4.1 The held-out subject drove training decisions ⛔

`run_new_wavelet_training.py` used subject 5 as the per-epoch validation set — for early
stopping *and* checkpoint selection — then reported subject 5 as the test result. The
reported number is a best-epoch-on-the-test-set figure. A deleted historical CV script
rotated the outer subject but still selected the epoch on it and reported the maximum.

### 4.2 Window-level K-fold ⛔

`run_random_forest_training.py` ran ordinary 5-fold CV over windows. At 0.5 s stride
adjacent windows overlap 50 %, and windows from the same recording and participant land in
different folds. That score cannot speak to generalisation across people. No K-fold
splitter now exists; a test asserts it.

### 4.3 Focal loss ⛔

```python
ce = cross_entropy(logits, targets, weight=alpha, reduction="none")
pt = torch.exp(-ce)          # not p_t once alpha != 1
```

The focusing term became a function of the class weights. Corrected to
`alpha_t · (1 − p_t)^γ · (−log p_t)` with `p_t` from the unweighted softmax; verified
against hand computation with and without alpha, and `γ=0` reduces exactly to weighted CE.

### 4.4 Wavelet bands were mislabelled ⛔

`pywt.wavedec` returns `[cA_L, cD_L, …, cD_1]`. The prototype's comment
`emg_detail_idx_high = 0  # highest frequency` names the **lowest** detail (D4, 37.5–75 Hz),
and its `details[2]` is **D2** (150–300 Hz) where the manuscript says "D3" (75–150 Hz).
Verified empirically — 50 Hz → 83 % in D4, 400 Hz → 92 % in D1. **Every band name in the
manuscript must be re-derived.**

### 4.5 Shallow checkpoint snapshot ⛔

`state_dict().copy()` is a shallow dict copy sharing live tensor storage, so the "best"
weights were whatever the model held at the end. A test demonstrates the original failure
directly.

Also corrected: the CWT block called `pywt.cwt` with a **discrete** wavelet, so it always
raised and its bare `except` emitted four zeros per channel — those features were always
zeros; the WPT declared depth 4 but read only two first-level nodes; "median frequency" was
not a spectral median; the 5 Hz high-pass after a 20 Hz bandpass was a no-op; ICA
reconstructed from all components (≈ identity); wavelets ran inside `forward()` forcing a
GPU→CPU→GPU round trip per batch.

**A latent bug of my own, caught by mypy:** the JSON log handler called
`self.formatException`, which lives on `Formatter`, not `Handler` — it would have raised
exactly when an exception was being logged. Fixed.

---

## 5. Legacy vs reproducible results

The legacy policy reproduces the prototype's window counts **exactly**:

| Class | Reproduced | Published matrix | Match |
|---|---:|---:|:--:|
| movement | 1,785 | 1,877 | ✗ |
| clench | 2,380 | 2,503 | ✗ |
| instructed grinding | 1,769 | 1,861 | ✗ |
| chewing | 5,316 | 5,604 | ✗ |
| rest | 595 | *(absent)* | — |
| **total** | **11,845** | **11,845** | ✓ |

The published four-class matrix totals exactly 11,845 — the count across **all five**
families **including 595 rest windows** — while having no rest class. No per-class support
matches.

**Conclusion: not reproducible under any labelling policy implemented here.** Documented as
irreproducible in `data_audit.json → historical_confusion_matrix_check`. The 85.0 %
accuracy derived from it must not be reused. No attempt was made to force agreement.

---

## 6. Proof the outer test data was not used for selection

Structural, not conventional:

```
release_test_ids(purpose="hyperparameter_search") → OuterFoldSealError
release_test_ids(purpose="early_stopping")        → OuterFoldSealError
release_test_ids(purpose="checkpoint_selection")  → OuterFoldSealError
release_test_ids(purpose="threshold_tuning")      → OuterFoldSealError
release_test_ids(purpose="final_evaluation")      → ok, once
release_test_ids(purpose="final_evaluation")      → OuterFoldSealError (already released)
```

From the executed smoke run (`outputs/runs/smoke/selection/fold_outcomes.json`):

| outer fold | held out | normalizer fitted on | inner val subjects | inner folds |
|---|---|---|---|---|
| 0 | S01 | S02,S03,S04,S05 | S02,S03,S04,S05 | 4 |
| 1 | S02 | S01,S03,S04,S05 | S01,S03,S04,S05 | 4 |
| 2 | S03 | S01,S02,S04,S05 | S01,S02,S04,S05 | 4 |
| 3 | S04 | S01,S02,S03,S05 | S01,S02,S03,S05 | 4 |
| 4 | S05 | S01,S02,S03,S04 | S01,S02,S03,S04 | 4 |

**Coverage:** 6,173 predictions for 6,173 windows, **0 duplicates**, one participant per
fold, each participant held out exactly once. Four inner folds everywhere.

---

## 7. Artifact paths

All under `Code/` (git-ignored; regenerable).

| Artifact | Path |
|---|---|
| Data audit bundle | `outputs/data_audit/46aae2d6394de668/` — `data_audit.md`, `.json`, `manifest.parquet/.csv`, `trigger_summary.csv`, `guard_sensitivity.csv`, `window_counts.csv`, `trigger_runs.csv`, `quality_figures/` |
| Analysis window index | `outputs/manifests/54593b12d44ffabe/` |
| Preprocessing examples | `outputs/preprocessing_examples/` — 5 raw-vs-filtered figures + `filter_chain.json` |
| Smoke run bundle | `outputs/runs/smoke/` — `predictions.parquet`, `metrics.json/.csv`, `folds.json`, `selection/`, `checkpoints/`, `logs/run.log.jsonl`, `resolved_config.yaml`, `environment.json`, `source_state.json` |
| Benchmarks | `outputs/benchmarks/benchmark.json` + `.csv` |
| Paper bundle | `outputs/paper_bundle/` — `figures/`, `tables/`, `predictions.parquet`, `metrics.json`, `manuscript_asset_map.json`, `paper_results.md` |

---

## 8. Benchmarks (measured, CPU, batch 1)

| model | trainable params | size (KiB, fp32) | forward (ms) | processing (ms) |
|---|---:|---:|---:|---:|
| **dual_branch_wavelet_cnn** | **7,485** | 29.2 | 2.88 | 2.97 |
| early_fusion_cnn | 19,141 | 74.8 | 0.20 | 0.29 |
| bilstm | 10,309 | 40.3 | 1.30 | 1.39 |

**The three latencies stay separate:**

| quantity | value |
|---|---|
| Input/context latency | **1000 ms** — the observation window; no decision can exist sooner |
| Decision update interval | **500 ms** — the stride |
| Processing latency | **~3 ms** — compute only |

The old README's "~15,000 parameters" is wrong by roughly 2×. Compute time is not detection
latency, and the acausal filter chain supports no real-time claim.

---

## 9. Claim-to-evidence summary

Full table (55 rows): `Code/docs/claim_to_evidence.md`.

| Status | Count |
|---|---:|
| supported | 18 |
| **contradicted** | **11** |
| blocked-human | 11 |
| blocked-compute | 13 |
| editorial | 2 |

**The eleven contradicted claims** — statements the manuscript currently makes that the
evidence does not support:

1. "two EMG channels" → four differential signals from two bilateral bipolar pairs
2. "3-minute trials repeated three times" → ~1 min per recording; metadata declares 60 s
3. the extra 5 Hz high-pass → a no-op, removed
4. wavelet band "D3" → it is D2 at level 4
5. "inner five-fold" → four, necessarily
6. "~15,000 parameters" → 7,485
7. "low-latency / real-time" → three separate latencies
8. "generalize to unseen individuals" → five participants cannot support this
9. "natural bruxing" → instructed grinding
10. 85.0 % four-class accuracy → **irreproducible**
11. total window count in Limitations → must match 6,173, not 11,845

---

## 10. Unresolved investigator questions and what they block

Full detail: `Code/docs/open_questions.md`.

**Resolved this session** — trigger semantics (Q1), channel montage (Q2), S05 metadata
conflict (Q7), rest definition (Q8).

**Still blocking:**

| ID | Question | Blocks |
|---|---|---|
| **Q1b** | Transition guard width — approve 0.25 s? | Methods transition sentence; **every sample count** |
| **Q3/Q9** | Hardware, gain, ADC, units; what `Index 143`/`Index 9` mean | Methods hardware paragraph; every axis label; whether offline filtering is doubling up |
| **Q4** | Recruitment evidence — provider-*indicated* grinding is not a formal diagnosis | Participant characterisation; the phenotype statement reviewers demanded |
| **Q5** | Actual duration and repetitions per condition | Protocol paragraph; total recording time |
| **Q10** | The correct IRB identifier among three conflicting values | Ethics statement |
| **Q11** | Which artifacts may be released | Data-availability statement |
| Q6 | Task naming (carrots/popcorn) | wording |
| Q12 | Venue; citation verification against DOIs (**not performed here**) | formatting, references |

---

## 11. What was deliberately not done

- **The definitive experiments.** You asked for a smoke test. `five_class_nested_loso`,
  `modality_and_no_chewing`, `baselines` and `secondary_tasks` are written, validated by
  `--validate-only`, and each is one command.
- **The manuscript.** `Paper/` was read, never edited.
- **Citation verification.** A separate scholarly task; no DOI was checked.
- **The architecture diagram.** `Figures/pipeline_5class.png` must be drawn by hand; the old
  `flowchart.png` does not depict this system.

The smoke run's numbers (subject-level macro-F1 0.31 ± 0.16) are **not a scientific result**
— one seed, ≤ 3 epochs, no hyperparameter search. They exist to prove the pipeline runs.

---

## 12. Reproduction record

| Item | Value |
|---|---|
| Starting commit | `a857f9b` (tagged `pre-refactor-audit`), clean tree |
| Final commit | `1f08209` on `refactor/reproducible-pipeline`, clean tree, 13 commits |
| Data manifest hash | `46aae2d6394de668` |
| Window index hash | `54593b12d44ffabe` |
| Smoke config hash | `8372b880ddaf7a62` |
| Run ID | `smoke` |
| Python | 3.12.12, Linux 6.17.0, glibc 2.42 |
| torch | 2.7.1+cu126, CUDA 12.6, RTX 4090 Laptop |
| Key libs | numpy 2.2.6 · scipy 1.18.0 · scikit-learn 1.9.0 · PyWavelets 1.9.0 · pandas 2.3.3 · pyarrow 22.0.0 · matplotlib 3.10.9 · opencv-headless 4.12.0 |
| Lock file | `Code/requirements.lock.txt` (64 pins; torch pinned separately by CUDA build) |

> The smoke run was launched from a dirty tree (26 files mid-refactor). That is recorded in
> `source_state.json` with the file list and a diff SHA-256, and a warning was logged. It is
> a smoke run, so this is acceptable; **launch the definitive runs from a clean tree.**

---

## 13. Next steps

**Yours, before definitive results:**

1. Approve or change the **0.25 s guard** (Q1b) — it changes every sample count.
2. Supply **hardware, units and acquisition filters** (Q3/Q9), the **IRB number** (Q10),
   **recruitment evidence** (Q4) and **protocol timing** (Q5).
3. Decide the **release scope** (Q11).

**Then, to produce the results — from a clean tree:**

```bash
cd Code
export BRUXISM_DATA_ROOT=../Data

bruxism-train      --config configs/experiments/five_class_nested_loso.yaml
bruxism-ablations  --config configs/experiments/modality_and_no_chewing.yaml   # the critical one
bruxism-baselines  --config configs/experiments/baselines.yaml
bruxism-ablations  --config configs/experiments/secondary_tasks.yaml
bruxism-benchmark  --output outputs/benchmarks
bruxism-report     --runs-root outputs/runs --output-root outputs/paper_bundle
```

Each is resumable; rerunning reuses completed folds and refuses to resume across a changed
config, manifest or window index. `bruxism-report` regenerates every figure, table, LaTeX
macro and `paper_results.md` from saved artifacts alone — rerun it any time without
retraining.

The analysis that decides the paper's contribution is the second command: **if the
fusion-minus-EMG-only gain survives on `no_chewing_four_class`, the audio claim stands; if
it only exists on `five_class`, the microphone is detecting eating** and the contribution
must be reframed as cheap eating rejection.
