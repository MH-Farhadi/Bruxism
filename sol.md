# Implementation Brief: Rebuild the Bruxism Project for a Reproducible Resubmission

## Your role and mission

You are the implementation model taking over this repository. Your job is to turn the existing research prototype into a professional, reproducible research codebase and then run a scientifically defensible evaluation that can supply the missing results for `Paper/K_Farhadi_Paper_Bruxism/Main_2.tex`.

This is not a cosmetic cleanup. The current manuscript revision proposes analyses that the current code does not implement, and several old reported results cannot be reconciled with the raw files. Preserve the historical work, but do not treat its metrics as verified.

Work primarily in `Code/`. Treat `Data/` as immutable raw input and `Paper/` as the documentary specification. Do not edit the paper or raw data unless the user separately authorizes that work. Generate machine-readable results, figures, and a manuscript-facing report that another step can use to finish the paper.

The target outcome is:

1. A clean, documented Python package with clearly separated data, preprocessing, models, training, evaluation, visualization, command-line scripts, tests, configuration, and generated outputs.
2. A validated dataset manifest and an explicit, reviewable labeling protocol.
3. Leakage-free nested leave-one-subject-out experiments for the paper's intended five-class task and its required secondary analyses.
4. Fully traceable predictions, metrics, figures, tables, runtime measurements, and environment metadata.
5. A clear list of manuscript statements that are supported, contradicted, or still blocked by missing human information.

Read this entire brief before changing anything.

---

## Repository map and source-of-truth hierarchy

The workspace root is the project umbrella:

```text
Bruxism/
├── Code/       # Current Python prototype; also a nested Git repository
├── Data/       # Raw participant recordings and associated private material
├── Paper/      # Manuscripts, review material, figures, and journal templates
└── sol.md      # This implementation brief
```

Use the following hierarchy when sources disagree:

1. Raw CSV/AVI/metadata files establish what was recorded.
2. The acquisition protocol or acquisition source code, if the investigator supplies it, establishes trigger and hardware semantics.
3. A documented, approved analysis protocol establishes labels, exclusions, splits, and endpoints.
4. Executed run artifacts establish numeric results.
5. `Main_2.tex` describes the intended revision but is not evidence that its unfinished methods were performed.
6. The original manuscript and old scripts provide provenance only; neither is a trusted source of final results.

Important paper files:

- `Paper/K_Farhadi_Paper_Bruxism/Main_2.tex` is the current target revision. The actual filename begins with a capital `M`.
- `Paper/K_Farhadi_Paper_Bruxism/misc/Main.tex` is the originally submitted manuscript.
- `Paper/K_Farhadi_Paper_Bruxism/REVISION.docx` contains revision notes.
- The rejection/decision PDF in the same manuscript directory contains the journal reviews.
- `Paper/K_Farhadi_Paper_Bruxism/Temp.md` is an existing and useful revision plan.
- `MBEC_Version.tex`, `Scientific_Reports_Version.tex`, `ntr-old.tex`, and `paper.tex` are historical manuscript variants. They can help identify provenance but must not override raw evidence or current instructions.

The current `Code/` repository was clean at commit `a857f9b` when this audit was prepared. Record the exact starting commit and working-tree state before implementation. Preserve unrelated user changes if the state has changed.

---

## Non-negotiable rules

### Scientific integrity

- Do not invent, interpolate, or preserve a number merely because it appears in a manuscript.
- Treat the old `85.0%` accuracy and its confusion matrix as unverified until a reproducible run produces matching predictions under a documented protocol.
- Never use an outer held-out subject to choose hyperparameters, stop training, select a checkpoint, set thresholds, fit preprocessing, compute class weights, or choose an epoch.
- Never use random window-level cross-validation for a claim about generalization to unseen people. Overlapping windows and repeated recordings make that split severely leaky.
- Distinguish instructed awake tasks from sleep bruxism, clinical bruxism diagnosis, spontaneous/natural bruxism, ambulatory validation, and real-world detection.
- Report classification of the recorded protocol. Do not call it clinical “detection” unless a clinically valid detection study is later performed.
- Chewing is a likely shortcut for the audio branch. The main result must be accompanied by no-chewing and modality analyses.
- With only five participants, emphasize subject-level descriptive performance and uncertainty. Do not imply population validation.

### Data protection

- `Data/` contains potentially identifiable video, participant photographs, health-related survey material, receipts, and reimbursement spreadsheets.
- Do not commit, publish, upload, or copy private raw assets into `Code/`, test fixtures, logs, plots, or generated reports.
- Administrative receipts and reimbursement files are not research features or public dataset components.
- The MIT license in `Code/` does not license the participant data.
- Use synthetic or irreversibly de-identified tiny fixtures for tests.
- Do not place participant names or other direct identifiers in manifests. Use canonical IDs such as `S01`.

### Preservation

- Do not delete or rewrite raw data.
- Do not move the 1.7 GB dataset into `Code/`. The requested separation means a dedicated data access layer and configurable data-root path, not duplication of private files.
- Do not remove the nested `.git` directory or rewrite Git history.
- Preserve legacy scripts until the new pipeline passes parity and provenance checks. Move them with `git mv` into an explicitly documented `legacy/` area only when it is safe to do so.
- Never silently “fix” contradictory metadata. Record the conflict and apply a named, reviewed resolution rule.

### Reproducibility

- No user-specific absolute paths.
- Every result must be recoverable from a configuration file, source commit, data-manifest hash, seed, and environment record.
- Generated paper figures and tables must be derived from saved predictions and metrics, not hand-edited values.
- Fail loudly on invalid data, unsupported options, missing artifacts, or feature-extraction errors. Do not silently substitute zeros.

---

## What the audit found

### Current code

`Code/` contains 11 Python files totaling roughly 5,900 lines, plus a README and two dependency specifications. All files parse, but the execution environment used for this audit did not have scikit-learn installed, so the full programs were not imported or run.

The current implementation is a research prototype with repeated copies of datasets, preprocessing, models, label maps, losses, and training loops across:

- `bruxism_dataset.py`
- `preprocessing_utils.py`
- `wavelet_features.py`
- `wavelet_dataset.py`
- `wavelet_cnn.py`
- `training_improvements.py`
- `run_new_wavelet_training.py`
- `run_wavelet_training.py`
- `run_feature_based_training.py`
- `run_random_forest_training.py`
- `sanity_check.py`

The main current path appears to be `run_new_wavelet_training.py`. It:

- Uses four EMG columns and one microphone column.
- Trains on Subjects 1–4 and holds out Subject 5.
- Drops the rest class and skips the first three seconds of each file.
- Uses Subject 5 during every epoch for validation, early stopping, and checkpoint choice, and then reports Subject 5 again as the final test result.
- Has no complete deterministic seeding policy.
- Uses shallow `state_dict().copy()` checkpoint snapshots.
- Writes outputs relative to the current working directory and can call `plt.show()`.
- Contains another copy of preprocessing, datasets, models, focal loss, and mappings.

The older training paths have similar issues. A deleted historical cross-validation script did rotate the outer subject, but still selected the epoch on that held-out subject and reported the maximum held-out accuracy. Do not restore those outputs as ground truth.

Key technical defects or uncertainties to address:

- `bruxism_dataset.py` labels an entire active-task file by its filename and ignores the `Trigger` column.
- It eagerly loads overlapping windows into memory and assumes a brittle directory and filename structure.
- Some preprocessing and normalization are performed independently on the test set in old helper paths.
- The test subject is used for training decisions.
- The random-forest path calls ordinary five-fold window-level cross-validation, allowing overlapping windows and recordings from the same person into different folds.
- The production preprocessing is zero-phase offline filtering (`filtfilt`), despite old “real-time” wording.
- The main chain uses a 60 Hz notch, 20–450 Hz bandpass, and another 5 Hz high-pass; whether acquisition hardware already filtered the signal is unknown.
- The optional ICA reconstructs all retained components and therefore may do almost nothing. It is disabled in the main path.
- The CWT feature code calls `pywt.cwt` with `db4`, a discrete wavelet that is not a suitable continuous-wavelet choice. Errors can be hidden by filling four zero features per channel.
- The WPT code only uses the `a` and `d` nodes at the first level despite declaring a deeper maximum level.
- One “median frequency” calculation is not a spectral median-frequency calculation.
- PyWavelets coefficient order is `[A_n, D_n, D_(n-1), ..., D_1]`. The current model indexes a coefficient as `details[2]` while the paper calls it `D3`; verify the actual band and correct either code or description.
- Wavelet transforms occur inside `forward()` through NumPy/PyWavelets loops, forcing GPU-to-CPU-to-GPU transfers.
- The focal loss derives `pt` from a class-weighted cross-entropy term. This does not equal the model probability of the true class when alpha weights are applied. Calculate the focal term from the unweighted true-class probability and apply alpha separately.
- A shallow copy of a PyTorch state dictionary does not safely freeze the best weights because tensor storage can continue changing. Deep-copy it or write a checkpoint immediately.
- Requirements use broad ranges and duplicate TXT/YAML specifications. The code appears to use options, such as `FastICA(whiten_solver="eigh")`, whose minimum compatible version should be verified.
- Model parameter counts must be calculated programmatically. The present dual-branch network is approximately 7,524 parameters for four output classes and 7,557 for five, not approximately 15,000 as suggested by old prose.
- The README overstates natural/real-time bruxism detection, contains a “~15 parameters” typo, and shows a flowchart that does not accurately represent the current dual-branch system.

### Raw data

The `Data/` directory is about 1.7 GB. It contains:

- 100 CSV recordings
- 100 AVI recordings
- 100 metadata files
- five scanned survey PDFs
- participant/setup images
- three NPY files
- administrative receipts and reimbursement spreadsheets

All CSVs use the same six-column schema:

```text
EMG1_1-2,EMG2_3-4,EMG3_5-6,EMG4_7-8,Trigger,Mic
```

The dataset has 7,167,600 numeric rows. The audit found no malformed numeric rows, missing values, non-finite values, or trigger values outside the expected binary set. All metadata files report 1,200 Hz and completion status. Most recordings have 72,000 rows, corresponding to 60 seconds. Two are shorter:

- Subject 2 `natural_bruxing`, repeat timestamp `143429`: 62,880 rows, about 52.4 seconds.
- Subject 5 `cheese`, repeat timestamp `151036`: 48,720 rows, about 40.6 seconds.

Their corresponding videos are also shorter. The remaining videos are generally about 59–60 seconds at 640×480 and 30 fps. Therefore, wording that each task was a three-minute trial repeated three times is not supported. The file layout instead suggests three recordings of roughly one minute per condition, for about three minutes total per condition. Confirm the protocol with the investigators.

Subjects 1–4 have 20 CSVs each in their primary directories. Subject 5 has 19 there. Its missing rest CSV/AVI/metadata and a missing protrusion metadata file are in:

```text
Data/More Data/Data/Subject_5/
```

Do not overlook these files and do not blindly move them. Resolve them through the manifest and document their physical locations.

`Data/README.txt` maps the four recorded EMG columns as:

- `EMG1_2`: left masseter
- `EMG3_4`: left temporalis
- `EMG5_6`: right masseter
- `EMG7_8`: right temporalis

The code uses all four columns. In contrast, manuscript tables and prose refer to two EMG channels or two unilateral bipolar pairs. Hardware photographs do not resolve this discrepancy. Exact electrode montage, differential-pair interpretation, laterality, units, electrode type, amplifier, ADC resolution/range, gain, and device filters require investigator confirmation.

The raw EMG values extend to approximately ±65,000, while microphone values are integer-like and roughly 50–227. Their calibrated physical units are not documented. Do not invent µV, Pa, or dB units.

Each active-task CSV contains a binary trigger with one or more active runs. Dedicated rest files have a trigger that is zero throughout. Across conditions, approximate trigger-on fractions are:

| Condition | Trigger on |
|---|---:|
| bite left | 46.6% |
| bite right | 52.8% |
| carrots | 76.7% |
| cheese | 64.9% |
| deviation | 42.1% |
| gum | 84.1% |
| incisor clench | 56.5% |
| molar clench | 58.3% |
| natural bruxing | 62.7% |
| open/close | 32.1% |
| protrusion | 41.7% |
| dedicated rest | 0% |

This strongly suggests that the trigger distinguishes active task intervals from other portions of each recording, but the acquisition source or protocol is missing. The trigger may mark a button press, an instruction interval, a repetition, device state, or something else. Do not infer its scientific meaning solely from its values. Label construction is blocked until the investigator confirms:

- who or what set the trigger;
- whether `1` means the participant was actively performing the named task;
- whether trigger-off intervals are rest, transitions, instruction periods, or unobserved activity;
- whether the trigger has onset/offset delay;
- how repeated runs correspond to repetitions;
- whether video was intended to adjudicate ambiguous periods.

The current whole-file windowing produces an exact and important discrepancy. With one-second windows and 0.5-second stride, including all complete raw recordings:

| Label family | Windows |
|---|---:|
| dedicated rest | 595 |
| movement: open/close, deviation, protrusion | 1,785 |
| clench: bite left/right, incisor, molar | 2,380 |
| instructed grinding (`natural_bruxing`) | 1,769 |
| chewing: carrots, cheese, gum | 5,316 |
| **All five classes** | **11,845** |
| **Four active classes only** | **11,250** |

The old paper's four-class confusion matrix also totals 11,845 windows, but distributes them as:

- movement: 1,877
- clench: 2,503
- grinding: 1,861
- chewing: 5,604

Its diagonal is 1,541, 1,725, 1,422, and 5,380, giving the reported 85.0% accuracy. The total is exactly the five-class all-file count, including 595 rest windows, even though the matrix has no rest class. The per-class supports do not match either the raw whole-file four-class counts or the current script after its three-second skip. This provenance must be investigated and documented. Do not reuse this matrix.

A preliminary diagnostic count of one-second windows wholly contained inside trigger-active runs and at least 0.5 seconds from run boundaries yielded far fewer examples—roughly 6,200 active windows plus the 595 dedicated-rest windows. This is not yet an approved count and must be regenerated by the new manifest/segmentation code after trigger semantics and boundary rules are confirmed.

Other data-quality findings:

- Subject 5's molar-clench metadata identifies the condition/key as incisor clench even though its filenames indicate molar clench.
- The metadata claims an NPY companion for every recording, but only three NPY files exist, all for Subject 1.
- In those NPY files, the first five columns correspond to EMG plus trigger, but the sixth column is all zeros and does not reproduce the CSV microphone channel. Treat NPY files as stale/incomplete caches and regenerate any caches from CSV.
- Signal starts frequently show large transients or initialization behavior. Blindly dropping three seconds is not a sufficient labeling or quality policy.
- Metadata entries such as `bandpass_filter: Index143` and `notch_filter: Index9` are not self-explanatory. Determine whether acquisition-side filtering occurred before adding or interpreting offline filters.
- The five dedicated rest files are the least ambiguous source for a rest class. Trigger-off intervals in active files must not be treated as rest unless their semantics are confirmed and a reviewable annotation policy is adopted.
- Participant surveys appear to ask whether a provider indicated that the participant grinds their teeth. This is not necessarily evidence of a formal, current clinical bruxism diagnosis. Do not upgrade this to “clinically diagnosed bruxism” without source documentation.
- Participant photographs, videos, and scanned surveys require restricted handling.

### Manuscripts and reviews

The original submission framed the work as natural or genuine bruxism detection, claimed leave-one-subject-out generalization and real-time clinical utility, omitted rest, and emphasized 85% four-class accuracy. Those claims are not supported by the current executable workflow.

The journal decision, dated July 3, 2026, rejected the submission. The reviews collectively require the revision to address:

- an outdated tooth-contact-only definition of bruxism;
- vague phenotype and diagnostic characterization;
- contemporary biopsychosocial and INfORM/TMD framing;
- newer terminology and educational material;
- home-to-home and temporal fluctuation;
- device validity;
- the small sample, even for a pilot;
- method ordering and reproducibility;
- ROC/AUC;
- chewing-driven inflation;
- clenching/grinding inconsistencies;
- comparison with stronger EMG studies;
- the distinction between classification and detection;
- the absence of rest;
- overgeneralization from five participants;
- the fact that tasks were elicited/emulated, not natural bruxism.

`Temp.md` proposes an appropriate direction:

- Reframe the study as a proof-of-concept classification of instructed awake tooth-contact tasks.
- Add rest.
- Report a no-chewing task.
- Add binary and possibly ternary clinical groupings.
- Test the audio contribution with chewing excluded.
- Report ROC-AUC and PR-AUC.
- Compare baselines fairly using equivalent inputs.
- Keep augmentation training-only.
- Distinguish model compute latency from the one-second observation window.
- Treat subject-level results as primary.
- Verify citations and clinical terminology.

`Main_2.tex` already moves substantially in this direction and places Methods before Results, but it is unfinished and presently contains approximately 206 `\TBD{...}` placeholders plus several author-action comments. It references missing five-class pipeline, t-SNE, confusion-matrix, and training-curve figures. Several of its proposed statements are not implemented:

- five-class rest classification;
- nested LOSO with inner selection;
- per-participant results;
- AUC/PR analysis;
- binary/ternary/no-chewing analyses;
- matched modality ablations and baselines;
- trustworthy latency and parameter measurements.

One methodological correction is essential: with five total participants, an outer fold leaves four training participants. A participant-grouped inner LOSO therefore has four folds, not five. The current manuscript's “inner five-fold” wording is impossible if folds are grouped by participant.

The manuscript also contains unresolved factual placeholders or conflicts concerning hardware, channel count, electrode placement, filters, task timing, recruitment/diagnosis, rest adjudication, hyperparameter selection, and IRB identifiers. Historical files contain at least `IRB22275690-2`, `IRB2425-139`, and a likely typo `IRB2275690-2`. Only an investigator or official approval record can determine the correct number.

---

## Required target structure

Refactor `Code/` toward this structure. Minor variations are acceptable when they improve packaging, but maintain the separation of concerns:

```text
Code/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   ├── data/
│   ├── experiments/
│   └── models/
├── src/
│   └── bruxism/
│       ├── __init__.py
│       ├── data/
│       │   ├── schema.py
│       │   ├── manifest.py
│       │   ├── labels.py
│       │   ├── segments.py
│       │   ├── dataset.py
│       │   ├── splits.py
│       │   └── quality.py
│       ├── preprocessing/
│       │   ├── filters.py
│       │   ├── normalization.py
│       │   ├── wavelets.py
│       │   └── augmentation.py
│       ├── features/
│       │   └── time_frequency.py
│       ├── models/
│       │   ├── dual_branch.py
│       │   ├── ablations.py
│       │   └── baselines.py
│       ├── training/
│       │   ├── engine.py
│       │   ├── losses.py
│       │   └── selection.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── aggregation.py
│       │   └── benchmark.py
│       ├── visualization/
│       │   └── paper_figures.py
│       └── utils/
│           ├── reproducibility.py
│           ├── io.py
│           └── logging.py
├── scripts/
│   ├── data/
│   │   ├── audit_dataset.py
│   │   ├── build_manifest.py
│   │   └── plot_preprocessing.py
│   ├── train/
│   │   ├── run_nested_loso.py
│   │   ├── run_ablations.py
│   │   └── run_baselines.py
│   └── evaluate/
│       ├── summarize_runs.py
│       ├── benchmark_models.py
│       └── make_paper_artifacts.py
├── tests/
│   ├── fixtures/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── data_dictionary.md
│   ├── experiment_protocol.md
│   ├── reproducibility.md
│   ├── legacy_crosswalk.md
│   └── open_questions.md
├── outputs/                 # ignored; generated run bundles only
└── legacy/                  # preserved prototype after safe migration
```

The terms “training scripts,” “testing scripts,” “dataset,” and “model definitions” must correspond to genuinely separate modules:

- Dataset access and label creation belong under `src/bruxism/data/`.
- Training orchestration belongs under `src/bruxism/training/` with thin entry points under `scripts/train/`.
- Held-out evaluation belongs under `src/bruxism/evaluation/` and `scripts/evaluate/`.
- Model definitions belong under `src/bruxism/models/`.
- Automated software tests belong under `tests/`; do not confuse them with held-out model evaluation.
- Raw data remains in the sibling `Data/` directory and is supplied through `--data-root`, an environment-specific config override, or both.

Avoid a generic `utils.py` dumping ground. Use typed configuration objects and small modules with explicit contracts. Keep entry-point scripts thin.

---

## Implementation sequence

Execute these phases in order. Do not start expensive definitive training until the data protocol is resolved and its manifest passes validation.

### Phase 0 — Preserve and specify

1. Record the starting Git commit and status.
2. Create a working branch if appropriate, without rewriting history.
3. Document the currently intended Python and platform support.
4. Replace duplicate dependency files with `pyproject.toml` as the authoritative package/build/tool configuration. Use a reproducible lock file compatible with the selected package manager.
5. Pin or constrain versions tightly enough to reproduce PyTorch, PyWavelets, NumPy, SciPy, scikit-learn, pandas, plotting, and video-metadata behavior.
6. Add console entry points or documented `python -m` commands.
7. Preserve old behavior in `legacy/` or a Git tag before removing duplicated modules.
8. Write `docs/legacy_crosswalk.md` mapping every old file/class/function to its replacement or explaining its retirement.

Do not combine behavioral changes, file moves, and result claims in one opaque commit. Prefer small, reviewable milestones.

### Phase 1 — Build a data manifest and validation gate

Create a deterministic scanner that discovers all primary files, including `Data/More Data/Data/Subject_5/`, without depending on directory ordering. It must never infer correctness merely because a file exists.

For every recording, the manifest should contain at least:

- canonical subject ID;
- source-relative paths for CSV, AVI, and metadata;
- original condition token and canonical condition;
- task family;
- repetition or timestamp token;
- row count;
- metadata and configured sampling rate;
- duration from samples;
- video duration/frame rate/codec/resolution when readable;
- channel schema and dtypes;
- trigger values, active fraction, number of runs, run boundaries, and transition counts;
- signal minima/maxima and basic quality flags;
- source checksums, preferably SHA-256;
- filename/metadata consistency flags;
- exclusion status, reason, and policy version.

Produce both machine-readable and human-readable artifacts such as:

```text
outputs/data_audit/<manifest_hash>/
├── manifest.parquet
├── manifest.csv
├── data_audit.json
├── data_audit.md
├── trigger_summary.csv
└── quality_figures/
```

Validation must explicitly catch:

- the two short recordings;
- Subject 5's files in `More Data`;
- the molar/incisor metadata conflict;
- missing or mismatched CSV/AVI/metadata triples;
- stale NPY claims;
- unexpected columns, trigger values, sample rates, durations, and numeric failures;
- accidental direct identifiers in generated public-facing outputs.

Do not use the supplied NPY files for experiments. If caching is needed, generate a versioned cache from a CSV checksum and preprocessing configuration. Include microphone data correctly.

Create `docs/data_dictionary.md` with known fields, units marked “unknown” where appropriate, the tentative channel mapping, condition taxonomy, privacy classification, and unresolved semantics.

Create `docs/open_questions.md` immediately and keep it updated. Questions requiring investigator answers are listed later in this brief.

### Phase 2 — Approve and encode the labeling protocol

The pipeline must support at least two explicit segmentation policies for provenance:

1. `whole_recording_legacy`: reproduces filename-based whole-file labeling only to diagnose historical counts. It must be labeled unsafe for final inference.
2. `trigger_constrained`: creates task examples only from approved stable trigger-active intervals.

Do not silently choose the second policy before trigger meaning is confirmed. Once confirmed, write the exact decision in `docs/experiment_protocol.md` and a versioned config.

For the intended trigger-constrained protocol:

- Filter each continuous recording before cutting windows so filter boundary artifacts are not introduced at every window.
- Use one-second windows and a 0.5-second stride unless the approved protocol changes them.
- Keep only windows fully contained inside a homogeneous approved segment.
- Exclude a configurable guard interval around trigger changes; begin with 0.5 seconds only if approved.
- Exclude recording startup/shutdown intervals through a declared quality rule, not an unexplained global skip.
- Use dedicated rest recordings as the primary rest source.
- Only use trigger-off intervals from active recordings as rest if their meaning is verified, transitions are excluded, and an auditable video/annotation policy supports them.
- Attach every window to subject, recording, start/end sample and seconds, trigger run, condition, task family, and exclusion provenance.
- Prevent an identical or overlapping window from entering multiple examples or folds.

Canonical task-family mapping should begin as:

```text
movement:
  - open_close
  - deviation
  - protrusion

clench:
  - bite_left
  - bite_right
  - incisor_clench
  - molar_clench

instructed_grinding:
  - natural_bruxing   # preserve raw token, rename scientifically in outputs

chewing:
  - carrots
  - cheese
  - gum

rest:
  - dedicated rest recording
  - other source only if explicitly approved
```

The old word `natural_bruxing` is a raw filename token, not evidence that the event was spontaneous or natural. Present it as instructed grinding.

### Phase 3 — Refactor preprocessing and loading

Create one authoritative preprocessing implementation used by training, evaluation, ablations, and visualization.

Requirements:

- Validate Nyquist constraints before filter design.
- Make notch, bandpass/high-pass, order, Q, zero-phase/causal mode, and edge handling configurable.
- Do not apply redundant filters without a documented rationale.
- State clearly that `filtfilt` is acausal/offline and cannot substantiate real-time streaming claims.
- Plot raw versus production-filtered examples using the exact production code.
- Fit normalization only on the inner-training or outer-training data appropriate to the current stage.
- Save fitted normalization parameters in each run.
- Apply augmentation only to training samples and only after the split.
- Seed augmentation deterministically per run/sample where feasible.
- Do not create all overlapping windows or all transformed tensors in RAM by default. Use indexed/lazy reads, memory maps, chunked caches, or a similarly scalable strategy.
- Cache deterministic expensive transforms using source checksum plus full transform configuration.

For wavelets:

- Unit-test the exact PyWavelets coefficient ordering and associated frequency bands at 1,200 Hz.
- Make selected decomposition level and coefficients explicit.
- Correct the code/paper mismatch around `details[2]` and “D3.”
- Use an appropriate continuous wavelet if CWT remains, or remove the unused broken feature path.
- Either implement WPT as documented or remove misleading depth settings.
- Implement true spectral median frequency if that feature remains.
- Never replace a feature-extraction exception with zeros without marking the sample/run invalid.
- Move non-differentiable PyWavelets transforms out of the neural network's hot `forward()` path, or replace them with a tested differentiable implementation.

### Phase 4 — Consolidate models and losses

Implement a single configurable dual-branch model:

- Four-channel EMG branch.
- One-channel microphone branch.
- Explicit feature fusion.
- Configurable classifier head and number of output classes.
- EMG-only and audio-only ablations using the same branch definitions where applicable.

Model code must:

- accept clearly documented tensor shapes;
- work on CPU and CUDA without hidden device copies;
- compute parameter counts programmatically;
- save architecture/config alongside checkpoints;
- expose embeddings for the exploratory t-SNE figure;
- return logits, not already-softmaxed probabilities, to standard PyTorch losses.

Correct focal loss and test it against a small manually computed/reference example. Class weights must be calculated from training data only and saved. Use standard cross-entropy as a baseline. Hyperparameter selection should determine whether focal loss is retained rather than assuming it is superior.

Implement baseline models with honest comparability:

- A bidirectional LSTM or similar sequence baseline.
- An early-fusion raw-signal CNN.
- A random forest and/or MLP on explicitly defined features.

The paper must not claim architecture superiority if a baseline receives fewer modalities or materially different information. Preferred comparison:

- dual-branch fusion: EMG + audio;
- early-fusion CNN: the same EMG + audio;
- BiLSTM: the same EMG + audio;
- feature baseline: both EMG and audio features.

If a baseline cannot accept equivalent inputs, call it a feature/modality comparison and say so. Never use ordinary random window folds.

### Phase 5 — Implement leakage-free nested LOSO

The primary evaluation is outer leave-one-subject-out over all five participants.

For each outer fold:

1. Reserve one participant as the untouched outer test set.
2. On the remaining four participants, perform inner leave-one-subject-out with four folds.
3. Fit every transform, normalization statistic, class weight, augmentation policy, model, threshold, and hyperparameter choice using inner-training participants only.
4. Select the prespecified objective, recommended as validation macro-F1 with a deterministic tie-break rule.
5. Estimate the final epoch budget from inner-fold results using a prespecified rule, such as the median best epoch.
6. Retrain the selected configuration on all four outer-training participants for that fixed epoch budget.
7. Evaluate the outer subject exactly once.
8. Save all predictions and metadata before advancing to the next fold.

An alternative is to prespecify all hyperparameters before the outer experiment. If so, still keep the held-out subject untouched and document how the configuration was selected. Do not add a hidden window-level validation split spanning subjects without justification.

Use multiple fully specified random seeds, preferably five if compute permits. Define in advance whether repeated-seed probabilities are averaged per outer sample or metrics are summarized across seeds. Do not select the best seed. Record:

- Python, NumPy, and PyTorch seeds;
- data-loader worker seeding;
- deterministic algorithm settings;
- hardware-dependent nondeterminism;
- inner-fold search space and every trial;
- selection objective and tie-break;
- final epoch rule;
- any failed run and its reason.

Checkpoint state must be deep-copied or serialized immediately. Training logs must never call the outer evaluator as a validation callback. Add a software-level guard that makes outer data unavailable to selection code.

### Phase 6 — Run the required task definitions

Each task is a separately configured experiment. Do not obtain all claims by relabeling a single confusion matrix unless the analysis is explicitly described as post hoc probability aggregation.

Required tasks:

1. **Primary five class**
   - rest
   - movement
   - clench
   - instructed grinding
   - chewing

2. **No-chewing four class**
   - rest
   - movement
   - clench
   - instructed grinding

3. **Binary tooth-contact activity**
   - positive: clench + instructed grinding
   - comparator: rest + movement + chewing

4. **Ternary task**, if retained in `Main_2.tex`
   - clench + instructed grinding
   - chewing
   - rest + movement

5. **Legacy active four class**, provenance only
   - movement
   - clench
   - instructed grinding
   - chewing

For the binary/ternary/no-chewing results, train purpose-specific models under the same outer protocol. You may additionally collapse five-class held-out probabilities as a clearly labeled post hoc secondary analysis, because that answers a different question.

Prespecify how class imbalance is handled. Report task-specific sample counts by participant and source condition.

### Phase 7 — Perform modality and chewing analyses

For the primary task and the no-chewing task, compare:

- EMG + audio fusion;
- EMG only;
- audio only.

This is central to the reviewer concern that microphone signals may mainly identify chewing. Keep splits, seeds, selection budget, window definitions, and evaluation code matched.

Report:

- the change from EMG-only to fusion on the primary five-class task;
- the same change after chewing is removed;
- class-specific clench and instructed-grinding effects;
- per-subject paired differences;
- uncertainty without overstating inferential significance.

Do not claim audio detects grinding merely because it improves pooled accuracy dominated by chewing.

### Phase 8 — Metrics and prediction ledger

The primary aggregation should treat participants as the units of generalization. Report:

- accuracy;
- balanced accuracy;
- macro precision, recall, and F1;
- per-class precision, recall, F1, and support;
- raw-count and row-normalized confusion matrices;
- one-vs-rest ROC-AUC by class and macro average;
- one-vs-rest average precision/PR-AUC by class and macro average;
- for binary tasks: sensitivity, specificity, PPV, NPV, F1, ROC-AUC, and PR-AUC.

Present subject-level mean, standard deviation, range, and all five individual results. Pooled-window metrics can be secondary and must be labeled as descriptive because windows within a participant and recording are correlated.

Do not use a naïve window bootstrap or ordinary window-level p-values as population uncertainty. With five participants, any participant bootstrap, permutation, or paired comparison is exploratory and must disclose the very small unit count. Prefer transparent individual participant results over decorative significance testing.

Every outer-test example must appear exactly once per task/model/seed in a prediction ledger with:

- stable sample ID;
- canonical subject and recording ID;
- start/end sample and time;
- true label;
- predicted label;
- probability for every class;
- outer fold and seed;
- task, model, and modality IDs;
- source commit;
- resolved configuration hash;
- data-manifest hash;
- checkpoint hash.

Metrics must be computed from this ledger, never separately from transient in-memory values. Include calibration or reliability analysis only if it is prespecified and interpretable at this sample size.

### Phase 9 — Generate paper artifacts

Use one command to regenerate the complete manuscript-facing bundle from saved ledgers and metrics. At minimum, generate:

- five-class raw and normalized confusion matrices;
- ROC curves and class/macro AUC table;
- PR curves and average-precision table;
- per-participant performance plot;
- training/inner-validation curves that do not include the outer subject;
- model/modality comparison table;
- no-chewing audio ablation plot or table;
- sample-count/flow diagram;
- exploratory held-out-embedding t-SNE plot with a fixed seed and an explicit “exploratory” label;
- parameter-count and latency table;
- LaTeX-ready tables/macros generated from JSON or CSV;
- a narrative `paper_results.md` that points every reported value to an artifact.

The current manuscript references missing assets resembling:

- `pipeline_5class`
- `tsne_5class`
- `confmatrx_5class`
- two training-curve figures

Map each placeholder in `Main_2.tex` to a generated artifact or a named human blocker. Do not edit the manuscript in this task unless separately requested.

For t-SNE, use held-out embeddings only, avoid presenting cluster appearance as independent validation, and save the exact preprocessing, perplexity, initialization, seed, and source checkpoint.

### Phase 10 — Benchmark runtime honestly

The current wavelet transform inside `forward()` makes latency sensitive to CPU/GPU transfers and Python loops. First refactor it into a reproducible preprocessing/cache or a tested tensor-native implementation.

Benchmark:

- deterministic preprocessing/filtering time;
- wavelet/feature-transform time;
- model-only inference;
- end-to-end preprocessing plus inference;
- batch size 1 and any relevant batch size;
- warm-up excluded from steady-state timing;
- repeated median and p95 latency;
- CPU and GPU synchronization where required;
- hardware, operating system, Python, framework, and numeric precision;
- programmatically computed parameter count and, if useful, model size.

The system requires at least a one-second observation window and produces decisions at a 0.5-second stride under the current design. A sub-second compute time does not create zero-latency detection. State separately:

- input/context latency;
- decision update interval;
- processing latency.

Do not claim wearable, streaming, embedded, or clinical real-time readiness without an actual implementation and validation.

### Phase 11 — Documentation and quality controls

The README should enable a future researcher to:

1. understand the study's narrow scope;
2. install the environment;
3. run tests without private data;
4. point the software to an authorized data root;
5. audit/build the manifest;
6. run a smoke experiment;
7. launch/resume the full nested LOSO experiment;
8. regenerate the paper artifact bundle;
9. understand privacy constraints and known limitations.

Add:

- type hints for public interfaces;
- concise docstrings explaining shapes, units, and fitting scope;
- structured logging;
- atomic writes for artifacts and checkpoints;
- resumable fold/seed execution;
- schema/version checks before resuming;
- formatter, linter, and type-check configuration;
- continuous integration for synthetic-data tests if the repository will be hosted.

Each production run should have an immutable bundle:

```text
outputs/runs/<run_id>/
├── resolved_config.yaml
├── environment.json
├── data_manifest.json
├── data_manifest.sha256
├── source_state.json
├── folds.json
├── selection/
├── logs/
├── checkpoints/
├── predictions.parquet
├── metrics.json
├── metrics.csv
├── figures/
└── paper_results.md
```

Include dirty-tree state or a source diff hash if a run is launched from uncommitted code.

---

## Minimum automated test suite

Use small synthetic fixtures and targeted metadata samples, not private raw data.

### Data and labels

- Parse the exact six-column schema.
- Reject missing, reordered, nonnumeric, or unexpected columns unless an explicit migration handles them.
- Detect a filename/metadata condition conflict.
- Discover Subject 5's secondary-location files.
- Verify sample count, duration, checksum, and pair/triple matching.
- Split trigger runs correctly.
- Exclude transition guards and keep windows fully within one approved segment.
- Verify one-second/0.5-second window arithmetic, including short recordings.
- Ensure sample IDs are stable and unique.
- Confirm task-family mappings and every binary/ternary collapse.
- Prevent overlap across train/validation/test groups.

### Leakage

- Assert outer-test subject IDs never enter training, inner validation, fitting, class weights, augmentation statistics, checkpoint selection, or threshold tuning.
- Assert normalization fitted values change when training data changes but not when only held-out data changes.
- Assert augmentation is invoked only for training.
- Assert every participant is outer-held-out exactly once.
- Assert every outer sample receives exactly one prediction per model/task/seed.
- Reject ordinary K-fold window splitting for production configurations.

### Signal processing and features

- Compare filters against a small known SciPy reference and test invalid cutoff handling.
- Test continuous-record filtering before segmentation.
- Test PyWavelets coefficient order and frequency-band labels at 1,200 Hz.
- Test cached and uncached transforms for numerical agreement.
- Test the chosen CWT/WPT path or remove it.
- Test true median-frequency computation on a known spectrum.
- Ensure feature errors are visible and never silently converted to zero vectors.

### Models and training

- Test input/output shapes for five-, four-, three-, and two-class configurations.
- Test CPU and, when available, CUDA device consistency.
- Test EMG-only, audio-only, and fusion paths.
- Test focal loss against a manually calculated example, with and without alpha.
- Test that saved “best” weights remain unchanged after subsequent optimizer steps.
- Test deterministic repeat behavior within documented platform limits.
- Test checkpoint resume and configuration mismatch rejection.

### Metrics and artifacts

- Compare confusion matrix, macro metrics, ROC-AUC, PR-AUC, and binary metrics with fixed reference examples.
- Handle an absent class in a fold explicitly without fabricating AUC.
- Recompute published summaries from the prediction ledger.
- Test raw versus normalized confusion matrices.
- Test that figure/table generation uses saved artifacts only.
- Test that no absolute private path or direct identifier leaks into public-facing results.
- Add a fast end-to-end CLI smoke test on synthetic participants.

---

## Human decisions that block definitive scientific results

Refactoring, tests, manifest construction, and legacy reproduction can proceed while these are open. Definitive labels, final experiments, and publication claims cannot.

Ask the project investigators to resolve and document:

1. **Trigger semantics:** who controlled it, the meaning of 0 and 1, timing delay, transition protocol, and whether videos should adjudicate it.
2. **Channel interpretation:** whether the four CSV EMG columns are four bipolar channels, two channels with paired fields, or another montage; exact muscle/laterality placement.
3. **Hardware:** manufacturer/model for electrodes, amplifier/DAQ, microphone, gain, ADC resolution/range, units, and acquisition-side filters.
4. **Study population:** what recruitment evidence supports provider-indicated grinding versus a formal clinical diagnosis, and what awake/sleep phenotype can truthfully be stated.
5. **Protocol timing:** actual duration and repetitions per condition; reconcile the files with “three-minute trials repeated three times.”
6. **Task naming:** confirm every instructed task and resolve any carrots/popcorn wording across manuscript versions.
7. **Metadata conflict:** approve the resolution for Subject 5's molar file labeled as incisor in metadata.
8. **Rest definition:** dedicated rest only versus verified trigger-off/video-annotated intervals, including transition and artifact exclusions.
9. **Signal preparation:** whether hardware filters were active and what `Index143`/`Index9` mean.
10. **Ethics:** the exact IRB identifier and the scope of consent for analysis, video use, data sharing, and publication.
11. **Privacy/public release:** which derived artifacts can be shared and whether any raw or de-identified dataset release is authorized.
12. **Target venue:** final journal/format requirements and whether all new clinical references and terminology have been verified.

Record signed-off answers with date and source. Do not resolve them by majority vote among inconsistent files.

---

## Manuscript claim controls

Before results are handed back for `Main_2.tex`, create a claim-to-evidence table with columns:

```text
claim_id
manuscript_location
proposed_claim
evidence_artifact
analysis_config
status: supported | contradicted | blocked | editorial
notes
```

At minimum, verify:

- participant count and characterization;
- channel count and placement;
- task names, durations, repeats, and rest source;
- sampling rate and filter descriptions;
- window/stride and transition exclusions;
- sample supports after exclusions;
- exact nested LOSO procedure;
- hyperparameter-selection objective and search space;
- architecture and parameter count;
- all overall/per-class/per-subject metrics;
- ROC-AUC and PR-AUC;
- modality and no-chewing conclusions;
- latency language;
- limitations and scope.

Keep the paper's language bounded:

- “instructed awake jaw/tooth-contact tasks,” not natural bruxism;
- “classification,” not clinical detection;
- “within this controlled five-participant dataset,” not population generalization;
- “proof of concept” only with an explicit small-sample limitation;
- no sleep-bruxism conclusion;
- no diagnostic, therapeutic, wearable, or home-monitoring claim without corresponding evidence.

Citation verification is a separate scholarly task. Do not invent DOI details or rely on unverified references in old `.tex` files.

---

## Execution commands to provide

The final interface should be simple and documented. Exact names may differ, but provide equivalents to:

```bash
# Install development environment
<package-manager> install --locked

# Run code quality and tests without private data
<package-manager> run lint
<package-manager> run typecheck
<package-manager> run pytest

# Audit raw data without modifying it
bruxism-audit --data-root ../Data --output-root outputs/data_audit

# Build a versioned analysis manifest after protocol approval
bruxism-build-manifest \
  --data-root ../Data \
  --config configs/data/trigger_constrained.yaml

# Fast synthetic or one-fold smoke run
bruxism-train \
  --config configs/experiments/smoke.yaml

# Full reproducible nested LOSO
bruxism-train \
  --config configs/experiments/five_class_nested_loso.yaml

# Required ablations and baselines
bruxism-ablations \
  --config configs/experiments/modality_and_no_chewing.yaml

# Evaluate saved outer predictions and generate the paper bundle
bruxism-report \
  --runs-root outputs/runs \
  --output-root outputs/paper_bundle
```

Commands must support `--help`, dry-run or validation-only modes where appropriate, noninteractive execution, actionable error messages, and resumability without silently mixing incompatible artifacts.

---

## Definition of done

The project is ready to supply the resubmission only when all applicable items below are true:

- The old code has an auditable preservation path and crosswalk.
- `Code/` has a standard installable package layout with distinct data, model, training, evaluation, visualization, script, and test areas.
- No source file contains a user-specific data path.
- No raw data, PII, surveys, videos, photographs, receipts, or reimbursements have been copied into the code repository or outputs intended for publication.
- A deterministic manifest accounts for all 100 CSV, 100 AVI, and 100 metadata files, including the secondary Subject 5 directory.
- The two short recordings, stale NPY files, and metadata conflict are explicitly reported.
- Trigger and rest semantics have investigator approval, or final training is clearly marked blocked.
- The historical 11,845-window confusion-matrix discrepancy is either reproduced and explained or explicitly documented as irreproducible.
- All preprocessing, wavelet selection, labels, and exclusions are tested and documented.
- All statistics, class weights, augmentation, and model selection are training-only.
- The outer test subject is inaccessible to the selection loop and evaluated once.
- The inner LOSO uses four participant folds when four training participants are available.
- Every outer held-out sample has one auditable prediction per configured task/model/seed.
- Five-class, no-chewing, binary, modality, and fair-baseline analyses have been run or explicitly blocked with reasons.
- Subject-level metrics are primary; pooled-window metrics are labeled secondary.
- ROC-AUC and PR-AUC are derived from saved probabilities.
- Parameter counts and runtime are measured by code, and context latency is separated from compute time.
- The full suite passes formatting, linting, type checks, unit tests, integration tests, and a synthetic smoke run.
- A fresh authorized researcher can reproduce the audit, experiments, and paper artifacts by following the README.
- Every numeric or graphical placeholder in `Main_2.tex` maps to a generated evidence artifact or a named unresolved human blocker.
- No unsupported clinical, naturalistic, real-time, wearable, or generalization claim is introduced.

---

## Required handoff from you

At the end of your implementation, provide:

1. A concise change summary grouped by data, code architecture, scientific protocol, tests, and documentation.
2. The final directory tree.
3. Exact commands executed and their outcomes.
4. The manifest/audit summary and every unresolved anomaly.
5. The nested-LOSO protocol and proof that outer test data was not used for selection.
6. Links/paths to prediction ledgers, metrics, figures, tables, benchmarks, and resolved configs.
7. A comparison between legacy reported results and reproducible new results, without forcing agreement.
8. The claim-to-evidence table for `Main_2.tex`.
9. A list of unresolved investigator questions and which manuscript sections they block.
10. The source commit, working-tree state, dependency lock, environment, manifest hash, and all run IDs.

If definitive experiments cannot run because trigger semantics or another human fact remains unresolved, finish every safe engineering and validation task, run only clearly labeled diagnostic/legacy/synthetic workflows, and stop before manufacturing a final scientific answer. A transparent blocked result is preferable to an impressive but invalid number.
