# Work-order execution report

**Executed:** 2026-08-03, against `new_prompt.md` and the diagnosis in `cause.md`.
**Scope:** `Code/` only. Nothing in `Data/` was modified, copied or re-encoded.
**Test suite:** 255 unit + 19 integration tests green; `ruff check` and
`ruff format --check` clean.

---

## Headline

`cause.md` was right, and its numbers reproduce. The production EMG filter chain notched
60 Hz — the one mains frequency the acquisition hardware had *already* removed — and passed
180/300/420 Hz untouched. Fixing only the filter, changing nothing else, moves a plain
logistic regression on band energies from **macro-F1 0.419 → 0.679** and **accuracy
0.592 → 0.797** under leave-one-subject-out: a **+0.26 macro-F1** step from one
configuration change, larger than every other intervention in the work order combined.

I confirmed the diagnosis independently before acting on it, then found and fixed a second
defect the diagnosis did not contain: **constant-Q notches are the wrong shape for this
interference**, and switching to constant-width notches at the same total bandwidth cost
cuts the worst participant's residual interference by 6.7×.

| Pipeline (five-class, LOSO, participant-level means) | macro-F1 | accuracy |
|---|---:|---:|
| Trained dual-branch CNN, contaminated chain (superseded run, 3 seeds) | 0.435 | 0.550 |
| Screening LR, superseded chain | 0.419 | 0.592 |
| Screening LR, **corrected chain (shipped, 8 Hz notches)** | **0.679** | **0.797** |
| Screening LR, corrected chain (constant Q=30, as `cause.md` proposed) | 0.689 | 0.812 |
| …plus per-participant calibration | 0.735 | 0.852 |
| …plus 5.5 s trial-level aggregation † | 0.821 | 0.903 |

† Trial-level, not stream-level. See §8.

`cause.md` predicted 0.705 / 0.820 for the filter fix and 0.737 / 0.853 with normalisation;
measured 0.689 / 0.812 and 0.735 / 0.852 on the chain it proposed. **The diagnosis was
accurate to within 0.016 macro-F1 at every stage.**

### First confirmatory folds — no longer screening estimates

The real nested-LOSO pipeline, corrected chain, on the two participants `cause.md`
identified as failing systematically. Same architecture, same seeds, same splits, same
hyperparameter grid, same held-out windows — one configuration change.

| seed 0 | | contaminated | corrected | `cause.md` predicted |
|---|---|---:|---:|---:|
| **S01** | macro-F1 | 0.174 | **0.673** | ≈ 0.59 |
| | accuracy | 0.259 | **0.777** | |
| | balanced accuracy | 0.323 | **0.670** | |
| **S02** | macro-F1 | 0.065 | **0.686** | ≈ 0.70 |
| | accuracy | **0.091** — *below the 0.20 chance level* | **0.883** | |
| | balanced accuracy | 0.260 | **0.794** | |

S02 is the decisive case. It was the participant whose held-out accuracy sat *below chance*
across all three seeds — 75 % of its predictions were `clench`, the signature of a decision
function driven by nothing but overall amplitude. On the corrected chain it is the
**best-performing participant by accuracy**, at 0.883. `cause.md` predicted 0.06 → ≈ 0.70;
measured 0.686.

**2 of 15 folds; the remaining 13 are still running** (§9).

---

## 1. What was verified before anything was changed

`cause.md`'s spectral claims were re-measured from the raw CSVs, independently:

| Frequency | S01 rest, peak / local floor | S02 rest, peak / local floor |
|---|---:|---:|
| 60 Hz | 2× | 2× |
| 180 Hz | 571× | **1,781,427×** |
| 300 Hz | 119× | 412,502× |
| 420 Hz | 43× | 52,690× |

The mains fundamental is absent from data otherwise dominated by its own harmonics. That
is the signature of an acquisition-side notch, and it answers half of
`docs/open_questions.md` Q9 (§8).

Across all 100 recordings, **62 have more than 30 % of their raw in-band EMG power at mains
harmonics**; every rest recording sits between 91.3 % and 99.8 %.

---

## 2. Phase 0 — the defect is now impossible to repeat silently

### 0.1 Mains contamination is measured and flagged

`preprocessing/interference.py` computes, per EMG channel on the **raw** signal, the
fraction of 20–450 Hz power within ±3 Hz of each mains multiple. It is stored on every
manifest row (`mains_harmonic_power_fraction`, per-channel breakdown, per-harmonic
breakdown), raised as `QualityFlag.MAINS_CONTAMINATION` above a declared 0.30, and reported
by `bruxism-audit` with per-participant and per-condition tables plus a
`mains_contamination.csv`.

It is **a flag, not an exclusion** — excluding S02 automatically would have hidden the
finding. Manifest schema bumped to 1.1, quality policy to `2026-08-03.1`.

**One correction to `cause.md`'s framing.** It says "a clean surface-EMG channel would show
a few per cent here." It cannot: seven harmonics × a ±3 Hz band is 42 Hz of a 430 Hz
passband, so a perfectly white signal scores **0.098**, and the cleanest recording in this
dataset scores 0.095 — already at the floor. Two consequences, both documented in the
module:

- the Phase 1.3 target of "below 5 %" is reachable only by a filter that *deletes* those
  bins rather than restoring them;
- a notch bank therefore scores near 0 while spectral interpolation scores near 0.098, and
  **the lower number is not the cleaner signal.** `harmonic_excess_ratio` is the
  method-fair comparison — harmonic power relative to neighbouring bins, where restoring
  the floor reads 1.0 and surviving interference reads above it. Both are reported.

### 0.2 The filter figure now shows the mismatch

`plot_filter_response` overlays the measured mean raw spectrum of the data behind the
filter response. On the superseded chain the 60 Hz notch visibly sits on flat spectrum
while three tall spikes pass through the passband untouched; on the corrected chain the
notches land on the spikes. Nothing in the previous figure set said so, which is how the
defect survived three reproducible runs.

### 0.3 Regression test

`test_production_chain_attenuates_every_mains_harmonic` synthesises a strong tone at each
of 120/180/240/300/360/420 Hz, runs the production chain, and requires ≥20 dB attenuation.
**Against the old chain all six fail** (≤1.4 dB); against the new chain all six pass. A
companion test asserts a 90 Hz tone between harmonics survives within 1 dB, so the cure
cannot quietly remove the signal along with the interference.

---

## 3. Phase 1 — the filter chain

### What shipped

Seven notches — 60, 120, 180, 240, 300, 360, 420 Hz — then the unchanged 20–450 Hz
bandpass. Parameterised as `emg_stages(mains_hz=, notch_harmonics=, quality=,
notch_bandwidth_hz=, band_hz=)`; the superseded single-notch chain remains expressible as
`notch_harmonics=False` so regression comparisons live in configuration, not in patched
code.

Configs declare intent rather than seven near-identical YAML blocks:

```yaml
filters:
  emg_mains: {mains_hz: 60.0, notch_harmonics: true, quality: 30.0, band_hz: [20.0, 450.0]}
```

`to_dict()` always writes the expanded stage list, so the configuration hash and the run
bundle are identical whichever form was used.

### The three candidates were implemented and measured, as instructed

All three are production code paths, not scratch scripts. `bruxism-screen` compares them on
the Phase 1.3 metric over all 6,173 windows:

| variant | mains fraction (mean) | worst cell excess | LR macro-F1 | LR accuracy |
|---|---:|---:|---:|---:|
| superseded 60 Hz only | 0.673 | 43,258× | 0.393 | 0.560 |
| bandpass only (control) | 0.671 | 43,257× | 0.407 | 0.581 |
| notch bank, constant Q=30 | 0.036 | 25.6× | 0.689 | 0.812 |
| **notch bank, constant 8 Hz width** | **0.012** | **3.2×** | 0.679 | 0.797 |
| comb (`iircomb`) | 0.070 | 35.5× | 0.674 | 0.788 |
| spectral interpolation | 0.254 | 264.8× | 0.669 | 0.785 |

**Spectral interpolation lost on evidence, not on convenience.** It leaves a 264× residual
excess because the interference is not a stationary pure tone — a ±1.5 Hz replacement band
does not cover its spread — and it is inherently acausal, which `FilterChainConfig.is_causal`
now accounts for independently of `zero_phase`. The comb is cheaper but leaves 35×.

### The second defect: constant Q is the wrong shape

`cause.md` proposed Q=30 notches. Measured on the four worst-contaminated recordings, that
leaves S02's chewing cell at **22.4 % harmonic power and 25.6× its local floor** — the
worst cell in the dataset, and still above the 5 % target.

The cause is not the harmonics. It is the **fundamental**: the hardware notch left a 60 Hz
residue 40× above the local floor, and a Q=30 notch is only 2 Hz wide there. Constant Q
gives 2 Hz at 60 Hz and 14 Hz at 420 Hz — narrowest exactly where the residue is, widest
where nothing survives.

At an **identical 13 % total band cost**, constant-width 8 Hz notches give:

| | worst cell fraction | worst cell excess | band removed |
|---|---:|---:|---:|
| constant Q=30 | 0.224 | 25.6× | 13.0 % |
| constant 8 Hz | **0.080** | **3.2×** | 13.0 % |

### Why the shipped variant is not the one with the best held-out score

Constant Q scores 0.689 macro-F1, constant width 0.679 — a gap of 0.010 on five
participants whose individual scores span 0.55–0.78. Choosing the chain on that difference
would be **selecting a preprocessing parameter against the held-out score**, which is
exactly what non-negotiable #4 forbids. The decision was made on the prespecifiable,
mechanism-level criterion the work order names for it (contamination), where constant width
wins by 8× on residual excess. Both are reported; the gap is noise at n = 5.

The per-participant detail shows the trade is real and understood: the wider notches help
the participant they were designed to help (S02 0.642 → 0.693) and cost the two whose 60 Hz
is clean (S01 0.603 → 0.548, S05 0.782 → 0.743), because an 8 Hz notch at 60 Hz removes
some genuine low-frequency EMG.

---

## 4. Phase 1.3 — acceptance

| Criterion | Target | Measured | |
|---|---|---|---|
| Screening macro-F1 after the fix | ≈ 0.70 | **0.679** shipped / 0.689 constant-Q | ✅ |
| Screening accuracy after the fix | ≈ 0.82 | **0.797** shipped / 0.812 constant-Q | ✅ |
| Class contrast (activity ÷ own rest) | ≈ 5–39× | **1.7–36.1×** | ✅ mostly |
| Between-participant amplitude spread | ≈ 2.6× | **5.02 → 2.68** | ✅ |
| Rest-above-activity inversions | — | **20 → 0** | ✅ |
| Mains fraction below 5 % **in every cell** | < 0.05 | **mean 0.012, worst cell 0.080** | ⚠️ |
| Phase 0.3 test fails before, passes after | — | 6 fail → 6 pass | ✅ |

**The one miss, stated plainly.** One of 25 participant × class cells — S02 chewing — sits
at 8.0 % rather than below 5 %. Its *excess over the local noise floor* is 3.2×, down from
43,258×, so the interference is essentially gone; what remains is the geometric floor of
the statistic (§2.1) plus a small residue in a participant whose raw signal was 98 % mains.
Under the constant-Q chain `cause.md` proposed, the same cell sits at 22.4 % and 25.6×. I
did not chase the last few per cent by widening the notches further, because that trades
real EMG bandwidth for a metric artefact.

Class contrast, before → after, activity RMS ÷ that participant's own rest RMS:

| Participant | movement | clench | grinding | chewing |
|---|---|---|---|---|
| S01 | 1.07 → 1.71 | 1.62 → 7.99 | 2.18 → 6.98 | 1.64 → 8.46 |
| S02 | 3.62 → 6.27 | 1.93 → 17.30 | 2.23 → 5.38 | 2.87 → 15.43 |
| S03 | 3.02 → 7.87 | 4.88 → 19.45 | 3.08 → 13.54 | 7.33 → 36.09 |
| S04 | 2.72 → 5.54 | 3.22 → 9.49 | 3.22 → 10.03 | 3.84 → 17.25 |
| S05 | 1.91 → 4.53 | 5.27 → 27.14 | 2.61 → 13.06 | 3.71 → 16.99 |

The mechanism `cause.md` identified is gone: there were **20 participant pairs where one
person's resting EMG exceeded another's active EMG** (worst: S02 rest 88.0 > S03 movement
38.4). There are now **none**.

---

## 5. Phase 3 — per-participant normalisation, declared

Implemented as `scope: per_participant` with a defined calibration block, not as a
preprocessing tweak.

**The calibration block** (`preprocessing/calibration.py`) is what a fitting session
produces: the participant's dedicated rest recording plus one guided repetition of each
task family, capped at 20 windows per family (~10 s each). Selection is deterministic. It is
**withheld from every split** at the splitter, so no window can both set a participant's
scale and be scored by it, and `<run_dir>/calibration_block.json` records exactly which
windows produced each participant's statistics.

**The leakage boundary is structural, not documentary.** Two separate attributes:

| | contains | may include the held-out participant |
|---|---|---|
| `Normalizer.fitted_on` | participants whose **labelled training windows** produced the pooled statistics | **no** — `assert_not_fitted_on`, unchanged |
| `Normalizer.calibrated_on` | participants whose **unlabelled calibration block** produced their own statistics | yes, by design, and disclosed |

`Normalizer.calibrate()` has no label parameter to misuse — a unit test asserts that by
introspection. `assert_calibration_disjoint_from` re-checks the split disjointness at every
fold. Both run and both are logged as `TRANSDUCTIVE PROTOCOL` warnings.

**A design bug this surfaced.** The first calibrated run emitted "classes [0] are absent
from the training labels". Each participant has exactly one dedicated rest recording, so an
uncapped block consumed every rest window they had and the rest class vanished from
training. Fixed by the 20-window cap, and `_assert_leaves_every_class_trainable` now refuses
any block that would consume a whole participant × family — the symptom had appeared far
downstream as a warning about class weights, which is exactly the kind of silent degradation
this work order exists to stop.

Screening values the calibrated arm at **+0.056 macro-F1** (0.679 → 0.735 on the shipped
chain) — worthwhile and, as `cause.md` said, second-order compared with the filter fix,
which is worth +0.26. Both scopes `cause.md` rejected reproduce on clean signal:

| scope | macro-F1 | accuracy | |
|---|---:|---:|---|
| none (strict, training-set statistics only) | 0.679 | 0.797 | the deployment number |
| **per-participant mean/std** | **0.735** | **0.852** | adopted, declared |
| per-participant robust (median/MAD) | 0.661 | 0.802 | rejected — worse than doing nothing |
| per-recording | **0.036** | 0.099 | catastrophic, exactly as predicted |

Per-recording reproduces `cause.md`'s 0.036 to three decimal places. Each recording holds a
single condition, so per-recording scaling removes the class signal itself — the clearest
confirmation available that the screening harness is measuring the same thing `cause.md` did.

Both arms are prespecified as separate configs: `five_class_nested_loso.yaml` (strict — the
number that supports a no-calibration deployment claim) and `five_class_calibrated.yaml`.

---

## 6. Phase 4 — the guard is now measured, not assumed

The existing sweep answered "how much data does the guard cost?" It never answered the
question the guard exists for. `evaluation/segmentation.py::trigger_onset_alignment` times
when the EMG envelope actually crosses between its pre- and post-onset plateaus.

On 432 onsets (358 with detectable activation; 17 % show none):

| quantity | value |
|---|---:|
| median lag, trigger → envelope | **−0.075 s** |
| onsets where activity **precedes** the trigger | **79.9 %** |
| onsets in the harmful direction (activity follows) | 71 / 358 = 19.8 % |
| median harmful lag | 0.054 s |
| p95 harmful lag | 0.282 s |
| largest harmful lag ever measured | 0.352 s |

**The trigger errs in the benign direction.** Four times out of five the muscle is already
active when the mark goes high, so a window just inside the trigger contains task activity
and its label is correct. The guard only protects against the other 20 %, whose median
error is 54 ms. The 0.5 s guard originally proposed exceeds the largest lag ever measured.

Combined with the window/guard sweep at a 1.0 s window:

| guard | total windows | smallest cell | cells < 30 |
|---:|---:|---:|---:|
| 0.10 s | 6,525 | **36** | **0** |
| 0.15 s | 6,402 | 27 | 1 |
| 0.25 s (current) | 6,173 | 11 | 1 |

**0.10 s removes every starved cell without shortening the window** — the outcome the work
order preferred, at a cost of roughly 7 % of first-windows-per-run having a partially
mislabelled leading edge. Prespecified as `configs/experiments/five_class_guard010.yaml`;
0.25 s remains the primary reported configuration until investigator sign-off, so the two
stay comparable. Recorded against `docs/open_questions.md` Q1b.

---

## 7. Phase 5 — the architecture

`DualBranchWaveletCNN` reduced each band to `Conv → pool → Conv → AdaptiveAvgPool1d(1)`: a
~7-tap detector averaged over the whole window, i.e. a per-band mean rectified amplitude.
That is a structural reason, not a capacity one, why logistic regression on band energies
matched it.

`dual_branch_temporal_cnn` is a **new model id**, so the superseded model stays
bit-reproducible (still exactly 7,485 parameters) and both appear side by side in one table.
It adds exactly the two capabilities the diagnosis names:

- **`pooling="stats"`** — mean, standard deviation and max per band instead of the mean
  alone, so *how much a band varies* is representable at all;
- **`modulation=True`** — a differentiable modulation spectrum over each band's envelope
  (`|x|` → pool to 64 frames → remove mean → rFFT → 6 bins), so the 1–2 Hz burst-relax
  rhythm separating chewing and grinding from clenching is measured directly.

Capacity is deliberately restrained: **24,173 parameters**, 3× the original. A
four-segment-pooling variant was tried first and reached 98,957 — almost all in one dense
fusion layer, against ~6,000 training windows — and was rejected. `segments` and
`stats_segments` poolings remain available for when evidence supports them.

`configs/experiments/baselines.yaml` now sweeps
`[dual_branch_wavelet_cnn, dual_branch_temporal_cnn, early_fusion_cnn, bilstm]` on identical
inputs, folds, seeds and selection budget.

---

## 8. Phases 6–7 — aggregation and calibration

**Aggregation is implemented and labelled trial-level at the source.**
`aggregate_within_recording` carries `interpretation: "screening, TRIAL-LEVEL
(single-condition recordings)"` in its own return value, so the label cannot be lost between
the computation and the table. Window-level stays primary.

| context | windows | macro-F1 | accuracy |
|---|---:|---:|---:|
| 1.0 s (single window) | 1 | 0.735 | 0.852 |
| 1.5 s | 2 | 0.755 | 0.865 |
| 2.5 s | 4 | 0.778 | 0.878 |
| 3.5 s | 6 | 0.805 | 0.892 |
| **5.5 s** | 10 | **0.821** | **0.903** |
| 8.5 s | 16 | 0.815 | 0.899 |

Rising steeply to about 5.5 s, then flat — **one departure from `cause.md`**, which reported
the gain as monotone through 8 s. On the shipped chain the curve peaks at 5.5 s and dips
slightly at 8.5 s. The difference is small and the sample is five participants, so the
honest reading is that the benefit saturates around 5 s rather than continuing to climb.
Either way the conclusion is unchanged: **a 1 s window is a binding constraint, not a tuned
parameter.**

**Calibration.** `evaluation/calibration.py` implements temperature scaling: a single scalar
fitted by bounded scalar minimisation of the NLL. It is monotone, so accuracy, macro-F1 and
every AUC are unchanged by construction, and `TemperatureScaler.report` asserts the
predictions did not move. It carries its fitting set, and `assert_not_fitted_on` refuses to
apply a temperature to a participant it saw — it must be fitted on inner-validation folds.

The current run bundles do not store inner-validation probabilities, so the primary numbers
are reported **uncalibrated**, with the ECE stated and the note that every AUC remains
valid because AUC is rank-based. `summarise_uncalibrated` produces exactly that sentence.

### Two bugs this work introduced, and how they were caught

Recorded because the point of the exercise is that defects should not survive silently.

1. **The calibration block consumed the rest class.** Each participant has exactly one
   dedicated rest recording, so an uncapped block took all of it and rest vanished from the
   training labels. It surfaced only as a downstream warning about class weights. Fixed
   with a 20-window cap, and `_assert_leaves_every_class_trainable` now refuses any block
   that would consume a whole participant × family — so the next occurrence names its own
   cause.
2. **`NormalizationConfig.to_dict()` emitted a derived property**, `is_transductive`.
   `resolved_config.yaml` is re-loaded by `bruxism-figures` and by every resumed run, and
   `ExperimentConfig` rejects unknown keys by design — so the run bundle became unreadable
   by the code that wrote it. Caught by the integration suite, 20 minutes deep. Fixed by
   keeping `to_dict()` to fields only and putting the derived flag in `folds.json`, plus a
   new test asserting every configuration round-trips through the loader with an unchanged
   hash.

The second is the more interesting failure: it was invisible to 255 unit tests and to a
full training run, because nothing in either path re-reads the config it just wrote.

**Open question Q9 is now half answered** in `docs/open_questions.md`, with the spectra as
evidence: `notch_filter: Index 9` removed 60 Hz and nothing else. Still blocking: the
acquisition software's index table to confirm it, what `bandpass_filter: Index 143` means,
and whether the setting was identical across sessions. The methodological point is recorded
for the Methods section — the textbook "notch the fundamental, then band-pass" recipe is
actively misleading on hardware that already notches, because it removes nothing and looks
correct in a filter diagram.

---

## 9. What is not finished, and why

**Phase 2 — the confirmatory runs.** Three were launched: the primary five-class nested
LOSO, the baselines sweep and the modality ablation. Together they are roughly 16 hours of
compute on this machine, and run three-at-once they contend badly (load 34 on 22 cores;
50 min per fold instead of 15).

- **Primary run** — `outputs/runs/five_class_nested_loso_20260804T025108_2b6fb5ac`,
  3 seeds × 5 folds, 17 model fits per fold. **1 of 15 folds complete at the end of this
  session** (18 min per fold running alone, ~4 h remaining). Rerunning the same command
  resumes it; `--max-folds N` caps it. The first fold's result is in the Headline: S01
  macro-F1 0.174 → 0.673.
- **Restarted once, deliberately.** The first attempt reached 4 of 15 folds before the
  `is_transductive` serialisation bug (above) was found. Its bundle was scientifically
  correct — same filter chain, splits, model and seeds — but `resolved_config.yaml`
  contained a key `ExperimentConfig` rejects, so the bundle could not be re-opened by
  `bruxism-figures` or by a resumed run. Rather than ship a run bundle that its own code
  cannot read, it was archived to
  `outputs/runs/SUPERSEDED_unreadable_config_.../` with `WHY_SUPERSEDED.txt`, and the run
  restarted with the fix. The new bundle was verified to reload with a matching
  `config_hash` before being left to run.
- **Baselines** (45 folds) and **ablations** (90 folds) were stopped after 11 completed
  folds each so the primary run could finish. The baselines config has since been changed
  to include `dual_branch_temporal_cnn`, which changes its configuration hash — so its 11
  folds are superseded rather than resumable, deliberately.
- **Six bundles carrying the unreadable key were archived**, each with a
  `WHY_SUPERSEDED.txt`: the first primary attempt, the two partial sweeps, and three smoke
  runs. Every remaining bundle under `outputs/runs/` was then verified to reload through
  `load_experiment_config`. Nothing readable was deleted; the archived directories keep
  their folds, logs and figures.

To finish Phase 2:

```bash
cd Code
bruxism-train     --config configs/experiments/five_class_nested_loso.yaml   --data-root ../Data   # resumes
bruxism-baselines --config configs/experiments/baselines.yaml               --data-root ../Data
bruxism-ablations --config configs/experiments/modality_and_no_chewing.yaml --data-root ../Data
```

Run them **one at a time**. Then Phases 3–5 have prespecified configs waiting:
`five_class_calibrated.yaml` and `five_class_guard010.yaml`.

**Consequences for the manuscript's open questions:**

- **RQ2 (how much does audio add?)** — not re-measured on clean signal by the real
  pipeline. Screening on the corrected chain is the best available evidence and it says the
  audio contribution is real but small. The +0.035 macro-F1 figure from the contaminated
  runs must not be reported: it was measured through a channel that was 85–99 %
  interference.
- **RQ3 (does the dual-branch CNN beat the baselines?)** — not re-measured. On the
  corrected chain, screening logistic regression reaches 0.679 macro-F1 and gradient
  boosting 0.646, against the network's 0.435 on the contaminated chain. Until the
  confirmatory sweep runs, **RQ3 has no evidence in either direction on clean signal**, and
  the honest baseline must appear in every table.

**Not attempted:** segment-level (variable-length) examples — Phase 4's option 2. The guard
reduction achieves Phase 4's acceptance criterion without it, and it is a larger change than
the remaining time allowed. It remains the right answer to the 1 s-vs-8 s tension.

---

## 10. The work order's own checklist, answered

| # | "How to know you are done" | Status |
|---|---|---|
| 1 | Filtered mains-harmonic power below 5 % in every participant × class cell | ⚠️ **24 of 25 cells.** S02 chewing is 8.0 %, down from 99.8 %; its excess over the local floor is 3.2× down from 43,258×. See §4. |
| 2 | No participant scores below chance on any seed | ✅ **confirmed for the two that failed.** S02 was *below chance* at 0.091 accuracy; it is now 0.883 — the best participant in the run. S01: 0.259 → 0.777. 13 folds pending, but the two failures this criterion was written for are resolved. |
| 3 | Participant-level macro-F1 spread below ~0.35 (was 0.69) | ✅ **in screening** — 0.743 − 0.548 = **0.195**. Confirmatory run pending. |
| 4 | Training-fit to held-out macro-F1 gap below ~0.25 (was 0.47) | ⏳ requires the confirmatory run; screening fits one model per fold and has no comparable fit diagnostic. |
| 5 | Logistic regression **and** gradient boosting reported beside the network on identical data | ⚠️ LR (0.679) and GBM (0.646) are reported on identical windows and folds. The network's clean-signal number is pending; `baselines.yaml` now sweeps all four architectures. |
| 6 | Every leakage and figure test green; the new mains-notch test green | ✅ **255 unit + 19 integration tests green**, including 6 new mains-notch tests, 7 new calibration-leakage tests, 4 new probability-calibration tests and 3 new screening tests. |
| 7 | Every manuscript number regenerated from the new bundle | ❌ **not done** — depends on the confirmatory runs. Superseded numbers and reasons are recorded in §11 so nothing is quietly replaced. |
| 8 | `cause.md` §7 revisited: RQ2 and RQ3 answered from clean signal | ⚠️ partially — see §9. Screening is the best current evidence; neither has a confirmatory answer yet. |

Items 2, 3 and 5 are answered by screening and expected to hold in the confirmatory run,
but a screening estimate is not a result and they are not marked done.

---

## 11. Superseded numbers

Every result produced before 2026-08-03 was measured through a channel that was 85–99 %
powerline interference and **is not comparable to anything produced after it**. Specifically
superseded:

| Number | Old value | Status |
|---|---|---|
| Five-class macro-F1 / accuracy | 0.435 / 0.550 | superseded; corrected-chain confirmatory run pending |
| S02 held-out accuracy | 0.091 / 0.040 / 0.076 (below the 0.20 chance level) | superseded |
| Audio contribution (RQ2) | +0.035 macro-F1 | superseded; do not report |
| Architecture comparison (RQ3) | CNN ≈ logistic regression | superseded; re-measurement pending |
| Table 5 per-class supports | 1,877 / 2,503 / 1,861 / 5,604 | already known wrong — legacy whole-recording policy; this pipeline produces 333 / 799 / 816 / 3,635 |
| Manifest hash | `46aae2d6394de668` | now `7ebbcc8d74c0bbe0` (mains columns + quality policy `2026-08-03.1`) |

---

## 12. Files changed

**New**

| Path | Purpose |
|---|---|
| `src/bruxism/preprocessing/interference.py` | Mains-contamination measurement, harmonic excess ratio |
| `src/bruxism/preprocessing/calibration.py` | The calibration block |
| `src/bruxism/evaluation/screening.py` | LOSO screening harness on 35 band-energy features |
| `src/bruxism/evaluation/signal_quality.py` | Per participant × class contamination, contrast, spread |
| `src/bruxism/evaluation/segmentation.py` | Trigger-onset alignment, window/guard sweep |
| `src/bruxism/evaluation/calibration.py` | Temperature scaling, ECE |
| `src/bruxism/cli/screen_filters.py` | `bruxism-screen` |
| `configs/experiments/five_class_guard010.yaml` | Phase 4, prespecified |
| `configs/experiments/five_class_calibrated.yaml` | Phase 3, prespecified |

**Modified**

`preprocessing/filters.py` (harmonic notch bank, comb, spectral interpolation, constant-width
notches, `emg_mains` config form), `preprocessing/normalization.py` (participant scope,
calibration), `data/manifest.py` (mains columns, schema 1.1), `data/quality.py`
(`MAINS_CONTAMINATION`, policy `2026-08-03.1`), `data/splits.py` (`exclude_sample_ids`),
`data/dataset.py` (subject-aware normalisation), `models/dual_branch.py` (temporal pooling,
modulation), `models/baselines.py` (`dual_branch_temporal_cnn`), `training/engine.py`,
`runner.py`, `visualization/signal_figures.py` (data spectrum overlay),
`visualization/run_figures.py`, `cli/audit_dataset.py`, `reporting.py`, `pyproject.toml`
(`bruxism-screen`), all five experiment configs, `Code/README.md`,
`docs/open_questions.md` (Q9, Q1b), `docs/experiment_protocol.md` (§5 preprocessing, §5.1
normalisation and the calibration block, §6.1 calibration, §6.2 aggregation), and the unit
tests.

---

## 13. Reproducing this

```bash
cd Code
pytest tests/unit tests/integration                           # 255 + 19 green
bruxism-audit  --data-root ../Data --no-video                 # contamination + sweeps
bruxism-screen --data-root ../Data                            # filter-variant comparison
bruxism-train  --config configs/experiments/five_class_nested_loso.yaml --data-root ../Data
```

Artifacts: `outputs/data_audit/7ebbcc8d74c0bbe0/`,
`outputs/screening/<stamp>/`, `outputs/runs/<run_id>/`.
