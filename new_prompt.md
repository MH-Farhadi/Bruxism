# Work order: fix the signal chain, then re-establish every result

**Read [`cause.md`](cause.md) first.** It contains the diagnosis and the measurements this plan
is built on. This file is the executable version: what to change, in what order, and what
result proves each step worked.

Everything lives in `Code/`. Nothing in `Data/` is ever modified, copied or re-encoded.

---

## Non-negotiables

These are the properties the project already has. Do not trade any of them for a better
number.

1. **No leakage.** Normalisation, class weights, augmentation minority sets and every
   hyperparameter are fitted on training participants only, and asserted against the held-out
   participant before evaluation. Inner LOSO has four folds, never five. `OuterFold` stays
   sealed. If a change makes a leakage assertion inconvenient, the change is wrong.
2. **Participants are the unit of generalisation.** Report participant-level means as primary
   and pooled-window numbers as descriptive, as the code already does.
3. **Every number is derived from a saved prediction ledger.** No metric is typed by hand or
   read off a training log.
4. **Prespecify before the confirmatory run.** The diagnosis in `cause.md` came from looking at
   these five participants. Everything below is therefore *exploratory* until it is written
   into a config and run once, as declared. Write the config, commit it, then run it. Do not
   iterate a config against the held-out score.
5. **Say what changed.** Every result produced after this work is not comparable to any result
   produced before it. The old numbers do not get quietly replaced; they get superseded with a
   stated reason.

---

## Phase 0 — Lock the diagnosis into the code

**Goal:** make it impossible for this class of defect to survive silently again.

### 0.1 Add a mains-contamination quality metric to the manifest

In `Code/src/bruxism/data/manifest.py`, for each recording compute — per EMG channel, on the
**raw** signal — the fraction of 20–450 Hz power lying within ±3 Hz of each multiple of the
mains frequency, and store it as `mains_harmonic_power_fraction` (plus the per-harmonic
breakdown). Add a `QualityFlag.MAINS_CONTAMINATION` in `Code/src/bruxism/data/quality.py`,
raised above a declared threshold (start at 0.30 for the pooled active-task windows), and
surface it in `bruxism-audit`.

Do **not** make it an automatic exclusion. It is a flag: it tells a human that a channel is
mostly interference. Excluding S02 automatically would hide the finding.

### 0.2 Make the filter figure show the mismatch

`Code/src/bruxism/visualization/signal_figures.py::plot_filter_response` currently draws the
filter response alone. Overlay the **measured mean spectrum of the data** on the same axes,
normalised to fit. A reader must be able to see, in one glance, whether the notches line up
with the peaks. In the current run they do not, and nothing in the figure set said so.

### 0.3 Regression test

In `Code/tests/unit/`, add a test that synthesises a signal with a strong 180 Hz component,
runs it through the production EMG chain, and asserts the 180 Hz power is attenuated by at
least 20 dB. This test must fail against today's `_default_emg_stages`.

**Acceptance:** the new test fails before Phase 1 and passes after it; `bruxism-audit` reports
a mains-contamination fraction per recording; the filter figure shows the data spectrum.

---

## Phase 1 — Fix the EMG filter chain  ← *do this before anything else*

**Goal:** stop training on powerline interference.

### 1.1 Change the production chain

`Code/src/bruxism/preprocessing/filters.py::_default_emg_stages` currently applies a 60 Hz
notch and a 20–450 Hz bandpass. Replace the single notch with a notch at **every multiple of
the mains frequency inside the passband** — 60, 120, 180, 240, 300, 360, 420 Hz — keeping the
bandpass. Parameterise it (`mains_hz: float = 60.0`, `notch_harmonics: bool = True`,
`quality: float = 30.0`) rather than hard-coding seven stages, and write the rationale into the
`FilterStage.rationale` fields: the hardware already removed the fundamental
(`notch_filter: Index 9` in every metadata sidecar), and the odd harmonics are 37,000–846,000×
above the local noise floor.

Then update `filters.emg_stages` in every config under `Code/configs/experiments/`. The
`FilterChainConfig` machinery already supports a list of notch stages, so no new filter type is
needed.

**Consider before implementing:** a bank of seven IIR notches at Q=30 removes ~10 % of the band
and adds phase distortion the zero-phase pass then cancels. Two alternatives are worth a quick
comparison, and whichever wins should be the one that ships:
- **spectral interpolation** — estimate and subtract the sinusoidal components, preserving
  bandwidth;
- **a single higher-Q comb**, which is cheaper and causal-friendly if a streaming claim is ever
  wanted.
Compare them on the metric in 1.3 and pick on evidence, not on convenience.

### 1.2 Answer open question Q9

`Code/docs/open_questions.md` Q9 asks what `bandpass_filter: Index 143` and
`notch_filter: Index 9` mean. The spectra now answer half of it: the hardware notch removed
60 Hz and nothing else. Record that finding against Q9, and ask the investigators to confirm
the hardware bandpass setting — if the hardware already band-limits, the offline bandpass may
be redundant too.

### 1.3 Verify

Re-measure, on the filtered signal, per participant and class:
- mains-harmonic power fraction — target **< 5 %** in every cell (currently 85–99.8 % at rest);
- class contrast (activity RMS ÷ that participant's own rest RMS) — expect roughly
  5–39× (currently 1.05–7.3×);
- between-participant amplitude spread — expect ≈ 2.6× (currently 4.1×).

Then re-run the screening comparison from `cause.md` §1. **Expected: macro-F1 0.428 → ≈ 0.70,
accuracy 0.595 → ≈ 0.82** for a plain logistic regression on band-energy features under LOSO.
If you do not see a jump of this size, stop and find out why before continuing.

**Acceptance:** filtered mains-harmonic fraction below 5 % everywhere; the Phase 0.3 test
passes; screening macro-F1 ≈ 0.70; the run's `04_filter_response` figure shows notches sitting
on the measured peaks.

---

## Phase 2 — Re-establish the baseline before touching the model

**Goal:** know what the fix alone is worth, with the real pipeline and the real protocol.

Run, unchanged except for the filter chain:

```bash
bruxism-train     --config configs/experiments/five_class_nested_loso.yaml --data-root ../Data
bruxism-baselines --config configs/experiments/baselines.yaml              --data-root ../Data
bruxism-ablations --config configs/experiments/modality_and_no_chewing.yaml --data-root ../Data
```

Every architecture and modality conclusion in the project was measured through a contaminated
channel and must be re-measured. In particular:

- **RQ2** — the audio contribution was **+0.035** macro-F1 on broken EMG and **+0.014** on
  clean EMG in screening. Re-run the modality ablation and report whatever it gives.
- **RQ3** — the dual-branch CNN currently ties a logistic regression. Re-run the baselines
  before claiming any architectural advantage.

**Acceptance:** a fresh run bundle per experiment, with figures, and a written comparison of
old vs new for every headline number. If the baseline now beats the proposed model, say so.

---

## Phase 3 — Cross-participant normalisation, declared honestly

**Goal:** close the residual participant gap without quietly changing the protocol.

Screening says this is worth ≈ **+0.03 macro-F1** once the filter is fixed (0.705 → 0.737) —
worthwhile but second-order. Treat it as a protocol change, not a preprocessing tweak.

### 3.1 Implement participant-scope normalisation

Add `scope: "per_participant"` to `NormalizationConfig` in
`Code/src/bruxism/preprocessing/normalization.py`. It standardises each participant's windows
by that participant's own statistics, computed **without labels**.

Record in the run bundle exactly which of the participant's data produced the statistics. The
existing `Normalizer.fitted_on` / `assert_not_fitted_on` machinery must keep working: using the
held-out participant's *unlabelled* signal is a declared calibration step, and it must be
impossible to confuse with using their labels.

### 3.2 Specify the calibration data

Do not use the whole labelled session — that is an upper bound, not a deployable procedure.
Define a **calibration block**: the participant's dedicated rest recording plus one guided
repetition of each task family, from which mean/std are computed and then frozen. Exclude
calibration windows from both training and evaluation so the comparison is clean.

Report both:
- **strict** — training-set normalisation only, no calibration (this is the number that
  supports a no-calibration deployment claim);
- **calibrated** — with the calibration block, clearly labelled as requiring a fitting session.

### 3.3 Do not

- **Do not normalise per recording.** Screening: macro-F1 **0.036**. Each recording holds one
  class, so per-recording scaling removes the label itself.
- **Do not use robust (median/MAD) participant statistics.** Screening: 0.409 vs 0.527. The
  distribution is multimodal and chewing-dominated.

**Acceptance:** both numbers reported; the calibration block defined in
`docs/experiment_protocol.md`; leakage tests still pass; a new test asserts that participant
scope never touches held-out **labels**.

---

## Phase 4 — Segmentation: stop starving the minority classes

**Goal:** remove the 11-window cells without shortening the observation window.

The current policy emits 6,173 windows, of which chewing is 58.9 % and S02's movement class is
**11 windows**, because S02 marked each repetition separately (median trigger run 1.3 s) while
S01 marked whole bouts (median 9.0 s). A 1.0 s window with a 0.25 s guard needs a 1.5 s run.

Measured options:

| window | guard | total | S02 movement | smallest non-chewing cell |
|---:|---:|---:|---:|---:|
| 1.00 s | 0.25 s (current) | 6,173 | 11 | 11 |
| 1.00 s | 0.10 s | 6,525 | 36 | 36 |
| 0.75 s | 0.10 s | 9,001 | 63 | 63 |
| 0.50 s | 0.10 s | 13,983 | 128 | 128 |

Shorter windows fix the starvation and break the rhythm evidence (§6 of `cause.md`), so do not
simply shrink the window. Preferred order:

1. **Reduce the guard to 0.10 s** and justify it from the data — measure how much label
   contamination a narrower guard actually admits, using the trigger-onset alignment, rather
   than assuming. Cheapest real gain.
2. **Move to segment-level examples**: treat each trigger run as one variable-length example
   (pad/mask, or pool over a variable number of frames) instead of forcing a fixed window onto
   it. This removes the starvation *and* preserves rhythm, and it is the change that makes the
   1 s-vs-8 s tension disappear.
3. **Keep 1.0 s as a reported secondary configuration** so the new results stay comparable to
   the manuscript's existing framing.

Whatever is chosen, run the guard/window sensitivity as a declared sweep and report the table —
it is evidence for the choice, and `docs/open_questions.md` Q1b already flags that the guard
width needs investigator sign-off.

**Acceptance:** no participant×class cell below ~30 windows for a class the task uses; the
sensitivity table in the run bundle; the choice justified in writing.

---

## Phase 5 — Give the architecture something to do

**Goal:** stop shipping a 7,485-parameter band-energy calculator.

`DualBranchWaveletCNN` reduces each band to `Conv → pool → Conv → AdaptiveAvgPool1d(1)` — a
~7-tap detector averaged over the entire window. It cannot represent rhythm, burst structure,
or onset shape, which is why logistic regression on band energies matches it.

Only attempt this **after** Phases 1–4, so the comparison is against clean signal.

- Replace the global average pool with something that keeps time: a few pooled segments per
  band, a temporal attention/statistics pool (mean **and** std **and** max), or a small
  recurrent/attention head over frames.
- Add explicit envelope-rhythm features or a modulation-spectrum branch — the chewing/grinding
  distinction is 1–2 Hz burst structure the current model provably discards.
- Grow capacity only where the evidence says it is used, and re-check the parameter count
  against the manuscript, which must quote it programmatically.
- **Keep the honest baseline in every table.** If gradient boosting on 35 features still ties
  the network, that is the result. `cause.md` §4 is what it looks like when it does.

**Acceptance:** the dual-branch model beats logistic regression *and* gradient boosting on
identical windows, folds, seeds and selection budget — or the manuscript's RQ3 claim is
rewritten to match reality.

---

## Phase 6 — Report the temporal-aggregation result without overselling it

Aggregating probabilities across consecutive windows lifts screening macro-F1 from 0.737 (1 s)
to 0.836 (8 s), monotonically. This is real, and it is **trial-level**: every recording contains
a single condition, so aggregating over 8 s inside one recording approaches a majority vote over
a homogeneous trial.

- Report **window-level as primary**.
- Report aggregation as a **clearly labelled secondary analysis**, stating the aggregation
  length and that it assumes a homogeneous trial.
- Do **not** present it as continuous-stream or event detection. That claim needs onset/offset
  evaluation on mixed-activity data, which this dataset does not contain.

---

## Phase 7 — Calibration, and the manuscript

- Held-out probabilities are badly calibrated (**ECE 0.276**, figure `19_calibration`). Either
  fit a temperature on inner-validation folds only and report calibrated ECE, or state
  explicitly that probabilities are uncalibrated. AUC is rank-based and unaffected either way.
- **Regenerate every number in `Main_2.tex` from the new bundle.** The per-class supports
  currently in Table 5 (movement 1,877 / clench 2,503 / grinding 1,861 / chewing 5,604) come
  from the legacy whole-recording policy and match nothing this pipeline produces
  (333 / 799 / 816 / 3,635). Use `tables/macros.tex` so the text cannot drift again.
- Add a Methods sentence recording the mains-harmonic finding and the corrected chain. It is a
  genuine methodological contribution: the interference sat at 180/300/420 Hz because the
  hardware had already removed the fundamental, and it survived a standard 60 Hz notch plus a
  20–450 Hz bandpass. Anyone reusing that textbook chain on this class of hardware has the
  same defect.

---

## Expected outcome

Screening estimates for the five-class task, LOSO, participant-level means (see
`cause.md` §9 for why these are optimistic and must be re-measured with the real nested
protocol):

| Stage | macro-F1 | accuracy |
|---|---:|---:|
| Today | 0.435 | 0.550 |
| After Phase 1 (filter only) | ≈ 0.70 | ≈ 0.82 |
| After Phase 3 (calibrated normalisation) | ≈ 0.74 | ≈ 0.85 |
| After Phases 4–5 (segments + architecture) | to be measured | to be measured |
| Trial-level, 8 s aggregation (secondary) | ≈ 0.84 | ≈ 0.91 |

The catastrophic participants recover: S02's macro-F1 goes from **0.06 to ≈ 0.70** and S01's
from **0.17 to ≈ 0.59** on the filter fix alone.

---

## How to know you are done

- [ ] Filtered mains-harmonic power below 5 % in every participant×class cell.
- [ ] No participant scores below chance on any seed.
- [ ] Participant-level macro-F1 spread (max − min) below ~0.35 — it is **0.69** today.
- [ ] The training-fit to held-out macro-F1 gap below ~0.25 — it is **0.47** today.
- [ ] Logistic regression and gradient boosting reported beside the network on identical data.
- [ ] Every leakage test and figure test still green; the new mains-notch test green.
- [ ] Every manuscript number regenerated from the new bundle, with the superseded numbers and
      the reason recorded.
- [ ] `cause.md` §7 revisited: RQ2 and RQ3 answered from clean signal, whatever the answers are.
