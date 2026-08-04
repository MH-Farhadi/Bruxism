# Analysis protocol

**Version:** 1.0 · **Date:** 2026-07-27 · **Quality policy:** `2026-07-27.1`

This document is the prespecification. Anything that could be chosen after seeing a result
is declared here first, and the code enforces it. Where a decision is still awaiting
investigator sign-off it is marked ⚠ and cross-referenced to `open_questions.md`.

---

## 1. Scope

This project performs **classification of instructed, awake jaw and tooth-contact tasks**
recorded from five participants in a single controlled laboratory session.

It is **not**:

- a clinical bruxism detection study,
- a sleep-bruxism study,
- a study of spontaneous or naturalistic behaviour,
- an ambulatory, wearable or real-time validation.

The acquisition token `natural_bruxing` is a filename label chosen at recording time. It is
reported throughout as **instructed grinding**. Performing grinding on command in a
laboratory is voluntary, emulated grinding, and is plausibly easier to detect than
spontaneous behaviour — a bias in the optimistic direction.

---

## 2. Labelling policy

### 2.1 Segmentation

Two policies exist. Only one may produce reported results.

| Policy | Use | Safe for inference |
|---|---|---|
| `trigger_constrained` | The approved scientific policy | **yes** |
| `whole_recording_legacy` | Reproducing historical counts | **no** |

Every window, prediction row and artifact carries `safe_for_inference`. Artifacts produced
under the legacy policy are stamped `false` and must never be quoted as findings.

### 2.2 The trigger-constrained policy

Confirmed by the investigator on 2026-07-27: `Trigger == 1` marks intervals during which
the participant was actively performing the named task.

1. **Filter first, window second.** The production filter chain is applied to the whole
   continuous recording, then windows are cut from the filtered array. Filtering each
   window separately injects a transient at both edges of every example — measured at more
   than **1× the signal standard deviation** in the first 50 samples
   (`tests/unit/test_signal_processing.py::test_filtering_per_window_corrupts_the_window`).
2. **Window / stride:** 1.0 s / 0.5 s (1200 / 600 samples at 1200 Hz).
3. **Containment:** a window is emitted only if it lies wholly inside one homogeneous
   approved segment. A window can never span two conditions or a trigger transition.
4. **Transition guard ⚠:** 0.25 s excluded on *each* side of every trigger edge. See
   `open_questions.md` Q1b for the sensitivity table and why 0.5 s was not adopted.
5. **Startup guard:** 0.5 s at the start of every recording. This replaces the prototype's
   unexplained blanket 3 s skip and is justified by measurement: amplifier settling
   transients were observed in 66/100 recordings, all settling within **0.40 s**, with peak
   excursions up to **12,580×** the recording's robust scale.
6. **Rest source:** the five dedicated rest recordings only. Trigger-off intervals inside
   active recordings are **not** rest (`allow_trigger_off_as_rest: false`).
7. **Provenance:** every window records subject, recording, start/end sample and second,
   trigger run index, condition, task family, segment source and policy.
8. **Uniqueness:** sample ids are unique by construction and asserted; a duplicate raises.

### 2.3 Task families

```
rest                 <- dedicated rest recordings
movement             <- open_close, deviation_left_right, protrusion_retrusion
clench               <- bite_left, bite_right, molar_clench, incisor_clench
instructed_grinding  <- natural_bruxing   (raw token preserved, renamed for reporting)
chewing              <- cheese, carrots, gum
```

---

## 3. Classification tasks

Each is a **separately trained** experiment under the same outer protocol. Binary and
ternary results are not obtained by relabelling the five-class confusion matrix; if a
post-hoc collapse is ever reported it must be labelled as such.

| `task_id` | Classes | Endpoint |
|---|---|---|
| `five_class` | rest, movement, clench, instructed_grinding, chewing | **primary** |
| `no_chewing_four_class` | rest, movement, clench, instructed_grinding | secondary |
| `binary_tooth_contact` | non_tooth_contact / tooth_contact | secondary |
| `ternary` | rest_or_movement / tooth_contact / chewing | secondary |
| `legacy_active_four_class` | movement, clench, instructed_grinding, chewing | **provenance only** |

`no_chewing_four_class` **excludes** chewing windows; it does not merge them.

---

## 4. Nested leave-one-subject-out

Per outer fold:

1. One participant is sealed as the outer test set.
2. Participant-grouped inner LOSO over the remaining four → **four inner folds**. Five is
   arithmetically impossible when four participants remain, and the code raises if asked.
3. Every transform, normalisation statistic, class weight, augmentation minority set and
   hyperparameter is fitted on inner-training participants only.
4. Best epoch per inner fold by the prespecified objective.
5. Epoch budget for the final refit from the inner best epochs.
6. Refit on all four outer-training participants for exactly that many epochs, with no
   early stopping and no validation set.
7. The held-out participant is evaluated **exactly once**.
8. The fold bundle is written before the next fold starts.

### 4.1 The seal

`OuterFold` stores held-out sample ids privately. `release_test_ids()` raises
`OuterFoldSealError` unless called with `purpose="final_evaluation"`, and raises again on a
second call. Selection code therefore cannot reach the outer participant even by accident —
this is a structural guarantee, not a convention.

### 4.2 Prespecified selection rules

| Decision | Rule |
|---|---|
| Objective | validation **macro-F1**, maximised (accuracy is dominated by chewing) |
| Tie-break | highest macro-F1 → lowest validation loss → **earliest** epoch |
| Epoch budget | **median** best epoch across inner folds, ceil, clamped to [min, max] |
| Trial choice | highest mean inner objective → lowest spread → lexicographic id |
| All trials fail | raises; no fallback configuration |

### 4.3 Seeds

Multiple fully specified seeds. Metrics are computed **per seed** and then summarised across
seeds as mean/std/min/max. Probabilities are **not** averaged across seeds and **no seed is
selected as best** (`MULTISEED_RULE` in `evaluation/aggregation.py`).

---

## 5. Preprocessing

**Revised 2026-08-03.** Everything produced before that date used the superseded chain in
the second table and is not comparable to anything produced after it.

| Stage | Setting | Rationale |
|---|---|---|
| Notch bank | 60, 120, 180, 240, 300, 360, 420 Hz — every mains multiple in the passband, each 8 Hz wide | The hardware had already removed the fundamental (`notch_filter: Index 9`, `open_questions.md` Q9); the surviving interference was entirely at the harmonics, at 37,000×–846,000× the local noise floor |
| Bandpass | 20–450 Hz, order 4 | surface-EMG band, below the 600 Hz Nyquist |
| Microphone | 20 Hz high-pass, order 2 | DC offset only; transducer response undocumented |
| Mode | zero-phase (`sosfiltfilt`) | **acausal / offline** |

**Superseded (every run before 2026-08-03):** a single 60 Hz notch at Q = 30, then the same
bandpass. It removed the one mains frequency the hardware had already taken out and passed
the three that dominated the recordings; 85–99 % of the in-band power of the "filtered"
signal was interference. See `cause.md` and `opus_report_1.md`.

**Notches are constant-width (8 Hz), not constant-Q.** Constant Q gives 2 Hz at 60 Hz and
14 Hz at 420 Hz — narrowest exactly where the hardware notch left a residue. At an identical
13 % total band cost, constant-width notches leave the worst-contaminated cell at 8.0 %
harmonic power and 3.2× its local floor, against 22.4 % and 25.6× for Q = 30. A comb and
spectral interpolation were both implemented and measured; both left more residue
(`bruxism-screen`, `outputs/screening/<stamp>/`).

The prototype's third stage — a 5 Hz high-pass after a 20–450 Hz bandpass — is a no-op and
was removed rather than carried forward.

**Zero-phase filtering reads samples from the future of each output sample.** It cannot
support any real-time, streaming or wearable claim. A causal mode exists
(`zero_phase: false`) and the mode used is recorded in every run bundle. A
spectral-interpolation stage, if ever selected, is acausal regardless of that setting and
`FilterChainConfig.is_causal` accounts for it.

### 5.1 Normalisation

**Strict (default, and the number that supports a no-calibration deployment claim).**
Per-channel z-scoring, fitted on training participants only, saved in every run bundle with
the list of participants that produced it, and asserted against the held-out participant
before every evaluation.

**Calibrated (`scope: per_participant`) — a protocol change, and reported as one.** Each
participant's windows are standardised by that participant's own statistics, *including the
held-out participant's*. This is transductive test-time adaptation: it uses the held-out
participant's **signal** and never their **labels**. Screening values it at about +0.04
macro-F1 once the filter is fixed (0.689 → 0.731), i.e. second-order compared with the
filter fix.

Two attributes keep the concession legible and impossible to confuse with leakage:

| Attribute | Meaning | May contain the held-out participant? |
|---|---|---|
| `Normalizer.fitted_on` | participants whose **labelled training windows** produced the pooled statistics | **No** — `assert_not_fitted_on` enforces it, unchanged |
| `Normalizer.calibrated_on` | participants whose **unlabelled calibration block** produced their own statistics | Yes, by design, and disclosed |

#### The calibration block

Defined in `preprocessing/calibration.py`. It is what a fitting session produces before any
diagnosis is attempted, and it is **not** the whole labelled session:

- the participant's **dedicated rest recording** (baseline), plus
- **one guided repetition of each task family** — the first trigger run of the first
  recording of that family (dynamic range).

Selection is deterministic (sorted order, never sampled). The block is **withheld from every
split** by `NestedLOSOSplitter(exclude_sample_ids=...)`, so no window can both set a
participant's scale and be scored by it; `assert_calibration_disjoint_from` re-checks this
at every fold. Exactly which windows produced each participant's statistics is written to
`<run_dir>/calibration_block.json`.

`calibration: all_windows_upper_bound` exists for measuring the ceiling and requires
`calibration_approved_by`, because it is an upper bound rather than a deployable procedure.

**Two scopes were measured and rejected:**

| Scope | Screening macro-F1 | Why rejected |
|---|---:|---|
| per-**recording** | 0.036 | Each recording holds one condition, so per-recording scaling removes the label itself |
| per-participant **robust** (median/MAD) | 0.409 | The distribution is multimodal and chewing-dominated; the MAD tracks the majority class |
| per-participant mean/std | **0.527** (contaminated) / **0.731** (clean) | Adopted |

### 5.2 Augmentation

Training only, enforced structurally: the augmenter raises for any stage other than
`"train"`, and `WindowDataset` refuses to accept one for a validation or test stage.
Randomness is seeded by `(run_seed, epoch, sample_id)`, so a sample's transformation is
independent of worker count, batch order and shuffling.

---

## 6. Metrics

Reported: accuracy, balanced accuracy, macro precision/recall/F1, per-class
precision/recall/F1/support, raw and row-normalised confusion matrices, one-vs-rest ROC-AUC
and average precision per class and macro, and for binary tasks sensitivity, specificity,
PPV, NPV, F1, ROC-AUC and PR-AUC.

- **Participant-level metrics are primary.** Computed within each participant, then
  summarised; all five individual values are always reported.
- **Pooled-window metrics are secondary and carry `interpretation: "descriptive_only"`.**
  Windows within a participant and recording are correlated.
- **A class absent from a fold gets `None`, never a fabricated AUC**, and the macro average
  reports how many classes it actually averaged.
- **No window bootstrap and no window-level p-values.** With five participants any
  participant-level bootstrap or permutation test is exploratory; the individual
  participant values are the evidence.

Every metric is computed from the saved prediction ledger, in which each held-out example
appears exactly once per task/model/modality/seed with its full probability vector and the
source commit, config hash, manifest hash and checkpoint hash.

### 6.1 Probability calibration

Held-out probabilities from the 2026-07/08 runs are **uncalibrated** (ECE 0.276, figure
`19_calibration`). Two consequences, both of which must appear in the manuscript:

- the probabilities must not be read as probabilities;
- **every AUC remains valid.** AUC is rank-based and no monotone rescaling of the scores can
  change it.

`evaluation/calibration.py` implements temperature scaling for runs that choose to fix it.
The temperature is a single scalar fitted by bounded scalar minimisation of the negative log
likelihood; it is monotone, so accuracy, macro-F1 and AUC are unchanged by construction and
`TemperatureScaler.report` asserts that the predictions did not move. It **must** be fitted
on inner-validation folds — participants in neither the final model's training set nor the
held-out set — and `assert_not_fitted_on` refuses to apply a temperature to a participant it
saw.

### 6.2 Temporal aggregation — trial-level, and labelled as such

Averaging held-out probabilities across consecutive windows of one recording lifts screening
macro-F1 from 0.731 (1 window) to 0.821 (16 windows ≈ 8.5 s), monotonically. This is a real
effect and a **trial-level** one: every recording in this dataset contains a single
condition, so averaging inside a recording approaches a majority vote over a homogeneous
trial.

- **Window-level is primary.**
- Aggregation is a **clearly labelled secondary analysis**, stating the aggregation length
  and the homogeneous-trial assumption. `aggregate_within_recording` carries that label in
  its own return value.
- It is **not** continuous-stream or event detection. That claim needs onset/offset
  evaluation on mixed-activity data, which this dataset does not contain.

---

## 7. Latency

Three quantities, always reported separately:

| Quantity | Meaning |
|---|---|
| Input/context latency | the 1.0 s observation window — no decision can exist sooner |
| Decision update interval | the 0.5 s stride |
| Processing latency | compute only: filtering, transform, forward pass |

Compute time alone is never described as detection latency.
