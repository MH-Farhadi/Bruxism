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

| Stage | Setting | Rationale |
|---|---|---|
| Notch | 60 Hz, Q = 30 | North American mains |
| Bandpass | 20–450 Hz, order 4 | surface-EMG band, below the 600 Hz Nyquist |
| Microphone | 20 Hz high-pass, order 2 | DC offset only; transducer response undocumented |
| Mode | zero-phase (`sosfiltfilt`) | **acausal / offline** |

The prototype's third stage — a 5 Hz high-pass after a 20–450 Hz bandpass — is a no-op and
was removed rather than carried forward. ⚠ Whether the acquisition hardware already applied
its own filtering is unknown (`open_questions.md` Q3/Q9).

**Zero-phase filtering reads samples from the future of each output sample.** It cannot
support any real-time, streaming or wearable claim. A causal mode exists
(`zero_phase: false`) and the mode used is recorded in every run bundle.

### 5.1 Normalisation

Per-channel z-scoring, fitted on training participants only, saved in every run bundle with
the list of participants that produced it, and asserted against the held-out participant
before every evaluation.

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

---

## 7. Latency

Three quantities, always reported separately:

| Quantity | Meaning |
|---|---|
| Input/context latency | the 1.0 s observation window — no decision can exist sooner |
| Decision update interval | the 0.5 s stride |
| Processing latency | compute only: filtering, transform, forward pass |

Compute time alone is never described as detection latency.
