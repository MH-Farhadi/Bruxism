# What is going wrong

**Run analysed:** `Code/outputs/runs/five_class_nested_loso_20260803T173724_a8d59c33`
(3 seeds × 5 outer folds, 3.77 h, commit `949d7504`, config hash `a8d59c33`, finished 2026-08-03 17:23)

---

## Headline

The pipeline is not broken. The **signal it is trained on is**.

About **85–99 % of the in-band power of the "filtered" EMG is powerline interference at
180 / 300 / 420 Hz**, which the production filter chain never removes. The chain notches
60 Hz — the one mains frequency the acquisition hardware had *already* removed — and lets the
odd harmonics through untouched. The network is therefore mostly measuring electrode
impedance and cable routing, which differ per participant and carry no information about jaw
activity.

Fixing only the filter, changing nothing else, takes a plain 35-feature logistic regression
from **macro-F1 0.428 → 0.705** and **accuracy 0.595 → 0.820** under the same
leave-one-subject-out protocol. The trained dual-branch CNN currently scores **0.435 / 0.550**.

| Pipeline (five-class, LOSO, participant-level means) | macro-F1 | accuracy | bal. acc |
|---|---:|---:|---:|
| **This run — dual-branch wavelet CNN, 3 seeds** | **0.435** | **0.550** | 0.540 |
| 35-feature logistic regression, same filters | 0.428 | 0.595 | 0.488 |
| …with every mains harmonic notched | **0.705** | **0.820** | 0.761 |
| …plus per-participant standardisation | **0.737** | **0.853** | 0.790 |
| …plus 3 s trial-level aggregation † | 0.794 | 0.891 | — |
| …plus 8 s trial-level aggregation † | 0.836 | 0.914 | — |

† Aggregation across consecutive windows *inside one recording*. Because each recording holds
a single condition, this is trial-level, not stream-level, performance and must be labelled as
such. See "Honest limits of these estimates".

---

## 1. Root cause: the filter chain removes the wrong mains frequency

### Evidence

Raw, **unfiltered** EMG, top spectral peaks in the 20–450 Hz band, every participant:

| Recording | Top peaks (Hz) |
|---|---|
| `S01_rest_20250804T102808` | 180.2, 179.9, 300.0, 300.3, 180.5, 179.6 |
| `S02_rest_20250805T142047` | 179.9, 180.2, 179.6, 300.0, 299.7, 300.3 |
| `S05_rest_20250807T144939` | 179.9, 180.2, 300.0, 179.6, 299.7, 300.3 |

Peak-to-local-noise-floor ratio in `S02`'s rest recording:

| Frequency | Peak / local floor |
|---|---:|
| 60 Hz | **0×** — absent from the raw data |
| 180 Hz | **846,000×** (≈ 59 dB) |
| 300 Hz | **194,000×** |
| 420 Hz | **37,000×** |

Every recording's metadata carries `notch_filter: Index 9` and `bandpass_filter: Index 143`.
The hardware notch removed 60 Hz. `docs/open_questions.md` Q9 flagged those fields as
unexplained; this is the answer — and the consequence is that
`preprocessing/filters.py::_default_emg_stages` notches a frequency that is already gone and
passes the three that dominate the recording.

Share of the **filtered** 20–450 Hz EMG power that sits within ±3 Hz of a mains harmonic:

| Participant | rest | clench | chewing |
|---|---:|---:|---:|
| S01 | 88.5 % | 21.6 % | 21.1 % |
| **S02** | **99.8 %** | **85.2 %** | **88.9 %** |
| S03 | 95.7 % | 28.1 % | 9.3 % |
| S04 | 94.7 % | 39.5 % | 13.8 % |
| S05 | 92.8 % | 10.1 % | 9.1 % |

A clean surface-EMG channel would show a few per cent here. **Every participant's "rest" class
is essentially a recording of mains interference.** S02's channel is interference in *every*
class — which is exactly why S02 is the participant that fails hardest.

### What removing it does

Adding notches at 120/180/240/300/360/420 Hz (10 % of the band, ~6 Hz each) and re-measuring:

Median EMG RMS per participant and class, **before → after**:

| Participant | rest | clench | chewing |
|---|---|---|---|
| S01 | 69.6 → **8.6** | 121.3 → 58.5 | 114.2 → 75.8 |
| S02 | 87.8 → **3.6** | 169.6 → 59.9 | 239.2 → 56.6 |
| S03 | 12.8 → 2.2 | 62.0 → 44.3 | 92.9 → 86.5 |

Class contrast (activity ÷ that participant's own rest), **before → after**:

| Participant | movement | clench | grinding | chewing |
|---|---|---|---|---|
| S01 | 1.05 → **1.53** | 1.74 → **6.82** | 2.16 → **7.00** | 1.64 → **8.84** |
| S02 | 3.63 → **6.29** | 1.93 → **16.69** | 1.89 → **5.28** | 2.72 → **15.78** |
| S03 | 3.20 → 9.35 | 4.86 → 19.83 | 2.91 → 12.77 | 7.28 → **38.75** |
| S04 | 2.53 → 6.78 | 3.24 → 9.69 | 3.19 → 10.05 | 3.56 → 18.33 |
| S05 | 1.84 → 3.78 | 6.07 → 28.88 | 2.73 → 13.86 | 3.66 → 18.12 |

Before the fix, **S01's resting EMG (69.6) was larger than S03's and S04's clenching EMG
(62.0 and 51.0)**. The classes were not separable in the feature the model actually uses.
After the fix, every participant shows a 5–39× rest-to-activity contrast.

Between-participant amplitude spread falls from **4.1× to 2.6×**.

---

## 2. The failure mode this produces

Held-out predictions, seed 0. The two failing participants do not make scattered errors — they
make one systematic error each:

**S01** (accuracy 0.259) — never predicts `rest` or `instructed_grinding` *at all*:

| true ↓ / predicted → | rest | movement | clench | grinding | chewing |
|---|---:|---:|---:|---:|---:|
| rest | 0 % | **70 %** | 30 % | 0 % | 0 % |
| movement | 0 % | 88 % | 12 % | 0 % | 0 % |
| clench | 0 % | 46 % | 54 % | 0 % | 0 % |
| grinding | 0 % | 49 % | 50 % | 0 % | 0 % |
| chewing | 0 % | 36 % | 45 % | 0 % | 19 % |

**S02** (accuracy 0.091 — **below the 0.20 chance level**) — calls almost everything `clench`:

| true ↓ / predicted → | rest | movement | clench | grinding | chewing |
|---|---:|---:|---:|---:|---:|
| rest | 0 % | 7 % | **92 %** | 0 % | 1 % |
| movement | 0 % | 0 % | 18 % | 82 % | 0 % |
| clench | 0 % | 0 % | 98 % | 2 % | 0 % |
| grinding | 0 % | 0 % | 70 % | 30 % | 0 % |
| chewing | 0 % | 0 % | 72 % | 25 % | 3 % |

75 % of all S02 predictions are `clench`. This is the signature of a decision function driven
by **overall amplitude**: when the held-out participant's interference level is high relative
to the training participants, every window slides up the amplitude ordering into one class.
Below-chance accuracy is not noise — it is a systematic mis-mapping.

The per-participant spread is the whole story:

| Participant | accuracy (seed 0/1/2) | macro-F1 |
|---|---|---|
| S01 | 0.259 / 0.281 / 0.267 | 0.17 / 0.21 / 0.16 |
| **S02** | **0.091 / 0.040 / 0.076** | 0.06 / 0.04 / 0.06 |
| S03 | 0.739 / 0.809 / 0.814 | 0.54 / 0.56 / 0.63 |
| S04 | 0.838 / 0.820 / 0.763 | 0.63 / 0.64 / 0.57 |
| S05 | 0.819 / 0.827 / 0.812 | 0.76 / 0.78 / 0.73 |

Three seeds agree to ±0.01. This is not variance — it is a reproducible property of two
participants' recordings.

---

## 3. Second cause: no cross-participant normalisation

Normalisation is a per-channel z-score fitted on the **pooled** training participants. That is
correctly leakage-free, but it does not equalise participants: a single mean/std for four
people cannot align a fifth whose scale differs by 4×.

The generalisation gap is entirely at the participant boundary:

| Measured on | macro-F1 |
|---|---:|
| Training data (final refit, fit diagnostic) | **0.907** |
| Inner-validation participants | 0.498 |
| Outer held-out participant | 0.435 |

The model fits its training participants almost perfectly and loses 0.41 macro-F1 the moment
it crosses a person. Inner-validation ≈ outer test, so **model selection is fine** — the model
simply cannot transfer. Figure `22_embedding_tsne` in the run's own figure folder shows the
same thing visually: held-out embeddings cluster by *participant*, not by class.

Standardising each participant's features by that participant's own statistics adds
**+0.099 macro-F1** on contaminated data and **+0.032** on clean data (0.705 → 0.737). Note
the ordering: **once the filter is fixed, normalisation is a second-order effect.** The filter
fix needs no protocol change and no calibration concession.

Two normalisation scopes were tested and rejected:

| Scope | macro-F1 | Why |
|---|---:|---|
| Per-**recording** | **0.036** | Each recording contains one class, so per-recording scaling removes the class signal itself. Catastrophic — do not do this. |
| Per-participant **robust** (median/MAD) | 0.409 | The distribution is multimodal and chewing-dominated; MAD tracks the majority class. |
| Per-participant mean/std | **0.527** | Best. |

---

## 4. Third cause: the architecture cannot use anything but amplitude

`DualBranchWaveletCNN` has **7,485 parameters**: EMG branch 1,656, mic branch 432, fusion
5,232, classifier 165. Each band branch is `Conv1d(k=3) → pool2 → Conv1d(k=3) →
AdaptiveAvgPool1d(1)` — a ~7-tap local detector whose output is then **averaged over the whole
window**. The representation is therefore a per-band mean rectified amplitude: 3 numbers ×
16 channels per modality. Every temporal structure inside the window is averaged away.

The proof is direct: a **logistic regression on 35 hand-written features** (log RMS in five
db4 bands per channel, waveform length, zero-crossing rate, six coif5 mic bands) matches the
CNN exactly — 0.428 vs 0.435 macro-F1, and beats it on accuracy (0.595 vs 0.550). The
wavelet-CNN is contributing nothing over hand-computed band energies. Under the same
conditions gradient boosting reaches 0.531 / 0.747.

This also means the manuscript's RQ3 ("does the dual-branch CNN outperform the baselines")
currently has the answer *no* — and the reason is that the architecture, as built, is a
band-energy calculator.

---

## 5. Fourth cause: the segmentation policy starves the minority classes

Participants marked the trigger with wildly different granularity. S01 marked one long run per
bout; S02 marked every repetition:

| | S01 clench | S02 clench | S02 movement |
|---|---:|---:|---:|
| trigger runs | 14 | 59 | 39 |
| median run length | 9.0 s | 1.7 s | **1.3 s** |
| windows emitted | 376 | 53 | **11** |

With a 1.0 s window and a 0.25 s guard on each side, a run must exceed 1.5 s to produce a
single window. Most of S02's repetitions produce **none**. The resulting class table is
grotesquely uneven — chewing is 58.9 % of all windows (3,635) while movement is 5.4 % (333),
and the smallest participant×class cell holds 11 windows.

Window/guard sensitivity, measured:

| window | guard | total windows | S02 movement | S02 clench | smallest non-chewing cell |
|---:|---:|---:|---:|---:|---:|
| 1.00 s | 0.25 s | 6,173 | 11 | 53 | 11 |
| 1.00 s | 0.10 s | 6,525 | 36 | 82 | 36 |
| 0.75 s | 0.10 s | 9,001 | 63 | 141 | 63 |
| **0.50 s** | **0.10 s** | **13,983** | **128** | **258** | **128** |

Shortening the window to 0.5 s with a 0.1 s guard yields **2.3× more data** and removes every
starved cell — at the cost of halving the observation window, which pulls against §6. The real
resolution is to stop forcing a fixed window onto variable-length trigger runs.

---

## 6. Fifth cause: one second is too short for the classes that matter

Chewing and grinding are distinguished from clenching by *rhythm* — burst-relax cycles at
roughly 1–2 Hz. A 1-second window cannot contain a cycle, and the architecture averages over
it anyway. Aggregating evidence across consecutive windows recovers a large amount:

| Context | macro-F1 | accuracy |
|---|---:|---:|
| 1.0 s (single window) | 0.737 | 0.853 |
| 2.0 s | 0.769 | 0.876 |
| 3.0 s | 0.794 | 0.891 |
| 5.0 s | 0.822 | 0.905 |
| 8.0 s | 0.836 | 0.914 |

Monotone across the whole range, i.e. the window length is a binding constraint, not a tuned
parameter. §5 and §6 pull in opposite directions and that tension is the argument for
segment-level rather than fixed-window classification.

---

## 7. What this changes in the manuscript

- **RQ2 (how much does audio add?) is currently answered on broken EMG.** With the
  contaminated EMG the microphone looked worth **+0.035** macro-F1 over EMG alone. With the
  EMG cleaned, the same comparison gives **+0.014** (EMG-only 0.723, mic-only 0.437, fusion
  0.737). The audio contribution is real but small, and reporting the larger number would be
  reporting an artefact of a broken filter.
- **RQ3 (does the dual-branch CNN beat the baselines?)** — as it stands, no. The architecture
  is matched by logistic regression on band energies.
- The **per-class supports in Table 5** (movement 1,877 / clench 2,503 / grinding 1,861 /
  chewing 5,604) belong to the legacy whole-recording policy, not to anything this pipeline
  produces (333 / 799 / 816 / 3,635). They must be regenerated, not carried over.
- The held-out probabilities are **badly calibrated** (ECE 0.276, figure `19_calibration`).
  Any AUC reported from them is still valid — AUC is rank-based — but the manuscript should
  state that no calibration was applied.

---

## 8. What is *not* wrong (ruled out)

These were checked and are clean; do not spend effort here.

- **No leakage.** The normaliser is asserted against the held-out participant on every fold;
  `OuterFold.release_test_ids` seals the test set; `assert_exactly_once` passes on the ledger;
  inner LOSO uses four folds, never five.
- **Not a training bug.** The model reaches 0.907 macro-F1 on its own training data. It
  optimises fine; it does not transfer.
- **Not seed noise.** Three seeds agree to ±0.008 macro-F1.
- **Not model selection.** Inner-validation (0.498) tracks outer test (0.435).
- **Not the loss or class weights.** Focal loss with balanced class weights is already on.
- **Not non-determinism.** This run reproduces the 2026-07-29 run's metrics to every decimal
  from an identical config hash. The pipeline's reproducibility machinery works.

---

## 9. Honest limits of these estimates

- The comparison numbers come from a **screening harness**: a single logistic
  regression / gradient boosting per fold, no nested hyperparameter selection, and a feature
  set chosen after looking at these five participants. They are directionally reliable and the
  effect sizes are far larger than the noise, but they are **optimistic** relative to a fully
  prespecified nested protocol. The definitive numbers must come from the real pipeline.
- **Per-participant standardisation uses the held-out participant's own unlabelled data.**
  That is transductive test-time adaptation, not leakage of labels — but it is a change of
  protocol and must be declared as a calibration step, with the calibration data specified. A
  scripted calibration block (one guided repetition per task) recovered a substantial part of
  the gain in testing. The filter fix requires none of this.
- **The context/aggregation numbers are trial-level.** Every recording contains one condition,
  so averaging over 8 s inside a recording approaches a majority vote over a homogeneous
  trial. It is a valid *trial-level* result and an invalid *stream-level* one.
- Five participants. Every number here describes this sample.

---

## 10. Priority order

| # | Fix | Expected gain | Cost | Risk |
|---|---|---|---|---|
| 1 | Notch every mains harmonic in the EMG chain | **+0.28 macro-F1, +0.23 accuracy** | one config change | none — 10 % of the band, no protocol change, no leakage concession |
| 2 | Add per-participant standardisation as a declared calibration step | +0.03 macro-F1 | moderate | must be declared; transductive |
| 3 | Replace the amplitude-summarising architecture, or admit the baseline wins | unlocks the rest | high | none |
| 4 | Re-tune window/guard, or move to segment-level classification | fixes the 11-window cells | moderate | changes the window index hash |
| 5 | Report trial-level aggregation as a labelled secondary analysis | +0.10 macro-F1 at 8 s | low | must not be presented as stream detection |

Do **1** before anything else, and re-run the whole comparison afterwards. Every other
conclusion in the project — including which architecture wins and how much audio contributes —
was measured through a channel that was 85–99 % powerline interference.

The concrete work order is in [`new_prompt.md`](new_prompt.md).
