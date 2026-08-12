# audio.md — implementation report

**Date:** 2026-08-12
**Input:** [`audio.md`](audio.md) — the microphone-channel audit.
**Scope delivered:** everything in `audio.md` §2–§6, plus the manuscript edits under the
"keep Table 4 with a caveat" decision.
**Result:** 37 files modified, 3 added, ~2,240 lines net. The manuscript compiles (29 pages,
0 undefined references, 0 errors). `ruff check` and `ruff format` clean; `mypy` reports the
same 5 pre-existing errors as `HEAD`, none new. **No EMG number changed, and the two
published run bundles still carry their original configuration hashes, `2b6fb5ac` and
`cead62e4`.**

---

## 1. What the audit found, in one paragraph

The `Mic` column of the 2025-08 collection is not per-participant audio. Across 100
recordings there are only **37 distinct microphone waveforms** — against 100 distinct
waveforms on each of the four EMG channels — and **83 recordings** carry a waveform that is
bit-identical, after a circular rotation of 0.2–8 s, to another *participant's* recording of
the same condition. All four S01–S04 quiet-rest recordings share one waveform. The channel
is also unaligned with the EMG (median zero-lag envelope correlation **−0.017**, median best
lag **18 s** over the 45 chewing recordings), and 96 % of its power lies below 10 Hz, so it
behaves as a sound-level envelope rather than an acoustic waveform. The production 20 Hz
high-pass then retains a median of **1.19 %** of its variance, and what survives is at or
below the 1-count quantisation floor for clenching and grinding.

Leave-one-subject-out therefore never held out the audio. RQ2 is unanswerable from these
files, and no second copy of the signal exists to recover.

---

## 2. What was built

### 2.1 A measurement module — `Code/src/bruxism/preprocessing/mic_integrity.py` (new, 481 lines)

Every number in the audit, computed once, by name, with declared thresholds:

| Function | Answers |
|---|---|
| `waveform_fingerprint` | SHA-256 of the **sorted** samples — invariant to circular rotation, so it catches the exact failure mode here. One pass per recording, then a dict lookup, instead of comparing every recording against every other. |
| `is_circular_rotation` | Confirms a shared fingerprint *exactly*: recovers the offset by FFT, then checks array equality. A `True` means same samples in the same cyclic order, not "well correlated". |
| `measure_envelope_alignment` | Circular envelope cross-correlation between two channels; returns `r_at_zero`, `best_lag_seconds`, `peak_r`. |
| `quantisation_step`, `power_fraction_below` | LSB and the sub-10 Hz power share. |
| `measure_mic_integrity` | All of the above for one recording, plus retained variance and SNR above the quantisation floor. |
| `duplicate_groups`, `summarise_duplication`, `confirm_rotations` | The cross-recording pass. |

Thresholds are module constants with their rationale attached, versioned as
`MIC_INTEGRITY_POLICY_VERSION`.

### 2.2 Five quality flags — `Code/src/bruxism/data/quality.py`

`mic_waveform_duplicated`, `mic_emg_unaligned`, `mic_at_quantisation_floor`,
`mic_bandwidth_implausible`, `mic_channel_dead`. `QUALITY_POLICY_VERSION` bumped to
`2026-08-12.1`.

**None of them excludes a recording.** Excluding 83 of 100 files would delete the dataset to
protect a channel the EMG results never read. They populate a new
`ExclusionPolicy.audio_blocking_flags` set instead, which gates audio-consuming *runs* —
the proportionate rule. Conflict rule `R4_mic_channel_is_not_analysable_audio` records the
decision in the same registry as R1–R3.

### 2.3 Manifest schema 1.2 — `Code/src/bruxism/data/manifest.py`

Sixteen new columns per recording: `mic_sorted_sha256`, `emg_sorted_sha256` (the control),
`trigger_sorted_sha256`, `mic_duplicate_group`, `mic_duplicate_of`, quantisation step and
unique-value count, sub-10 Hz power fraction, retained variance, SNR above the floor, the
three envelope-alignment figures, the mic mains-harmonic fraction, and the chain and policy
versions the numbers were computed under.

**The structural change is a second pass.** Every check in `_iter_records` looks at one file
in isolation, and that is why the defect survived: each affected recording is individually
well-formed. `flag_shared_waveforms` now groups all recordings by channel fingerprint after
the per-file pass and raises `mic_waveform_duplicated` on any group spanning two
participants. It runs before the manifest hash is computed, so a dataset that gains a
duplicated waveform gets a different identity.

The EMG and trigger channels go through the same grouping. EMG duplication would invalidate
the whole project, so it is measured on every build rather than assumed absent — it reports
zero. The trigger is reported as **informational only**: a binary channel's sorted-sample
fingerprint collides on duty cycle alone, so a shared trigger fingerprint is not evidence of
copied data, and asserting on it would make the check fire on a correct dataset.

### 2.4 Two run-level guards — `Code/src/bruxism/runner.py`

- `assert_modality_is_supported_by_data` refuses a `fusion` or `audio_only` run when the
  manifest flags the microphone, unless the config declares `mic_defect_acknowledged_by`.
- `assert_bands_are_inside_their_passband` refuses a branch configured to read a wavelet
  band its own filter chain has already removed, unless the config declares
  `stopband_bands_acknowledged_by`.

Both run *before* the run directory is created, so a refused run leaves no half-written
bundle. Both record their findings — including the per-band passband gains — in
`data_manifest.json`.

### 2.5 The stopband check — `Code/src/bruxism/preprocessing/wavelets.py`

`assert_bands_within_passband` computes each configured band's mean power gain through the
chain that feeds it and raises `BandStopbandError` below 5 %. This is `cause.md`'s lesson on
the wavelet side: there a filter was verified against its own design rather than the data;
here a band list was chosen without checking it against the filter. Measured on the shipped
configuration:

| Branch | Band | Retained power |
|---|---|---|
| mic | **A5 (0–18.75 Hz)** | **2.95 %** ← inside the 20 Hz high-pass stopband |
| mic | D3 (75–150 Hz) | 99.7 % |
| mic | D1 (300–600 Hz) | 100 % |
| emg | A4 / D3 / D1 | 38.5 % / 73.9 % / 30.9 % — all clear |

A third of the audio branch was reading the high-pass roll-off residue, which the
`BatchNorm1d` behind it then rescaled to unit variance.

### 2.6 Supporting changes

| Area | Change |
|---|---|
| `preprocessing/filters.py` | `chain_magnitude()` and `white_noise_power_gain()` (exact factor for scaling a quantisation floor); `mic_envelope_stages()` (0.2–20 Hz, the chain this transducer actually needs); the 20 Hz high-pass rationale now records its measured 1.19 % retained variance. |
| `evaluation/signal_quality.py` | `modality="mic"` parameter and `both_modalities_quality_table()`. Pooling the two modalities in `contrast_table`/`spread_summary` now raises instead of silently dividing a mic RMS by an EMG RMS. |
| `preprocessing/normalization.py` | Opt-in `mic_scope="per_recording"`, default off, absent from `to_dict()` when default so no published hash moves. |
| `data/dataset.py` | `RecordingCache.mic_statistics()` (memoised); the `emg_only`/`audio_only` docstring reconciled with the model's "branch not constructed" behaviour and annotated with what the conditions do *not* isolate on this data. |
| `evaluation/screening.py` | `emg_only_feature_mask()` — 7 of the 35 screening features are mic-derived. |
| `models/ablations.py` | A `.. warning::` at the design-intent docstring: the harness is correct, the input is not. |
| `cli/audit_dataset.py`, `reporting.py` | New `mic_integrity` section in `data_audit.json` / `.md`, plus a `mic_integrity.csv` artifact. |
| `configs/` | All seven experiment configs carry both acknowledgements, with the reason in the file. `configs/models/dual_branch.yaml` annotates the A5 band as a declared defect. |

---

## 3. Verification

### 3.1 The pipeline reproduces the audit exactly

`bruxism-audit --data-root ../Data` now prints, from the code rather than from an ad-hoc
script:

```
| channel | distinct waveforms | of recordings | cross-subject groups |
| mic     |  37 | 100 | 20 |
| emg1    | 100 | 100 |  0 |
| emg2    | 100 | 100 |  0 |
| emg3    | 100 | 100 |  0 |
| emg4    | 100 | 100 |  0 |
| trigger |  95 | 100 |  1 |   (informational; the one group is all-zero rest triggers)

Recordings whose mic waveform also appears under a different participant: 83 of 100
  — S01 20, S02 19, S03 20, S04 20, S05 4

Quantisation step: [1.0] counts
Median power below 10 Hz:           0.960
Median variance retained by chain:  0.0119
Median zero-lag envelope r:        -0.013      Median |best lag|: 18.7 s

flags: mic_bandwidth_implausible 100, mic_waveform_duplicated 83,
       mic_emg_unaligned 45, mic_at_quantisation_floor 34, mic_channel_dead 0
```

One number differs from `audio.md`: recordings at the quantisation floor read **34**, not
42. The manifest measures over the whole recording (it predates windowing and must not
depend on a segmentation policy); the 42 in `audio.md` was measured over trigger-active
samples, which is what windows are actually cut from. Pooling active and inactive intervals
inflates the variance, so the whole-recording rule is the conservative one. Both numbers are
now stated in the flag description rather than one silently replacing the other.

### 3.2 The microphone contrast table, which no code could produce before

Each activity's median RMS divided by *that participant's own* rest RMS:

| | chewing | clenching | grinding | movement |
|---|---|---|---|---|
| **EMG** | 9.9–35.7× | 9.6–27.9× | 5.6–15.3× | 1.6–9.2× |
| **Mic** | 2.0–7.5× | **0.99–1.36×** | 1.17–1.91× | 1.2–2.3× |

An acoustic channel must put every activity above quiet rest. This one puts S01's clenching
*below* it. That single row is the cheapest possible statement that the channel is not
measuring sound, and it was unobtainable until `signal_quality.py` learned the word "mic".

### 3.3 Screening, EMG-only (audio.md §4.4)

Identical windows and folds, 6,173 windows. The 35-feature numbers **exactly reproduce the
manuscript's**, which validates the harness:

| Model | All 35 features | EMG-only (28) | Δ macro-F1 |
|---|---|---|---|
| Logistic regression | 79.7 % acc / **67.9 %** macro-F1 | 78.3 % / 63.9 % | **−4.0 pp** |
| Gradient boosting | 77.7 % acc / **64.6 %** macro-F1 | 74.8 % / 61.3 % | **−3.3 pp** |

So the seven microphone features are worth 3–4 points of the screening macro-F1 quoted in
the paper for direction — and they read the duplicated channel, so that much is
leakage-inflated. Written to `Code/outputs/screening/emg_only_contrast.json` and now stated
in the manuscript.

### 3.4 Tests

`tests/unit/test_leakage.py` gains the test that would have caught this a year ago, plus the
tests that prove it can fail:

- `test_no_measured_channel_waveform_is_shared_across_subjects` — fingerprints every EMG and
  mic channel of every recording; fails if any spans two participants.
- `test_the_channel_identity_test_actually_detects_a_planted_duplicate` — a leakage test
  that cannot fail is decoration. The synthetic fixture gained an opt-in
  `duplicate_mic_condition`, which replays one waveform across participants **with different
  rotations** (a byte-comparison would miss it, which is the point).
- `test_planted_duplicates_are_exact_rotations_not_merely_similar`
- `test_audio_run_is_refused_on_flagged_data` — and passes for `emg_only`, and passes once
  signed.
- `test_acknowledgements_do_not_change_the_configuration_hash`

`tests/unit/test_signal_processing.py` gains 11 tests: fingerprint invariance, exact-rotation
confirmation, quantisation-floor detection, dead-channel handling, envelope alignment
recovering a known shift, envelope-vs-waveform discrimination, the measured cost of the
published mic chain, the envelope chain keeping what it discards, `white_noise_power_gain`
against a measured pass-through, and the stopband assertion in both directions.

Two problems surfaced while wiring this up and were fixed rather than worked around. The
synthetic chewing microphone was a bare 4 Hz tone, so the new bandwidth check fired on it —
correctly. Rather than loosen the threshold I fixed the fixture: it now modulates a
broadband carrier, which is what an impact produces, so the default synthetic dataset is a
clean control and defects are opt-in. And the 16 new manifest fields were initially required
arguments, which broke every hand-constructed `RecordingRecord` in the suite; they now carry
defaults, which is also the right call for a schema addition.

**Suite status:** 272 unit tests passed (75 s) and 19 integration tests passed (13 min 20 s)
— 291 total, 0 failures. The integration suite trains a real nested-LOSO run per test, so
both new guards are exercised end to end rather than only in isolation.

### 3.5 Nothing published moved

| Artifact | Before | After |
|---|---|---|
| `five_class_nested_loso` config hash | `2b6fb5ac` | `2b6fb5ac` |
| `modality_and_no_chewing` config hash | `cead62e4` | `cead62e4` |
| EMG cross-subject duplicate groups | — | 0 (asserted every build) |
| Manuscript build | 29 pages | 29 pages, 0 undefined refs |

The acknowledgements are deliberately **excluded from the configuration hash**
(`ExperimentConfig._UNHASHED_KEYS`): an acknowledgement records who authorised a run, not
what it computes, so two runs differing only in who signed for them must produce identical
results and therefore identical identity. Without that, adding the required declaration to
the published configs would have silently invalidated the provenance section of the paper.
The manifest hash *does* change, which is correct — the data description genuinely gained
content.

---

## 4. The manuscript

Table 4 and Fig. `modality_ablation` are unchanged, per your decision. What changed around
them:

**Added** — a new Results subsection, `\label{sec:mic_defect}` *A second measurement-chain
defect, in the microphone channel*, placed immediately before the RQ2 subsection so no
reader reaches Table 4 first. Four paragraphs: the measurements; what follows for Table 4
(numbers correct as a record, but not a LOSO evaluation on that channel, with the bias
non-uniform in direction); and why it generalises — the defect is invisible in any single
file, and the audit that caught the mains problem was thorough and pointed at one modality.

**Changed** — 13 passages, each because it asserted something about *acoustics* that the
channel cannot support:

| Location | Was | Now |
|---|---|---|
| Abstract | "Audio was therefore informative… its contribution was specific rather than global" | Ablation reported as run; bounds a duplicated channel; RQ2 recorded unanswered |
| RQ list | "RQ1 and RQ2 are answered below" | RQ1 answered; RQ2 and RQ3 not |
| Sensors | one omnidirectional microphone, 1200 Hz | + the measured character: 1-count step, 96 % sub-10 Hz, 600 Hz Nyquist below the band of interest |
| Filter chain | "DC offset only" | + the measured 1.19 % retained variance and why it is kept anyway |
| Wavelets | "A5 (<19 Hz)" | A5 (<18.75 Hz) **and** that it sits in the 20 Hz stopband, retaining 2.95 % |
| Screening (RQ3) | 67.9 % / 64.6 % | + EMG-only 63.9 % / 61.3 %, so the quoted values are inflated by 3–4 pp |
| Audio-only results | "The microphone hears whether the mouth is quiet" | withdrawn; the shared rest waveform, and the single-scalar reproduction (rest 0.852 vs 0.950) |
| "Two effects were consistent" | presented as the microphone's specific contribution | withdrawn — consistency across seeds is what a duplicated channel produces |
| Discussion, "three reasons to keep it" | all three | two withdrawn; the parameter-cost reason stands because it is about the model |
| Signal-comparison figure | "large acoustic transient roughly five times" grinding | a measured **3.3×** median filtered RMS ratio (4.2× in raw counts), described as an envelope excursion, and **not time-aligned with the EMG row above it**. The old 5× appears to have been read off one excerpt; it is not the dataset value either post-filter or raw, so I replaced it with a number that can be checked. |
| Limitations | "underpowered in a way no amount of running would fix" | not underpowered — invalid input; more participants fix the first, only a new acquisition fixes the second |
| Conclusion | "The microphone earned its place on a narrower ground" | RQ2 unanswered; the case for a microphone is untested rather than supported |
| Methodological conclusion | one recommendation (interference statistic) | two, the second being the channel fingerprint — with the note that the first audit was thorough and scoped to one modality |

Four more pointers were added so no route into Table 4 bypasses the caveat: the Methods
*Modality ablation* subsection, the RQ2 Results subsection opening, Table 4's caption and
Fig. `modality_ablation`'s caption. The code-availability section now explains why both
configuration hashes are still valid despite the manifest schema change.

**One thing I want to flag once, then leave with you.** §5.1 of `audio.md` listed what a
caveat can defend, and the rest-false-alarm claim (7.7 % → 6.1 %, "in every seed") is the
one place where the wording had to change rather than be qualified. It reads as consistency
evidence, and all four S01–S04 rest recordings are the same waveform — so every seed is
scoring the same audio, and reproducibility there is a property of the duplication rather
than of the microphone. I rewrote it to say that explicitly. Table 4 still reports the
numbers.

---

## 5. Documentation

| File | Change |
|---|---|
| `Code/docs/open_questions.md` | **Q13 — Microphone channel provenance ⛔ BLOCKING**, with the evidence, what is encoded where, and four open items. The first is the acquisition source: the same writer produced the **trigger** channel, which is load-bearing for every label in the paper, and nothing measured so far suggests it is affected — but that audit has not been run. |
| `Code/docs/data_dictionary.md` | `Mic` row relabelled "not analysable audio"; the full character of the channel; the `.npy` note upgraded from "stale cache" to what it actually is — direct evidence the acquisition array held no microphone data. |
| `Code/docs/experiment_protocol.md` | Microphone filter row carries its measured cost. |
| `Code/docs/claim_to_evidence.md` | C39 and C40 moved from **blocked-compute** to **blocked-data**; C40 noted as no longer make-or-break because neither the hypothesis nor its negation is evaluable. |
| `Code/docs/audio_collection_spec.md` | **New.** Nine normative requirements for the next collection, each traced to a specific failure, plus acceptance criteria the ingest audit must pass before modelling. |
| `cause.md` | **§11 — the same mistake, in the other modality.** The shape is identical: a correct check, correctly run, scoped one step too narrowly — to the filter rather than the data, and to the file rather than the dataset. |
| `README.md`, `Code/README.md` | RQ2 status corrected from "Done" to "ran; unanswerable from this data". |

---

## 6. What was *not* done, and why

- **The `Mic` column was not repaired, de-rotated, or deleted.** The offset relative to the
  EMG is unknown per recording, so any re-alignment would be a guess presented as a fix.
- **No published number was regenerated.** The A5 band and the 20 Hz high-pass are both
  documented defects that stay in place, because changing them would silently move Table 1's
  fused row.
- **`mic_scope="per_recording"` is off by default.** It would change results, and it does not
  repair the channel.
- **The trigger channel has not been audited.** It was written by the same acquisition
  software that produced the microphone column, and every label in the paper rests on it.
  Nothing I measured suggests it is affected — its fingerprints are distinct except for the
  degenerate all-zero rest triggers — but a targeted audit has not been run. Q13 item 1.
- **RQ2 is not answered.** It cannot be, from these files. `docs/audio_collection_spec.md`
  states what a collection would need.

---

## 7. Reproducing this

```bash
cd Code
python -m pytest tests/ -q                              # full suite
bruxism-audit --data-root ../Data --no-video --no-figures   # writes mic_integrity.csv + §7c
python -c "                                             # the guard, firing
import sys; sys.path.insert(0,'src')
from bruxism.config import ExperimentConfig
from bruxism.data.manifest import build_manifest
from bruxism.runner import assert_modality_is_supported_by_data
m = build_manifest('../Data', probe_video=False)
assert_modality_is_supported_by_data(ExperimentConfig(name='x', modality='fusion'), m)
"
```

The audit bundle lands in `Code/outputs/data_audit/<manifest_hash>/`; §7c of `data_audit.md`
is the microphone section, and `mic_integrity.csv` is the per-recording table.
