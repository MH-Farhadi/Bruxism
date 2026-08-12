# Open questions blocking definitive scientific results

**Status date:** 2026-07-27
**Maintainer rule:** an answer is only recorded here when it comes with a named source and a
date. Do not resolve an item by taking a majority vote among inconsistent files.

Legend — **RESOLVED**: answered and encoded in code/config. **BLOCKING**: prevents a
specific manuscript statement. **OPEN**: needed for completeness but does not block a
result already produced.

---

## Q1 — Trigger semantics ✅ RESOLVED

**Question.** Who or what set the `Trigger` column, what do 0 and 1 mean, is there onset or
offset delay, and how do repeated runs correspond to repetitions?

**Answer (project owner, 2026-07-27).** `Trigger == 1` marks intervals during which the
participant was actively performing the named task.

**Encoded as.** `configs/data/trigger_constrained.yaml`; `SegmentSource.TRIGGER_ACTIVE` in
`src/bruxism/data/segments.py`.

**Residual risk.** Trigger granularity differs markedly between participants — some marked
one long run per bout, others marked each repetition. Onset/offset delay is still
unquantified, which is why a transition guard is applied.

---

## Q1b — Transition guard width ⚠️ BLOCKING (methods prose + sample counts)

**Question.** How wide should the guard around each trigger transition be?

**Why it matters.** A one-second window needs a trigger run of at least `1.0 + 2 × guard`
seconds. Because participants marked the trigger at different granularities, the guard
directly controls how many examples each participant contributes — and at 0.5 s two
participant × class cells collapse to 1–3 windows, which makes those held-out folds
uninformative for those classes.

Measured (`outputs/data_audit/<hash>/guard_sensitivity.csv`):

| guard (s) | total windows | smallest participant × class cell | cells < 10 |
|---:|---:|---:|---:|
| 0.000 | 6,779 | 50 | 0 |
| 0.125 | 6,458 | 34 | 0 |
| **0.250** | **6,173** | **11** | **0** |
| 0.375 | 5,863 | 2 | 1 |
| 0.500 | 5,610 | 1 | 2 |

**Current default.** 0.25 s, chosen as the widest guard that leaves no degenerate
participant × class cell. **This needs investigator sign-off.** It is one line in
`configs/data/trigger_constrained.yaml` and in every experiment config.

### 2026-08-03: the guard is now measured, not assumed

The table above answers "how much data does the guard cost?" It never answered the question
the guard exists for: **how far is the trigger mark from the actual condition change?**
`evaluation/segmentation.py::trigger_onset_alignment` measures it, by timing when the EMG
envelope actually crosses the midpoint between its pre-onset and post-onset plateaus.

On 432 trigger onsets (358 with a detectable activation; 17 % show none):

| quantity | value |
|---|---:|
| median lag (trigger → envelope) | **−0.075 s** |
| onsets where activity **precedes** the trigger | **79.9 %** |
| onsets where activity **follows** the trigger (the harmful direction) | 71 / 358 = 19.8 % |
| median positive lag | 0.054 s |
| p95 positive lag | 0.282 s |
| max positive lag | 0.352 s |

**Interpretation.** The trigger is not a precise onset marker, but it errs in the benign
direction: four times out of five the muscle is already active when the mark goes high, so a
window placed just inside the trigger contains task activity and its label is correct. The
guard only protects against the other 20 %, whose median error is 54 ms.

**What that buys.** From the window/guard sweep
(`outputs/data_audit/<hash>/window_guard_sweep.csv`), at a 1.0 s window:

| guard (s) | total windows | smallest cell | cells < 30 | S02 movement |
|---:|---:|---:|---:|---:|
| 0.10 | 6,525 | **36** | **0** | 36 |
| 0.15 | 6,402 | 27 | 1 | 27 |
| 0.25 (current) | 6,173 | 11 | 1 | 11 |

Reducing the guard from 0.25 s to 0.10 s removes every starved participant × class cell
**without shortening the window**, at the cost of exposing roughly 7 % of first-windows-per-run
to a partially mislabelled leading edge (20 % of onsets are harmful × ~35 % of those exceed
0.10 s). The 0.5 s guard originally proposed exceeds the largest positive lag ever measured
(0.352 s) and is not supported by the data.

**Recommendation for sign-off:** 1.0 s window, **0.10 s guard**. Prespecified as
`configs/experiments/five_class_guard010.yaml`; the 0.25 s configuration remains the
primary reported one until sign-off, so the two are comparable.

**Blocks.** The Methods sentence describing transition exclusion, and every sample count in
the manuscript.

---

## Q2 — EMG channel montage ✅ RESOLVED (partially)

**Question.** Are the four CSV EMG columns four bipolar channels, two channels with paired
fields, or another montage?

**Answer (project owner, 2026-07-27).** Four recorded differential signals arranged as two
bilateral bipolar pairs — one pair per side of the head. `Data/README.txt` maps them to
left masseter, left temporalis, right masseter, right temporalis.

**Encoded as.** `EMG_MUSCLE_MAP` in `src/bruxism/data/schema.py`; all four channels are
used, and `data.emg_channels` exposes a subset if that ever changes.

**Consequence for the manuscript.** Manuscript tables that say "two EMG channels" must be
corrected to describe four differential signals from two bilateral bipolar pairs.

**Still open.** Exact electrode positions, inter-electrode distance, skin preparation and
reference placement are undocumented.

---

## Q3 — Hardware, units and acquisition-side filtering ⛔ BLOCKING

**Question.** Manufacturer and model of the electrodes, amplifier/DAQ and microphone; gain;
ADC resolution and range; physical units; and what `bandpass_filter: Index 143` /
`notch_filter: Index 9` in every metadata file actually mean.

**Why it matters.** Raw EMG spans roughly ±65,000 and the microphone is integer-valued in
roughly 50–227. Without calibration those are **arbitrary ADC units** and nothing in this
project labels them µV, Pa or dB (`SIGNAL_UNITS` in `schema.py`). More seriously, if the
acquisition hardware already applied a bandpass and a notch, the offline chain in
`configs/` is filtering an already-filtered signal.

**Blocks.** The Methods hardware paragraph; any unit label on any axis; the justification
for the offline filter chain.

---

## Q4 — Study population and phenotype ⛔ BLOCKING

**Question.** What recruitment evidence exists? The participant surveys appear to ask
whether a provider *indicated* that the participant grinds their teeth, which is not the
same as a formal, current clinical bruxism diagnosis.

**Why it matters.** The original manuscript claimed clinically confirmed bruxers. That
claim cannot be made from a self-reported survey item.

**Blocks.** Every sentence characterising the participants; the phenotype statement
(awake/sleep, tooth-contact vs bracing/thrusting) that reviewers specifically demanded.

---

## Q5 — Protocol timing ⛔ BLOCKING

**Question.** What was the actual duration and number of repetitions per condition?

**Evidence against the manuscript.** The manuscript describes "3-minute continuous trials
repeated three times". The files show **three recordings of about one minute each** per
chewing/grinding condition and a **single ~1-minute recording** for the other conditions —
about three minutes total per condition, not nine. Every metadata file declares
`target_duration_seconds: 60`.

**Blocks.** The protocol paragraph and the total-recording-time figure.

---

## Q6 — Task naming ⚠️ OPEN

Confirm every instructed task name and resolve the carrots/popcorn wording that differs
across manuscript versions. Filenames use `carrots`.

---

## Q7 — S05 molar/incisor metadata conflict ✅ RESOLVED

**Conflict.** `molar_clench_5_20250807_145916_metadata.txt` reports
`condition_key: incisor_clench` — the same value as S05's *separate* incisor-clench
recording at 15:00:41.

**Resolution (rule `R1_filename_wins_for_condition`, project owner, 2026-07-27).** The
filename wins. Trusting the metadata would give S05 two incisor recordings and no molar
recording, contradicting the protocol described in the manuscript. Both values are retained
in the manifest and the recording carries `metadata_condition_conflict`.

---

## Q8 — Rest definition ✅ RESOLVED (conservative default)

**Decision.** Rest comes **only** from the five dedicated rest recordings, whose trigger is
zero throughout. Trigger-off intervals inside active recordings are *not* treated as rest,
because their meaning (transition? instruction period? unobserved activity?) is unconfirmed.

**Encoded as.** `allow_trigger_off_as_rest: false`. Setting it true without recording
`trigger_off_rest_approved_by` raises.

**Consequence.** Rest is a small class (585–590 windows, ~117 per participant) relative to
real deployment where rest dominates. The manuscript must state this limitation.

---

## Q9 — Signal preparation ⚠️ HALF-ANSWERED FROM THE DATA (2026-08-03), remainder blocking

**Question.** Whether hardware filters were active, and what `bandpass_filter: Index 143` /
`notch_filter: Index 9` refer to.

### Answered by the spectra: `notch_filter: Index 9` removed 60 Hz, and nothing else

Measured on the **raw** recordings, peak-to-local-noise-floor ratio in the 20–450 Hz band:

| Frequency | S01 rest | S02 rest |
|---|---:|---:|
| 60 Hz  | 2× | 2× |
| 180 Hz | 571× | **1,781,427×** |
| 300 Hz | 119× | 412,502× |
| 420 Hz | 43× | 52,690× |

The mains fundamental is absent from data that is otherwise dominated by its own harmonics.
Nothing but an acquisition-side notch explains that, so `notch_filter: Index 9` is a 60 Hz
notch and it was active on every recording.

**This was not a harmless curiosity.** The offline chain notched 60 Hz — the one mains
frequency already gone — and band-passed 20–450 Hz, letting 180/300/420 Hz through. Across
all 100 recordings, 62 have more than 30 % of their raw in-band EMG power at mains
harmonics; every rest recording is between 91 % and 99.8 %. Corrected on 2026-08-03: the
chain now notches every mains multiple inside the passband. See `cause.md`, `opus_report_1.md`
and `preprocessing/filters.py`.

**Methodological point worth stating in the manuscript.** The textbook chain — notch the
mains fundamental, then band-pass — is *actively misleading* on hardware that already
notches: it removes nothing and leaves the harmonics, while looking correct in a filter
diagram. `04_filter_response` now plots the measured data spectrum behind the filter
response so the mismatch is visible rather than inferable.

### Still blocking

1. **Confirm the hardware notch** from the acquisition software's index table, rather than
   from our inference. The spectra are unambiguous but they are an inference.
2. **What is `bandpass_filter: Index 143`?** If the hardware already band-limits, the
   offline 20–450 Hz bandpass may be redundant, and the manuscript should say which stage
   shaped the band. The recordings show no obvious roll-off below 450 Hz, so if a hardware
   bandpass was active its upper edge is at or above ours — but that is a weaker inference
   than the notch and it is not recorded as answered.
3. **Was the notch setting identical for every participant and session?** All five show the
   same 60 Hz absence, which is consistent with a fixed setting, but it is unconfirmed.

**Asked of the investigators:** the acquisition software's filter index table, and
confirmation that both settings were unchanged across all sessions.

---

## Q10 — Ethics ⛔ BLOCKING

**Question.** The exact IRB identifier and the scope of consent for analysis, video use,
data sharing and publication.

**Conflicting values found in historical files:** `IRB22275690-2`, `IRB2425-139`, and a
likely typo `IRB2275690-2`. Only an investigator or the official approval record can settle
this. **Do not pick one.**

**Blocks.** The ethics statement.

---

## Q11 — Privacy and public release ⛔ BLOCKING

Which derived artifacts may be shared, and whether any raw or de-identified dataset release
is authorised. The MIT licence in `Code/` covers the software only — it does **not** license
the participant data.

**Blocks.** The data-availability statement.

---

## Q12 — Target venue ⚠️ OPEN

Final journal and format requirements, and whether the new clinical references and
terminology in `Temp.md` §8 have been independently verified against their DOIs. Citation
verification is a separate scholarly task and was not performed here.

---

## Q13 — Microphone channel provenance ⛔ BLOCKING

**Question.** What produced the `Mic` column, and does any real audio from these sessions
survive anywhere?

**Evidence from the data (`audio.md`, 2026-08-12).** The column is not per-participant
audio. Measured over all 100 recordings:

- **37 distinct microphone waveforms**, against 100 on each of the four EMG channels.
  83 recordings carry a waveform that is bit-identical, after a circular rotation of
  0.2–8 s, to another *participant's* recording of the same condition. All four S01–S04
  quiet-rest recordings share one waveform. `S05_protrusion_retrusion` carries the
  waveform of the other participants' `incisor_clench` — a different condition.
- **Unaligned with the EMG.** Over the 45 chewing recordings the envelope
  cross-correlation at zero lag has a median of −0.017 and peaks at a median absolute lag
  of 18 s.
- **Not a waveform.** 96 % of raw power below 10 Hz (range 91.4–98.7 %), integer-valued,
  quantisation step exactly 1.0 count. The production 20 Hz high-pass retains a median of
  1.19 % of its variance; clenching and grinding are then at or below the quantisation
  floor.
- **Corroboration.** The three `.npy` acquisition companions reproduce EMG and Trigger
  exactly and carry an **all-zero** microphone column (see Q9 and rule R2). The `.avi`
  files contain no audio stream (`vids` 1, `auds` 0).

**Why it matters.** Leave-one-subject-out does not hold out the audio, so the modality
ablation bounds what a duplicated channel contributes rather than measuring a microphone.
RQ2 is unanswerable from these files. The EMG is unaffected — all four channels are
distinct across every recording — and no other result in the manuscript reads the
microphone.

**Encoded as.** Five `QualityFlag` entries (`mic_waveform_duplicated`,
`mic_emg_unaligned`, `mic_at_quantisation_floor`, `mic_bandwidth_implausible`,
`mic_channel_dead`), measured by `bruxism.preprocessing.mic_integrity`, stored as manifest
schema 1.2 columns, raised by the cross-recording pass in
`manifest.flag_shared_waveforms`, and enforced by
`runner.assert_modality_is_supported_by_data`. Conflict rule
`R4_mic_channel_is_not_analysable_audio`. Regression test
`tests/unit/test_leakage.py::test_no_measured_channel_waveform_is_shared_across_subjects`.

**Still open — and this is what BLOCKING refers to.**

1. **Acquisition software.** Who wrote the recorder and does the source survive? The
   ring-buffer signature and the all-zero `.npy` column should be traceable to a specific
   bug in minutes with the source in hand. The same writer produced the **trigger**
   channel, which is load-bearing for every label in the manuscript. Nothing measured so
   far suggests the trigger is affected, but the audit that would confirm it has not been
   run.
2. **Hardware.** Transducer model, output type (waveform vs envelope/AGC), preamp gain,
   which DAQ input. Overlaps Q3, which has been blocking since 2026-07-27; Q13 now needs
   it for the *correct description of the channel*, not just for a units label.
3. **Was the microphone verified during collection?** Did anyone see a live trace? Decides
   whether a future collection needs an ingest-time liveness check on top of the post-hoc
   flags.
4. **Session order.** Confirm S01 was recorded first (2025-08-04) and that the shared
   waveforms originate from that session. If so, S01's own microphone data may be genuine
   and the manuscript could say whose audio is real. The metadata does not settle it.

**Answered from the data (2026-08-12).** Whether a second copy of the audio exists: no.
Project owner confirmed the CSV column is all there ever was; the `.avi` files carry no
audio stream and the `.npy` companions are zero. The channel is therefore not recoverable
and Q13 cannot be closed by reanalysis — only by a new acquisition.

---

## Summary

| ID | Topic | Status | Blocks |
|---|---|---|---|
| Q1 | Trigger semantics | RESOLVED | — |
| Q1b | Transition guard width | MEASURED 2026-08-03 (onset alignment); needs sign-off | Methods, all sample counts |
| Q2 | Channel montage | RESOLVED (partial) | electrode detail only |
| Q3 | Hardware / units / filters | BLOCKING | Methods, axis labels |
| Q4 | Population and phenotype | BLOCKING | Participants, phenotype |
| Q5 | Protocol timing | BLOCKING | Protocol paragraph |
| Q6 | Task naming | OPEN | wording only |
| Q7 | S05 metadata conflict | RESOLVED | — |
| Q8 | Rest definition | RESOLVED | — |
| Q9 | Acquisition filtering | HALF-ANSWERED (notch confirmed from spectra 2026-08-03; bandpass index still open) | Methods |
| Q10 | IRB identifier | BLOCKING | Ethics statement |
| Q11 | Release authorisation | BLOCKING | Data availability |
| Q12 | Venue and citations | OPEN | formatting |
| Q13 | Microphone channel provenance | BLOCKING | RQ2; modality ablation; sensor description |
