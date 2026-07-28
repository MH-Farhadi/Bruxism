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

## Q9 — Signal preparation ⛔ BLOCKING (see Q3)

Whether hardware filters were active, and what `Index 143` / `Index 9` refer to. Until
answered, the offline chain is documented as "applied on top of unknown acquisition-side
filtering".

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

## Summary

| ID | Topic | Status | Blocks |
|---|---|---|---|
| Q1 | Trigger semantics | RESOLVED | — |
| Q1b | Transition guard width | BLOCKING | Methods, all sample counts |
| Q2 | Channel montage | RESOLVED (partial) | electrode detail only |
| Q3 | Hardware / units / filters | BLOCKING | Methods, axis labels |
| Q4 | Population and phenotype | BLOCKING | Participants, phenotype |
| Q5 | Protocol timing | BLOCKING | Protocol paragraph |
| Q6 | Task naming | OPEN | wording only |
| Q7 | S05 metadata conflict | RESOLVED | — |
| Q8 | Rest definition | RESOLVED | — |
| Q9 | Acquisition filtering | BLOCKING | Methods |
| Q10 | IRB identifier | BLOCKING | Ethics statement |
| Q11 | Release authorisation | BLOCKING | Data availability |
| Q12 | Venue and citations | OPEN | formatting |
