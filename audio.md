# Audio audit — what has to change

**Date:** 2026-08-12
**Scope:** data handling, data quality, preprocessing and classification of the microphone
channel, and everything downstream of it (RQ2, Table 4, Fig. `modality_ablation`).
**Verdict:** the `Mic` column in `Data/**/*.csv` is not a per-participant acoustic recording.
It is a small set of 37 waveforms replayed across 100 recordings with a small circular
rotation, temporally unaligned with the EMG, band-limited below 10 Hz, and reduced to the
ADC quantisation floor by the production filter chain before the model sees it. RQ2 as
currently posed — *how much does audio add to EMG* — has not been answered by this dataset
and cannot be answered from the files that exist.

The EMG side is clean. All four EMG channels are distinct across all 100 recordings; no EMG
duplication, no EMG alignment problem. **Nothing in this document affects the EMG-only or
five-class EMG results.** The problem is confined to the microphone channel and to the
claims that rest on it.

---

## 1. Findings

Every number below is measured from the files in `Data/`, not inferred. Section 8 has the
commands to reproduce each one.

### 1.1 The microphone channel is duplicated across participants

Fingerprinting each channel by the MD5 of its **sorted** sample values (a rotation-invariant
signature) gives:

| Channel | Distinct waveforms | Recordings sharing a waveform with another file |
|---|---|---|
| `EMG1_1-2` … `EMG4_7-8` | 100 / 100 | 0 |
| `Trigger` | 95 / 100 | 7 (benign: 5 all-zero rest triggers, 2 all-one gum triggers) |
| **`Mic`** | **37 / 100** | **83** |

The duplicates are not merely similar. For all 63 duplicate pairs tested,
`np.array_equal(np.roll(a, -k), b)` is `True` — the two mic columns are **bit-identical
after a circular rotation**. Normalised cross-correlation at the recovered lag is
1.000000000.

The grouping is by *condition*, across participants:

```
f9a74278 x5: bite_left_1, bite_left_2, bite_left_3, bite_left_4, bite_left_5
8bfd97d6 x5: bite_right_1 … bite_right_5
f524dbaf x5: deviation_left_right_1 … deviation_left_right_5
c9533596 x5: incisor_clench_1, incisor_clench_2, incisor_clench_3, incisor_clench_4,
             protrusion_retrusion_5          <-- cross-condition
42296c32 x4: rest_1, rest_2, rest_3, rest_4  <-- all four S01-S04 rest recordings
… 15 more groups of 4 covering carrots, cheese, gum, molar_clench, natural_bruxing,
  open_close, protrusion_retrusion
```

Exposure per participant, counting recordings whose mic waveform also appears under a
different subject:

| Held-out participant | Recordings with a duplicated mic waveform |
|---|---|
| S01 | 20 / 20 |
| S02 | 19 / 20 |
| S03 | 20 / 20 |
| S04 | 20 / 20 |
| S05 | 4 / 20 |

**Consequence: leave-one-subject-out does not hold out the audio.** For S01–S04, essentially
every held-out audio window is drawn from a waveform that is also in the training set under
the same class label. The four S01–S04 rest recordings share one single waveform, so in any
fold that holds out one of them, the model has already been trained on that exact rest audio
three times.

Two further details pin the mechanism:

- The rotation offsets are small and wrap in both directions (0.2 s to 8 s), which is the
  signature of a ring-buffer read pointer, not of a resampling or export step.
- S05's four duplicated recordings carry offsets *identical* to S02's
  (`bite_left` 5586, `bite_right` 69009, `deviation` 2279, `protrusion` 69728), i.e. S02's
  mic column was copied verbatim into four S05 files. And S05's `protrusion_retrusion`
  carries S01–S04's `incisor_clench` waveform, so that recording's audio is labelled
  *movement* while carrying the *clench* condition's sound.

### 1.2 The microphone is not time-aligned with the EMG

Over all 45 chewing recordings — where every chew burst should produce a simultaneous EMG
burst and an acoustic transient — the Hilbert-envelope cross-correlation between EMG1 and
the mic gives:

- correlation at lag 0: **median −0.017** (i.e. none)
- best-lag |offset|: **median 18.0 s**, no recording under 0.1 s
- peak correlation even at the best lag: 0.14–0.42

This holds for S05's *unique* mic waveforms too, so it is not a side effect of the
duplication. There is no offset that fixes it: the misalignment differs per recording.

**Consequence:** the mic samples in a window labelled *grinding* are, in general, not the
sound that occurred during that grinding. Window-level fusion of two unaligned channels
cannot work by construction.

### 1.3 The channel is a sub-10 Hz envelope, not an audio waveform

Welch PSD of the raw `Mic` column across all 100 recordings:

- power below 10 Hz: **median 96.0 %** (range 91.4–98.7 %)
- quantisation step: exactly **1.0 count** in every recording, integer-valued, range ≈ 50–227

Band power (raw counts²) by condition family, trigger-active samples only:

| Family | 1–3 Hz | 3–10 Hz | 20–600 Hz |
|---|---|---|---|
| chewing | 63.2 | 25.7 | 1.82 |
| movement | 41.7 | 10.1 | 1.68 |
| rest | 16.6 | 3.4 | 1.04 |
| clenching | 4.2 | 1.1 | 0.14 |
| grinding | 3.8 | 1.6 | 0.18 |

All of the condition separation lives at **1–3 Hz** — the chew-rhythm band. That is the
signature of an analog sound-level / envelope-detector output sampled at 1200 Hz, not of a
microphone waveform. A real acoustic recording of tooth contact would put its energy in the
kilohertz range; at a 1200 Hz sampling rate there is no kilohertz range to put it in
(Nyquist = 600 Hz).

### 1.4 The production filter chain discards 99 % of what is there

`_default_mic_stages()` in `Code/src/bruxism/preprocessing/filters.py:403` applies a single
second-order 20 Hz high-pass. Measured on the actual recordings, the variance surviving that
stage is **median 1.19 %** (range 0.50–3.25 %). The 20 Hz high-pass removes precisely the
1–3 Hz band that carries all the discriminative content in §1.3 and keeps the band that
does not.

What survives is at or below the quantisation floor. With a 1.0-count step, quantisation
noise contributes ≈ 0.081 counts² of variance to the retained 20–600 Hz band. Median
post-filter variance of trigger-active windows:

| Family | post-filter variance | quantisation floor | verdict |
|---|---|---|---|
| chewing | 0.906 | 0.081 | ~10 dB above floor |
| movement | 0.874 | 0.081 | ~10 dB above floor |
| rest | 0.521 | 0.081 | ~7 dB above floor |
| **grinding** | **0.085** | 0.081 | **at the floor** |
| **clenching** | **0.066** | 0.081 | **below the floor** |

**42 of 100 recordings** have retained mic content within 3 dB of pure quantisation noise.
For the two tooth-contact classes the paper exists to study, the audio branch is being fed
ADC dither.

### 1.5 One third of the audio branch is wired into the filter's stopband

The mic branch (`Code/src/bruxism/models/dual_branch.py:148-153`, mirrored in
`baselines.py:78-80` and `configs/models/dual_branch.yaml:17`) uses `coif5`, level 5, bands
`("A5", "D3", "D1")`. At 1200 Hz those are:

| Band | Nominal range | Status |
|---|---|---|
| `A5` | 0 – 18.75 Hz | **entirely inside the 20 Hz high-pass stopband** |
| `D3` | 75 – 150 Hz | quantisation noise (§1.4) |
| `D1` | 300 – 600 Hz | quantisation noise (§1.4) |

`A5` therefore receives the high-pass roll-off residue — the very DC drift the stage exists
to remove — attenuated but not eliminated by a 2nd-order filter. Its share of the branch's
input energy swings wildly between recordings (5.4 %, 41.7 %, 47.6 %, 94.7 % on four
recordings I measured), because it is tracking per-recording baseline wander rather than
anything the participant did. The `BatchNorm1d` immediately after each band's first
convolution then rescales that residue back to unit variance, so the branch spends a third
of its capacity amplifying a recording-identity artefact.

The paper's own text (`Main_2.tex:227`) reports `A5 (<19 Hz)` next to a 20 Hz high-pass
(`Main_2.tex:211`) without flagging the contradiction.

### 1.6 What this did to the RQ2 numbers

Read from the ablation ledger
`Code/outputs/runs/modality_and_no_chewing_20260810T020642_cead62e4/predictions.parquet`,
five-class task, three-seed means:

| Held out | mic duplication | audio-only F1 | audio-only AUC | EMG-only F1 | EMG-only AUC |
|---|---|---|---|---|---|
| S01 | 20/20 | 0.423 | 0.781 | 0.632 | 0.904 |
| S02 | 19/20 | 0.425 | 0.779 | 0.726 | 0.957 |
| S03 | 20/20 | 0.522 | 0.839 | 0.713 | 0.967 |
| S04 | 20/20 | 0.414 | 0.787 | 0.722 | 0.981 |
| **S05** | **4/20** | **0.354** | **0.714** | **0.805** | **0.980** |

S05 is the *worst* participant on audio and the *best* on EMG. It is also the only
participant whose audio is mostly not in the training set. That dissociation is what
leakage looks like; it is not explained by S05 being a hard subject, because on EMG S05 is
the easiest.

Two more diagnostics:

- **The audio branch is a loudness meter.** A single scalar — the log RMS of the filtered
  mic window, no learning at all — recovers most of the trained CNN's one-vs-rest AUC:
  rest 0.852 vs 0.950, chewing 0.865 vs 0.898, clenching 0.777 vs 0.829. The wavelet
  decomposition and the convolutional stack are adding very little on top of "how loud is
  this recording".
- **The one cross-condition swap behaves as predicted.** `S05_protrusion_retrusion` (true
  class *movement*) carries S01–S04's *incisor_clench* waveform. Its 63 held-out audio-only
  predictions across three seeds: 23 clenching, 20 grinding, 20 rest, **0 movement**. Too
  small to move the aggregate AUC (0.427 → 0.441 when removed) but a clean illustration of
  the failure mode.

### 1.7 Corroborating evidence that the mic capture path was broken

- **The `.npy` companions have a zero mic column.** Three exist (all S01). Their EMG and
  Trigger columns match the CSV *exactly*; column 5 (`Mic`) is all zeros in every one. The
  manifest already flags these as `stale_npy_companion`
  (`Code/docs/data_dictionary.md:207` notes it) but treats it as a stale cache rather than
  as evidence about the mic path. It is the strongest single clue: the acquisition array had
  no microphone data in it, and the CSV's `Mic` column was filled from somewhere else.
- **The `.avi` files contain no audio stream.** Parsing the RIFF headers gives one `strh`
  chunk of type `vids` and zero `auds`. There is no second copy of the audio to recover.
- **`Code/docs/open_questions.md` Q3** already lists the microphone hardware, gain, ADC
  resolution and units as ⛔ BLOCKING and unanswered. This audit converts that from a
  documentation gap into a data-validity finding.

---

## 2. What has to change — data layer

### 2.1 Treat `Mic` as a failed channel of record

Add a documented, versioned decision that the `Mic` column of the 2025-08 collection is
**not analysable audio**. Do not delete it, do not repair it, do not attempt to
de-rotate it: the rotation offset is unknown relative to the EMG (§1.2), so any
re-alignment would be a guess presented as a fix.

### 2.2 New quality flags — `Code/src/bruxism/data/quality.py`

Add to `QualityFlag` and `_FLAG_DESCRIPTIONS`, and bump `QUALITY_POLICY_VERSION`
(currently `"2026-08-03.1"`, line 29) — that invalidates every manifest, which is correct
here:

| Flag | Detection |
|---|---|
| `mic_waveform_duplicated` | The recording's mic column is a circular rotation of another recording's, and that other recording belongs to a different subject. |
| `mic_emg_unaligned` | Envelope cross-correlation between mic and EMG peaks at \|lag\| > 0.25 s, or is < 0.1 at lag 0, in a recording whose condition should produce simultaneous EMG and sound. |
| `mic_at_quantisation_floor` | Post-filter mic variance in the analysis band is within 3 dB of `step² / 12`. |
| `mic_bandwidth_implausible` | More than 90 % of raw mic power below 10 Hz — i.e. an envelope, not a waveform. |
| `mic_channel_dead` | Mic column is constant, or its variance is below the quantisation floor. |

`ExclusionPolicy` should **not** auto-exclude on these by default — the EMG in those
recordings is fine and excluding them would destroy the five-class results. They must
instead make any *audio-consuming* run refuse to start (§4.2).

### 2.3 Manifest columns — `Code/src/bruxism/data/manifest.py`

`RecordingRecord` currently stores only `mic_min` / `mic_max` (lines 317–318). Add, and
populate in `_iter_records` (around line 517 where `mic` is already read):

- `mic_sorted_sha256` — the rotation-invariant fingerprint. This is the one column that
  makes §1.1 impossible to miss again.
- `mic_quantisation_step`, `mic_n_unique_values`
- `mic_power_fraction_below_10hz`
- `mic_variance_retained_by_chain` — variance after `mic_stages`, as a fraction of raw
- `mic_snr_above_quantisation_db`
- `mic_emg_envelope_lag_seconds`, `mic_emg_envelope_r_at_zero`

Then add a **manifest-level** cross-recording check (there is currently no such pass — every
check in `_iter_records` looks at one file in isolation): after building the records, group
by `mic_sorted_sha256` and raise `mic_waveform_duplicated` on every group of size > 1 whose
members span more than one `subject_id`. Bump `MANIFEST_SCHEMA_VERSION` (line 66) to `1.2`
and add the new columns to `_summarise_for_hash` so the manifest hash changes.

### 2.4 Mains contamination is measured on EMG only

`measure_mains_contamination` is called on `emg` alone
(`Code/src/bruxism/data/manifest.py:589`), and
`Code/src/bruxism/preprocessing/interference.py` has no mic path. The same audit that caught
the harmonic defect on EMG was never run on the mic. Run it on the mic channel too and store
the result; it costs nothing and the omission is exactly the class of gap the paper's own
methodological argument is about.

### 2.5 Signal-quality tables cover EMG only

`Code/src/bruxism/evaluation/signal_quality.py` computes contamination, class contrast and
between-participant spread from the EMG columns. Extend `window_quality_table` and
`contrast_table` to emit the same three measurements for the mic. A mic contrast table would
have shown rest louder than clenching and grinding (§1.4) — physically backwards, and
visible in one glance.

---

## 3. What has to change — preprocessing

### 3.1 `Code/src/bruxism/preprocessing/filters.py:403` — `_default_mic_stages`

The current chain is a 20 Hz high-pass justified as "remove the large DC offset". It removes
98.8 % of the channel's variance including all of its information. Replace with a chain that
is stated as a *choice about what the channel is*:

- **If the channel is treated as an envelope** (which is what it measures, §1.3): a 0.2 Hz
  high-pass for drift removal and a 20 Hz low-pass anti-alias, then decimate. Not for
  publication from this dataset — see §5 — but this is the chain that would make the channel
  analysable at all, and it is what a re-collection with the same transducer would need.
- **If the channel is treated as audio**: it cannot be. Say so in the module docstring
  rather than leaving a 20 Hz high-pass that reads as a considered design.

Whichever is chosen, the `rationale` string on the stage must record the measured fraction
of variance the stage removes. The `FilterStage.rationale` field already exists and is
mandatory for exactly this reason; it is currently the weakest rationale in the file.

### 3.2 The filter/wavelet contradiction

`A5` at 0–18.75 Hz behind a 20 Hz high-pass (§1.5) must not survive in any form. Fix in all
four places that declare it:

- `Code/src/bruxism/models/dual_branch.py:150`
- `Code/src/bruxism/models/baselines.py:80`
- `Code/src/bruxism/evaluation/screening.py:59` (`_MIC_WAVELET`, 6 bands)
- `configs/models/dual_branch.yaml:17`

Add a construction-time assertion — in `BranchConfig.__post_init__` or in
`WaveletConfig.band_frequency_table` — that raises when a configured band lies entirely
inside the stopband of the filter chain that feeds it. That check is cheap, it is the
wavelet analogue of "plot the filter response over the measured spectrum", and it is the
generalisable lesson from §1.5.

### 3.3 Normalisation — `Code/src/bruxism/preprocessing/normalization.py`

The mic gets a single scalar `mic_center` / `mic_scale` pooled over all training
participants (lines 197, 297). Given that the surviving mic content is a per-recording noise
floor (§1.4, §1.6), one global scale converts recording identity directly into a feature.
If audio is ever used again, the mic needs at minimum a per-recording gain reference, and
the normaliser needs the same `fitted_on` audit the EMG side has. Note this is a *design*
change; it does not rescue the current data.

---

## 4. What has to change — classification and the RQ2 harness

### 4.1 `emg_only` zeroes the tensor, the model removes the branch

`Code/src/bruxism/data/dataset.py:348-352` zeroes the unused modality tensor;
`Code/src/bruxism/models/dual_branch.py:345-354` does not construct the unused branch. Both
are true and they are consistent in effect, but the dataset docstring (lines 242-246) says
"zeroes … so all three modality conditions see identical windows" while the model docstring
(line 338) says "the unused branch is not constructed at all". The paper repeats the second
(`Main_2.tex:240`, `266`). Reconcile the two docstrings so a reader cannot conclude the
ablation was run one way when it was run the other. This is a documentation defect, not a
results defect.

### 4.2 An audio-consuming run must refuse to start on flagged data

Add a guard in `Code/src/bruxism/runner.py` (or in `ExperimentConfig.__post_init__`): if
`modality` is `fusion` or `audio_only` and the resolved manifest carries any
`mic_waveform_duplicated` flag, raise unless the config declares an explicit
`mic_defect_acknowledged_by: "<name>, <date>"`. This mirrors the existing pattern for
`allow_trigger_off_as_rest` / `trigger_off_rest_approved_by`
(`Code/src/bruxism/config.py:69-70`) and `calibration_approved_by`
(`normalization.py:98-103`). The project already has the idiom for "this concession must be
signed for"; the mic defect needs it more than either of those.

### 4.3 The leakage test suite has no data-identity test

`Code/tests/unit/test_leakage.py` covers split construction, normaliser provenance,
calibration disjointness and augmentation staging — all of it about *indices and
statistics*. Nothing checks that the held-out participant's **signal** is absent from
training. Add:

```
test_no_channel_waveform_is_shared_across_subjects
```

which fingerprints every channel of every recording in the manifest (rotation-invariant, as
in §1.1) and fails if any fingerprint spans more than one subject, with a documented
allow-list for degenerate cases (all-zero rest triggers, all-one triggers). Run it in CI.
This one test, written a year ago, would have caught the entire finding.

### 4.4 `screening.py` mic features

`_window_features` (line 91) computes log RMS of six coif5 mic bands plus overall log RMS —
seven of the 35 screening features. Given §1.6, those seven features are a recording-noise-
floor detector and the screening macro-F1 quoted for direction in the paper
(`Main_2.tex:321`, 67.9 % LR / 64.6 % GB) includes them. Re-run screening EMG-only and
report the difference; if it is small, the quoted screening numbers stand and you have said
so from measurement rather than assumption.

### 4.5 `Code/src/bruxism/models/ablations.py`

The module docstring (lines 3-6) states the ablation's purpose as isolating the modality:
"same windows, same folds … identical in every respect except which modality the model
sees". That is true of the *code* and false of the *data*, because the audio in the held-out
fold is in the training fold (§1.1). Add the caveat to the docstring where the design intent
is stated, so the next person reading the harness does not inherit the assumption.

---

## 5. What has to change — the manuscript

You've decided to keep the Table 4 numbers with a caveat. I'll say once, plainly, what that
does and does not cover, then give you the text.

**What a caveat can defend:** every number in Table 4 is a correct report of what the run
produced. The run is real, the ledger is real, the protocol was followed. Reporting those
numbers with a statement of what the input channel turned out to be is defensible.

**What a caveat cannot defend:** any sentence that reads the numbers as a measurement of
*acoustics*, of *the microphone's contribution*, or of *generalisation to a new
participant's audio*. Those are the sentences listed in §5.2. Under the duplication in §1.1,
the audio-only and fusion conditions were not evaluated leave-one-subject-out at all on the
audio channel, so a reader who takes them as modality contrasts is being misled regardless
of how carefully the caveat is worded. Those sentences have to change even under
keep-with-caveat. Everything else can stay.

### 5.1 May stay unchanged

- Table 4 (`Main_2.tex:348-372`) and its footnotes, including the existing rule that the
  fusion row is the primary run's and that differences are never taken across runs.
- Fig. `modality_ablation` (`Main_2.tex:420-426`) and
  `Code/scripts/evaluate/make_ablation_figure.py`.
- The run-bundle identifiers and hashes in Data/Code availability (`Main_2.tex:617`).
- Every EMG-only and five-class number in Tables 1–3, 5, 6 and the associated figures.

### 5.2 Must change even under keep-with-caveat

| Line(s) | Current text | Problem |
|---|---|---|
| 68 (abstract) | "Audio was therefore informative but far weaker on its own" / "its contribution was specific rather than global" | Both are claims about the microphone. Neither survives §1.1. |
| 94 | RQ2: "how much does audio add to EMG" | The question is fine; the answer must become "not measurable from this collection". |
| 98 | "RQ1 and RQ2 are answered below" | RQ2 is not. |
| 144 | "One omnidirectional microphone was positioned near the temporomandibular joint … All signals were sampled at 1200 Hz." | Needs the measured character of the channel (§1.3): sub-10 Hz, integer, 1-count step, 600 Hz Nyquist. |
| 211 | mic filter chain paragraph | Must state the 1.19 % median variance retained. |
| 227 | "`A5` (<19 Hz)" | Contradicts the 20 Hz high-pass two paragraphs earlier (§1.5). |
| 410 | "The microphone hears whether the mouth is quiet and whether the participant is eating" | A physical claim about a channel that is 96 % sub-10 Hz and duplicated across participants. |
| 416, 418 | "Two effects were nevertheless consistent" / "the microphone is a complementary channel rather than a redundant or a sufficient one" | The rest-false-alarm effect (7.7 % → 6.1 %) is consistent across seeds *because* all four S01–S04 rest recordings share one waveform. This is the single most affected claim in the paper. |
| 537–543 | "three reasons to keep it" | Reasons 1 and 2 are the ones §1.1 removes. |
| 541, 556 | "a large acoustic transient roughly five times the amplitude seen during grinding" | The 4.2× ratio is real but it is a 1–3 Hz envelope excursion, not an acoustic transient. |
| 591 | "The modality ablation (RQ2) is likewise complete, but it is underpowered" | Not underpowered — invalidated at the input. Different failure, different remedy. |
| 601 | "the modality comparison is algorithm-specific" | Understates it; it is channel-specific in a way that has nothing to do with the algorithm. |
| 605 (conclusion) | "The microphone earned its place on a narrower ground" | Withdraw. |

### 5.3 Drop-in caveat

Suggested placement: a new subsection immediately before *Modality ablation (RQ2)*
(`Main_2.tex:406`), with a one-sentence forward reference from the abstract and a pointer in
*Validity and limitations*. Sized to sit alongside the mains-harmonic audit, which it
parallels exactly.

> **A second measurement-chain defect, in the microphone channel.** The interference audit
> that produced the corrected filter chain was applied to the electromyographic channels
> only. Repeating it on the microphone channel after the modality ablation had been run
> revealed a defect of a different kind, and it is reported here for the same reason the
> first one was. Fingerprinting each channel by a rotation-invariant signature of its
> samples shows that the 100 recordings contain 100 distinct waveforms on each of the four
> EMG channels and only 37 on the microphone channel: 83 recordings carry a microphone
> waveform that is bit-identical, after a circular rotation of between 0.2 and 8 s, to that
> of another participant's recording of the same condition, and all four of the S01–S04
> quiet-rest recordings share a single waveform. The channel is also unaligned with the
> electromyogram — over the 45 chewing recordings the envelope cross-correlation at zero lag
> has a median of −0.017 and peaks at a median absolute lag of 18 s — and 96 % of its power
> lies below 10 Hz, so it behaves as an envelope output rather than as an acoustic waveform;
> the 20 Hz high-pass of the production chain retains a median of 1.19 % of its variance,
> and for the clenching and grinding conditions what remains is at or below the
> analog-to-digital quantisation floor. The three `.npy` acquisition companions that exist
> reproduce the electromyographic and trigger columns exactly and contain an all-zero
> microphone column, and the video files carry no audio stream, so no second copy of the
> signal exists to recover.
>
> The consequence for Table~\ref{tab:modality_ablation} is specific and it is not a question
> of statistical power. Every number in that table is a correct report of what the run
> produced under the protocol described above, and the electromyographic conditions are
> unaffected: the EMG channels are distinct across every recording, and the five-class,
> secondary and participant-level results elsewhere in this paper do not read the microphone
> at all. But because a held-out participant's microphone waveform is, for four of the five
> participants, already present in that fold's training data, the audio-only and fused
> conditions were not evaluated leave-one-subject-out on the audio channel. The numbers
> therefore bound what a duplicated, unaligned, envelope-band channel can contribute under
> this architecture; they do not measure what a microphone contributes, and the direction of
> the bias is not uniform — where the duplicated waveform carries the matching condition
> label it is optimistic, and where it does not, as in the one recording whose microphone
> waveform belongs to a different condition entirely, it is pessimistic. We therefore report
> the ablation as run, withdraw the reading that the microphone is a complementary channel
> whose value lies in separating quiet rest from tooth contact, and record RQ2 as
> unanswered. Notably, the one participant whose microphone waveforms are largely unique
> (S05, 4 of 20 recordings duplicated against 19–20 of 20 for the others) is the weakest of
> the five on every audio-only metric and the strongest on every EMG-only metric, which is
> the pattern this defect predicts and is not otherwise explained.

Then in *Validity and limitations* (`Main_2.tex:587-601`), replace the "underpowered" framing
at line 591 with a pointer to the above, and in the *Conclusion* (line 605) replace the
microphone sentence with a statement that the audio channel of this collection was found to
be defective on audit and that a microphone remains untested rather than tested-and-weak.

### 5.4 Documentation

- `Code/docs/open_questions.md` — add **Q13, microphone channel provenance**, ⛔ BLOCKING
  (Q1–Q12 are taken; Q9 is the mains defect, Q3 the unanswered hardware question),
  with §1 as its evidence. Cross-reference Q3 (hardware, still unanswered) and Q9 (the mains
  defect), since this is the same failure mode at a different point in the chain.
- `Code/docs/data_dictionary.md:46,52` — the `Mic` row says "microphone; integer-valued on
  disk" with units "unknown". Add the measured character and the duplication.
  The `.npy` note at lines 206–207 should point at §1.7 rather than dismissing the zero
  column as a stale cache.
- `Code/docs/experiment_protocol.md:149` — the microphone row of the filter table needs the
  retained-variance figure.
- `Code/docs/claim_to_evidence.md:95-96` — C39 and C40 are marked **blocked-compute**. They
  are now **blocked-data**; C40's note ("the make-or-break analysis") should say why it can
  no longer be made or broken from this collection.
- `cause.md` — this is the natural companion to the mains-harmonic write-up. The generalisable
  lesson is the same one, one level up: *the audit that found the first defect was applied to
  one modality and not the other.*

---

## 6. Next collection — audio specification

Nothing in §2–§5 recovers RQ2. Answering it needs new audio. Minimum requirements, each one
traceable to a specific failure above:

| Requirement | Because |
|---|---|
| Sample audio at ≥ 16 kHz on its own clock, in its own file (WAV, 16-bit or better) | §1.3 — 1200 Hz cannot represent tooth-contact acoustics at all; 600 Hz Nyquist is below the band of interest |
| Never share a buffer or a writer between the audio and the physiological stream | §1.1 — the duplication is a buffer-reuse artefact |
| Record a hardware sync marker on **both** streams (shared trigger edge, clap, or a tone burst at session start and end) and verify alignment per recording before analysis | §1.2 — there is currently no way to know what the offset was |
| Log transducer model, placement, preamp gain, AGC state and ADC bit depth per session | §1.7 / `open_questions.md` Q3 — none of this is known for the existing data |
| Record a calibration tone or a known-level reference at session start | §3.3 — enables a real per-recording gain normalisation instead of a pooled z-score over noise floors |
| Record ≥ 30 s of room tone with the participant silent and still, per session | Gives a real per-session noise floor, which is the denominator every audio SNR statement needs |
| Store the raw audio unfiltered; do all shaping offline | §1.4 — the current chain's information loss is invisible because no raw copy survives |
| Run the §2.3 fingerprint and §2.2 flags on ingest, before anyone trains anything | §4.3 |

If the same envelope-output transducer is used again, that is a defensible choice — it is
cheap and it separates chewing from rest — but it must then be described as a
**sound-level sensor**, analysed in the 0.5–20 Hz modulation band, and not called a
microphone or fed to a wavelet decomposition designed for waveforms.

---

## 7. Suggested order of work

1. §4.3 — write the failing test first. It is ~30 lines and it pins the finding.
2. §2.2, §2.3 — flags and manifest columns; rebuild the manifest. Manifest hash changes,
   so every run bundle's `manifest_hash` will no longer match. That is correct and it is
   the mechanism that stops stale artefacts being reused.
3. §4.2 — the run guard, so no new audio run can start unacknowledged.
4. §3.2 — the stopband assertion, plus the four config sites.
5. §4.4 — re-run screening EMG-only; report the delta.
6. §5 — manuscript edits, with §5.3 as the anchor text.
7. §2.4, §2.5, §3.1, §3.3 — the remaining measurement and preprocessing work.
8. §6 — the collection spec, for whenever new data is gathered.

Steps 1–5 touch no EMG result and change no number in Tables 1–3, 5 or 6.

---

## 8. Reproducing this audit

Run from `Code/`. Each block prints the numbers quoted above.

**§1.1 — duplication**

```python
import numpy as np, pandas as pd, glob, hashlib, collections, os
files = sorted(glob.glob('../Data/Subject_*/*.csv')) + \
        sorted(glob.glob('../Data/More Data/Data/Subject_*/*.csv'))
def fp(v):
    return hashlib.md5(np.ascontiguousarray(np.sort(np.asarray(v, float))).tobytes()).hexdigest()[:8]
g = collections.defaultdict(list)
for f in files:
    g[fp(pd.read_csv(f, usecols=['Mic'])['Mic'])].append(os.path.basename(f))
print(len(g), 'distinct mic waveforms of', len(files))
for h, v in sorted(g.items(), key=lambda x: -len(x[1])):
    if len(v) > 1:
        print(f'  {h} x{len(v)}: ' + ', '.join(x.split("_20")[0] for x in v))
```

**§1.1 — the rotation is exact**

```python
a = pd.read_csv('../Data/Subject_1/bite_left_1_20250804_103436.csv', usecols=['Mic'])['Mic'].to_numpy(float)
b = pd.read_csv('../Data/Subject_2/bite_left_2_20250805_142631.csv', usecols=['Mic'])['Mic'].to_numpy(float)
cc = np.fft.irfft(np.fft.rfft(a - a.mean()) * np.conj(np.fft.rfft(b - b.mean())), n=len(a))
k = int(np.argmax(cc))
print('lag', k, 'exact after roll:', np.array_equal(np.roll(a, -k), b))   # -> lag 5586, True
```

**§1.2 — alignment**

```python
from scipy.signal import butter, sosfiltfilt, hilbert
fs = 1200
def env(x, lo, hi):
    x = np.asarray(x, float) - np.mean(x)
    y = sosfiltfilt(butter(4, [lo/(fs/2), hi/(fs/2)], btype='band', output='sos'), x)
    return sosfiltfilt(butter(2, 3/(fs/2), btype='low', output='sos'), np.abs(hilbert(y)))
df = pd.read_csv('../Data/Subject_1/gum_1_20250804_105539.csv')
a, b = env(df['EMG1_1-2'], 20, 450), env(df['Mic'], 20, 590)
a, b = a - a.mean(), b - b.mean()
cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a)) / (np.linalg.norm(a) * np.linalg.norm(b))
print('r at lag 0:', cc[0], ' best lag (s):', (np.argmax(cc) if np.argmax(cc) < len(a)//2 else np.argmax(cc)-len(a))/fs)
```

**§1.3 / §1.4 — bandwidth, retained variance, quantisation floor**

```python
import sys; sys.path.insert(0, 'src')
from scipy.signal import welch
from bruxism.preprocessing.filters import FilterChainConfig, apply_filter_chain
cfg = FilterChainConfig()
for f in files:
    m = pd.read_csv(f, usecols=['Mic'])['Mic'].to_numpy(float)
    fr, P = welch(m - m.mean(), fs=1200, nperseg=4096)
    y = apply_filter_chain(m, cfg, 1200, modality='mic')
    step = np.min(np.diff(np.unique(m)))
    print(os.path.basename(f)[:34].ljust(36),
          f'<10Hz {100*P[fr<10].sum()/P.sum():5.1f}%  kept {100*y.var()/m.var():5.2f}%  '
          f'step {step:g}  var/qfloor {y.var()/(step**2/12*580/600):6.2f}')
```

**§1.5 — where the mic branch's energy goes**

```python
from bruxism.preprocessing.wavelets import WaveletConfig, decompose
w = WaveletConfig(wavelet='coif5', level=5, bands=('A5', 'D3', 'D1'))
y = apply_filter_chain(pd.read_csv('../Data/Subject_1/gum_1_20250804_105539.csv',
                                   usecols=['Mic'])['Mic'].to_numpy(float), cfg, 1200, modality='mic')
W = y[:len(y)//1200*1200].reshape(-1, 1200)[:, None, :]
d = decompose(W, w, check_level=False)
tot = sum((d[b]**2).sum() for b in w.bands)
print({b: round(100*float((d[b]**2).sum())/tot, 1) for b in w.bands})
```

**§1.6 — per-participant audio-only vs EMG-only**

```python
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
d = pd.read_parquet('outputs/runs/modality_and_no_chewing_20260810T020642_cead62e4/predictions.parquet')
d = d[d.task_id == 'five_class']
P = ['prob_rest','prob_movement','prob_clench','prob_instructed_grinding','prob_chewing']
def m(g):
    return pd.Series(dict(acc=accuracy_score(g.true_label, g.predicted_label),
                          f1=f1_score(g.true_label, g.predicted_label, average='macro',
                                      labels=list(range(5)), zero_division=0),
                          auc=roc_auc_score(g.true_label, g[P], multi_class='ovr',
                                            average='macro', labels=list(range(5)))))
print(d.groupby(['modality','subject_id','seed']).apply(m, include_groups=False)
       .groupby(level=[0,1]).mean().round(3).unstack(0).to_string())
```

**§1.7 — the `.npy` mic column and the `.avi` streams**

```python
for n in sorted(glob.glob('../Data/Subject_*/*.npy')):
    a = np.load(n); c = pd.read_csv(n.replace('.npy', '.csv'))
    print(os.path.basename(n), 'mic range', a[:, 5].min(), a[:, 5].max(),
          '| EMG matches CSV:', all(np.allclose(a[:, i], c.iloc[:, i], atol=1e-3) for i in range(5)))
b = open('../Data/Subject_1/cheese_1_20250804_104906.avi', 'rb').read(200000)
print('video streams', b.count(b'vids'), 'audio streams', b.count(b'auds'))
```

---

## 9. Open questions for the team

1. **Acquisition software.** Who wrote the recorder, and does the source still exist? The
   ring-buffer signature in §1.1 and the all-zero `.npy` mic column in §1.7 should be
   traceable to a specific bug in tens of minutes with the source in hand. Worth knowing
   because the same writer produced the trigger channel, and the trigger is load-bearing for
   every label in the paper. Nothing I measured suggests the trigger is affected — but the
   audit that would confirm it has not been run either.
2. **Microphone hardware.** Model, output type (waveform vs envelope/AGC), and which DAQ
   input it went into. `open_questions.md` Q3 has been ⛔ BLOCKING since 2026-07-27; §1.3
   now depends on it for the correct description of the channel, not just for a units label.
3. **Was the microphone verified during collection?** Did anyone see a live trace, or was
   the channel assumed to be recording? This determines whether §6 needs an ingest-time
   liveness check on top of the post-hoc flags.
4. **Session order.** Confirm S01 was recorded first (2025-08-04) and that the shared
   waveforms originate from that session. If so, the canned waveforms are S01's — which
   would mean S01's own mic data may be genuine, and the paper could say which participant's
   audio is real. I did not find enough in the metadata to settle it.
5. **Table 4 footnote.** The existing footnote already carries the rule that the fusion row
   comes from the primary run and that differences are never taken across runs. Should the
   §5.3 caveat be folded into that footnote as well as the body, or is the body pointer
   enough for the journal's format?
