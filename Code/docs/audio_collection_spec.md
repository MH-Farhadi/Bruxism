# Audio acquisition specification for the next collection

**Status:** normative for any future recording session in this project.
**Origin:** `audio.md` §6, written after the audit that found the 2025-08 microphone channel
to be unusable. Every requirement below traces to a specific way that collection failed;
none is generic best practice included for completeness.

Nothing in this document repairs the existing data. The 2025-08 microphone channel is not
recoverable — there is no second copy, the `.avi` files carry no audio stream, and the
offset of the microphone column relative to the EMG is unknown per recording. RQ2 needs new
audio.

---

## 1. What went wrong, in one table

| # | Failure | Evidence | Requirement |
|---|---|---|---|
| F1 | The microphone column is not per-participant data — 37 distinct waveforms across 100 recordings, 83 of them a circular rotation of another participant's same-condition recording | `audio.md` §1.1 | R1, R2, R8 |
| F2 | The channel is unaligned with the EMG — median zero-lag envelope correlation −0.017, median best lag 18 s over 45 chewing recordings | `audio.md` §1.2 | R3 |
| F3 | It is an envelope, not a waveform — 96 % of power below 10 Hz; and at 1200 Hz the 600 Hz Nyquist is below the band tooth contact radiates in | `audio.md` §1.3 | R1, R4 |
| F4 | The production filter kept 1.19 % of its variance and discarded the only informative band | `audio.md` §1.4 | R7 |
| F5 | Surviving content is at the 1-count quantisation floor for clenching and grinding | `audio.md` §1.4 | R1, R5 |
| F6 | No transducer, gain, output-type or ADC documentation exists | `open_questions.md` Q3, Q13 | R4, R5 |
| F7 | Nobody noticed for a year | `audio.md` §1.7 | R8, R9 |

---

## 2. Requirements

### R1 — Sample audio at ≥ 16 kHz, in its own file, on its own clock

WAV, 16-bit or better, one file per recording. Do not fold audio into the physiological CSV.

*Because* (F3, F5): tooth contact radiates most of its energy above 1 kHz, and a 1200 Hz
stream cannot represent any of it — the 600 Hz Nyquist sits below the band of interest, so
no amount of processing recovers what was never sampled. A dedicated file also gives the
audio its own bit depth: the shared column carried 1 count of resolution, which put
clenching and grinding under the quantiser.

### R2 — Never share a buffer, array or writer between the audio and the physiological stream

Allocate the audio buffer inside the recording session and let it go out of scope at the
end. Do not reuse a module-level array. Assert at write time that the buffer's identity
differs from the previous recording's.

*Because* (F1): the duplication is a buffer-reuse artefact. Rotations cluster near zero and
near the recording length, in both directions — the signature of a ring-buffer read pointer
— and four of S05's files carry S02's microphone column with S02's exact offsets.

### R3 — Record a hardware sync marker on both streams, and verify alignment per recording

A shared trigger edge into both devices is best. Failing that, a clap or a 1 kHz tone burst
at the start *and* end of every session, audible on the microphone and visible on the
physiological trigger. Verify before analysis, per recording, not per session:
`measure_envelope_alignment` from `bruxism.preprocessing.mic_integrity` reports the offset
and the zero-lag correlation.

*Because* (F2): there is currently no way to know what the offset was, so nothing can be
re-aligned after the fact. Two markers rather than one also catch clock drift, which a
single marker cannot distinguish from a fixed offset.

### R4 — Log the transducer and the signal path, per session

Model and manufacturer; output type (waveform, or envelope/AGC — these need different
analysis and must not be confused); preamplifier gain and whether AGC was enabled; ADC bit
depth and full-scale range; placement, with a photograph.

*Because* (F3, F6): the 2025-08 channel was described as a microphone throughout and is an
envelope output. That is a legitimate instrument, but it must then be analysed in its
modulation band and never decomposed with a waveform wavelet.

### R5 — Record a calibration reference at session start

A known-level tone, or a calibrator if one is available. Note the level in the session
metadata.

*Because* (F5, F6): without it, every amplitude is in arbitrary units and the only available
normalisation is a pooled z-score — which, on this dataset, turned per-recording noise floor
into a usable feature. `NormalizationConfig.mic_scope="per_recording"` is the interim
mitigation; a real reference is the fix.

### R6 — Record ≥ 30 s of room tone per session, participant silent and still

*Because*: every SNR statement needs a denominator. The audit had to infer the noise floor
from the quantisation step, which bounds it but does not measure it.

### R7 — Store audio raw and unfiltered; do all shaping offline

*Because* (F4): the information loss of the 20 Hz high-pass was invisible for a year
precisely because no raw copy survived the acquisition step in an analysable form. Offline
shaping is reversible; acquisition-time shaping is not.

### R8 — Run the integrity checks at ingest, before anyone trains anything

`bruxism-audit` computes all of them and writes `mic_integrity.csv`. Specifically:

- rotation-invariant fingerprint per channel per recording, grouped — no fingerprint may
  span two participants (`manifest.flag_shared_waveforms`);
- envelope alignment with the EMG, per recording (R3);
- power fraction below 10 Hz, quantisation step, and SNR above the quantisation floor.

A run that reads the microphone refuses to start when any of these flags is raised, unless
the experiment configuration declares `mic_defect_acknowledged_by`
(`runner.assert_modality_is_supported_by_data`).

### R9 — Look at a live trace during the session, and say so in the log

*Because* (F7): no automated check substitutes for someone seeing that the microphone
responds when the participant chews. Record who verified it and when.

---

## 3. If the same envelope transducer is used again

That is a defensible choice — it is cheap, and it separates chewing from rest cleanly
(chewing 63.2 vs rest 16.6 counts² at 1–3 Hz). It then must be:

- **described as a sound-level sensor**, not a microphone, in the paper and the code;
- **filtered with `mic_envelope_stages()`** (0.2–20 Hz), not the 20 Hz high-pass, which
  removes its entire signal;
- **analysed in its modulation band**, with features that describe rhythm — not decomposed
  with a wavelet chosen for transient acoustic structure;
- **not expected to separate clenching from grinding**, since neither produces the
  low-frequency envelope structure this sensor measures.

It cannot answer RQ2 as posed. Answering "how much does audio add to EMG" needs R1.

---

## 4. Acceptance criteria for the next collection

Before any modelling, the ingest audit must show, on every recording:

| Check | Threshold |
|---|---|
| Distinct microphone waveforms | equal to the recording count |
| Cross-participant fingerprint groups, any channel | 0 |
| Envelope alignment with EMG on chewing recordings | \|lag\| ≤ 0.25 s and r(0) ≥ 0.10 |
| SNR above the quantisation floor, per condition family | > 10 dB, including clenching and grinding |
| Power below 10 Hz | < 0.90 for a waveform transducer; documented as an envelope sensor otherwise |
| Room tone recorded | present, ≥ 30 s |
| Sync markers | present at session start and end, on both streams |

The five microphone quality flags in `bruxism.data.quality` implement these; a clean
collection raises none of them.
