# Data dictionary

**Applies to:** the raw acquisition files under the authorized data root, and the manifest
this project derives from them.
**Manifest schema version:** `1.0` · **Quality policy version:** `2026-07-27.1`

---

## 1. Privacy classification

The data root mixes research recordings with material that is **not** research data and
must never be read, copied, hashed or referenced by any generated artifact.

| Asset | Class | Handling |
|---|---|---|
| `Subject_<n>/*.csv` | research signal | read-only input |
| `Subject_<n>/*.avi` | **identifiable video** | inventoried only: presence, duration, frame rate, resolution, codec. Never decoded to frames, never copied. |
| `Subject_<n>/*_metadata.txt` | research metadata | read-only input |
| `Subject_<n>/*.npy` | stale cache | never read as input (rule `R2`) |
| `*_Survey.pdf` | **health-related, identifiable** | never opened by this software |
| `More Data/Pics/*` | **participant / setup photographs** | never opened; the directory is excluded from scanning |
| `Receipts.pdf`, `Reconciliation*.xlsx` | **administrative** | not research features; never opened |
| `README.txt` | documentation | read for the channel map |

Generated artifacts contain canonical subject IDs (`S01`…`S05`), data-root-relative POSIX
paths and numeric summaries only. `bruxism-audit` asserts this before finishing and fails
if a private token appears in any output; the same check runs in
`tests/integration/test_end_to_end.py`.

The MIT licence in `Code/` covers the software. **It does not license the participant
data.**

---

## 2. Recording CSV schema

Exactly six columns, in this order. Any deviation raises `SchemaError`.

| Column | Type | Meaning | Units |
|---|---|---|---|
| `EMG1_1-2` | float64 | differential EMG, electrode pair 1–2 | **unknown** |
| `EMG2_3-4` | float64 | differential EMG, electrode pair 3–4 | **unknown** |
| `EMG3_5-6` | float64 | differential EMG, electrode pair 5–6 | **unknown** |
| `EMG4_7-8` | float64 | differential EMG, electrode pair 7–8 | **unknown** |
| `Trigger` | float64 | task-active marker; only `{0.0, 1.0}` observed | dimensionless |
| `Mic` | float64 | **not analysable audio — see below and `audio.md`**; integer-valued on disk, 1-count step | **unknown** |

### 2.1 Units

Physical units were never documented by the acquisition chain. Everything in this project
reports `SIGNAL_UNITS = "arbitrary_adc_units"`. **Do not label these values µV, Pa or dB.**
Observed ranges: EMG approximately ±65,000; microphone approximately 50–227.

**The `Mic` column is a failed channel of record (rule `R4`, `audio.md`, Q13).** It is
not per-participant audio: the 100 recordings hold only 37 distinct microphone
waveforms (against 100 on each EMG channel), 83 of them bit-identical to another
*participant's* same-condition recording after a circular rotation of 0.2–8 s, and all
four S01–S04 rest recordings share one waveform. It is unaligned with the EMG (median
zero-lag envelope correlation −0.017 over the chewing recordings; median best lag 18 s)
and 96 % of its power is below 10 Hz, so it behaves as a sound-level/envelope output
rather than a waveform. It is retained verbatim, never repaired and never de-rotated —
the offset relative to the EMG is unknown per recording. Manifest schema 1.2 records
`mic_sorted_sha256`, `mic_duplicate_group`, `mic_quantisation_step`,
`mic_power_fraction_below_10hz`, `mic_variance_retained_fraction`,
`mic_snr_above_quantisation_db`, `mic_emg_envelope_*` and
`mic_mains_harmonic_power_fraction` per recording, plus `emg_sorted_sha256` and
`trigger_sorted_sha256` as the controls.

### 2.2 Channel map

Confirmed by the investigator (2026-07-27): four differential signals from **two bilateral
bipolar pairs**, one pair per side of the head.

| Column | Tentative muscle site (from `Data/README.txt`) |
|---|---|
| `EMG1_1-2` | left masseter |
| `EMG2_3-4` | left temporalis |
| `EMG3_5-6` | right masseter |
| `EMG4_7-8` | right temporalis |

Still unconfirmed: exact electrode positions, inter-electrode distance, electrode type,
reference placement, amplifier/DAQ model, gain, ADC resolution and range
(`open_questions.md` Q2/Q3).

### 2.3 Trigger

`Trigger == 1` marks intervals during which the participant was actively performing the
named task (investigator, 2026-07-27). Dedicated rest recordings have a trigger that is
zero throughout. Onset/offset delay is unquantified, which is why a transition guard is
applied.

Trigger granularity varies markedly between participants (some marked one long run per
bout, some each repetition). Measured run durations, in seconds:

| Task family | n runs | median | 25th | 75th | fraction ≥ 2.0 s |
|---|---:|---:|---:|---:|---:|
| chewing | 159 | 11.59 | 8.21 | 14.79 | 0.96 |
| instructed grinding | 122 | 2.74 | 2.14 | 3.95 | 0.82 |
| clench | 196 | 2.04 | 1.71 | 3.38 | 0.53 |
| movement | 146 | 2.24 | 1.64 | 2.73 | 0.66 |
| rest | 0 | — | — | — | — |

---

## 3. Metadata sidecar

`<stem>_metadata.txt`, `key: value` per line.

| Key | Used for | Trust |
|---|---|---|
| `subject_id` | cross-check against the filename | informational |
| `condition` / `condition_key` | cross-check | **filename wins** on conflict (rule `R1`) |
| `sampling_rate` | validated against the configured rate | authoritative if consistent |
| `target_duration_seconds` | the short-recording threshold | authoritative |
| `samples_saved` / `expected_samples` | cross-check | **CSV row count wins** (rule `R3`) |
| `status` | exclusion if not `COMPLETED` | authoritative |
| `bandpass_filter` / `notch_filter` | recorded verbatim | **meaning unknown** — `Index 143` / `Index 9` are unexplained (`open_questions.md` Q9) |
| `npy_file` | presence check | **claims an `.npy` that usually does not exist** |
| `csv_file` / `video_file` | presence check | informational |

---

## 4. Condition taxonomy

| Raw filename token | Canonical condition | Task family |
|---|---|---|
| `rest` | `rest` | rest |
| `open_close` | `open_close` | movement |
| `deviation_left_right` | `deviation` | movement |
| `protrusion_retrusion` | `protrusion` | movement |
| `bite_left` | `bite_left` | clench |
| `bite_right` | `bite_right` | clench |
| `molar_clench` | `molar_clench` | clench |
| `incisor_clench` | `incisor_clench` | clench |
| **`natural_bruxing`** | **`instructed_grinding`** | instructed_grinding |
| `cheese` | `cheese` | chewing |
| `carrots` | `carrots` | chewing |
| `gum` | `gum` | chewing |

`natural_bruxing` is a filename token, not evidence that the behaviour was spontaneous. An
unknown token raises rather than being silently dropped.

---

## 5. Manifest columns

One row per discovered recording.

**Identity** — `recording_id` (`S01_rest_20250804T102808`), `subject_id`,
`condition_token`, `condition`, `task_family`, `repetition_token`.

**Location** — `csv_relpath`, `avi_relpath`, `metadata_relpath`, `npy_relpath` (all
data-root-relative POSIX; companions are resolved across directories, so a metadata file in
`More Data` is still found).

**Size and timing** — `n_samples`, `sampling_rate_hz`, `metadata_sampling_rate_hz`,
`duration_seconds`, `metadata_samples_saved`, `metadata_target_duration_seconds`.

**Trigger** — `trigger_values`, `trigger_active_samples`, `trigger_active_fraction`,
`n_trigger_runs`, `n_trigger_transitions`, `trigger_run_boundaries`.

**Signal quality** — `emg_min`, `emg_max`, `mic_min`, `mic_max`,
`startup_transient_seconds`, `startup_transient_peak_ratio`.

**Video** — `video_frame_count`, `video_fps`, `video_duration_seconds`, `video_width`,
`video_height`, `video_codec`, `video_readable`. Container metadata only; frames are never
decoded.

**Integrity** — `csv_sha256`, `avi_sha256` (opt-in), `metadata_sha256`, `npy_claimed`,
`npy_agrees_with_csv`, `npy_disagreement`.

**Policy** — `quality_flags`, `excluded`, `exclusion_reason`, `conflict_rules_applied`,
`quality_policy_version`, `manifest_schema_version`.

---

## 6. Quality flags

| Flag | Meaning | Effect |
|---|---|---|
| `short_recording` | below 98 % of the metadata's own target duration | retained, reported |
| `secondary_location` | file lives outside the participant's primary directory | retained, path recorded, **never moved** |
| `metadata_condition_conflict` | metadata condition ≠ filename condition | filename wins (rule `R1`) |
| `missing_npy_companion` | metadata claims an `.npy` that does not exist | harmless |
| `stale_npy_companion` | an `.npy` exists but does not reproduce the CSV | never read |
| `incomplete_triple` | a CSV/AVI/metadata member is absent anywhere under the root | retained if the CSV exists |
| `sampling_rate_mismatch` | metadata rate ≠ configured rate | **excluded** |
| `sample_count_mismatch` | `samples_saved` ≠ CSV rows | CSV wins (rule `R3`) |
| `video_duration_mismatch` | video/CSV durations differ > 3 s | informational |
| `video_unreadable` | container could not be probed | informational |
| `startup_transient` | opening excursion above 12× the robust scale | handled by the startup guard |
| `no_trigger_activity` | active-task recording whose trigger never fires | **excluded** |
| `unexpected_trigger_in_rest` | dedicated rest recording whose trigger fires | **excluded** |
| `not_completed` | metadata `status` ≠ `COMPLETED` | **excluded** |

---

## 7. Observed dataset state (2026-07-27)

100 CSV + 100 AVI + 100 metadata, 5 participants × 20 recordings, 7,167,600 samples
(1.66 h). No recording excluded.

| Flag | Count |
|---|---:|
| `startup_transient` | 66 |
| `missing_npy_companion` | 97 |
| `stale_npy_companion` | 3 |
| `short_recording` | 2 |
| `secondary_location` | 2 |
| `metadata_condition_conflict` | 1 |

Named anomalies:

- **Short:** `S02_natural_bruxing_20250805T143429` (52.4 s) and
  `S05_cheese_20250807T151036` (40.6 s), against a declared 60 s target.
- **Secondary location:** `S05_rest_20250807T144939` (CSV, AVI and metadata) and the
  metadata for `S05_protrusion_retrusion_20250807T145430`, all in
  `More Data/Data/Subject_5/`.
- **Metadata conflict:** `S05_molar_clench_20250807T145916` — metadata says
  `incisor_clench`, which is also S05's *separate* incisor recording's key. Filename wins.
- **`.npy`:** only three exist, all for S01, and none reproduces the CSV — the sixth column
  is all zeros rather than the microphone channel. EMG and Trigger match the CSV
  *exactly*. This is not merely a stale cache: it is direct evidence that the acquisition
  array held no microphone data and the CSV column was filled from elsewhere. See
  `audio.md` §1.7 and Q13.
- **Video:** all 100 readable, 640×480, 30 fps, MJPG.
