# Claim-to-evidence table for `Main_2.tex`

Every numeric or factual claim the revision needs, mapped to the artifact that supplies it
or to the named human decision that blocks it.

**Status legend**

| Status | Meaning |
|---|---|
| `supported` | The artifact exists and the claim follows from it. |
| `contradicted` | Evidence disagrees with what the manuscript currently says. **Must be changed.** |
| `blocked-human` | Needs an investigator decision; no amount of code resolves it. |
| `blocked-compute` | Code and configuration exist; the experiment has not been run yet. |
| `editorial` | Wording only; no new evidence required. |

`Main_2.tex` currently carries **206** `\TBD{}` placeholders and **6** author-action
comments. Their distribution:

| Section | `\TBD` count |
|---|---:|
| Five-class performance and modality ablation | 85 |
| Secondary analyses with rest | 24 |
| Participant-level results | 24 |
| Training and evaluation | 15 |
| Class-level diagnostics | 15 |
| Training behavior and computational measurements | 10 |
| Abstract / preamble | 9 |
| Conclusion | 7 |
| Effect of the chewing class | 6 |
| Sensors and dataset | 4 |
| Engineering interpretation | 4 |
| Validity and limitations | 1 |
| Experimental procedure | 1 |
| Model architecture | 1 |

---

## Study description and protocol

| claim_id | manuscript_location | proposed_claim | evidence_artifact | analysis_config | status | notes |
|---|---|---|---|---|---|---|
| C01 | Sensors and dataset | Five participants, 100 recordings | `outputs/data_audit/<hash>/data_audit.md` §1 | — | **supported** | 100 CSV + 100 AVI + 100 metadata; 7,167,600 samples (1.66 h). |
| C02 | Sensors and dataset | Two EMG channels | `docs/data_dictionary.md` §2.2 | — | **contradicted** | Four differential signals from two bilateral bipolar pairs (investigator, 2026-07-27). Rewrite the sentence. |
| C03 | Sensors and dataset | Electrode placement | `Data/README.txt` map in `schema.EMG_MUSCLE_MAP` | — | **blocked-human** | Site labels are tentative; positions, inter-electrode distance and reference are undocumented. `open_questions.md` Q2. |
| C04 | Sensors and dataset | Hardware, gain, ADC, units | — | — | **blocked-human** | Q3. Every artifact says `arbitrary_adc_units`; **no µV may appear anywhere.** |
| C05 | Experimental procedure | "3-minute trials repeated three times" | manifest `duration_seconds`, `metadata_target_duration_seconds` | — | **contradicted** | Files show ~1 min per recording, ×3 for chewing/grinding conditions. Every metadata declares `target_duration_seconds: 60`. Q5. |
| C06 | Experimental procedure | Sampling rate 1200 Hz | manifest `sampling_rate_hz` | — | **supported** | All 100 recordings agree. |
| C07 | Experimental procedure | Rest source | `docs/experiment_protocol.md` §2.2 | `trigger_constrained.yaml` | **supported** | Five dedicated rest recordings, trigger flat zero. ~118 windows per participant. |
| C08 | Experimental procedure | Rest adjudication / manual exclusion | — | — | **blocked-human** | The `\TBD{state whether excluded manually…}` placeholder. Nothing here excluded any interval manually. |
| C09 | Participants | Clinically confirmed bruxers | — | — | **blocked-human** | Q4. Surveys appear to ask whether a provider *indicated* grinding — not a formal diagnosis. |
| C10 | Ethics | IRB identifier | — | — | **blocked-human** | Q10. Three conflicting values in historical files. **Do not pick one.** |
| C11 | Data availability | Public deposit | — | — | **blocked-human** | Q11. MIT covers software only. |

## Signal processing

| claim_id | manuscript_location | proposed_claim | evidence_artifact | analysis_config | status | notes |
|---|---|---|---|---|---|---|
| C12 | Signal processing | 60 Hz notch + 20–450 Hz bandpass | `resolved_config.yaml` → `filters` | any experiment config | **supported** | Single authoritative implementation; Nyquist-validated. |
| C13 | Signal processing | Additional 5 Hz high-pass | — | — | **contradicted** | Removed: a no-op after a 20 Hz bandpass edge. Delete the sentence. |
| C14 | Signal processing | Causal vs zero-phase | `filter_chain.json`; `FilterChainConfig.describe()` | — | **supported** | Zero-phase (`sosfiltfilt`), **acausal/offline**. Answers the reproducibility author-action at line 220. |
| C15 | Signal processing | Filtering applied before windowing | `docs/experiment_protocol.md` §2.2 | — | **supported** | Per-window filtering measured to corrupt the leading edge by >1× the signal SD. |
| C16 | Signal processing | Acquisition-side filtering | — | — | **blocked-human** | Q9: `bandpass_filter: Index 143` / `notch_filter: Index 9` unexplained. |
| C17 | Model architecture | Wavelet band "D3" | `model.band_frequency_table()`; `outputs/paper_bundle/tables/` | — | **contradicted** | The prototype's `details[2]` is **D2** (150–300 Hz) at level 4, not D3 (75–150 Hz). Re-derive every band name. |
| C18 | Signal processing | Window 1 s, stride 0.5 s | `resolved_config.yaml` → `data` | `trigger_constrained.yaml` | **supported** | — |
| C19 | Signal processing | Transition exclusion | `outputs/data_audit/<hash>/guard_sensitivity.csv` | `guard_seconds` | **blocked-human** | Q1b: 0.25 s default needs sign-off; 0.5 s produces degenerate participant × class cells. |
| C20 | Experimental procedure | Startup handling | manifest `startup_transient_seconds` | `startup_guard_seconds: 0.5` | **supported** | Replaces the unexplained 3 s skip. 66/100 recordings, all settling ≤ 0.40 s. |

## Model and training

| claim_id | manuscript_location | proposed_claim | evidence_artifact | analysis_config | status | notes |
|---|---|---|---|---|---|---|
| C21 | Model architecture | Exact trainable parameter count | `outputs/benchmarks/benchmark.csv`; `parameter_counts()` | — | **supported** | **7,485** for five classes. The "~15,000" in the old README is wrong. Answers the author-action at line 244. |
| C22 | Model architecture | Layer-by-layer table | `architecture_record()` in every checkpoint | — | **supported** | Kernel sizes, padding, pooling and band edges are all recorded. |
| C23 | Model architecture | Five-output head diagram | — | — | **blocked-human** | `Figures/pipeline_5class.png` must be drawn by hand. The old `flowchart.png` does not depict this system. |
| C24 | Training and evaluation | Nested LOSO, inner folds grouped by participant | `folds.json`; `docs/experiment_protocol.md` §4 | any | **supported** | Answers the author-action at line 267. |
| C25 | Training and evaluation | "inner five-fold" | `folds.json` → `n_inner_folds_per_outer` | — | **contradicted** | With four training participants a participant-grouped inner LOSO has **four** folds. The code raises on five. |
| C26 | Training and evaluation | Hyperparameter grid, seeds, runs per fold | `selection/fold_outcomes.json`; `resolved_config.yaml` | any | **supported** | Every trial and every epoch history is saved. |
| C27 | Training and evaluation | Selection objective and tie-break | `docs/experiment_protocol.md` §4.2 | any | **supported** | Macro-F1; tie-break loss then earliest epoch. |
| C28 | Training and evaluation | γ, patience, min/max epochs, LR, batch size, weight decay | `resolved_config.yaml`; `selection/fold_outcomes.json` | `five_class_nested_loso.yaml` | **blocked-compute** | Values are *selected on inner folds*, so they exist only after the full run. |
| C29 | Training and evaluation | Augmentation is training-only | `docs/experiment_protocol.md` §5.2; `test_leakage.py` | — | **supported** | Enforced structurally: the augmenter raises for any non-training stage. |
| C30 | Training and evaluation | Normalisation is training-only | `fold_outcomes.json` → `normalizer.fitted_on` | — | **supported** | Asserted against the held-out participant before every evaluation. |

## Results

| claim_id | manuscript_location | proposed_claim | evidence_artifact | analysis_config | status | notes |
|---|---|---|---|---|---|---|
| C31 | Five-class performance | Five-class accuracy / macro-F1 / precision / recall | `paper_bundle/metrics.json`; `tables/macros.tex` | `five_class_nested_loso.yaml` | **blocked-compute** | Pipeline validated end to end; the full run has not been executed. |
| C32 | Five-class performance | Confusion matrix | `figures/five_class_confusion_matrix.png` | same | **blocked-compute** | Generator verified on the smoke run. |
| C33 | Five-class performance | ROC-AUC per class + macro | `figures/five_class_roc_curves.png`; `metrics.json` | same | **blocked-compute** | Derived from saved probabilities only — never from a confusion matrix. Answers the author-action at line 277. |
| C34 | Five-class performance | PR-AUC / average precision | `figures/five_class_pr_curves.png` | same | **blocked-compute** | — |
| C35 | Participant-level results | Per-participant mean ± SD and all five values | `figures/five_class_per_participant.png` | same | **blocked-compute** | Participant-level is **primary**; pooled windows are labelled descriptive. |
| C36 | Five-class performance | Total window count | `data_manifest.json` → `n_windows` | `trigger_constrained.yaml` | **supported** | **6,173** at a 0.25 s guard (chewing 3,635 · clench 799 · grinding 816 · movement 333 · rest 590). Depends on C19. |
| C37 | Sensors and dataset | Per-class window counts | `paper_bundle/tables/sample_counts.csv` | same | **supported** | Chewing is ~59 % of windows — *higher* than the legacy 45 %, because chewing bouts are long. |
| C38 | Effect of the chewing class | No-chewing four-class result | `metrics.json` → `no_chewing_four_class` | `modality_and_no_chewing.yaml` | **blocked-compute** | Purpose-trained, not a relabelled matrix. |
| C39 | Modality ablation | Fusion vs EMG-only vs audio-only | `figures/modality_comparison.png`; `metrics.json` → `modality_contrast` | same | **blocked-compute** | Matched windows, folds, seeds and budget. |
| C40 | Modality ablation | Audio benefit **excluding chewing** | `metrics.json` → `chewing_contrast` | same | **blocked-compute** | **The make-or-break analysis** (`Temp.md` item E). If the gain vanishes without chewing, the contribution is "cheap eating rejection", not grinding detection. |
| C41 | Secondary analyses | Binary tooth-contact metrics | `metrics.json` → `binary_tooth_contact` | `secondary_tasks.yaml` | **blocked-compute** | Sensitivity, specificity, PPV, NPV, ROC-AUC, PR-AUC. |
| C42 | Secondary analyses | Ternary result | `metrics.json` → `ternary` | same | **blocked-compute** | — |
| C43 | Engineering interpretation | "outperformed" the baselines | `paper_bundle/tables/condition_comparison.csv` | `baselines.yaml` | **blocked-compute** | All models receive identical EMG + audio. If the margin is not there, **delete the word**. |
| C44 | Class-level diagnostics | t-SNE description | `figures/five_class_tsne.png` + `_settings.json` | `five_class_nested_loso.yaml` | **blocked-compute** | Held-out embeddings only; labelled EXPLORATORY. Cluster appearance is not validation. |
| C45 | Training behavior | Training/validation curves | `figures/training_curves.png` | same | **blocked-compute** | Excludes the outer participant by construction. |

## Latency, size and framing

| claim_id | manuscript_location | proposed_claim | evidence_artifact | analysis_config | status | notes |
|---|---|---|---|---|---|---|
| C46 | Training behavior | Inference latency | `outputs/benchmarks/benchmark.csv` | — | **supported** | Measured: 1.53 ms forward, 1.67 ms end-to-end per window (CPU, batch 1). |
| C47 | Training behavior | "low-latency / real-time" | `benchmark.json` → `latency_budget` | — | **contradicted** | Three latencies must stay separate: 1000 ms context, 500 ms decision interval, ~1.7 ms compute. Compute time is **not** detection latency. |
| C48 | Training behavior | Model size | `benchmark.csv` → `model_size_kib_fp32` | — | **supported** | ~29 KiB at fp32. |
| C49 | Conclusion | "generalize to unseen individuals" | — | — | **contradicted** | Five participants cannot support this. Delete or rewrite (`Temp.md` item D). |
| C50 | Title / Abstract | "bruxism detection" | — | — | **editorial** | Reframe as classification of instructed awake tooth-contact tasks. |
| C51 | Throughout | "natural bruxing" | `labels.RAW_TOKEN_TO_CONDITION` | — | **contradicted** | The token is a filename label. Report as **instructed grinding**. |
| C52 | Results | 85.0 % four-class accuracy | `data_audit.json` → `historical_confusion_matrix_check` | `whole_recording_legacy.yaml` | **contradicted** | **Irreproducible.** The published matrix totals 11,845 — the all-five-family count *including* 595 rest windows — yet has no rest class, and no per-class support matches. Must not be reused. |
| C53 | Validity and limitations | Total window count in limitations | `data_manifest.json` | — | **supported** | Must match C36, not the legacy 11,845. |
| C54 | Discussion | Comparison with Sonmezocak & Kurt | — | — | **editorial** | Argument is about class sets, validation rigour and hardware; needs no new run. |
| C55 | References | 2025 consensus and INfORM/TMD citations | — | — | **blocked-human** | Q12. Citation verification against DOIs was **not** performed here. |

---

## Summary

| Status | Count |
|---|---:|
| supported | 18 |
| contradicted | 11 |
| blocked-human | 11 |
| blocked-compute | 13 |
| editorial | 2 |
| **total** | **55** |

**The eleven `contradicted` rows are the important ones.** Each is a statement the current
manuscript makes that the evidence does not support: the channel count, the trial timing,
the redundant filter stage, the wavelet band name, "inner five-fold", the parameter count,
the latency framing, the generalisation sentence, "natural bruxing", and the 85 % headline.

The thirteen `blocked-compute` rows all become available from one command each — the
pipeline is validated end to end and the configs are written. See the README.
