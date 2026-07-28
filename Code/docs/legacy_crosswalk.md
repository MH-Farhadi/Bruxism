# Legacy crosswalk

Every file, class and function of the research prototype, mapped to its replacement or to
the reason it was retired.

**Preservation.** Nothing was deleted. The prototype was moved with `git mv` into
`legacy/`, and the state immediately before the refactor is tagged
**`pre-refactor-audit`** at commit **`a857f9b`**. `git show pre-refactor-audit:<path>`
retrieves any original file.

> **The legacy code must not be used to produce results.** It is preserved for provenance:
> to explain where historical numbers came from and to prove that no behaviour was lost by
> accident.

---

## 1. File-level map

| Legacy file | Replacement | Notes |
|---|---|---|
| `bruxism_dataset.py` | `data/manifest.py`, `data/segments.py`, `data/dataset.py`, `data/splits.py` | Split into discovery, labelling, loading and splitting. |
| `preprocessing_utils.py` | `preprocessing/filters.py` | One authoritative implementation; SOS-based; Nyquist-validated. |
| `wavelet_features.py` | `preprocessing/wavelets.py`, `features/time_frequency.py` | Band naming, real WPT, true median frequency; broken CWT block removed. |
| `wavelet_dataset.py` | `data/dataset.py` | Lazy, memory-mapped, cache-keyed. |
| `wavelet_cnn.py` | `models/dual_branch.py`, `models/dwt.py` | Named bands; differentiable on-device DWT. |
| `training_improvements.py` | `training/engine.py`, `training/losses.py`, `training/selection.py` | Focal loss corrected; selection rules prespecified. |
| `run_new_wavelet_training.py` | `cli/run_nested_loso.py` + `runner.py` | Was the main path; its evaluation protocol was invalid (see §3). |
| `run_wavelet_training.py` | same | Superseded. |
| `run_feature_based_training.py` | `cli/run_baselines.py`, `models/baselines.py` | Matched-input baselines. |
| `run_random_forest_training.py` | `models/baselines.py` (`random_forest`) | **Window-level 5-fold CV removed** (see §3). |
| `sanity_check.py` | `cli/audit_dataset.py`, `tests/` | Split into a real audit command and an automated test suite. |
| `requirements.txt`, `requirements.yaml` | `pyproject.toml` | Two overlapping dependency specs replaced by one authoritative file. |
| `dataset_specs_report.txt` | `outputs/data_audit/<hash>/data_audit.md` | Regenerated, hashed, reproducible. |
| `flowchart.png` | — | Retained at the repository root; **it does not depict the current dual-branch system** and must be redrawn before reuse. |

---

## 2. Symbol-level map

| Legacy symbol | Replacement | Change |
|---|---|---|
| `BruxismDataset` | `WindowIndex` + `WindowDataset` + `RecordingCache` | Index/loading separated; lazy instead of eager. |
| `BruxismDataset.condition_mapping` | `labels.RAW_TOKEN_TO_CONDITION`, `CONDITION_TO_FAMILY` | `natural_bruxing` → `instructed_grinding`. |
| `create_train_test_split` | `NestedLOSOSplitter` | Fixed 1–4/5 split → outer LOSO with sealed test folds. |
| `create_dataloaders` | `NestedLOSOTrainer._make_loader` | Seeded generator and worker init. |
| `CLASS_REDUCTION_STRATEGIES` | `labels.TASK_DEFINITIONS` | Typed tasks with declared endpoints and notes. |
| `ReducedWaveletCoefficientDataset` | `ClassificationTask.label_for_family` | Mapping is data, not control flow. |
| `AugmentedWaveletDataset` | `preprocessing/augmentation.Augmenter` | Stage-guarded, deterministically seeded. |
| `ImprovedDualBranchWaveletCNN` | `models.dual_branch.DualBranchWaveletCNN` | Named bands; DWT out of the Python loop. |
| `..._wavelet_decompose_emg/_mic` | `models.dwt.WaveletDecompose1d` | Differentiable conv cascade, exact to 1e-15 vs `pywt`. |
| `FocalLoss` | `training.losses.FocalLoss` | **Corrected** (see §3). |
| `bandpass_filter` / `notch_filter` / `remove_baseline_drift` | `filters.FilterStage` + `apply_filter_chain` | Configurable, validated, single copy. |
| `apply_ica` / `reconstruct_from_ica` | — | **Retired** (see §3). |
| `extract_wavelet_features` | `features.FeatureExtractor` | Named features; failures raise. |
| `get_wavelet_feature_dimension` | `FeatureExtractor.n_features()` | Computed by extraction, not by arithmetic on paper. |

---

## 3. Defects corrected

### 3.1 The held-out subject was used for training decisions ⛔

`run_new_wavelet_training.py` trained on subjects 1–4 and used subject 5 as the
**validation set for every epoch** — for early stopping and checkpoint selection — and then
reported subject 5 again as the final test result. The reported number is therefore a
best-epoch-on-the-test-set figure, not a held-out result.

**Now:** `OuterFold.release_test_ids()` raises unless `purpose="final_evaluation"`, and
raises on a second call. Selection uses inner folds only.

### 3.2 Window-level K-fold cross-validation ⛔

`run_random_forest_training.py` used ordinary 5-fold CV over windows. With 0.5 s stride,
adjacent windows overlap by 50 %, and windows from the same recording and the same
participant land in different folds. Such a score does not measure generalisation to unseen
people.

**Now:** only participant-grouped splitting exists.
`tests/unit/test_leakage.py::test_random_window_kfold_is_not_available` asserts that no
K-fold splitter is exported.

### 3.3 Focal loss ⛔

```python
ce   = cross_entropy(logits, targets, weight=alpha, reduction="none")
pt   = torch.exp(-ce)                 # NOT the model probability when alpha != 1
focal = (1 - pt) ** gamma * ce
```

With class weights, `exp(-ce)` is not `p_t`, so the focusing term became a function of the
class weights and gamma/alpha interacted uncontrollably.

**Now:** `p_t = softmax(logits)[target]`, `loss = alpha_t · (1 − p_t)^γ · (−log p_t)`.
Verified against hand computation with and without alpha, and `γ=0` reduces exactly to
weighted cross-entropy.

### 3.4 Shallow checkpoint snapshot ⛔

`best_model_state = model.state_dict().copy()` is a shallow dict copy: the tensors are the
*live* parameters and keep changing as training continues. The "best" weights were whatever
the model happened to hold at the end.

**Now:** deep-copied, detached, cloned and written atomically.
`test_deep_copied_best_weights_survive_further_optimizer_steps` demonstrates the original
failure directly.

### 3.5 Wavelet band indexing ⛔

`pywt.wavedec` returns `[cA_L, cD_L, …, cD_1]`. The prototype commented
`emg_detail_idx_high = 0  # Use level 0 (highest frequency)` — but `details[0]` is `cD_L`,
the **lowest**-frequency detail (37.5–75 Hz at level 4). It also used `details[2]`, which
at level 4 is **D2** (150–300 Hz), while the manuscript called it "D3" (75–150 Hz).

**Now:** bands are named (`"A4"`, `"D3"`, `"D1"`) and resolved by `band_index()`. Verified
empirically: a 50 Hz tone puts 83 % of its energy in D4, a 400 Hz tone 92 % in D1.

**Manuscript action:** any sentence naming a wavelet band must be re-derived from
`model.band_frequency_table()`.

### 3.6 Continuous wavelet transform on a discrete wavelet ⛔

`pywt.cwt(sig, scales, 'db4', 1.0/fs)` — `db4` is a discrete wavelet and is not a valid
continuous-wavelet argument. The call raised, and the bare `except` appended four zeros per
channel. The "CWT features" were therefore **always zeros**.

**Now:** removed. Nothing in the manuscript depends on it.

### 3.7 Wavelet packet transform read only two nodes ⚠

`pywt.WaveletPacket(sig, wavelet, maxlevel=4)` was created, then only nodes `'a'` and `'d'`
— the two **first-level** nodes — were read. The declared depth of 4 was never used.

**Now:** `wavelet_packet_energies` walks all `2**max_level` nodes at the requested depth.

### 3.8 "Median frequency" was not a median frequency ⚠

The prototype's statistic was not the frequency below which half the power lies.

**Now:** `median_frequency()` computes it from a Welch PSD with interpolation, tested
against a known two-tone spectrum and against a direct Welch integration.

### 3.9 Silent zero-filling on feature failure ⛔

Bare `except: features.extend([0]*4)` made a broken transform indistinguishable from a
genuinely flat signal.

**Now:** failures raise `WaveletError` / `ValueError` and the sample is excluded.

### 3.10 Wavelets inside `forward()` ⚠

The decomposition ran as a NumPy/PyWavelets double loop inside `forward()`, forcing a
GPU→CPU→GPU round trip on every batch and making any latency measurement meaningless.

**Now:** `WaveletDecompose1d` is a fixed grouped-convolution cascade — device-native,
differentiable, vectorised, and exact to 1e-15 against `pywt.wavedec`.

### 3.11 Redundant filter stage ⚠

`notch(60) → bandpass(20–450) → highpass(5)`. The 5 Hz high-pass is a no-op after a 20 Hz
bandpass edge. Removed.

### 3.12 ICA that could not do anything ⚠

`apply_ica` reconstructed from **all** retained components, so
`inverse_transform(fit_transform(x)) ≈ x` — the step was close to identity. It was disabled
in the main path anyway. **Retired**; if source separation is wanted later it needs a
component-selection criterion and a validation.

### 3.13 Unexplained 3-second skip ⚠

`skip_initial_seconds=3.0` discarded 3 s of every recording without stated justification.

**Now:** a measured startup guard of 0.5 s. 66/100 recordings show a settling transient, all
settling within **0.40 s**, peak excursion up to **12,580×** the robust scale. The
measurement is in the manifest (`startup_transient_seconds`, `startup_transient_peak_ratio`).

### 3.14 Reproducibility gaps ⚠

No complete seeding policy; outputs written relative to the working directory; `plt.show()`
in a training script; a Windows absolute path hard-coded in two files
(`r"C:\Users\mhfar\Desktop\Depo\Brusxism_data"`).

**Now:** `SeedBundle`, `--data-root`/`$BRUXISM_DATA_ROOT`, `matplotlib.use("Agg")` and no
`plt.show()` anywhere, atomic writes into a versioned run bundle.

### 3.15 Parameter-count and README claims ⚠

The README claimed "~15,000 parameters" (and contained a "~15 parameters" typo), described
real-time natural-bruxism detection, and showed a flowchart that does not match the
dual-branch system.

**Now:** counts are computed by `model.parameter_counts()`. The current model is **7,485**
trainable parameters for five classes. The README was rewritten.

---

## 4. Behaviour deliberately preserved

| Behaviour | Where | Why |
|---|---|---|
| Whole-recording filename labelling | `SegmentationPolicy.WHOLE_RECORDING_LEGACY` | Reproduces the historical 11,845-window count exactly. |
| The four active classes without rest | task `legacy_active_four_class` | Lets the old framing be compared, labelled provenance-only. |
| db4 / coif5 wavelet choices | default `BranchConfig` | Continuity with the prototype; now explicit and configurable. |
| 1 s / 0.5 s windowing | protocol default | Unchanged; the protocol says so explicitly. |

---

## 5. The historical 85 % result

`whole_recording_legacy` reproduces the prototype's window counts **exactly**:

| Class | Reproduced (whole-recording) | Published matrix support | Match |
|---|---:|---:|:--:|
| movement | 1,785 | 1,877 | ✗ |
| clench | 2,380 | 2,503 | ✗ |
| instructed grinding | 1,769 | 1,861 | ✗ |
| chewing | 5,316 | 5,604 | ✗ |
| rest | 595 | *(absent from the matrix)* | — |
| **total** | **11,845** (all five families) | **11,845** | ✓ |

The published four-class confusion matrix totals exactly 11,845 — which is the
whole-recording count across **all five** families **including the 595 rest windows** — even
though the matrix has no rest class. No per-class support matches.

**Conclusion: the published matrix is not reproducible from these recordings under any
labelling policy implemented here, and must not be reused.** The 85.0 % accuracy derived
from it is treated as unverified. This is recorded in `data_audit.json` under
`historical_confusion_matrix_check`.
