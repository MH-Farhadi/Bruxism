# Reproducibility

Every number this project publishes must be recoverable from: a configuration file, a
source commit, a data-manifest hash, a seed and an environment record. This document
explains what is captured, where it lives, and what still limits exact reproduction.

---

## 1. The run bundle

Each production run writes one immutable directory:

```
outputs/runs/<run_id>/
├── resolved_config.yaml     fully resolved configuration (the hash source)
├── environment.json         interpreter, libraries, OS, CPU, CUDA/cuDNN, GPU names
├── source_state.json        commit, branch, dirty flag, dirty file list, diff SHA-256
├── data_manifest.json       manifest hash, policy version, segmentation, window counts
├── data_manifest.sha256     the manifest hash alone, for quick comparison
├── data_manifest.csv        the full per-recording manifest
├── run_bundle.json          run id and the three identifying hashes
├── folds.json               the complete split plan, written BEFORE training
├── folds/                   one JSON + one Parquet per completed (condition, seed, fold)
├── selection/               every hyperparameter trial and every epoch history
├── logs/run.log.jsonl       one JSON object per log record
├── checkpoints/             weights + architecture + normalizer + class weights
├── predictions.parquet      the prediction ledger
├── metrics.json / .csv      recomputed from the ledger
└── figures/                 the run's own figure set (PNG + PDF), README.md and
                             figure_index.json naming every figure and every skip reason
```

`figures/` is written after the last fold, from the artifacts above plus the raw recordings
and this run's own checkpoints. It is a **display** artifact: it enters no hash, no metric
and no claim, so `--no-figures` (or a failure while drawing) changes nothing a result depends
on. `bruxism-figures --run-dir outputs/runs/<run_id> --data-root ...` regenerates it from the
bundle without retraining, which is also how a run that predates a figure acquires it.

Three hashes identify a run:

| Hash | Covers | Changes when |
|---|---|---|
| `config_hash` | the resolved configuration | any setting changes |
| `manifest_hash` | recording checksums, conditions, exclusions, policy version | the data or the quality policy changes |
| `window_index_hash` | manifest hash + segmentation config + every sample id | the segmentation policy changes |

A resume that finds a mismatch on any of the three **raises** rather than mixing artifacts.

---

## 2. Seeding

`SeedBundle.from_base(seed)` derives every library seed from one integer, so quoting one
number reproduces the run:

| Stream | Derivation |
|---|---|
| Python `random` | `seed` |
| NumPy global | `(seed · 2654435761 + 1) mod 2³¹` |
| PyTorch (CPU + all CUDA) | `(seed · 40503 + 7) mod 2³¹` |
| DataLoader generator | `(seed · 2246822519 + 13) mod 2³¹` |
| Augmentation | `blake2b(run_seed \| epoch \| sample_id)` |

Augmentation is seeded **per sample**, not per step, so a given window in a given epoch of a
given run receives the same transformation regardless of worker count, batch order or
shuffling.

DataLoader workers are re-seeded from the parent's torch seed by `worker_init_fn`; without
it every worker inherits the same NumPy state.

`deterministic: true` sets `CUBLAS_WORKSPACE_CONFIG`, disables cuDNN benchmarking and calls
`torch.use_deterministic_algorithms(True, warn_only=False)`. An operation with no
deterministic implementation therefore **raises** rather than silently falling back.

### What still limits exactness

- A different GPU model, CUDA version or cuDNN version can change floating-point reduction
  order. All are recorded in `environment.json`.
- CPU and GPU results agree to roughly float32 tolerance, not bit-for-bit.
- Changing `num_workers` does not change results (augmentation is sample-seeded) but does
  change wall-clock timing.

---

## 3. Environment

Runtime dependencies and their version floors live in `pyproject.toml`; each floor is the
oldest release whose behaviour the code relies on, and the reason is written next to it.

For an exact environment, freeze the resolved versions:

```bash
python -m pip install -e ".[dev,video]"
python -m pip freeze --exclude-editable > requirements.lock.txt
```

`requirements.lock.txt` pins the exact resolved set for a given platform. `environment.json`
inside every run bundle records what was *actually* loaded, which is the authoritative
record for that run.

The legacy `requirements.txt` / `requirements.yaml` pair is preserved under `legacy/` and is
no longer authoritative — the two files disagreed with each other.

---

## 4. Data provenance

- The data root is supplied by `--data-root` or `$BRUXISM_DATA_ROOT`. **No source file
  contains a machine-specific path.**
- The manifest stores a streaming SHA-256 of every CSV, so a changed recording changes the
  manifest hash and invalidates every downstream artifact.
- Raw data is never copied into the repository. The `.npy` files shipped with the data are
  never read; caches are regenerated and keyed by `(csv_sha256, filter config, channel
  selection, sampling rate)`.
- `outputs/` is git-ignored in its entirety.

---

## 5. Atomicity and resumption

Every artifact is written to a temporary file in the destination directory and then
`os.replace`d into position, so a killed run never leaves a half-written metrics file that a
later stage would read as complete.

Execution resumes at the granularity of one `(condition, seed, outer fold)` triple. A
completed fold's ledger and outcome JSON are written **before** the next fold starts, so an
interrupted run loses at most one fold.

---

## 6. Verifying a reproduction

```bash
# 1. Same data?
cat outputs/runs/<run_id>/data_manifest.sha256
bruxism-audit --data-root <root> --validate-only | grep manifest_hash

# 2. Same code?
python - <<'PY'
import json; s = json.load(open("outputs/runs/<run_id>/source_state.json"))
print(s["commit"], "dirty:", s["is_dirty"], s.get("diff_sha256"))
PY

# 3. Do the published metrics follow from the saved predictions?
bruxism-summarize --runs-root outputs/runs --run-id <run_id> --output /tmp/check
diff <(jq -S . /tmp/check/metrics.json) <(jq -S . outputs/runs/<run_id>/metrics.json)
```

Step 3 is the important one: it recomputes every metric from `predictions.parquet` alone. If
it disagrees with `metrics.json`, the published summary does not follow from the saved
predictions and the run must not be reported.

---

## 7. Quality gates

```bash
python -m ruff format --check src scripts tests
python -m ruff check src scripts tests
python -m mypy
python -m pytest -m "not slow"     # fast unit tests, no data needed
python -m pytest                    # + integration on synthetic participants
```

The whole suite runs on synthetic fixtures and needs **no access to the private data root**,
so it is safe to run in CI.

---

## 8. Known reproducibility limits

| Limit | Effect | Mitigation |
|---|---|---|
| Hardware-dependent float reduction order | last-digit metric differences | full hardware record in `environment.json` |
| Dirty working tree | commit alone does not identify the source | dirty flag + file list + diff SHA-256 recorded, and a warning is logged |
| Non-deterministic ops under `deterministic: true` | run raises | intentional: a silent fallback would be worse |
| t-SNE | needs the data root to recompute held-out embeddings | exact perplexity/init/seed/checkpoint saved beside the figure |
| Video probing | depends on installed codecs | excluded from the manifest hash |
