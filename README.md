# Audio–EMG classification of instructed jaw activities

Proof-of-concept study and reproducible analysis pipeline for classifying **instructed, awake** jaw and tooth-contact tasks from surface EMG and a near-TMJ microphone. Five adults with a prior clinical diagnosis of bruxism completed a single controlled laboratory session.

Manuscript working title: *Audio–EMG Fusion with Dual-Branch Wavelet CNNs for Classifying Instructed Jaw Activities Relevant to Tooth-Contact Bruxism*.

> **Scope.** This is not a clinical bruxism detector, a sleep study, or a validated ambulatory/real-time system. Labels describe experimental task conditions (rest, jaw movement, clench, grinding, chewing), not independently adjudicated bruxism episodes. See [`Code/README.md`](Code/README.md) for the full “what this is / is not” table.

---

## Repository layout

```text
Bruxism/
├── Code/          Reproducible Python package, configs, tests, and docs
├── Data/          Raw recordings (local only — not in Git)
├── Paper/         Manuscript sources, figures, and review materials
├── REPORT.md      Implementation report for the Code/ rebuild
├── sol.md         Original implementation brief that drove the rebuild
└── README.md      This file
```

| Path | What it contains |
|---|---|
| **[`Code/`](Code/)** | Installable `bruxism` package: data manifest & labelling, preprocessing, dual-branch wavelet CNN and baselines, nested LOSO training, evaluation, figure/table generation, CLI, and ~196 tests on synthetic fixtures. Detailed quick start lives in [`Code/README.md`](Code/README.md). |
| **`Data/`** | Per-participant CSV/AVI/metadata recordings, surveys, and admin files. Kept on disk for local analysis; **excluded from Git** (see root `.gitignore`). Point the software at it with `BRUXISM_DATA_ROOT` or `--data-root`. |
| **[`Paper/`](Paper/)** | LaTeX manuscript (`K_Farhadi_Paper_Bruxism/`, primary draft `Main_2.tex`), figures, author photos, and `Reviews/` correspondence. |
| **[`REPORT.md`](REPORT.md)** | Engineering status of the rebuild: what was fixed, what the smoke tests cover, and which scientific claims remain blocked. |
| **[`sol.md`](sol.md)** | Full implementation brief (protocol, defects in the prototype, required analyses). Historical context for why `Code/` looks the way it does. |

A small nested `Bruxism/` folder holds only the GitHub template `LICENSE` / `README` stubs; the real software licence is [`Code/LICENSE`](Code/LICENSE) (MIT, **software only**).

---

## Study at a glance

| | |
|---|---|
| **Participants** | 5 adults (prior clinical bruxism diagnosis) |
| **Setting** | Awake, instructed laboratory tasks — not spontaneous or sleep recordings |
| **Modalities** | Bipolar surface EMG (masseter / temporalis) + omnidirectional microphone (~1200 Hz) |
| **Primary task** | Five-class: quiet rest, jaw movement, clench, grinding, chewing |
| **Evaluation** | Outer leave-one-subject-out; participant-grouped inner folds for selection |
| **Model** | Dual-branch wavelet CNN (EMG + audio fusion), plus modality and architecture baselines |

Secondary analyses (modality ablations, no-chewing sensitivity, binary/ternary endpoints) are configured under `Code/configs/experiments/`.

---

## Getting started (software)

All runnable work lives under `Code/`. From the repo root:

```bash
cd Code
python -m pip install -e ".[dev,video]"

# Quality gates — no private data required
python -m pytest -m "not slow"

# Point at your local data root (never committed)
export BRUXISM_DATA_ROOT=../Data

bruxism-audit --data-root "$BRUXISM_DATA_ROOT" --output-root outputs/data_audit
bruxism-train --config configs/experiments/smoke.yaml --max-folds 1
```

Full command tables, figure regeneration, leakage controls, and limitations: **[`Code/README.md`](Code/README.md)**.

Protocol and reproducibility docs:

| Document | Role |
|---|---|
| [`Code/docs/experiment_protocol.md`](Code/docs/experiment_protocol.md) | Prespecified labelling, splits, selection, metrics |
| [`Code/docs/data_dictionary.md`](Code/docs/data_dictionary.md) | Schema, taxonomy, privacy classification |
| [`Code/docs/reproducibility.md`](Code/docs/reproducibility.md) | Run bundles, hashes, seeding |
| [`Code/docs/open_questions.md`](Code/docs/open_questions.md) | Claims blocked on human decisions (IRB, hardware, etc.) |
| [`Code/docs/legacy_crosswalk.md`](Code/docs/legacy_crosswalk.md) | Prototype → current mapping; irreproducible historical 85% result |

---

## Data and privacy

`Data/` may contain identifiable video, photographs, health-related surveys, and reimbursement material.

- It is **not** tracked in this repository.
- Analysis code reads it only via `--data-root` / `$BRUXISM_DATA_ROOT`.
- Manifests use canonical IDs (`S01`…) and data-root-relative paths.
- The MIT licence covers the software only; it does **not** license participant data.

---

## Licence

Software under `Code/` is MIT. Participant data and paper review materials are not covered by that licence.
