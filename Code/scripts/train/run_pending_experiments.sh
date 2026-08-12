#!/usr/bin/env bash
#
# Run the confirmatory experiments on the corrected (mains-harmonic) filter chain:
#
#   RQ3  bruxism-baselines   architecture comparison with matched inputs   -- STILL PENDING
#   RQ2  bruxism-ablations   fusion / EMG-only / audio-only, +/- chewing   -- DONE
#
# RQ2 completed on 2026-08-10 as run modality_and_no_chewing_20260810T020642_cead62e4 and
# is written up in the manuscript. It was launched through bruxism-ablations directly, so
# its run id carries the auto-generated timestamp_confighash suffix rather than a RUN_TAG.
# Re-running 'ablations' here therefore starts a NEW run rather than resuming that one --
# pass 'baselines' unless you deliberately want to repeat RQ2.
#
# Both runs are resumable. If one is interrupted, re-run this script with the same
# arguments and it picks up at the first fold that has no saved prediction file; nothing
# already computed is repeated. The run ids are fixed for exactly that reason.
#
# Usage, from Code/ :
#
#   ./scripts/train/run_pending_experiments.sh              # both, sequentially
#   ./scripts/train/run_pending_experiments.sh baselines    # RQ3 only
#   ./scripts/train/run_pending_experiments.sh ablations    # RQ2 only
#   ./scripts/train/run_pending_experiments.sh check        # validate configs, run nothing
#
# Environment overrides:
#   DATA_ROOT   path to the recordings          (default ../Data)
#   RUN_TAG     suffix appended to the run ids  (default the date, YYYYMMDD)
#
# Rough cost. The reported five-class run did 255 model fits in 5.0 h on an RTX 4090
# laptop GPU, so ~71 s per fit. Baselines are 300 fits and the ablations about 320
# fit-equivalents (the no-chewing task has 2,538 windows against 6,173), which puts each
# job in the 6-10 h range and both together at roughly an overnight run. Progress lines
# and a running ETA are written to the log file named below.
#
# What NOT to do: these runs are confirmatory. Do not re-run one of them after seeing its
# score, and do not edit a config between attempts. If a config has to change, that is a
# new experiment with a new name, and the superseded result is reported as superseded.

set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_ROOT="${DATA_ROOT:-../Data}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
LOG_DIR="outputs/logs"
WHAT="${1:-all}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "error: DATA_ROOT '$DATA_ROOT' does not exist. Set DATA_ROOT=/path/to/Data." >&2
  exit 1
fi

for command in bruxism-baselines bruxism-ablations bruxism-report; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: '$command' is not on PATH. Install the package first: pip install -e ." >&2
    exit 1
  fi
done

# A dirty tree means the exact source of a run cannot be recovered from its commit alone.
# The runner records this either way; the warning is here so it is seen before 8 h of GPU
# time, not afterwards in the run bundle.
if git rev-parse --git-dir >/dev/null 2>&1 && ! git diff --quiet HEAD -- src configs; then
  echo "warning: src/ or configs/ have uncommitted changes; the run bundle will be marked dirty."
  echo "         Commit first if you want these results to be reproducible from the commit."
  echo
fi

run_baselines() {
  local run_id="baselines_${RUN_TAG}"
  echo "==> RQ3 architecture comparison  ->  outputs/runs/${run_id}"
  bruxism-baselines \
    --config configs/experiments/baselines.yaml \
    --data-root "$DATA_ROOT" \
    --run-id "$run_id" \
    --progress plain \
    2>&1 | tee -a "${LOG_DIR}/${run_id}.log"
}

run_ablations() {
  local run_id="modality_and_no_chewing_${RUN_TAG}"
  echo "==> RQ2 modality and no-chewing ablation  ->  outputs/runs/${run_id}"
  bruxism-ablations \
    --config configs/experiments/modality_and_no_chewing.yaml \
    --data-root "$DATA_ROOT" \
    --run-id "$run_id" \
    --progress plain \
    2>&1 | tee -a "${LOG_DIR}/${run_id}.log"
}

case "$WHAT" in
  check)
    # --validate-only still writes a dry-run plan directory. Send those to a scratch root
    # so a preflight never leaves a half-empty run in outputs/runs.
    echo "==> validating configs (no training)"
    scratch="outputs/runs/_validate"
    rm -rf "$scratch"
    bruxism-baselines --config configs/experiments/baselines.yaml \
      --data-root "$DATA_ROOT" --set "output.runs_root=$scratch" --validate-only
    bruxism-ablations --config configs/experiments/modality_and_no_chewing.yaml \
      --data-root "$DATA_ROOT" --set "output.runs_root=$scratch" --validate-only
    rm -rf "$scratch"
    echo "configs validate. Re-run without 'check' to train."
    exit 0
    ;;
  baselines) run_baselines ;;
  ablations) run_ablations ;;
  all)       run_baselines; run_ablations ;;
  *)
    echo "usage: $0 [all|baselines|ablations|check]" >&2
    exit 2
    ;;
esac

echo
echo "==> regenerating one paper bundle per run"
# One bundle per run, not one pooled bundle. Several runs legitimately contain the same
# condition -- the primary five-class run and the ablation both hold
# five_class::dual_branch_wavelet_cnn::fusion -- and the ledger asserts that every held-out
# window is predicted exactly once per configuration. Pooling them across runs therefore
# aborts with a duplicate-row assertion, and it would be the wrong comparison anyway:
# those two conditions differ in protocol, not only in what they measure.
for run_id in $(
  case "$WHAT" in
    baselines) echo "baselines_${RUN_TAG}" ;;
    ablations) echo "modality_and_no_chewing_${RUN_TAG}" ;;
    all)       echo "baselines_${RUN_TAG} modality_and_no_chewing_${RUN_TAG}" ;;
  esac
); do
  [[ -f "outputs/runs/${run_id}/predictions.parquet" ]] || continue
  echo "  ${run_id}  ->  outputs/paper_bundle/${run_id}"
  bruxism-report \
    --runs-root outputs/runs \
    --run-id "$run_id" \
    --output-root "outputs/paper_bundle/${run_id}" \
    --data-root "$DATA_ROOT" \
    2>&1 | tee -a "${LOG_DIR}/report_${RUN_TAG}.log"
done

cat <<'EOF'

Done. Next steps:

  1. Read outputs/paper_bundle/<run id>/paper_results.md - one bundle per run, each
     listing that run's conditions recomputed from its saved prediction ledger. The
     ablation bundle also carries the modality contrast (RQ2).

  2. Fold the results into Paper/K_Farhadi_Paper_Bruxism/Main_2.tex. RQ2 is already
     written up from modality_and_no_chewing_20260810T020642_cead62e4. What still says
     "pending re-measurement" is RQ3 only:
       - Table 3 baseline rows
       - the RQ3 paragraph in Results (Section 4.1)
       - the RQ3 sentences in Discussion 6.1 and in Validity and limitations
       - the RQ3 sentence in the Conclusion and in the Abstract

  3. Regenerate the run-dependent manuscript figures if the run they depict changes:
       python scripts/evaluate/make_manuscript_figures.py \
           --run-dir outputs/runs/<five-class run> --data-root ../Data
       python scripts/evaluate/make_ablation_figure.py \
           --run-dir outputs/runs/<ablation run>
EOF
