#!/usr/bin/env bash
#
# Run the two confirmatory experiments the manuscript still marks as pending, on the
# corrected (mains-harmonic) filter chain:
#
#   RQ3  bruxism-baselines   architecture comparison with matched inputs
#   RQ2  bruxism-ablations   fusion / EMG-only / audio-only, with and without chewing
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
echo "==> regenerating the paper bundle across every run in outputs/runs"
bruxism-report \
  --runs-root outputs/runs \
  --output-root outputs/paper_bundle \
  --data-root "$DATA_ROOT" \
  2>&1 | tee -a "${LOG_DIR}/report_${RUN_TAG}.log"

cat <<'EOF'

Done. Next steps:

  1. Read outputs/paper_bundle/paper_results.md - it lists every condition, the
     modality contrast (RQ2) and the architecture comparison (RQ3), each recomputed
     from the saved prediction ledgers.

  2. Hand the new run ids to Claude and ask it to fold the results into
     Paper/K_Farhadi_Paper_Bruxism/Main_2.tex. Three places currently say
     "pending re-measurement" and must be replaced with measured values:
       - Table 3 baseline rows (RQ3)
       - Table 4 modality ablation rows (RQ2)
       - the Results, Discussion and Limitations paragraphs that mark them pending

  3. Regenerate the run-dependent manuscript figures if the headline five-class run
     changes:
       python scripts/evaluate/make_manuscript_figures.py \
           --run-dir outputs/runs/<five-class run> --data-root ../Data
EOF
