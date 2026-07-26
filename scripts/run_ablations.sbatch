#!/bin/bash
#SBATCH --job-name=asymduo_ablations
#SBATCH --output=logs/ablations_%j.out
#SBATCH --error=logs/ablations_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# NOTE: no --gres=gpu. These ablations are post-hoc on the saved prediction
# NPZs, so nothing here loads a model or touches a GPU. Adjust the partition /
# account directives below to match the other scripts in this repo.
# #SBATCH --partition=cpu
# #SBATCH --account=<account>

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override by exporting before sbatch, or edit here)
# ---------------------------------------------------------------------------
WORK_DIR="${WORK_DIR:-data/results}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/ablations}"

# Which experiment directories to analyse: <WORK_DIR>/<dataset>/<MODEL_TAG>/<METHOD>
MODEL_TAG="${MODEL_TAG:-qwen2_7b_qwen2_15b}"
METHOD="${METHOD:-lora_tempscaling_duo_unconstrained}"
DATASETS="${DATASETS:-arc_c arc_e csqa obqa race sciq}"
SEEDS="${SEEDS:-1 2 3 4 5}"

# Which ablations: "all", "priority", or an explicit list such as "E2 E3 A1".
ABLATIONS="${ABLATIONS:-all}"

# Resampling budgets. Defaults match the agreed plan: 100 permutation draws for
# the A2 nulls, 10 subset draws per proportion for C1.
NULL_DRAWS="${NULL_DRAWS:-100}"
DOSE_DRAWS="${DOSE_DRAWS:-10}"
CALIB_DRAWS="${CALIB_DRAWS:-10}"
DECOMP_NULL_DRAWS="${DECOMP_NULL_DRAWS:-20}"
RNG_SEED="${RNG_SEED:-0}"

METRICS="${METRICS:-nll ece auroc_error aurc}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# module load python/3.11          # uncomment / adapt to the cluster
# source "${HOME}/venvs/asymduos/bin/activate"

mkdir -p logs "${OUT_DIR}"

# BLAS threads: the fits are small, and oversubscribing them slows the
# permutation sweeps down rather than speeding them up.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "=============================================================="
echo "Asymmetric Duos -- post-hoc ablations"
echo "job          : ${SLURM_JOB_ID:-<interactive>}"
echo "host         : $(hostname)"
echo "work dir     : ${WORK_DIR}"
echo "out dir      : ${OUT_DIR}"
echo "model tag    : ${MODEL_TAG}"
echo "method       : ${METHOD}"
echo "datasets     : ${DATASETS}"
echo "seeds        : ${SEEDS}"
echo "ablations    : ${ABLATIONS}"
echo "started      : $(date)"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Preflight: resolve result dirs and fail loudly if none exist
# ---------------------------------------------------------------------------
RESULT_DIRS=()
MISSING=()
for ds in ${DATASETS}; do
    dir="${WORK_DIR}/${ds}/${MODEL_TAG}/${METHOD}"
    if [[ -d "${dir}/base" && -d "${dir}/sidekick" ]]; then
        RESULT_DIRS+=("${dir}")
        echo "  [found]   ${dir}"
    else
        MISSING+=("${dir}")
        echo "  [MISSING] ${dir}"
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo
    echo "WARNING: ${#MISSING[@]} experiment director(ies) missing base/ or sidekick/;"
    echo "         they will be skipped. Check MODEL_TAG and METHOD if this is unexpected."
fi

if [[ ${#RESULT_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no usable experiment directories found under ${WORK_DIR}." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo
echo "--- running ablations ---"
python tools/run_ablations.py \
    --result-dirs "${RESULT_DIRS[@]}" \
    --seeds ${SEEDS} \
    --out-dir "${OUT_DIR}" \
    --ablations ${ABLATIONS} \
    --null-draws "${NULL_DRAWS}" \
    --dose-draws "${DOSE_DRAWS}" \
    --calib-draws "${CALIB_DRAWS}" \
    --decomposition-null-draws "${DECOMP_NULL_DRAWS}" \
    --rng-seed "${RNG_SEED}"
STATUS=$?

echo
echo "--- rendering markdown tables ---"
python tools/make_ablation_tables.py \
    --in-dir "${OUT_DIR}" \
    --out "${OUT_DIR}/report.md" \
    --metrics ${METRICS}

# ---------------------------------------------------------------------------
# VERIFY summary
#
# Printed inline because it is the check to read first: it compares a fresh
# float64 refit against the duo scales stored during training. Scales stored as
# alpha = beta = 1.0000 mean the original fit never moved off its starting
# point, in which case the stored duo is an unweighted logit sum rather than a
# fitted one.
# ---------------------------------------------------------------------------
if [[ -f "${OUT_DIR}/VERIFY.csv" ]]; then
    echo
    echo "--- VERIFY: stored vs refitted duo scales ---"
    python - "${OUT_DIR}/VERIFY.csv" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
keep = ['alpha_stored', 'beta_stored', 'alpha_refit', 'beta_refit',
        'alpha_rel_diff', 'beta_rel_diff']
pivot = (df[df.metric.isin(keep)]
         .pivot_table(index='dataset', columns='metric', values='value')
         .reindex(columns=keep))
print(pivot.round(4).to_string())

stored_at_one = ((pivot['alpha_stored'] - 1.0).abs() < 1e-6) & \
                ((pivot['beta_stored'] - 1.0).abs() < 1e-6)
if stored_at_one.any():
    print()
    print('!! Stored scales are alpha = beta = 1 for: '
          + ', '.join(pivot.index[stored_at_one]))
    print('!! The training-time fit did not move off its starting point, so the '
          'stored duo is an unweighted logit sum. See duo_fit.STALL_WARNING.')
PY
fi

echo
echo "=============================================================="
echo "finished     : $(date)"
echo "tidy CSVs    : ${OUT_DIR}/*.csv"
echo "report       : ${OUT_DIR}/report.md"
echo "=============================================================="
exit ${STATUS}
