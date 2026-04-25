#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -J postprocess_blob_asymduos

# Post-process all BLoB asymmetric-duos runs produced by the
# bayesian-peft fork's scripts/asymmetric_duos/blob_*.sbatch jobs.
#
# For each dataset:
#   1. optimize_weights_blob(..., constrained=False) per seed 1..5
#   2. Deep ensemble (size 2) over same-model seed pairs
#   3. Mean ± 1 SD summary CSV + pretty test table
#
# This is CPU-only (no GPU needed). A GPU is requested only because the
# queue/partition expects it; remove --gres if your partition allows it.

set -euo pipefail

cd "$HOME/asymmetric-llm-duos"

source "$HOME/miniforge3/bin/activate"
eval "$(mamba shell hook --shell bash)"
mamba activate adl-asymmetric-lm-duos

pip install -e . >/dev/null

RESULTS_ROOT="$HOME/results"
DATASETS=(arc_c arc_e obqa csqa race sciq)

python tools/postprocess_blob_asymduos.py \
    --results-root "$RESULTS_ROOT" \
    --datasets "${DATASETS[@]}" \
    --seeds 1 2 3 4 5 \
    --deep-ensemble-size 2

echo "Post-processing complete. Summaries under:"
echo "  $RESULTS_ROOT/overall_metrics_summary_blob/<dataset>/metrics_summary_sd.csv"
