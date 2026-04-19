#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00

set -euo pipefail

cd asymmetric-llm-duos

source $HOME/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate adl-asymmetric-lm-duos

pip install -e .

CFG_BASE=configs/arc_c_qwen2_7b/blob_arc_c_qwen2_7b.yaml
CFG_SIDE=configs/arc_c_qwen2_15b/blob_arc_c_qwen2_15b.yaml
WORK_DIR=$HOME/results
DATASET=arc_c

for SEED in 1 2 3 4 5; do
    echo "=============================================================="
    echo "[BLoB arc_c] seed=$SEED — training"
    echo "=============================================================="
    python tools/train.py \
        --config_base     "$CFG_BASE" \
        --config_sidekick "$CFG_SIDE" \
        -w "$WORK_DIR" \
        --blob --duo \
        --seed $SEED
    echo "[BLoB arc_c] seed=$SEED — done."
done

echo "=============================================================="
echo "[BLoB arc_c] aggregating metrics (DE size 2)"
echo "=============================================================="
python tools/average_metrics.py \
    --config_base     "$CFG_BASE" \
    --config_sidekick "$CFG_SIDE" \
    -w "$WORK_DIR" -d "$DATASET" \
    --blob --duo \
    --deep-ensemble --deep-ensemble-size 2
