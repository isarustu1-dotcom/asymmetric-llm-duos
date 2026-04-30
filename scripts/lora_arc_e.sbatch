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

CFG_BASE=configs/arc_e_qwen2_7b/lora_arc_e_qwen2_7b.yaml
CFG_SIDE=configs/arc_e_qwen2_15b/lora_arc_e_qwen2_15b.yaml
WORK_DIR=$HOME/results
DATASET=arc_e

for SEED in 1 2 3 4 5; do
    echo "=============================================================="
    echo "[LoRA arc_e] seed=$SEED — training (duo + temperature-scaling)"
    echo "=============================================================="
    python tools/train.py \
        --config-base     "$CFG_BASE" \
        --config-sidekick "$CFG_SIDE" \
        -w "$WORK_DIR" \
        --duo --temperature-scaling \
        --seed $SEED
    echo "[LoRA arc_e] seed=$SEED — done."
done

echo "=============================================================="
echo "[LoRA arc_e] aggregating metrics (duo + TS + DE size 2)"
echo "=============================================================="
python tools/average_metrics.py \
    --config-base     "$CFG_BASE" \
    --config-sidekick "$CFG_SIDE" \
    -w "$WORK_DIR" -d "$DATASET" \
    --duo --temperature-scaling \
    --deep-ensemble --deep-ensemble-size 2
