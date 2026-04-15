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

CFG_BASE=configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml
CFG_SIDE=configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml
WORK_DIR=$HOME/results
DATASET=arc_c

BETAS=(0.00009 0.00015 0.0005 0.001)
TUNE_SEED=1
FINAL_SEEDS=(2 3 4 5 6)

for BETA in "${BETAS[@]}"; do
    echo "=============================================================="
    echo "[tune] beta=$BETA (seed $TUNE_SEED)"
    echo "=============================================================="
    python tools/train.py \
        --config_base     "$CFG_BASE" \
        --config_sidekick "$CFG_SIDE" \
        -w "$WORK_DIR" \
        --ib-edl \
        --seed $TUNE_SEED \
        --beta-base $BETA \
        --beta-sidekick $BETA
done

echo "=============================================================="
echo "[pick-base] best base beta by base_ece"
echo "=============================================================="
BEST_BASE=$(python tools/pick_best_beta.py \
    --config_base     "$CFG_BASE" \
    --config_sidekick "$CFG_SIDE" \
    -w "$WORK_DIR" -d "$DATASET" \
    --ib-edl \
    --tune base \
    --seed $TUNE_SEED \
    --betas "${BETAS[@]}")
echo "[pick-base] winner: $BEST_BASE"

echo "=============================================================="
echo "[pick-sidekick] best sidekick beta by sidekick_ece"
echo "=============================================================="
BEST_SIDE=$(python tools/pick_best_beta.py \
    --config_base     "$CFG_BASE" \
    --config_sidekick "$CFG_SIDE" \
    -w "$WORK_DIR" -d "$DATASET" \
    --ib-edl \
    --tune sidekick \
    --seed $TUNE_SEED \
    --betas "${BETAS[@]}")
echo "[pick-sidekick] winner: $BEST_SIDE"

echo "[summary] base=$BEST_BASE  sidekick=$BEST_SIDE"

for SEED in "${FINAL_SEEDS[@]}"; do
    echo "=============================================================="
    echo "[final] seed=$SEED beta_base=$BEST_BASE beta_sidekick=$BEST_SIDE"
    echo "=============================================================="
    python tools/train.py \
        --config_base     "$CFG_BASE" \
        --config_sidekick "$CFG_SIDE" \
        -w "$WORK_DIR" \
        --ib-edl --duo \
        --seed $SEED \
        --beta-base $BEST_BASE \
        --beta-sidekick $BEST_SIDE
done

echo "=============================================================="
echo "[aggregate] beta_base=$BEST_BASE beta_sidekick=$BEST_SIDE"
echo "=============================================================="
python tools/average_metrics.py \
    --config_base     "$CFG_BASE" \
    --config_sidekick "$CFG_SIDE" \
    -w "$WORK_DIR" -d "$DATASET" \
    --ib-edl --duo \
    --beta-base $BEST_BASE \
    --beta-sidekick $BEST_SIDE \
    --deep-ensemble --deep-ensemble-size 2
