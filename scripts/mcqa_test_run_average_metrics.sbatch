#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
s#SBATCH -t 00:20:00
source $HOME/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/average_metrics.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml --work-dir data/results --dataset_name obqa --temperature-scaling --duo --deep-ensemble --deep-ensemble-size 2
