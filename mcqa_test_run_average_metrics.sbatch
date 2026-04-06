#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
s#SBATCH -t 00:20:00
source ~/.bashrc
mamba activate adl-asymmetric-lm-duos

pip install -e .


python tools/average_metrics.py --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml --work-dir data/results --dataset_name arc_c --ib-edl --duo --duo-constrained --deep-ensemble --deep-ensemble-size 2
python tools/average_metrics.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml --work-dir data/results --dataset_name obqa --duo --temperature-scaling --deep-ensemble --deep-ensemble-size 2
python tools/average_metrics.py --config_base configs/csqa_qwen2_7b/lora_csqa_qwen2_7b.yaml --work-dir data/results --dataset_name csqa --deep-ensemble --deep-ensemble-size 2