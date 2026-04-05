#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
source ~/.bashrc
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --duo --temperature-scaling --seed 1
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --duo --temperature-scaling --seed 2
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --duo --temperature-scaling --seed 3
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --duo --temperature-scaling --seed 4
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --duo --temperature-scaling --seed 5
