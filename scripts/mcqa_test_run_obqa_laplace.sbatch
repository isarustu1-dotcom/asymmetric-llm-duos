#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
source $HOME/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --laplace-lora --test-run --seed 1
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --laplace-lora --test-run --seed 2
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --laplace-lora --test-run --seed 3
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --laplace-lora --test-run --seed 4
python tools/train.py --config_base configs/obqa_qwen2_7b/lora_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/lora_obqa_qwen2_15b.yaml -w data/results --laplace-lora --test-run --seed 5