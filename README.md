# Asymmetric LLM Duos

Code for calibrating multiple-choice QA language models with LoRA fine-tuning, Information-Theoretic Evidential Deep Learning (IB-EDL), post-hoc temperature scaling, and asymmetric two-model logit fusion.

This repository applies the asymmetric duo idea to LLMs. The original asymmetric duo formulation used here comes from the computer vision work *Asymmetric Duos: Sidekicks Improve Uncertainty*, and this codebase adapts that idea to language-model calibration and multiple-choice QA.

The repository is organized around a single training entrypoint, a single final-analysis entrypoint, and a lightweight Python package (`ib_edl`) that contains the reusable dataset, model, training, and calibration code.

## What This Repository Does

The core workflow is:

1. Fine-tune one model, or a base/sidekick pair of models, on a multiple-choice QA dataset.
2. Save validation and test predictions as logits.
3. Optionally fit post-hoc calibration on the validation set:
   - per-model temperature scaling
   - asymmetric duo weights for combining base and sidekick logits
4. Aggregate metrics across multiple seeds and optionally compute deep-ensemble results.

The repository currently includes experiment configs for:

- `ARC-Challenge` (`arc_c`)
- `CommonsenseQA` (`csqa`)
- `OpenBookQA` (`obqa`)

Base model configs are provided for:

- `Qwen2-7B`
- `Qwen2-1.5B` (stored under `qwen2_15b.yaml`)
- `Qwen3-0.6B`
- `Qwen3-1.7B`
- `Qwen3-8B`

## Repository Layout

```text
.
├── configs/                # Experiment and shared config files
│   ├── _base_/             # Reusable dataset/model/schedule config fragments
│   ├── arc_c_qwen2_7b/     # Dataset+model experiment configs
│   ├── arc_c_qwen2_15b/
│   ├── csqa_qwen2_7b/
│   ├── csqa_qwen2_15b/
│   ├── obqa_qwen2_7b/
│   └── obqa_qwen2_15b/
├── ib_edl/                 # Python package with datasets, trainers, utils
│   ├── datasets/
│   ├── models/
│   ├── train_eval/
│   └── utils/
├── scripts/                # Example Slurm launch scripts
├── tools/
│   ├── train.py            # Main training / calibration entrypoint
│   └── average_metrics.py  # Final aggregation across seeds
├── setup.cfg               # Package metadata and dependencies
└── pyproject.toml
```

## Main Concepts

### Plain LoRA

The simplest setup fine-tunes a single causal LM with PEFT/LoRA and evaluates it directly.

### IB-EDL

`IB-EDL` replaces plain cross-entropy training with an evidential objective. The model predicts positive evidence for answer classes, which is converted into Dirichlet-style uncertainty estimates. The implementation also supports a variational information bottleneck (VIB) term.

### Temperature Scaling

Temperature scaling is a post-hoc calibration baseline. After training, the code fits one scalar temperature per model on the validation logits and applies it to validation and test predictions.

### Asymmetric Duo

An asymmetric duo is a pair of independently fine-tuned models:

- `base`
- `sidekick`

Their logits are combined after training by learning validation-set fusion weights. Two modes are supported:

- unconstrained: independent positive scales for base and sidekick
- constrained: weights in `[0, 1]` that sum to `1`

The duo is "asymmetric" because the two models can differ in size or capacity.

In this repository, that idea is applied to LLMs. The motivation comes from the computer vision asymmetric-duo setup of Zhou, Shelhamer, and Pleiss, but the implementation here is tailored to causal language models, token-level answer selection, and calibration-focused evaluation.

## Installation

Create an environment with Python and PyTorch that supports your GPU setup, then install the package in editable mode:

```bash
pip install -e .
```

Key dependencies are declared in `setup.cfg` and include:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `mmengine`
- `scipy`
- `wandb`

## Quick Start

### 1. Single-model LoRA

```bash
python tools/train.py \
  --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \
  --work-dir data/results \
  --seed 1
```

### 2. Two-model LoRA + Temperature Scaling + Duo

```bash
python tools/train.py \
  --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \
  --config_sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --temperature-scaling \
  --duo \
  --seed 1
```

### 3. IB-EDL + Constrained Duo

```bash
python tools/train.py \
  --config_base configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml \
  --config_sidekick configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --ib-edl \
  --duo \
  --duo-constrained \
  --seed 1
```

### 4. Aggregate Final Results Across Seeds

Run the training command for several seeds first, then summarize:

```bash
python tools/average_metrics.py \
  --config_base configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml \
  --config_sidekick configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --dataset_name arc_c \
  --ib-edl \
  --duo \
  --duo-constrained
```

To include deep-ensemble results:

```bash
python tools/average_metrics.py \
  --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \
  --config_sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --dataset_name arc_c \
  --temperature-scaling \
  --duo \
  --deep-ensemble \
  --deep-ensemble-size 2
```

## Training Entry Point

`tools/train.py` is the main script for:

- loading configs
- building datasets and models
- running fine-tuning
- evaluating on validation and test splits
- saving predictions
- optionally running post-hoc temperature scaling
- optionally running duo optimization

Important arguments:

- `--config_base`: config for the mandatory base model
- `--config_sidekick`: optional config for the sidekick model
- `--work-dir`: root directory for outputs
- `--seed`: random seed
- `--gpu-id`: CUDA device index
- `--skip-ft`: skip fine-tuning and reuse existing PEFT checkpoints from config
- `--ib-edl`: use the evidential trainer instead of plain fine-tuning
- `--duo`: run duo optimization after prediction files are saved
- `--duo-constrained`: use constrained fusion instead of unconstrained fusion
- `--temperature-scaling`: run per-model post-hoc temperature scaling
- `--no-wandb`: disable Weights & Biases logging
- `--cfg-options key=value ...`: override config fields from the CLI

## Result Aggregation

`tools/average_metrics.py` reconstructs the experiment output directory from the same config and method flags used in training, then:

- loads per-seed metrics
- computes mean and standard deviation
- writes tidy CSV summaries
- optionally computes deep ensembles from pairs or groups of same-model seeds

If no `metrics/<dataset>.npz` file exists yet, the script can rebuild metrics directly from raw prediction files.

## Output Structure

Outputs are grouped under the `--work-dir` root using:

```text
<work-dir>/<dataset>/<model-tag>/<method>/
```

Examples:

```text
data/results/arc_c/qwen2_7b/lora/
data/results/arc_c/qwen2_7b_qwen2_15b/lora_tempscaling_duo_unconstrained/
data/results/arc_c/qwen2_7b_qwen2_15b/ib_edl_duo_constrained/
```

A typical experiment directory contains:

```text
base/
  checkpoints/
  val_preds/
  test_preds/
sidekick/
  checkpoints/
  val_preds/
  test_preds/
temperature_scaling/
  base/
  sidekick/
duo/
  val_preds/
  test_preds/
metrics/
overall_metrics_summary/
```

## Config System

Configs are built from reusable pieces in `configs/_base_`.

Typical composition:

- dataset config, for example `configs/_base_/arc_c.yaml`
- model config, for example `configs/_base_/qwen2_7b.yaml`
- misc config, for prediction saving and plotting
- training schedule, either:
  - `non_edl_schedule.yaml`
  - `edl_schedule.yaml`

Experiment configs then add method-specific settings, for example:

- `process_preds.npz_file`
- `edl_loss_cfg`
- `vib`

### Current Model Base Configs

- `configs/_base_/qwen2_7b.yaml`
- `configs/_base_/qwen2_15b.yaml`
- `configs/_base_/qwen3_06b.yaml`
- `configs/_base_/qwen3_1_7B.yaml`
- `configs/_base_/qwen3_8b.yaml`

The repository now includes Qwen3 base configs and matching dataset-specific experiment configs following the same structure as the Qwen2 runs.

## How Datasets Are Formatted

The code treats multiple-choice QA as next-token classification over answer labels like `A`, `B`, `C`, etc.

Each dataset wrapper:

- loads a Hugging Face dataset
- formats a prompt containing the question and choices
- maps class labels to token IDs
- returns prompt/label pairs compatible with `transformers.Trainer`

Implemented dataset wrappers include:

- `ARCDataset`
- `CSQADataset`
- `OBQADataset`
- `SciQDataset`
- `RaceDataset`
- `MMLUDataset`

Only some of these currently have ready-made experiment configs.

## How IB-EDL Is Implemented

The package `ib_edl` contains the reusable method implementation:

- `ib_edl/models/`
  - model and tokenizer loading
  - LoRA / PEFT setup
- `ib_edl/datasets/`
  - dataset wrappers and prompt formatting
- `ib_edl/train_eval/`
  - `FTTrainer` for plain cross-entropy fine-tuning
  - `EvidentialTrainer` for evidential objectives
  - evidential losses and evaluation metrics
- `ib_edl/utils/`
  - prediction saving
  - uncertainty metrics
  - temperature scaling
  - duo optimization

The `EvidentialTrainer`:

- converts logits to positive evidence
- computes evidential loss terms
- optionally applies a variational information bottleneck
- returns logits derived from normalized Dirichlet parameters for evaluation

## Temperature Scaling Details

Temperature scaling is implemented in `ib_edl/utils/temperature_scaling.py`.

It:

- loads saved validation and test logits
- fits a single scalar temperature on validation NLL
- applies the learned scale to validation and test logits
- computes accuracy, F1, NLL, Brier score, ECE, LPPD, and mean uncertainty
- saves calibrated predictions and updates the shared metrics file

This makes it easy to compare:

- raw base model
- raw sidekick model
- temperature-scaled base model
- temperature-scaled sidekick model

## Asymmetric Duo Details

Duo optimization is implemented in `ib_edl/utils/duo_optimizer.py`.

It:

- loads saved base and sidekick predictions
- checks that validation and test example indices match across both models
- optimizes fusion weights on validation logits
- evaluates the fused model on validation and test data
- saves combined logits and uncertainty metrics

The fused probability distribution is computed from a weighted combination of base and sidekick logits, not from probability averaging.

## Slurm Scripts

The `scripts/` directory contains example Slurm launchers for:

- ARC-C runs
- CSQA runs
- OBQA runs
- metrics aggregation

These scripts are useful as templates, but they should be reviewed before reuse because they reflect a specific cluster environment and a few paths/config combinations may need updating.

## Practical Notes

- `--duo` requires `--config_sidekick`.
- `--duo-constrained` only makes sense together with `--duo`.
- `--skip-ft` requires `model.peft_path` to be set in the config.
- Weights & Biases logging is enabled by default unless `--no-wandb` is passed.
- Prediction saving depends on `process_preds.npz_file` being set in the config.
- The code expects CUDA and uses `cuda:<gpu-id>` directly.

## Extending The Repository

### Add a New Experiment Config

Create a new config folder that combines:

- a dataset config from `configs/_base_`
- a model config from `configs/_base_`
- `misc.yaml`
- either `non_edl_schedule.yaml` or `edl_schedule.yaml`

Then add method-specific fields such as:

- `process_preds.npz_file`
- `edl_loss_cfg`
- `vib`

### Add a New Dataset

1. Implement a dataset wrapper under `ib_edl/datasets/`.
2. Register it in the dataset registry.
3. Add a base dataset config under `configs/_base_/`.
4. Create experiment configs that compose it with a model config.

### Add a New Model

1. Create a new base model config in `configs/_base_/`.
2. Point `model_name_or_path` to the Hugging Face model.
3. Adjust LoRA target modules if needed for the architecture.
4. Create dataset-specific experiment configs that reference it.

## Known Caveats

- There is no generated documentation site yet; the source of truth is the code and configs.
- `configs/_base_/qwen2_15b.yaml` currently points to `Qwen/Qwen2-1.5B`, so the filename reads like `15b` while the underlying model is `1.5B`.
- The Qwen3 configs use the same experiment structure as the Qwen2 configs, so any new model family should follow that same pattern to work cleanly with the training and aggregation scripts.

## Reference

The asymmetric duo idea used in this repository is adapted from the following computer vision paper and applied here to LLM calibration:

- Tim G. Zhou, Evan Shelhamer, and Geoff Pleiss. *Asymmetric Duos: Sidekicks Improve Uncertainty*. arXiv:2505.18636, 2025. NeurIPS 2025. https://arxiv.org/abs/2505.18636

DOI: https://doi.org/10.48550/arXiv.2505.18636

## Suggested Reading Order

If you are new to the repo, this order works well:

1. `tools/train.py`
2. `tools/average_metrics.py`
3. one experiment config, for example `configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml`
4. the matching base configs in `configs/_base_/`
5. `ib_edl/train_eval/evidential_trainer.py`
6. `ib_edl/utils/temperature_scaling.py`
7. `ib_edl/utils/duo_optimizer.py`

## Citation

If you are using this repository in academic work, cite both:

- Zhou, Shelhamer, and Pleiss, *Asymmetric Duos: Sidekicks Improve Uncertainty* (the original computer vision asymmetric-duo paper)
- this LLM adaptation / project repository
