# Asymmetric LLM Duos

Code for *Pushing the Calibration-Compute Frontier of LLMs via Asymmetric Duos*. We adapt the Asymmetric Duos framework of Zhou, Shelhamer, and Pleiss (NeurIPS 2025) from computer vision to large language models, evaluated in a multiple-choice question answering (MCQA) setting.

An Asymmetric Duo pairs a strong **base** model with a much smaller **sidekick** model and combines their predictions through a simple logit-space fusion whose global scaling parameters are fit on a validation set. Across six MCQA benchmarks, this yields best or second-best calibration (ECE, NLL) on five — RACE being the lone exception, where it trails the strongest baseline by a small margin — at a fraction of the inference FLOPs of stronger baselines such as BLoB, placing the method on, and in some cases improving, the calibration–compute Pareto frontier. The fusion also composes additively with existing calibration-aware methods (Laplace-LoRA, IB-EDL, BLoB).

This repository implements the LoRA fine-tuning, post-hoc calibration (temperature scaling, Laplace-LoRA), evidential training (IB-EDL), and asymmetric logit fusion used in the paper, organized around a single training entry point, a single aggregation entry point, and a Python package `asym_duos`.

## What This Repository Does

The core workflow is:

1. Fine-tune one model, or a base / sidekick pair, on a multiple-choice QA dataset.
2. Save validation and test predictions as logits.
3. Optionally fit a post-hoc calibration on the validation set:
   - per-model temperature scaling
   - Laplace-LoRA posterior approximation
   - asymmetric duo weights for combining base and sidekick logits
4. Aggregate metrics across seeds and optionally compute deep-ensemble baselines.

Datasets with ready-made experiment configs:

- `ARC-Easy` (`arc_e`) and `ARC-Challenge` (`arc_c`)
- `CommonsenseQA` (`csqa`)
- `OpenBookQA` (`obqa`)
- `SciQ` (`sciq`)
- `RACE` (`race`)

Base-model configs:

- `Qwen2-7B`, `Qwen2-1.5B`

## Repository Layout

```text
.
├── configs/                # Experiment and shared config files
│   ├── _base_/             # Reusable dataset/model/schedule fragments
│   └── <dataset>_<model>/  # Composed experiment configs
├── asym_duos/              # Python package
│   ├── datasets/
│   ├── models/
│   ├── train_eval/
│   └── utils/
├── scripts/                # Slurm launch templates
├── tools/
│   ├── train.py                     # Training + post-hoc calibration entry point
│   ├── average_metrics.py           # Aggregate metrics across seeds (+ deep ensembles)
│   ├── compute_flops.py             # MACs / FLOPs analysis for each method
│   ├── pick_best_ib_edl_beta.py     # Pick winning IB-EDL beta from a tuning sweep
│   └── postprocess_blob_asymduos.py # Post-hoc duo + ensemble for BLoB predictions
├── setup.cfg
└── pyproject.toml
```

## Main Concepts

### Plain LoRA

Standard cross-entropy LoRA fine-tuning of a causal LM, evaluated directly on the answer-token logits.

### IB-EDL

Replaces cross-entropy with an evidential objective: the model predicts positive Dirichlet evidence over answer classes, optionally perturbed through a variational information bottleneck. Built on the [IB-EDL reference implementation](https://github.com/sandylaker/ib-edl).

### Temperature Scaling

A post-hoc scalar `T` per model fitted on validation NLL and applied to val and test logits (Guo et al., ICML 2017).

### Laplace-LoRA

Post-hoc Laplace approximation over the LoRA parameters via the [`bayesian-lora`](https://github.com/MaximeRobeyns/bayesian_lora) package, following Yang et al. (ICLR 2024) and Daxberger et al. (NeurIPS 2021).

### Asymmetric Duo

A pair of independently fine-tuned models — `base` and `sidekick` — whose logits are fused on validation:

- **unconstrained**: independent positive scales for base and sidekick (L-BFGS-B).
- **constrained**: weights in `[0, 1]` that sum to `1` (SLSQP).

The fusion is over logits, not probabilities. The duo is "asymmetric" because the two models can differ in size or capacity.

## Installation

```bash
pip install -e .
```

Key dependencies are declared in `setup.cfg` (see `install_requires`).

## Quick Start

### 1. Single-model LoRA

```bash
python tools/train.py \
  --config-base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \
  --work-dir data/results \
  --seed 1
```

### 2. Two-model LoRA + Temperature Scaling + Duo

```bash
python tools/train.py \
  --config-base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \
  --config-sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --temperature-scaling \
  --duo \
  --seed 1
```

### 3. IB-EDL + Constrained Duo

```bash
python tools/train.py \
  --config-base configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml \
  --config-sidekick configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --ib-edl \
  --duo --duo-constrained \
  --seed 1
```

### 4. Aggregate across seeds

```bash
python tools/average_metrics.py \
  --config-base configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml \
  --config-sidekick configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml \
  --work-dir data/results \
  --dataset-name arc_c \
  --ib-edl --duo --duo-constrained
```

Add `--deep-ensemble --deep-ensemble-size N` to also report deep-ensemble metrics built from same-model seeds.

## Training Entry Point

`tools/train.py` loads configs, builds datasets and models, fine-tunes, evaluates val/test, saves predictions, and runs the requested post-hoc steps.

Important arguments:

| Flag | Purpose |
| --- | --- |
| `--config-base` | Config for the base model (required). |
| `--config-sidekick` | Config for the sidekick model. Required for `--duo`. |
| `--work-dir`, `-w` | Root directory for outputs. |
| `--seed`, `--gpu-id` | Standard. |
| `--skip-ft`, `-s` | Skip fine-tuning and reuse `model.peft_path` from config. |
| `--ib-edl` | Use the evidential trainer instead of plain LoRA. |
| `--temperature-scaling` | Run per-model temperature scaling after training. |
| `--laplace-lora` | Run Laplace-LoRA after training. |
| `--duo` / `--duo-constrained` | Run duo fusion (constrained or unconstrained). |
| `--beta-base`, `--beta-sidekick` | Override `vib.beta` per model (IB-EDL only). |
| `--no-wandb` | Disable Weights & Biases. |
| `--cfg-options key=value ...` | Override config fields from the CLI. |

## Other Tools

- `tools/average_metrics.py` — reconstructs the experiment output directory from the same flags used in training, computes mean ± std across seeds, writes tidy CSVs, and optionally builds deep ensembles. Falls back to rebuilding `metrics/<dataset>.npz` from raw prediction files if it is missing.
- `tools/compute_flops.py` — empirical MACs/FLOPs analysis for each method (LoRA / IB-EDL / Laplace / BLoB / duo).
- `tools/pick_best_ib_edl_beta.py` — given a sweep over `--beta-base` (or `--beta-sidekick`) values, picks the winner under a chosen target metric.
- `tools/postprocess_blob_asymduos.py` — reads BLoB predictions saved under `blob/{model_type}/` and runs the duo + ensemble pipeline on top of them.

## Output Structure

Outputs are grouped under `--work-dir` as `<dataset>/<model-tag>/<method>/`, e.g.:

```text
data/results/arc_c/qwen2_7b/lora/
data/results/arc_c/qwen2_7b_qwen2_15b/lora_tempscaling_duo_unconstrained/
data/results/arc_c/qwen2_7b_qwen2_15b/ib_edl_duo_constrained/
```

Each experiment directory contains:

```text
base/         { checkpoints/, val_preds/, test_preds/ }
sidekick/     { checkpoints/, val_preds/, test_preds/ }   # if applicable
temperature_scaling/  { base/, sidekick/ }                # if --temperature-scaling
duo/          { val_preds/, test_preds/ }                 # if --duo
metrics/
overall_metrics_summary/
```

## Config System

Configs are composed from reusable fragments under `configs/_base_/`:

- a dataset config (`arc_c.yaml`, `csqa.yaml`, `obqa.yaml`, `sciq.yaml`, `race.yaml`)
- a model config (`qwen2_7b.yaml`, `qwen2_15b.yaml`; `qwen3_06b.yaml`, `qwen3_1_7B.yaml`, `qwen3_8b.yaml` are also available as fragments, though only the Qwen2 ones have composed experiment configs)
- a misc config (prediction saving, plotting)
- a training schedule (`non_edl_schedule.yaml` or `edl_schedule.yaml`)

Experiment configs add method-specific fields such as `process_preds.npz_file`, `edl_loss_cfg`, and `vib`.

> Note: `qwen2_15b.yaml` points to `Qwen/Qwen2-1.5B` — the file name reads "15b" but the model is 1.5B.

## Datasets

Each dataset wrapper (under `asym_duos/datasets/`):

- loads a Hugging Face dataset
- formats a prompt with question and choices (preamble defined per class)
- maps class labels to single answer-token IDs
- returns prompt/label pairs compatible with `transformers.Trainer`

Wrappers: `ARCDataset`, `CSQADataset`, `OBQADataset`, `SciQDataset`, `RaceDataset`, `MMLUDataset`.

## Practical Notes

- `--duo` requires `--config-sidekick`.
- `--duo-constrained` only makes sense with `--duo`.
- `--skip-ft` requires `model.peft_path` in the config.
- W&B is on by default; pass `--no-wandb` to disable.
- Prediction saving requires `process_preds.npz_file` to be set in the config.
- The code uses `cuda:<gpu-id>` directly.

## Suggested Reading Order

1. `tools/train.py`
2. one experiment config under `configs/<dataset>_<model>/`
3. `asym_duos/train_eval/ft_trainer.py`
4. `asym_duos/utils/duo_optimizer.py`
5. `tools/average_metrics.py`

## Citations and Acknowledgements

This repository builds on, and reuses code from, several earlier works:

- **IB-EDL** — the trainers, losses, and MCQA prompt format are adapted from <https://github.com/sandylaker/ib-edl>.  
  *Calibrating LLMs with Information-Theoretic Evidential Deep Learning* (arXiv:2502.06351).
- **Asymmetric Duos** — Zhou, Shelhamer, Pleiss, *Asymmetric Duos: Sidekicks Improve Uncertainty*, NeurIPS 2025 (arXiv:2505.18636).
- **Temperature Scaling** — Guo, Pleiss, Sun, Weinberger, *On Calibration of Modern Neural Networks*, ICML 2017.
- **Laplace-LoRA** — Yang, Robeyns et al., *Bayesian Low-Rank Adaptation for Large Language Models*, ICLR 2024 (via the [`bayesian-lora`](https://github.com/MaximeRobeyns/bayesian_lora) package).

