"""Unified training entry point for Asymmetric LLM Duos.

Usage examples
--------------
# Plain LoRA, single model only
python tools/train.py --config_base configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    -w data/results --seed 1

# Plain LoRA + unconstrained duo + temperature scaling
python tools/train.py \\
    --config_base    configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml \\
    --config_sidekick configs/arc_c_qwen2_15b/lora_arc_c_qwen2_15b.yaml \\
    -w data/results --duo --temperature-scaling --seed 1

# IB-EDL + constrained duo
python tools/train.py \\
    --config_base    configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml \\
    --config_sidekick configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml \\
    -w data/results --ib-edl --duo --duo-constrained --seed 1
"""

import argparse
import os
import os.path as osp
from datetime import datetime

import mmengine
import torch
import wandb
from mmengine.runner.utils import set_random_seed
from transformers import TrainingArguments

from ib_edl import (
    DATASETS,
    ClassificationMetric,
    EvidentialTrainer,
    FTTrainer,
    UpdateRegWeightCallback,
    get_model_and_tokenizer,
    optimize_temperature_scaling,
    optimize_weights,
    plot_predictions,
    save_predictions,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train and evaluate Asymmetric LLM Duos.')

    # --- model configs (base is mandatory, sidekick is optional) ---
    p.add_argument('--config_base', required=True,
                   help='Config file for the base model (mandatory).')
    p.add_argument('--config_sidekick', default=None,
                   help='Config file for the sidekick model (optional).')

    # --- output ---
    p.add_argument('--work-dir', '-w', required=True,
                   help='Root directory on disk for all outputs (outside the repo).')

    # --- training control ---
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--skip-ft', '-s', action='store_true',
                   help='Skip fine-tuning and use existing LoRA checkpoints.')

    # --- method flags ---
    p.add_argument('--ib-edl', action='store_true',
                   help='Use IB-EDL (EvidentialTrainer) instead of plain LoRA (FTTrainer).')
    p.add_argument('--duo', action='store_true',
                   help='Run duo optimisation after training. Requires --config_sidekick.')
    p.add_argument('--duo-constrained', action='store_true',
                   help='Use constrained duo (SLSQP, w_base+w_side=1). '
                        'Default is unconstrained (L-BFGS-B).')
    p.add_argument('--temperature-scaling', action='store_true',
                   help='Run per-model temperature scaling after training.')

    # --- wandb ---
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--run-name', '-n', default=None)
    p.add_argument('--run-group', '-g', default=None)

    # --- config overrides ---
    p.add_argument('--cfg-options', '-o', nargs='+', action=mmengine.DictAction,
                   help='Override config values using key=value syntax.')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_args(args):
    if args.duo and args.config_sidekick is None:
        raise ValueError('--duo requires --config_sidekick to be specified.')
    if args.duo_constrained and not args.duo:
        raise ValueError('--duo-constrained only makes sense with --duo.')


def _validate_cfg(cfg, args, model_type):
    if args.skip_ft:
        assert cfg.model.peft_path is not None, \
            f'peft_path must be set in config when --skip-ft is used ({model_type}).'
    if args.ib_edl:
        assert cfg.get('edl_loss_cfg', None) is not None, \
            f'edl_loss_cfg missing in config — required for --ib-edl ({model_type}).'


# ---------------------------------------------------------------------------
# Work-dir naming: encode the full experiment configuration
# ---------------------------------------------------------------------------

def _build_work_dir(root: str, cfg_base_path: str, cfg_sidekick_path, args) -> str:
    """Construct a descriptive output path that encodes all key settings."""
    def _model_tag(cfg_path):
        # e.g. 'configs/arc_c_qwen2_7b/lora_arc_c_qwen2_7b.yaml'
        # -> folder name 'arc_c_qwen2_7b'
        # The model name always starts with a known prefix (e.g. 'qwen2').
        # Split at the first part that begins with the model family name.
        folder = cfg_path.split('/')[1]           # arc_c_qwen2_7b
        parts = folder.split('_')
        # Find the split index: first part that belongs to the model name.
        # Model names currently always contain 'qwen2' as a component.
        known_model_starts = ('qwen2', 'llama', 'mistral', 'gemma')
        split_idx = next(
            (i for i, p in enumerate(parts) if p in known_model_starts),
            2,  # fallback: old behaviour (parts[:2] = dataset)
        )
        dataset = '_'.join(parts[:split_idx])     # arc_c  or  csqa
        model = '_'.join(parts[split_idx:])       # qwen2_7b
        return dataset, model

    dataset, base_model = _model_tag(cfg_base_path)
    model_tag = base_model
    if cfg_sidekick_path:
        _, side_model = _model_tag(cfg_sidekick_path)
        model_tag = f'{base_model}_{side_model}'

    method_parts = ['ib_edl' if args.ib_edl else 'lora']
    if args.temperature_scaling:
        method_parts.append('tempscaling')
    if args.duo:
        method_parts.append('duo_constrained' if args.duo_constrained else 'duo_unconstrained')
    method = '_'.join(method_parts)

    return osp.join(root, dataset, model_tag, method)


# ---------------------------------------------------------------------------
# Single-model training loop
# ---------------------------------------------------------------------------

def _train_one_model(model_type: str, cfg, args, work_dir: str,
                     timestamp: str, device: torch.device):
    logger = setup_logger(
        name='ib-edl',
        filepath=osp.join(work_dir, model_type, f'{timestamp}.log'),
    )
    logger.info(f'Config for {model_type}:\n{"="*60}\n{cfg.pretty_text}\n{"="*60}')
    cfg.dump(osp.join(work_dir, model_type,
                      f'{osp.splitext(osp.basename(cfg.filename))[0]}_{timestamp}.yaml'))

    model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)

    train_set = DATASETS.build(cfg.data['train'], default_args=dict(tokenizer=tokenizer))
    val_set   = DATASETS.build(cfg.data['val'],   default_args=dict(tokenizer=tokenizer))
    test_set  = DATASETS.build(cfg.data['test'],  default_args=dict(tokenizer=tokenizer))

    train_tids = train_set.target_ids
    val_tids   = val_set.target_ids
    test_tids  = test_set.target_ids

    assert torch.all(train_tids == val_tids), \
        f'target_ids mismatch between train and val for {model_type}.'

    target_ids = train_tids if type(train_set) is type(test_set) else test_tids

    training_args = TrainingArguments(
        output_dir=osp.join(work_dir, model_type, 'checkpoints'),
        logging_dir=osp.join(work_dir, model_type),
        report_to='wandb' if not args.no_wandb else 'none',
        remove_unused_columns=False,
        seed=args.seed,
        run_name=timestamp if args.run_name is None else args.run_name,
        **cfg.train_cfg,
    )

    callbacks = []
    if args.ib_edl:
        reg_cfg = cfg.edl_loss_cfg['reg_weight_cfg']
        callbacks.append(UpdateRegWeightCallback(
            start_epoch=reg_cfg['start_epoch'],
            final_reg_weight=reg_cfg['final_reg_weight'],
        ))
        trainer_cls = EvidentialTrainer
    else:
        trainer_cls = FTTrainer

    trainer = trainer_cls(
        cfg=cfg,
        target_ids=target_ids,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=ClassificationMetric(
            num_classes=target_ids.shape[-1], **cfg.get('metric_cfg', {})),
        data_collator=train_set.get_collate_fn(),
        callbacks=callbacks if callbacks else None,
    )

    if not args.skip_ft:
        trainer.train()
        logger.info(f'Fine-tuning finished for {model_type}.')

    for split_name, split_set, prefix in [('val', val_set, 'val'),
                                           ('test', test_set, 'test')]:
        split_metrics = trainer.evaluate(eval_dataset=split_set,
                                         metric_key_prefix=prefix)
        for k, v in split_metrics.items():
            logger.info(f'[{model_type}] {k}: {v}')

    if cfg.process_preds.get('npz_file') is not None:
        val_idx  = val_set.get_data_indices()
        test_idx = test_set.get_data_indices()
        preds_val  = trainer.predict(val_set)
        preds_test = trainer.predict(test_set)

        npz = cfg.process_preds['npz_file']
        save_predictions(preds_val,
                         osp.join(work_dir, model_type, 'val_preds', npz),
                         logger=logger, seed=args.seed,
                         data_idx=val_idx, input_text=val_set.get_input_text())
        save_predictions(preds_test,
                         osp.join(work_dir, model_type, 'test_preds', npz),
                         logger=logger, seed=args.seed,
                         data_idx=test_idx, input_text=test_set.get_input_text())

    return cfg   # return last cfg for npz filename lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _validate_args(args)

    set_random_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}')
    timestamp = datetime.now().strftime('%m%d_%H%M_%S')

    has_sidekick = args.config_sidekick is not None

    work_dir = _build_work_dir(args.work_dir, args.config_base,
                               args.config_sidekick, args)
    mmengine.mkdir_or_exist(work_dir)

    print(f'\n[train] Seed={args.seed} | output -> {work_dir}\n')

    model_specs = [('base', args.config_base)]
    if has_sidekick:
        model_specs.append(('sidekick', args.config_sidekick))

    last_cfg = None
    for model_type, cfg_path in model_specs:
        cfg = mmengine.Config.fromfile(cfg_path)
        if args.cfg_options:
            cfg.merge_from_dict(args.cfg_options)
        _validate_cfg(cfg, args, model_type)

        mmengine.mkdir_or_exist(osp.join(work_dir, model_type))

        if not args.no_wandb:
            wandb.init(
                project=f'asym-duo-{model_type}',
                dir=osp.join(work_dir, model_type),
                name=args.run_name or timestamp,
                group=args.run_group,
            )
            wandb.config.update({f'{model_type}_config': cfg.to_dict()})

        last_cfg = _train_one_model(model_type, cfg, args, work_dir, timestamp, device)

        if not args.no_wandb:
            wandb.finish()

    # --- post-hoc optimisation ------------------------------------------
    npz_file = last_cfg.process_preds.get('npz_file') if last_cfg else None
    if npz_file is None:
        print('[train] No npz_file in config — skipping post-hoc optimisation.')
        return

    if args.temperature_scaling:
        print('\n[train] Running temperature scaling ...')
        optimize_temperature_scaling(
            work_dir, npz_file,
            seed=args.seed,
            has_sidekick=has_sidekick,
        )

    if args.duo:
        print('\n[train] Running duo optimisation ...')
        optimize_weights(
            work_dir, npz_file,
            seed=args.seed,
            constrained=args.duo_constrained,
        )


if __name__ == '__main__':
    main()
