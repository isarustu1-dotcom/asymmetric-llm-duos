import os.path as osp
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import transformers
from transformers.tokenization_utils_base import BatchEncoding
from copy import deepcopy
import peft

from bayesian_lora import (
    calculate_kronecker_factors,
    cholesky_decompose_small_factors,
    model_evidence,
    variance,
    stable_cholesky,
)
from bayesian_lora.main import jacobian_mean

from .optimizer_helpers import (
    _merge_dictionaries,
    _read_or_create_path,
    _sanity_check_on_probs,
    _save_val_test_results,
    _sort_by_idx,
    _logsumexp,
    _softmax_from_log
)
from .uncertainty_metrics import compute_uncertainty_metrics


def _find_factors(model, trainer, tokenizer, tok_cfg, target_ids, n_kfac, lr_threshold_kfac, device):
    # --- forward hook to find factors -----------------------------------------------
    
    def fwd_call(model, batch_prompts):
        prompts = batch_prompts['prompts']
        inputs = tokenizer(prompts, **tok_cfg).to(device)
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, target_ids]  # Get the last token logits
        return logits

    ## Get train loader
    train_loader = trainer.get_train_dataloader()

    ## Find Kronecker factors
    factors = calculate_kronecker_factors(
        model,            
        fwd_call,         
        train_loader,     
        n_kfac,
        lr_threshold_kfac,
        target_module_keywords = ["lora"]
    )

    ## Cholesky decompose small factors
    factors = cholesky_decompose_small_factors(
            factors, lr_threshold_kfac, device,torch.float32
        )
    return factors

def _find_LL(model, tokenizer, val_loader, tok_cfg, target_ids, device):
    LL = 0
    
    with torch.no_grad(), torch.inference_mode():
        for batch in val_loader:
            prompts = batch['prompts']
            labels = batch['labels'].to(device)
            inputs = tokenizer(prompts, **tok_cfg).to(device)
            logits = model(**inputs).logits[:, -1, target_ids]
            probs = logits.softmax(-1)
            LL += probs.gather(1, labels[:, None].to(device)).log().sum()

    return LL

def _find_optimal_prior_var(prior_var, trainer, model, tokenizer, tok_cfg, val_dataset, target_ids, factors, peft_r, n_kfac, device):
    s2 = torch.tensor(prior_var, requires_grad=True)
    opt = torch.optim.Adam([s2], lr=1e-2)
    val_loader = trainer.get_eval_dataloader(val_dataset)

    LL = _find_LL(model, tokenizer, val_loader, tok_cfg, target_ids, device)

    for _ in range(100):
        opt.zero_grad()
        loss = model_evidence(model, LL, factors, peft_r, n_kfac, s2).log()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([s2], 1.0)
        opt.step()
    
    return s2

def _get_raw_model_tokenizer(cfg):
    ## Get raw model
    model_cfg = deepcopy(cfg.model.model_cfg)
    model_cls_name = model_cfg.pop('type')
    model_cls = getattr(transformers, model_cls_name)
    model_torch_dtype = model_cfg.get('torch_dtype', None)

    model_laplace = model_cls.from_pretrained(cfg.model.model_name_or_path, **model_cfg)
    
    ## Get raw tokenizer
    tokenizer_cfg = deepcopy(cfg.model.tokenizer_cfg)
    tokenizer_cls = getattr(transformers, tokenizer_cfg.pop('type'))
    tokenizer_laplace = tokenizer_cls.from_pretrained(cfg.model.model_name_or_path, **tokenizer_cfg)
    tokenizer_special_tokens = {
        k: getattr(tokenizer_laplace, v.split('.')[-1]) if isinstance(v, str) and v.startswith('tokenizer') else v
        for k,
        v in cfg.model.special_tokens.items()
    }
    if len(tokenizer_special_tokens) > 0:
        tokenizer_laplace.add_special_tokens(tokenizer_special_tokens)
    if tokenizer_laplace.pad_token is None:
        tokenizer_laplace.pad_token = tokenizer_laplace.eos_token

    return model_laplace, tokenizer_laplace

def _get_laplace_predictions(model, tokenizer, tok_cfg, trainer, dataset, target_ids, factors, s2, peft_r, n_kfac, device):
    # --- output callback for Laplace predictions -----------------------------------------------
    def output_callback(outputs):
        """Post process model outputs.

        This function will be passed the results of model(**batch_inputs), and
        should return the relevant logits. For multiple-choice tasks, this is
        the class logits, but for full next-token prediction, this would just
        be all the logits.
        """
        # Get the last token for CausalLM
        logits = outputs.logits
        # Select the logits corresponding to our target classes
        target_logits = logits[:, -1, target_ids]
        return target_logits
    
    loader = trainer.get_eval_dataloader(dataset)

    # Release fragmented allocations left over from training before the
    # Jacobian loop, which is the most memory-intensive part of Laplace.
    torch.cuda.empty_cache()

    pred_logits_list = []
    pred_mu_list     = []
    pred_var_list    = []
    true_labels_list = []

    with torch.no_grad():
        for batch in loader:
            prompts = batch['prompts']
            labels = batch['labels'].to(device)

            batch_inputs = tokenizer(prompts, **tok_cfg).to(device)

            # Predict the output logit locations
            jacobian, f_mu = jacobian_mean(
                model, batch_inputs, output_callback=output_callback
            )
            pred_mu_list.append(f_mu.clone().cpu())

            # Predict the output logit variances.
            # Work-around for a bug in bayesian_lora: variance() calls
            # .squeeze() on every Jacobian tensor without specifying a dim,
            # so when batch_size == 1 the batch dimension is removed and the
            # subsequent shape assertion fails.  We duplicate the single
            # sample to batch_size=2 before passing to variance() so that
            # squeeze() leaves the batch dim intact, then slice [:1] back.
            actual_bs = batch_inputs.input_ids.size(0)
            if actual_bs == 1:
                var_inputs   = BatchEncoding(
                    {k: v.expand(2, *v.shape[1:]) for k, v in batch_inputs.items()}
                )
                var_jacobian = {k: v.expand(2, *v.shape[1:]) for k, v in jacobian.items()}
                f_var = variance(
                    var_inputs, var_jacobian, factors, s2, len(target_ids), peft_r, n_kfac, device
                )[:1]
            else:
                f_var = variance(
                    batch_inputs, jacobian, factors, s2, len(target_ids), peft_r, n_kfac, device
                )
            pred_var_list.append(f_var.clone().cpu())

            # Sample logits from N(f_mu, f_var) via reparameterisation
            # Cast to float32: f_mu may be bfloat16, L is float32 from stable_cholesky
            L = stable_cholesky(f_var)
            samples = 100_000
            f_mu_exp = f_mu.float().expand(samples, *f_mu.shape)
            L_exp    = L.float().expand(samples, *L.shape)
            eps      = torch.randn_like(f_mu_exp).unsqueeze(-1)  # float32 via randn_like
            logits   = f_mu_exp[..., None] + L_exp @ eps
            logits   = logits.squeeze(-1).mean(0)  # (batch, C) — MC average

            pred_logits_list.append(logits.cpu())
            true_labels_list.append(labels.cpu())

    all_logits = torch.cat(pred_logits_list, dim=0)          # (N, C)
    all_labels = torch.cat(true_labels_list, dim=0).numpy()

    kfac_avg_probs   = torch.softmax(all_logits, dim=-1).float().numpy().astype(np.float32)
    kfac_avg_logits  = np.log(np.clip(kfac_avg_probs, 1e-12, None)).astype(np.float32)
    kfac_true_labels = all_labels.astype(int)

    kfac_pred_mu  = torch.cat(pred_mu_list,  dim=0).float().numpy().astype(np.float32)  # (N, C)
    kfac_pred_var = torch.cat(pred_var_list, dim=0).float().numpy().astype(np.float32)

    return kfac_avg_logits, kfac_avg_probs, kfac_true_labels, kfac_pred_mu, kfac_pred_var


def _build_metrics(model_type_logits, laplace_logits, labels, model_type):

    probs_model_type = _softmax_from_log(model_type_logits)
    _sanity_check_on_probs(probs_model_type, model_type)
    pred_model_type = probs_model_type.argmax(axis=1)

    probs_laplace = _softmax_from_log(laplace_logits)
    _sanity_check_on_probs(probs_laplace, f'{model_type}_laplace')
    pred_laplace = probs_laplace.argmax(axis=1)

    prefix = f"{model_type}_laplace"
    metrics = {
        f"{model_type}_accuracy": float(accuracy_score(labels, pred_model_type)),
        f"{model_type}_f1_macro": float(f1_score(labels, pred_model_type, average="macro")),
        f"{prefix}_accuracy": float(accuracy_score(labels, pred_laplace)),
        f"{prefix}_f1_macro": float(f1_score(labels, pred_laplace, average="macro")),
    }

    metrics.update({
        f"{model_type}_{k}": v
        for k, v in compute_uncertainty_metrics(probs_model_type, labels).items()
    })
    metrics.update({
        f"{prefix}_{k}": v
        for k, v in compute_uncertainty_metrics(probs_laplace, labels).items()
    })
    return metrics, pred_laplace


def optimize_laplace(
    result_dir: str,
    saved_file_name: str,
    model_type: str,
    trainer,
    val_dataset,
    test_dataset,
    checkpoint_path: str,
    seed: int = 42,
    prior_var: Optional[float] = 0.1,
    n_kfac: int = 100,
    lr_threshold_kfac: int = 100,
) -> None:
    """Fit diagonal Laplace over LoRA params, evaluate on val/test, persist.

    Args
    ----
    result_dir      Experiment output directory (same root as train.py uses).
    saved_file_name NPZ filename, e.g. 'arc_e.npz'.
    model_type      'base' or 'sidekick' — controls the output subdirectory.
    trainer         The FTTrainer (or EvidentialTrainer) after training.
                    Provides: model, processing_class, cfg, args.device,
                    get_train_dataloader(), get_eval_dataloader().
    val_dataset     Used for prior precision tuning + val evaluation.
    test_dataset    Used for test evaluation.
    seed            Written as the key into the NPZ (same as other optimisers).
    prior_precision If None, tuned via grid search on val NLL (recommended).
    """
    model      = trainer.model
    device     = trainer.args.device
    target_ids = trainer.target_ids
    cfg        = trainer.cfg
    tokenizer  = trainer.processing_class
    tok_cfg    = cfg.tokenizer_run_cfg
    peft_r     = cfg.model.peft_cfg.r


    # --- load MAP predictions -----------------------------------------------
    def _load(split):
        path  = osp.join(result_dir, model_type, f"{split}_preds", saved_file_name)
        data  = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data["idx"])
        return data, order

    map_val, mv_ord = _load("val")
    map_val_logits = _sort_by_idx(map_val['logits'], mv_ord)
    map_tst, mt_ord = _load("test")
    map_tst_logits = _sort_by_idx(map_tst['logits'], mt_ord)

    print(f"\n[laplace] Snapshotting MAP weights for {model_type} ...")
    map_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    n_lora = sum(p.numel() for p in map_params.values())
    print(f"  LoRA parameter count: {n_lora:,}")

    ## Find Kronecker factors
    factors = _find_factors(model, trainer, tokenizer, tok_cfg, target_ids, n_kfac, lr_threshold_kfac, device)

    ## Find optimal prior
    s2 = _find_optimal_prior_var(
        prior_var, 
        trainer, 
        model, 
        tokenizer, 
        tok_cfg, 
        val_dataset, 
        target_ids, 
        factors, 
        peft_r, 
        n_kfac, 
        device
    )
    print(f"Optimized prior variance is: {s2.item()}")

    # model is no longer needed after this point — delete it and make space in cache
    # before loading model_laplace so we don't hold two full model copies on the GPU.
    del model
    torch.cuda.empty_cache()

    ## Get raw model and tokenizer
    model_laplace, tokenizer_laplace = _get_raw_model_tokenizer(cfg)

    ## Get peft model
    model_laplace = peft.PeftModel.from_pretrained(model_laplace, checkpoint_path, is_trainable=True)
    model_laplace = model_laplace.to(device)

    ## Freeze non-lora params
    non_lora = [p for n, p in model_laplace.named_parameters()
                if p.requires_grad and "lora" not in n.lower()]
    for p in non_lora:
        p.requires_grad_(False)

    

    print(f"[laplace] Generating val predictions...")
    val_logits, val_probs, val_labels, val_pred_mu, val_pred_var = _get_laplace_predictions(
        model_laplace, tokenizer_laplace, tok_cfg, trainer, val_dataset, target_ids, factors,
        s2, peft_r, n_kfac, device
    )

    print(f"[laplace] Generating test predictions...")
    tst_logits, tst_probs, tst_labels, tst_pred_mu, tst_pred_var = _get_laplace_predictions(
        model_laplace, tokenizer_laplace, tok_cfg, trainer, test_dataset, target_ids, factors,
        s2, peft_r, n_kfac, device
    )

    val_metrics, val_preds = _build_metrics(map_val_logits, 
                                            _sort_by_idx(val_logits, mv_ord), 
                                            _sort_by_idx(val_labels, mv_ord), 
                                            model_type)
    tst_metrics, tst_preds = _build_metrics(map_tst_logits, 
                                            _sort_by_idx(tst_logits, mt_ord), 
                                            _sort_by_idx(tst_labels, mt_ord), 
                                            model_type)
    print(f"  [laplace] val  acc={val_metrics[f'{model_type}_laplace_accuracy']:.4f}  "
          f"ece={val_metrics[f'{model_type}_laplace_ece']:.4f}  nll={val_metrics[f'{model_type}_laplace_nll']:.4f}")
    print(f"  [laplace] test acc={tst_metrics[f'{model_type}_laplace_accuracy']:.4f}  "
          f"ece={tst_metrics[f'{model_type}_laplace_ece']:.4f}  nll={tst_metrics[f'{model_type}_laplace_nll']:.4f}")


    metrics_path = osp.join(result_dir, "metrics", saved_file_name)
    lap_val_path = osp.join(result_dir, "laplace_lora", model_type, "val_preds",  saved_file_name)
    lap_tst_path = osp.join(result_dir, "laplace_lora", model_type, "test_preds", saved_file_name)

    ex_val = _read_or_create_path(lap_val_path)
    ex_tst = _read_or_create_path(lap_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(
        seed, saved_file_name,
        _sort_by_idx(map_val["idx"], mv_ord),
        _sort_by_idx(map_val["input"], mv_ord),
        _sort_by_idx(val_logits, mv_ord),
        _sort_by_idx(val_probs, mv_ord),
        _sort_by_idx(val_preds, mv_ord),
        _sort_by_idx(map_val["true_labels"], mv_ord),
        ex_val, lap_val_path,
    )
    _save_val_test_results(
        seed, saved_file_name,
        _sort_by_idx(map_tst["idx"], mt_ord),
        _sort_by_idx(map_tst["input"], mt_ord),
        _sort_by_idx(tst_logits, mt_ord),
        _sort_by_idx(tst_probs, mt_ord),
        _sort_by_idx(tst_preds, mt_ord),
        _sort_by_idx(map_tst["true_labels"], mt_ord),
        ex_tst, lap_tst_path,
    )

    new_metrics = {
        "data": saved_file_name,
        "laplace_optimized_prior_variance": s2.item(),
        "laplace_n_kfac": n_kfac,
        "laplace_peft_r": peft_r,
        "val": val_metrics,
        "test": tst_metrics,
    }
    seed_key = str(seed)
    existing = ex_met.get(seed_key, {})
    if hasattr(existing, "item"):
        existing = existing.item()
    ex_met[seed_key] = _merge_dictionaries(existing, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)
    print(f"[laplace] Results saved → {lap_val_path}")
    print(f"[laplace] Results saved → {lap_tst_path}")
    print(f"[laplace] Metrics saved → {metrics_path}")