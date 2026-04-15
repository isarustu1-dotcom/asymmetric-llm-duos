import os.path as osp
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .optimizer_helpers import (
    _merge_dictionaries,
    _read_or_create_path,
    _sanity_check_on_probs,
    _save_val_test_results,
    _sort_by_idx,
    _logsumexp,
)
from .uncertainty_metrics import compute_uncertainty_metrics


def _compute_diagonal_fisher(
    model: torch.nn.Module,
    trainer,
    target_ids: torch.LongTensor,
    device: torch.device,
    tokenizer,
    tokenizer_run_cfg: dict,
) -> dict:
    """Accumulate the diagonal empirical Fisher over LoRA parameters.

    F_ii ≈ (1/N) Σ_n (d log p(y_n|x_n,θ) / d θ_i)²

    Uses the true observed labels (empirical Fisher), which is standard for
    post-hoc Laplace over classification models.

    Returns
    -------
    dict  name → Tensor of same shape as the parameter, on *device*.
    """
    model.eval()
    diag_F = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad  # LoRA adapters only
    }
    n_total = 0

    # get_train_dataloader() uses the dataset's collate_fn + TrainingArguments
    # batch size — identical to how the trainer itself iterates during training.
    train_dl = trainer.get_train_dataloader()

    for batch in train_dl:
        # Batch format matches FTTrainer.compute_loss: {'prompts': [...], 'labels': Tensor}
        prompts = batch["prompts"]
        labels  = batch["labels"].to(device)
        B       = labels.size(0)

        inputs = tokenizer(prompts, **tokenizer_run_cfg).to(device)

        model.zero_grad()
        logits = model(**inputs).logits[:, -1, target_ids]  # (B, C)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                diag_F[name] += param.grad.data ** 2 * B  # accumulate before normalising

        n_total += B

    # Normalise → per-sample expectation
    for name in diag_F:
        diag_F[name] /= n_total

    return diag_F

@torch.no_grad()
def _laplace_predict(
    model: torch.nn.Module,
    trainer,
    dataset,                  # val_set or test_set
    target_ids: torch.LongTensor,
    device: torch.device,
    tokenizer,
    tokenizer_run_cfg: dict,
    map_params: dict,
    diag_F: dict,
    prior_precision: float,
    n_samples: int,
    eval_type = 'val'
) -> tuple:
    """MC-average softmax probs over *n_samples* posterior weight draws.

    Posterior variance per parameter:
        σ²_i = 1 / (F_ii + prior_precision)

    Returns
    -------
    avg_logits : np.ndarray  (N, C) float32  — log of averaged probs (pseudo-logits)
    avg_probs  : np.ndarray  (N, C) float32
    true_labels: np.ndarray  (N,)   int
    """
    model.eval()

    # Pre-compute posterior std — same shape as each LoRA parameter
    post_std = {
        name: (1.0 / (diag_F[name] + prior_precision)).sqrt()
        for name in map_params
    }

    # get_eval_dataloader respects eval batch size + collate_fn from TrainingArguments
    if eval_type == 'val':
        dl = trainer.get_eval_dataloader(dataset)
    elif eval_type == 'test':
        dl = trainer.get_test_dataloader(dataset)
    else:
        raise ValueError(f"Invalid eval_type: {eval_type}")

    # Accumulate one list of (N, C) tensors per sample draw
    all_sample_probs = []  # length = n_samples, each entry (N, C)
    true_labels_collected = False
    true_labels_all = []

    for s in range(n_samples):
        # Perturb weights: θ_s ~ N(θ_MAP, diag(σ²))
        for name, mu in map_params.items():
            noise = torch.randn_like(mu) * post_std[name]
            # Access parameter by name and overwrite in-place
            param_ref = dict(model.named_parameters())[name]
            param_ref.data.copy_(mu + noise)

        sample_probs = []
        for batch in dl:
            prompts = batch["prompts"]
            inputs  = tokenizer(prompts, **tokenizer_run_cfg).to(device)
            logits  = model(**inputs).logits[:, -1, target_ids]  # (B, C)
            sample_probs.append(F.softmax(logits, dim=-1).cpu())

            if not true_labels_collected:
                true_labels_all.append(batch["labels"].numpy())

        true_labels_collected = True
        all_sample_probs.append(torch.cat(sample_probs, dim=0))  # (N, C)

    # Restore MAP weights so the model is unchanged after this call
    for name, mu in map_params.items():
        dict(model.named_parameters())[name].data.copy_(mu)

    # Average over posterior samples  →  (N, C)
    avg_probs  = torch.stack(all_sample_probs, dim=0).mean(dim=0)  # (S, N, C) → (N, C)
    avg_logits = torch.log(avg_probs.clamp(min=1e-12))             # pseudo-logits

    true_labels = np.concatenate(true_labels_all, axis=0)
    return (
        avg_logits.numpy().astype(np.float32),
        avg_probs.numpy().astype(np.float32),
        true_labels,
    )

def _tune_prior_precision(
    model, trainer, val_dataset, target_ids, device,
    tokenizer, tokenizer_run_cfg,
    map_params, diag_F,
    n_samples: int,
    grid=None,
) -> float:
    """Return the prior_precision with lowest NLL on the validation set."""
    if grid is None:
        grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    best_prec, best_nll = grid[0], float("inf")
    for prec in grid:
        logits, probs, labels = _laplace_predict(
            model, trainer, val_dataset, target_ids, device,
            tokenizer, tokenizer_run_cfg,
            map_params, diag_F, prec,
            n_samples=min(n_samples, 5),  # cheaper grid: fewer samples
        )
        log_p_norm = logits - _logsumexp(logits, axis=1, keepdims=True)
        nll = float(-np.mean(log_p_norm[np.arange(len(labels)), labels]))
        print(f"  [laplace] prior_precision={prec:.4g}  val_NLL={nll:.6f}")
        if nll < best_nll:
            best_nll  = nll
            best_prec = prec

    print(f"  [laplace] Best prior_precision={best_prec:.4g}  (val_NLL={best_nll:.6f})")
    return best_prec

def _build_metrics(probs, labels, model_type):
    prefix = f"{model_type}_laplace"
    _sanity_check_on_probs(probs, prefix)
    preds = probs.argmax(axis=1)
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(labels, preds)),
        f"{prefix}_f1_macro": float(f1_score(labels, preds, average="macro")),
    }
    metrics.update({
        f"{prefix}_{k}": v
        for k, v in compute_uncertainty_metrics(probs, labels).items()
    })
    return metrics, preds

def optimize_laplace(
    result_dir: str,
    saved_file_name: str,
    model_type: str,
    trainer,
    val_dataset,
    test_dataset,
    seed: int = 42,
    prior_precision: Optional[float] = None,
    n_samples: int = 20,
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
    n_samples       Number of MC weight draws at inference.
    """
    model      = trainer.model
    device     = trainer.args.device
    target_ids = trainer.target_ids
    tokenizer  = trainer.processing_class
    tok_cfg    = trainer.cfg.tokenizer_run_cfg

    print(f"\n[laplace] Snapshotting MAP weights for {model_type} ...")
    map_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    n_lora = sum(p.numel() for p in map_params.values())
    print(f"  LoRA parameter count: {n_lora:,}")

    print(f"[laplace] Computing diagonal Fisher (train set) ...")
    diag_F = _compute_diagonal_fisher(
        model, trainer, target_ids, device, tokenizer, tok_cfg,
    )

    if prior_precision is None:
        print(f"[laplace] Tuning prior_precision on val ...")
        prior_precision = _tune_prior_precision(
            model, trainer, val_dataset, target_ids, device,
            tokenizer, tok_cfg, map_params, diag_F, n_samples,
        )

    print(f"[laplace] Generating val predictions  (n_samples={n_samples}) ...")
    val_logits, val_probs, val_labels = _laplace_predict(
        model, trainer, val_dataset, target_ids, device,
        tokenizer, tok_cfg, map_params, diag_F, prior_precision, n_samples
    )

    print(f"[laplace] Generating test predictions (n_samples={n_samples}) ...")
    tst_logits, tst_probs, tst_labels = _laplace_predict(
        model, trainer, test_dataset, target_ids, device,
        tokenizer, tok_cfg, map_params, diag_F, prior_precision, n_samples, eval_type = 'test'
    )

    val_metrics, val_preds = _build_metrics(val_probs, val_labels, model_type)
    tst_metrics, tst_preds = _build_metrics(tst_probs, tst_labels, model_type)
    print(f"  [laplace] val  acc={val_metrics['laplace_accuracy']:.4f}  "
          f"ece={val_metrics['laplace_ece']:.4f}  nll={val_metrics['laplace_nll']:.4f}")
    print(f"  [laplace] test acc={tst_metrics['laplace_accuracy']:.4f}  "
          f"ece={tst_metrics['laplace_ece']:.4f}  nll={tst_metrics['laplace_nll']:.4f}")


    metrics_path = osp.join(result_dir, "metrics", saved_file_name)
    lap_val_path = osp.join(result_dir, "laplace_lora", model_type, "val_preds",  saved_file_name)
    lap_tst_path = osp.join(result_dir, "laplace_lora", model_type, "test_preds", saved_file_name)

    def _load_map_preds(split):
        path  = osp.join(result_dir, model_type, f"{split}_preds", saved_file_name)
        data  = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data["idx"])
        return data, order

    map_val, mv_ord = _load_map_preds("val")
    map_tst, mt_ord = _load_map_preds("test")

    ex_val = _read_or_create_path(lap_val_path)
    ex_tst = _read_or_create_path(lap_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(
        seed, saved_file_name,
        _sort_by_idx(map_val["idx"], mv_ord),
        _sort_by_idx(map_val["input"], mv_ord),
        _sort_by_idx(val_logits, mv_ord).astype(np.float32),
        _sort_by_idx(val_probs, mv_ord),
        _sort_by_idx(val_preds, mv_ord),
        _sort_by_idx(map_val["true_labels"], mv_ord),
        ex_val, lap_val_path,
    )
    _save_val_test_results(
        seed, saved_file_name,
        _sort_by_idx(map_tst["idx"], mt_ord),
        _sort_by_idx(map_tst["input"], mt_ord),
        _sort_by_idx(tst_logits, mt_ord).astype(np.float32),
        _sort_by_idx(tst_probs, mt_ord),
        _sort_by_idx(tst_preds, mt_ord),
        _sort_by_idx(map_tst["true_labels"], mt_ord),
        ex_tst, lap_tst_path,
    )

    new_metrics = {
        "data": saved_file_name,
        "laplace_prior_precision": prior_precision,
        "laplace_n_samples": n_samples,
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