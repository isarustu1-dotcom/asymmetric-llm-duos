"""Asymmetric duo optimizer: combines base + sidekick logits via learned scales.

Two modes:
  - Unconstrained (default): L-BFGS-B, alpha >= 1e-6 and beta >= 1e-6 independently.
      p_duo = softmax(alpha * z_base + beta * z_sidekick)
  - Constrained: SLSQP, w_base + w_sidekick = 1, both in [0, 1].
      p_duo = softmax(w_base * z_base + w_sidekick * z_sidekick)

Both are optimised to minimise NLL on the validation set.
"""
import os.path as osp
from typing import Dict

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score

from .optimizer_helpers import (
    _logsumexp, _merge_dictionaries, _read_or_create_path,
    _sanity_check_on_probs, _save_val_test_results,
    _softmax_from_log, _sort_by_idx,
)
from .uncertainty_metrics import compute_uncertainty_metrics


# ---------------------------------------------------------------------------
# NLL losses
# ---------------------------------------------------------------------------

def _nll_unconstrained(scales: np.ndarray, logits_base: np.ndarray,
                       logits_sidekick: np.ndarray, y: np.ndarray) -> float:
    alpha, beta = scales
    log_p = alpha * logits_base + beta * logits_sidekick
    log_p_norm = log_p - _logsumexp(log_p, axis=1, keepdims=True)
    return float(-np.mean(log_p_norm[np.arange(len(y)), y]))


def _nll_constrained(weights: np.ndarray, logits_base: np.ndarray,
                     logits_sidekick: np.ndarray, y: np.ndarray) -> float:
    w_base, w_sidekick = weights
    log_p = w_base * logits_base + w_sidekick * logits_sidekick
    log_p_norm = log_p - _logsumexp(log_p, axis=1, keepdims=True)
    return float(-np.mean(log_p_norm[np.arange(len(y)), y]))


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------

def _fit_unconstrained(logits_base: np.ndarray, logits_sidekick: np.ndarray,
                       y_true: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """L-BFGS-B: alpha, beta >= 1e-6, no sum constraint."""
    x0 = np.array([1.0, 1.0], dtype=np.float64)
    bounds = [(1e-6, None), (1e-6, None)]

    cb = (lambda xk: print(f'  scales -> base={xk[0]:.4f}, sidekick={xk[1]:.4f}')) if verbose else None
    res = minimize(_nll_unconstrained, x0, args=(logits_base, logits_sidekick, y_true),
                   method='L-BFGS-B', bounds=bounds, callback=cb,
                   options={'disp': verbose})
    if not res.success:
        raise RuntimeError(f'Unconstrained duo optimisation failed: {res.message}')

    alpha, beta = res.x
    total = alpha + beta
    return {
        'scale_base': float(alpha),
        'scale_sidekick': float(beta),
        'weight_base': float(alpha / total),
        'weight_sidekick': float(beta / total),
        'temperature_base': float(1.0 / alpha),
        'temperature_sidekick': float(1.0 / beta),
    }


def _fit_constrained(logits_base: np.ndarray, logits_sidekick: np.ndarray,
                     y_true: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """SLSQP: w_base + w_sidekick = 1, both in [0, 1]."""
    x0 = np.array([0.5, 0.5], dtype=np.float64)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1.0}

    cb = (lambda xk: print(f'  weights -> base={xk[0]:.4f}, sidekick={xk[1]:.4f}')) if verbose else None
    res = minimize(_nll_constrained, x0, args=(logits_base, logits_sidekick, y_true),
                   method='SLSQP', bounds=bounds, constraints=constraints, callback=cb,
                   options={'disp': verbose})
    if not res.success:
        raise RuntimeError(f'Constrained duo optimisation failed: {res.message}')

    w_base, w_sidekick = res.x
    return {
        'scale_base': float(w_base),
        'scale_sidekick': float(w_sidekick),
        'weight_base': float(w_base),
        'weight_sidekick': float(w_sidekick),
        'temperature_base': float(1.0 / w_base) if w_base > 0 else float('inf'),
        'temperature_sidekick': float(1.0 / w_sidekick) if w_sidekick > 0 else float('inf'),
    }


# ---------------------------------------------------------------------------
# Metrics builder
# ---------------------------------------------------------------------------

def _build_metrics(scale_base: float, scale_sidekick: float,
                   logits_base: np.ndarray, logits_sidekick: np.ndarray,
                   y_true: np.ndarray):
    probs_base = _softmax_from_log(logits_base)
    _sanity_check_on_probs(probs_base, 'base')
    pred_base = probs_base.argmax(axis=1)

    probs_sidekick = _softmax_from_log(logits_sidekick)
    _sanity_check_on_probs(probs_sidekick, 'sidekick')
    pred_sidekick = probs_sidekick.argmax(axis=1)

    duo_logits = scale_base * logits_base + scale_sidekick * logits_sidekick
    probs_duo = _softmax_from_log(duo_logits)
    _sanity_check_on_probs(probs_duo, 'duo')
    pred_duo = probs_duo.argmax(axis=1)

    metrics = {
        'base_accuracy': accuracy_score(y_true, pred_base),
        'sidekick_accuracy': accuracy_score(y_true, pred_sidekick),
        'duo_accuracy': accuracy_score(y_true, pred_duo),
        'base_f1_macro': f1_score(y_true, pred_base, average='macro'),
        'sidekick_f1_macro': f1_score(y_true, pred_sidekick, average='macro'),
        'duo_f1_macro': f1_score(y_true, pred_duo, average='macro'),
    }
    metrics.update({f'base_{k}': v for k, v in compute_uncertainty_metrics(probs_base, y_true).items()})
    metrics.update({f'sidekick_{k}': v for k, v in compute_uncertainty_metrics(probs_sidekick, y_true).items()})
    metrics.update({f'duo_{k}': v for k, v in compute_uncertainty_metrics(probs_duo, y_true).items()})

    return metrics, duo_logits, probs_duo, pred_duo


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_weights(result_dir: str, saved_file_name: str, seed: int = 42,
                     constrained: bool = False, verbose: bool = True) -> None:
    """Fit duo scales on val logits, evaluate on test, and persist all results.

    Args:
        result_dir:       Experiment output directory (contains base/ and sidekick/).
        saved_file_name:  NPZ filename, e.g. 'arc_c.npz'.
        seed:             Which seed's predictions to load and store results under.
        constrained:      If True, use SLSQP with w_base+w_sidekick=1 constraint.
                          If False (default), use L-BFGS-B with independent scales.
        verbose:          Print optimiser iterations.
    """
    # --- load predictions -----------------------------------------------
    def _load(subdir, split):
        path = osp.join(result_dir, subdir, split, saved_file_name)
        data = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data['idx'])
        return data, order

    base_val, bv_ord = _load('base', 'val_preds')
    side_val, sv_ord = _load('sidekick', 'val_preds')
    base_tst, bt_ord = _load('base', 'test_preds')
    side_tst, st_ord = _load('sidekick', 'test_preds')

    # --- index alignment checks -----------------------------------------
    if not np.array_equal(_sort_by_idx(base_val['idx'], bv_ord),
                          _sort_by_idx(side_val['idx'], sv_ord)):
        raise ValueError('Val set indices of base and sidekick do not match.')
    if not np.array_equal(_sort_by_idx(base_tst['idx'], bt_ord),
                          _sort_by_idx(side_tst['idx'], st_ord)):
        raise ValueError('Test set indices of base and sidekick do not match.')

    bv_logits = _sort_by_idx(base_val['logits'], bv_ord)
    sv_logits = _sort_by_idx(side_val['logits'], sv_ord)
    bt_logits = _sort_by_idx(base_tst['logits'], bt_ord)
    st_logits = _sort_by_idx(side_tst['logits'], st_ord)
    y_val = _sort_by_idx(base_val['true_labels'], bv_ord)
    y_tst = _sort_by_idx(base_tst['true_labels'], bt_ord)

    # --- optimise -------------------------------------------------------
    fit_fn = _fit_constrained if constrained else _fit_unconstrained
    mode_label = 'constrained' if constrained else 'unconstrained'
    print(f'\n[duo_optimizer] Fitting {mode_label} duo for seed {seed} ...')
    params = fit_fn(bv_logits, sv_logits, y_val, verbose=verbose)
    alpha, beta = params['scale_base'], params['scale_sidekick']
    print(f'  scale_base={alpha:.4f}, scale_sidekick={beta:.4f}, '
          f'w_base={params["weight_base"]:.4f}, w_sidekick={params["weight_sidekick"]:.4f}')

    # --- build metrics --------------------------------------------------
    val_metrics, duo_val_logits, duo_val_probs, duo_val_pred = \
        _build_metrics(alpha, beta, bv_logits, sv_logits, y_val)
    tst_metrics, duo_tst_logits, duo_tst_probs, duo_tst_pred = \
        _build_metrics(alpha, beta, bt_logits, st_logits, y_tst)

    # --- persist --------------------------------------------------------
    metrics_path = osp.join(result_dir, 'metrics', saved_file_name)
    duo_val_path = osp.join(result_dir, 'duo', 'val_preds', saved_file_name)
    duo_tst_path = osp.join(result_dir, 'duo', 'test_preds', saved_file_name)

    ex_val = _read_or_create_path(duo_val_path)
    ex_tst = _read_or_create_path(duo_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_val['idx'], bv_ord),
                           _sort_by_idx(base_val['input'], bv_ord),
                           duo_val_logits.astype(np.float32), duo_val_probs, duo_val_pred,
                           _sort_by_idx(base_val['true_labels'], bv_ord),
                           ex_val, duo_val_path)

    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_tst['idx'], bt_ord),
                           _sort_by_idx(base_tst['input'], bt_ord),
                           duo_tst_logits.astype(np.float32), duo_tst_probs, duo_tst_pred,
                           _sort_by_idx(base_tst['true_labels'], bt_ord),
                           ex_tst, duo_tst_path)

    new_metrics = {
        'data': saved_file_name,
        'duo_mode': mode_label,
        'duo_weights': {'base': params['weight_base'], 'sidekick': params['weight_sidekick']},
        'duo_effective_scales': {'base': alpha, 'sidekick': beta},
        'duo_temperatures': {'base': params['temperature_base'], 'sidekick': params['temperature_sidekick']},
        'val': val_metrics,
        'test': tst_metrics,
    }

    seed_key = str(seed)
    existing_seed = ex_met.get(seed_key, {})
    if hasattr(existing_seed, 'item'):
        existing_seed = existing_seed.item()
    ex_met[seed_key] = _merge_dictionaries(existing_seed, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)
