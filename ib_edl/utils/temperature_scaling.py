"""Per-model temperature scaling baseline.

Optimises a single scalar T > 0 per model so that softmax((1/T) * logits)
minimises NLL on the validation set, then applies it to both val and test.
"""
import os.path as osp

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
# NLL loss
# ---------------------------------------------------------------------------

def _nll_single(temperature: np.ndarray, logits: np.ndarray, y: np.ndarray) -> float:
    T = float(temperature[0])
    log_p = (1.0 / T) * logits
    log_p_norm = log_p - _logsumexp(log_p, axis=1, keepdims=True)
    return float(-np.mean(log_p_norm[np.arange(len(y)), y]))


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def _fit_temperature(logits: np.ndarray, y_true: np.ndarray,
                     verbose: bool = True, model_label: str = '') -> dict:
    """Fit a single temperature scalar via L-BFGS-B."""
    x0 = np.array([1.0], dtype=np.float64)
    bounds = [(1e-6, None)]
    cb = (lambda xk: print(f'  T_{model_label}={xk[0]:.4f}')) if verbose else None
    res = minimize(_nll_single, x0, args=(logits, y_true),
                   method='L-BFGS-B', bounds=bounds, callback=cb,
                   options={'disp': verbose})
    if not res.success:
        raise RuntimeError(f'Temperature scaling failed for {model_label}: {res.message}')
    T = float(res.x[0])
    return {'temperature': T, 'scale': 1.0 / T}


# ---------------------------------------------------------------------------
# Metrics builder
# ---------------------------------------------------------------------------

def _build_metrics(scale: float, logits: np.ndarray, y_true: np.ndarray,
                   model_type: str = 'base'):
    new_logits = scale * logits
    probs = _softmax_from_log(new_logits)
    _sanity_check_on_probs(probs, f'{model_type}_tempscaled')
    preds = probs.argmax(axis=1)

    prefix = f'{model_type}_ts'
    metrics = {
        f'{prefix}_accuracy': float(accuracy_score(y_true, preds)),
        f'{prefix}_f1_macro': float(f1_score(y_true, preds, average='macro')),
    }
    metrics.update({f'{prefix}_{k}': v
                    for k, v in compute_uncertainty_metrics(probs, y_true).items()})
    return metrics, new_logits, probs, preds


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize_temperature_scaling(result_dir: str, saved_file_name: str,
                                 seed: int = 42, verbose: bool = True,
                                 has_sidekick: bool = True) -> None:
    """Fit per-model temperature on val, evaluate on test, persist results.

    Args:
        result_dir:      Experiment output directory.
        saved_file_name: NPZ filename, e.g. 'arc_c.npz'.
        seed:            Seed whose predictions to load.
        verbose:         Print optimiser iterations.
        has_sidekick:    Whether a sidekick model was trained alongside the base.
    """
    def _load(subdir, split):
        path = osp.join(result_dir, subdir, split, saved_file_name)
        data = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data['idx'])
        return data, order

    base_val, bv_ord = _load('base', 'val_preds')
    base_tst, bt_ord = _load('base', 'test_preds')
    y_val = _sort_by_idx(base_val['true_labels'], bv_ord)
    y_tst = _sort_by_idx(base_tst['true_labels'], bt_ord)
    bv_logits = _sort_by_idx(base_val['logits'], bv_ord)
    bt_logits = _sort_by_idx(base_tst['logits'], bt_ord)

    print(f'\n[temperature_scaling] Fitting base temperature for seed {seed} ...')
    base_params = _fit_temperature(bv_logits, y_val, verbose=verbose, model_label='base')
    scale_base = base_params['scale']

    bv_m, bv_log, bv_prb, bv_prd = _build_metrics(scale_base, bv_logits, y_val, 'base')
    bt_m, bt_log, bt_prb, bt_prd = _build_metrics(scale_base, bt_logits, y_tst, 'base')

    val_metrics = bv_m
    tst_metrics = bt_m

    # --- optional sidekick ----------------------------------------------
    side_params = {}
    if has_sidekick:
        side_val, sv_ord = _load('sidekick', 'val_preds')
        side_tst, st_ord = _load('sidekick', 'test_preds')
        sv_logits = _sort_by_idx(side_val['logits'], sv_ord)
        st_logits = _sort_by_idx(side_tst['logits'], st_ord)
        y_val_side = _sort_by_idx(side_val['true_labels'], sv_ord)
        y_tst_side = _sort_by_idx(side_tst['true_labels'], st_ord)

        print(f'[temperature_scaling] Fitting sidekick temperature for seed {seed} ...')
        side_params = _fit_temperature(sv_logits, y_val_side, verbose=verbose, model_label='sidekick')
        scale_side = side_params['scale']

        sv_m, sv_log, sv_prb, sv_prd = _build_metrics(scale_side, sv_logits, y_val_side, 'sidekick')
        st_m, st_log, st_prb, st_prd = _build_metrics(scale_side, st_logits, y_tst_side, 'sidekick')
        val_metrics.update(sv_m)
        tst_metrics.update(st_m)

    # --- persist --------------------------------------------------------
    metrics_path = osp.join(result_dir, 'metrics', saved_file_name)
    ts_base_val_path = osp.join(result_dir, 'temperature_scaling', 'base', 'val_preds', saved_file_name)
    ts_base_tst_path = osp.join(result_dir, 'temperature_scaling', 'base', 'test_preds', saved_file_name)

    ex_met = _read_or_create_path(metrics_path)
    ex_bv = _read_or_create_path(ts_base_val_path)
    ex_bt = _read_or_create_path(ts_base_tst_path)

    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_val['idx'], bv_ord),
                           _sort_by_idx(base_val['input'], bv_ord),
                           bv_log.astype(np.float32), bv_prb, bv_prd,
                           _sort_by_idx(base_val['true_labels'], bv_ord),
                           ex_bv, ts_base_val_path)
    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_tst['idx'], bt_ord),
                           _sort_by_idx(base_tst['input'], bt_ord),
                           bt_log.astype(np.float32), bt_prb, bt_prd,
                           _sort_by_idx(base_tst['true_labels'], bt_ord),
                           ex_bt, ts_base_tst_path)

    if has_sidekick:
        ts_side_val_path = osp.join(result_dir, 'temperature_scaling', 'sidekick', 'val_preds', saved_file_name)
        ts_side_tst_path = osp.join(result_dir, 'temperature_scaling', 'sidekick', 'test_preds', saved_file_name)
        ex_sv = _read_or_create_path(ts_side_val_path)
        ex_st = _read_or_create_path(ts_side_tst_path)
        _save_val_test_results(seed, saved_file_name,
                               _sort_by_idx(side_val['idx'], sv_ord),
                               _sort_by_idx(side_val['input'], sv_ord),
                               sv_log.astype(np.float32), sv_prb, sv_prd,
                               _sort_by_idx(side_val['true_labels'], sv_ord),
                               ex_sv, ts_side_val_path)
        _save_val_test_results(seed, saved_file_name,
                               _sort_by_idx(side_tst['idx'], st_ord),
                               _sort_by_idx(side_tst['input'], st_ord),
                               st_log.astype(np.float32), st_prb, st_prd,
                               _sort_by_idx(side_tst['true_labels'], st_ord),
                               ex_st, ts_side_tst_path)

    new_metrics = {
        'data': saved_file_name,
        'base_temperature': base_params['temperature'],
        'base_ts_scale': base_params['scale'],
        'sidekick_temperature': side_params.get('temperature'),
        'sidekick_ts_scale': side_params.get('scale'),
        'val': val_metrics,
        'test': tst_metrics,
    }

    seed_key = str(seed)
    existing_seed = ex_met.get(seed_key, {})
    if hasattr(existing_seed, 'item'):
        existing_seed = existing_seed.item()
    ex_met[seed_key] = _merge_dictionaries(existing_seed, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)
