"""Asymmetric duo optimizer: combines base + sidekick logits via learned scales.

Two modes:
  - Unconstrained (default): L-BFGS-B, alpha >= 1e-6 and beta >= 1e-6 independently.
      p_duo = softmax(alpha * z_base + beta * z_sidekick)
  - Constrained: SLSQP, w_base + w_sidekick = 1, both in [0, 1].
      p_duo = softmax(w_base * z_base + w_sidekick * z_sidekick)

Both are optimised to minimise NLL on the validation set.

Method: Zhou, Shelhamer, and Pleiss, "Asymmetric Duos: Sidekicks Improve
Uncertainty", NeurIPS 2025 (arXiv:2505.18636). The original formulation is for
vision classifiers; here it is adapted to causal LMs and answer-token logits.
"""
import os.path as osp
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import warnings

from .duo_fit import STALL_WARNING, duo_params, fit_duo, legacy_kwargs
from .optimizer_helpers import (
    _merge_dictionaries, _read_or_create_path,
    _sanity_check_on_probs, _save_val_test_results,
    _softmax_from_log, _sort_by_idx,
)
from .uncertainty_metrics import compute_uncertainty_metrics


# ---------------------------------------------------------------------------
# Optimisers
#
# Both delegate to the shared core in ``duo_fit``, pinned to the original
# float32 / finite-difference numerics via ``legacy_kwargs()`` so the numbers
# behind the published tables stay reproducible. Post-hoc analyses under
# ``asym_duos.analysis`` call the same core with its float64 defaults instead.
# ---------------------------------------------------------------------------

def _fit_unconstrained(logits_base: np.ndarray, logits_sidekick: np.ndarray,
                       y_true: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """L-BFGS-B: alpha, beta >= 1e-6, no sum constraint."""
    fit = fit_duo(logits_base, logits_sidekick, y_true, constrained=False, **legacy_kwargs())
    if not fit.success:
        raise RuntimeError(f'Unconstrained duo optimisation failed: {fit.message}')
    if fit.stalled:
        warnings.warn(STALL_WARNING, RuntimeWarning, stacklevel=2)
    if verbose:
        print(f'  scales -> base={fit.scales[0]:.4f}, sidekick={fit.scales[1]:.4f} '
              f'(nll={fit.nll:.6f}, iters={fit.n_iter})')
    return duo_params(fit)


def _fit_constrained(logits_base: np.ndarray, logits_sidekick: np.ndarray,
                     y_true: np.ndarray, verbose: bool = True) -> Dict[str, float]:
    """SLSQP: w_base + w_sidekick = 1, both in [0, 1]."""
    fit = fit_duo(logits_base, logits_sidekick, y_true, constrained=True, **legacy_kwargs())
    if not fit.success:
        raise RuntimeError(f'Constrained duo optimisation failed: {fit.message}')
    if fit.stalled:
        warnings.warn(STALL_WARNING, RuntimeWarning, stacklevel=2)
    if verbose:
        print(f'  weights -> base={fit.scales[0]:.4f}, sidekick={fit.scales[1]:.4f} '
              f'(nll={fit.nll:.6f}, iters={fit.n_iter})')
    params = duo_params(fit)
    # Under the simplex constraint the scales *are* the weights.
    params['weight_base'] = params['scale_base']
    params['weight_sidekick'] = params['scale_sidekick']
    return params


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
        'duo_base_weight': params['weight_base'],
        'duo_sidekick_weight': params['weight_sidekick'],
        'duo_base_scale': alpha,
        'duo_sidekick_scale': beta,
        'duo_base_temperature': params['temperature_base'],
        'duo_sidekick_temperature': params['temperature_sidekick'],
        'val': val_metrics,
        'test': tst_metrics,
    }

    seed_key = str(seed)
    existing_seed = ex_met.get(seed_key, {})
    if hasattr(existing_seed, 'item'):
        existing_seed = existing_seed.item()
    ex_met[seed_key] = _merge_dictionaries(existing_seed, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)


def optimize_weights_blob(result_dir: str, saved_file_name: str, seed: int = 42,
                          constrained: bool = False, verbose: bool = True) -> None:
    """Fit duo scales on BLoB val logits, evaluate on BLoB test logits.

    Loads predictions from ``blob/{base|sidekick}/{val|test_preds}/`` and
    saves combined predictions to ``duo_blob/{val|test_preds}/``.

    Args:
        result_dir:       Experiment output directory.
        saved_file_name:  NPZ filename, e.g. 'arc_c.npz'.
        seed:             Which seed's predictions to load and store results under.
        constrained:      If True, use SLSQP with w_base+w_side=1 constraint.
        verbose:          Print optimiser iterations.
    """
    def _load(model, split):
        path = osp.join(result_dir, 'blob', model, split, saved_file_name)
        data = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data['idx'])
        return data, order

    base_val, bv_ord = _load('base', 'val_preds')
    side_val, sv_ord = _load('sidekick', 'val_preds')
    base_tst, bt_ord = _load('base', 'test_preds')
    side_tst, st_ord = _load('sidekick', 'test_preds')

    if not np.array_equal(_sort_by_idx(base_val['idx'], bv_ord),
                          _sort_by_idx(side_val['idx'], sv_ord)):
        raise ValueError('Val set indices of base_blob and sidekick_blob do not match.')
    if not np.array_equal(_sort_by_idx(base_tst['idx'], bt_ord),
                          _sort_by_idx(side_tst['idx'], st_ord)):
        raise ValueError('Test set indices of base_blob and sidekick_blob do not match.')

    bv_logits = _sort_by_idx(base_val['logits'], bv_ord)
    sv_logits = _sort_by_idx(side_val['logits'], sv_ord)
    bt_logits = _sort_by_idx(base_tst['logits'], bt_ord)
    st_logits = _sort_by_idx(side_tst['logits'], st_ord)
    y_val = _sort_by_idx(base_val['true_labels'], bv_ord)
    y_tst = _sort_by_idx(base_tst['true_labels'], bt_ord)

    fit_fn = _fit_constrained if constrained else _fit_unconstrained
    mode_label = 'constrained' if constrained else 'unconstrained'
    print(f'\n[duo_optimizer] Fitting {mode_label} BLoB duo for seed {seed} ...')
    params = fit_fn(bv_logits, sv_logits, y_val, verbose=verbose)
    alpha, beta = params['scale_base'], params['scale_sidekick']
    print(f'  scale_base={alpha:.4f}, scale_sidekick={beta:.4f}, '
          f'w_base={params["weight_base"]:.4f}, w_sidekick={params["weight_sidekick"]:.4f}')

    val_metrics, duo_val_logits, duo_val_probs, duo_val_pred = \
        _build_metrics(alpha, beta, bv_logits, sv_logits, y_val)
    tst_metrics, duo_tst_logits, duo_tst_probs, duo_tst_pred = \
        _build_metrics(alpha, beta, bt_logits, st_logits, y_tst)

    metrics_path     = osp.join(result_dir, 'metrics', saved_file_name)
    duo_blob_val_path = osp.join(result_dir, 'duo_blob', 'val_preds', saved_file_name)
    duo_blob_tst_path = osp.join(result_dir, 'duo_blob', 'test_preds', saved_file_name)

    ex_val = _read_or_create_path(duo_blob_val_path)
    ex_tst = _read_or_create_path(duo_blob_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_val['idx'], bv_ord),
                           _sort_by_idx(base_val['input'], bv_ord),
                           duo_val_logits.astype(np.float32), duo_val_probs, duo_val_pred,
                           _sort_by_idx(base_val['true_labels'], bv_ord),
                           ex_val, duo_blob_val_path)
    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_tst['idx'], bt_ord),
                           _sort_by_idx(base_tst['input'], bt_ord),
                           duo_tst_logits.astype(np.float32), duo_tst_probs, duo_tst_pred,
                           _sort_by_idx(base_tst['true_labels'], bt_ord),
                           ex_tst, duo_blob_tst_path)

    new_metrics = {
        'data': saved_file_name,
        'duo_blob_mode': mode_label,
        'duo_blob_base_weight': params['weight_base'],
        'duo_blob_sidekick_weight': params['weight_sidekick'],
        'duo_blob_base_scale': alpha,
        'duo_blob_sidekick_scale': beta,
        'duo_blob_base_temperature': params['temperature_base'],
        'duo_blob_sidekick_temperature': params['temperature_sidekick'],
        'val': val_metrics,
        'test': tst_metrics,
    }

    seed_key = str(seed)
    existing_seed = ex_met.get(seed_key, {})
    if hasattr(existing_seed, 'item'):
        existing_seed = existing_seed.item()
    ex_met[seed_key] = _merge_dictionaries(existing_seed, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)


def optimize_weights_ts(result_dir: str, saved_file_name: str, seed: int = 42,
                        constrained: bool = False, verbose: bool = True) -> None:
    """Fit duo scales on temperature-scaled val logits, evaluate on TS test logits.

    Loads predictions from temperature_scaling/{base|sidekick}/{val|test_preds}/
    and saves combined duo_ts predictions to duo_ts/{val|test_preds}/.

    Args:
        result_dir:       Experiment output directory.
        saved_file_name:  NPZ filename, e.g. 'arc_c.npz'.
        seed:             Which seed's predictions to load and store results under.
        constrained:      If True, use SLSQP with w_base+w_side=1 constraint.
        verbose:          Print optimiser iterations.
    """
    def _load(model, split):
        path = osp.join(result_dir, 'temperature_scaling', model, split, saved_file_name)
        data = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data['idx'])
        return data, order

    base_val, bv_ord = _load('base', 'val_preds')
    side_val, sv_ord = _load('sidekick', 'val_preds')
    base_tst, bt_ord = _load('base', 'test_preds')
    side_tst, st_ord = _load('sidekick', 'test_preds')

    if not np.array_equal(_sort_by_idx(base_val['idx'], bv_ord),
                          _sort_by_idx(side_val['idx'], sv_ord)):
        raise ValueError('Val set indices of base_ts and sidekick_ts do not match.')
    if not np.array_equal(_sort_by_idx(base_tst['idx'], bt_ord),
                          _sort_by_idx(side_tst['idx'], st_ord)):
        raise ValueError('Test set indices of base_ts and sidekick_ts do not match.')

    bv_logits = _sort_by_idx(base_val['logits'], bv_ord)
    sv_logits = _sort_by_idx(side_val['logits'], sv_ord)
    bt_logits = _sort_by_idx(base_tst['logits'], bt_ord)
    st_logits = _sort_by_idx(side_tst['logits'], st_ord)
    y_val = _sort_by_idx(base_val['true_labels'], bv_ord)
    y_tst = _sort_by_idx(base_tst['true_labels'], bt_ord)

    fit_fn = _fit_constrained if constrained else _fit_unconstrained
    mode_label = 'constrained' if constrained else 'unconstrained'
    print(f'\n[duo_optimizer] Fitting {mode_label} Duo TS for seed {seed} ...')
    params = fit_fn(bv_logits, sv_logits, y_val, verbose=verbose)
    alpha, beta = params['scale_base'], params['scale_sidekick']
    print(f'  scale_base={alpha:.4f}, scale_sidekick={beta:.4f}, '
          f'w_base={params["weight_base"]:.4f}, w_sidekick={params["weight_sidekick"]:.4f}')

    # Build duo_ts metrics (individual TS metrics are already in the metrics file)
    def _compute_duo_ts(logits_base, logits_sidekick, y_true):
        duo_logits = alpha * logits_base + beta * logits_sidekick
        probs_duo = _softmax_from_log(duo_logits)
        _sanity_check_on_probs(probs_duo, 'duo_ts')
        pred_duo = probs_duo.argmax(axis=1)
        m = {
            'duo_ts_accuracy': float(accuracy_score(y_true, pred_duo)),
            'duo_ts_f1_macro': float(f1_score(y_true, pred_duo, average='macro')),
        }
        m.update({f'duo_ts_{k}': v for k, v in compute_uncertainty_metrics(probs_duo, y_true).items()})
        return m, duo_logits, probs_duo, pred_duo

    val_metrics, duo_val_logits, duo_val_probs, duo_val_pred = _compute_duo_ts(bv_logits, sv_logits, y_val)
    tst_metrics, duo_tst_logits, duo_tst_probs, duo_tst_pred = _compute_duo_ts(bt_logits, st_logits, y_tst)

    metrics_path = osp.join(result_dir, 'metrics', saved_file_name)
    duo_ts_val_path = osp.join(result_dir, 'duo_ts', 'val_preds', saved_file_name)
    duo_ts_tst_path = osp.join(result_dir, 'duo_ts', 'test_preds', saved_file_name)

    ex_val = _read_or_create_path(duo_ts_val_path)
    ex_tst = _read_or_create_path(duo_ts_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_val['idx'], bv_ord),
                           _sort_by_idx(base_val['input'], bv_ord),
                           duo_val_logits.astype(np.float32), duo_val_probs, duo_val_pred,
                           _sort_by_idx(base_val['true_labels'], bv_ord),
                           ex_val, duo_ts_val_path)
    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_tst['idx'], bt_ord),
                           _sort_by_idx(base_tst['input'], bt_ord),
                           duo_tst_logits.astype(np.float32), duo_tst_probs, duo_tst_pred,
                           _sort_by_idx(base_tst['true_labels'], bt_ord),
                           ex_tst, duo_ts_tst_path)

    new_metrics = {
        'data': saved_file_name,
        'duo_ts_mode': mode_label,
        'duo_ts_base_weight': params['weight_base'],
        'duo_ts_sidekick_weight': params['weight_sidekick'],
        'duo_ts_base_scale': alpha,
        'duo_ts_sidekick_scale': beta,
        'duo_ts_base_temperature': params['temperature_base'],
        'duo_ts_sidekick_temperature': params['temperature_sidekick'],
        'val': val_metrics,
        'test': tst_metrics,
    }

    seed_key = str(seed)
    existing_seed = ex_met.get(seed_key, {})
    if hasattr(existing_seed, 'item'):
        existing_seed = existing_seed.item()
    ex_met[seed_key] = _merge_dictionaries(existing_seed, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)


def optimize_weights_laplace(result_dir: str, saved_file_name: str, seed: int = 42,
                        constrained: bool = False, verbose: bool = True) -> None:
    """Fit duo scales on temperature-scaled val logits, evaluate on TS test logits.

    Loads predictions from temperature_scaling/{base|sidekick}/{val|test_preds}/
    and saves combined duo_ts predictions to duo_ts/{val|test_preds}/.

    Args:
        result_dir:       Experiment output directory. 
        saved_file_name:  NPZ filename, e.g. 'arc_c.npz'.
        seed:             Which seed's predictions to load and store results under.
        constrained:      If True, use SLSQP with w_base+w_side=1 constraint.
        verbose:          Print optimiser iterations.
    """
    def _load(model, split):
        path = osp.join(result_dir, 'laplace_lora', model, split, saved_file_name)
        data = dict(np.load(path, allow_pickle=True))[str(seed)].item()
        order = np.argsort(data['idx'])
        return data, order

    base_val, bv_ord = _load('base', 'val_preds')
    side_val, sv_ord = _load('sidekick', 'val_preds')
    base_tst, bt_ord = _load('base', 'test_preds')
    side_tst, st_ord = _load('sidekick', 'test_preds')

    if not np.array_equal(_sort_by_idx(base_val['idx'], bv_ord),
                          _sort_by_idx(side_val['idx'], sv_ord)):
        raise ValueError('Val set indices of base_laplace and sidekick_laplace do not match.')
    if not np.array_equal(_sort_by_idx(base_tst['idx'], bt_ord),
                          _sort_by_idx(side_tst['idx'], st_ord)):
        raise ValueError('Test set indices of base_laplace and sidekick_laplace do not match.')

    bv_logits = _sort_by_idx(base_val['logits'], bv_ord)
    sv_logits = _sort_by_idx(side_val['logits'], sv_ord)
    bt_logits = _sort_by_idx(base_tst['logits'], bt_ord)
    st_logits = _sort_by_idx(side_tst['logits'], st_ord)
    y_val = _sort_by_idx(base_val['true_labels'], bv_ord)
    y_tst = _sort_by_idx(base_tst['true_labels'], bt_ord)

    fit_fn = _fit_constrained if constrained else _fit_unconstrained
    mode_label = 'constrained' if constrained else 'unconstrained'
    print(f'\n[duo_optimizer] Fitting {mode_label} Duo Laplace LoRA for seed {seed} ...')
    params = fit_fn(bv_logits, sv_logits, y_val, verbose=verbose)
    alpha, beta = params['scale_base'], params['scale_sidekick']
    print(f'  scale_base={alpha:.4f}, scale_sidekick={beta:.4f}, '
          f'w_base={params["weight_base"]:.4f}, w_sidekick={params["weight_sidekick"]:.4f}')

    # Build duo_ts metrics (individual TS metrics are already in the metrics file)
    def _compute_duo_laplace(logits_base, logits_sidekick, y_true):
        duo_logits = alpha * logits_base + beta * logits_sidekick
        probs_duo = _softmax_from_log(duo_logits)
        _sanity_check_on_probs(probs_duo, 'duo_laplace_lora')
        pred_duo = probs_duo.argmax(axis=1)
        m = {
            'duo_laplace_lora_accuracy': float(accuracy_score(y_true, pred_duo)),
            'duo_laplace_lora_f1_macro': float(f1_score(y_true, pred_duo, average='macro')),
        }
        m.update({f'duo_laplace_lora_{k}': v for k, v in compute_uncertainty_metrics(probs_duo, y_true).items()})
        return m, duo_logits, probs_duo, pred_duo

    val_metrics, duo_val_logits, duo_val_probs, duo_val_pred = _compute_duo_laplace(bv_logits, sv_logits, y_val)
    tst_metrics, duo_tst_logits, duo_tst_probs, duo_tst_pred = _compute_duo_laplace(bt_logits, st_logits, y_tst)

    metrics_path = osp.join(result_dir, 'metrics', saved_file_name)
    duo_laplace_val_path = osp.join(result_dir, 'duo_laplace', 'val_preds', saved_file_name)
    duo_laplace_tst_path = osp.join(result_dir, 'duo_laplace', 'test_preds', saved_file_name)

    ex_val = _read_or_create_path(duo_laplace_val_path)
    ex_tst = _read_or_create_path(duo_laplace_tst_path)
    ex_met = _read_or_create_path(metrics_path)

    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_val['idx'], bv_ord),
                           _sort_by_idx(base_val['input'], bv_ord),
                           duo_val_logits.astype(np.float32), duo_val_probs, duo_val_pred,
                           _sort_by_idx(base_val['true_labels'], bv_ord),
                           ex_val, duo_laplace_val_path)
    _save_val_test_results(seed, saved_file_name,
                           _sort_by_idx(base_tst['idx'], bt_ord),
                           _sort_by_idx(base_tst['input'], bt_ord),
                           duo_tst_logits.astype(np.float32), duo_tst_probs, duo_tst_pred,
                           _sort_by_idx(base_tst['true_labels'], bt_ord),
                           ex_tst, duo_laplace_tst_path)

    new_metrics = {
        'data': saved_file_name,
        'duo_laplace_mode': mode_label,
        'duo_laplace_base_weight': params['weight_base'],
        'duo_laplace_sidekick_weight': params['weight_sidekick'],
        'duo_laplace_base_scale': alpha,
        'duo_laplace_sidekick_scale': beta,
        'duo_laplace_base_temperature': params['temperature_base'],
        'duo_laplace_sidekick_temperature': params['temperature_sidekick'],
        'val': val_metrics,
        'test': tst_metrics,
    }

    seed_key = str(seed)
    existing_seed = ex_met.get(seed_key, {})
    if hasattr(existing_seed, 'item'):
        existing_seed = existing_seed.item()
    ex_met[seed_key] = _merge_dictionaries(existing_seed, new_metrics)
    np.savez_compressed(metrics_path, **ex_met)
