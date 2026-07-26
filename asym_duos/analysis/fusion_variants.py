"""Group A: the nested fusion ladder and the smoothing nulls.

A1 -- fusion ladder
-------------------
Every variant is a set of non-negative scales over logit components, fitted on
validation NLL and evaluated on test:

===  ===========================  ======  ============================
 #   variant                      params  form
===  ===========================  ======  ============================
 1   ``base_raw``                      0  ``z_b``
 2   ``base_ts``                       1  ``a * z_b``
 3   ``equal_unscaled``                0  ``z_b + z_s``
 4   ``symmetric_learned``             1  ``g * (z_b + z_s)``
 5   ``duo``                           2  ``a * z_b + b * z_s``
 6   ``duo_oracle``                    2  ``a * z_b + b * z_s`` fit on test
===  ===========================  ======  ============================

Contrasts: ``5 - 2`` is the total value of the sidekick, ``5 - 4`` the value of
*asymmetric* weighting given that you already fuse and scale, ``4 - 3`` the value
of scaling given fusion, and ``6 - 5`` what is lost by fitting on finite
validation data.

The constrained duo (``w * z_b + (1 - w) * z_s``) is deliberately absent: the
code exists in ``duo_optimizer`` but is not part of the reported results.

A2 -- smoothing nulls
---------------------
Any sidekick surrogate that is an affine function of the base logits collapses
to base temperature scaling exactly, since
``a * z_b + b * (c * z_b + d) = (a + bc) * z_b + bd`` and softmax is
shift-invariant. Such controls are therefore already covered by the ``base_ts``
row and are not re-run here.

The informative nulls are the ones that preserve the sidekick's *smoothness* but
destroy its alignment with the input:

``class_permuted``
    Per example, permute the sidekick's class ordering. Preserves each example's
    logit spread exactly; destroys which class it favours. This is the tightest
    smoothness-matched null and the one to lead with.
``row_permuted``
    Permute sidekick rows across examples. Preserves the marginal distribution;
    destroys per-example alignment.
``matched_gaussian``
    Resample sidekick logits from a Gaussian fitted to the real sidekick's
    column moments. Preserves first and second moments; destroys all input
    dependence.
``zero``
    ``z_s = 0``. Reduces to base TS; a sanity anchor.

Under a pure smoothing account the real sidekick should sit inside the null
distribution, and the fitted ``beta`` should not distinguish it from the nulls.
"""
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..utils.duo_fit import fit_component_scales, fuse
from ..utils.optimizer_helpers import _softmax_from_log
from ..utils.preds_io import DuoPreds
from .metrics_panel import full_panel

# Variant name -> (component builder, number of fitted scales, fit-on-test flag)
LADDER_ORDER = ('base_raw', 'base_ts', 'equal_unscaled', 'symmetric_learned', 'duo', 'duo_oracle')


def _components(variant: str, z_base: np.ndarray, z_side: np.ndarray) -> List[np.ndarray]:
    if variant in ('base_raw', 'base_ts'):
        return [z_base]
    if variant in ('equal_unscaled', 'symmetric_learned'):
        return [z_base + z_side]
    if variant in ('duo', 'duo_oracle'):
        return [z_base, z_side]
    raise ValueError(f'Unknown fusion variant {variant!r}')


def _is_fitted(variant: str) -> bool:
    return variant not in ('base_raw', 'equal_unscaled')


# ---------------------------------------------------------------------------
# A1: the ladder
# ---------------------------------------------------------------------------

def evaluate_variant(variant: str, val: DuoPreds, test: DuoPreds,
                     **fit_kwargs) -> Dict[str, object]:
    """Fit one ladder variant and score it on both splits.

    ``duo_oracle`` fits on the test split on purpose: it is the upper bound that
    shows how much the honest validation-fitted duo gives up.
    """
    val_components = _components(variant, val.base.logits, val.sidekick.logits)
    test_components = _components(variant, test.base.logits, test.sidekick.logits)

    if not _is_fitted(variant):
        scales = np.ones(len(val_components))
        fit_nll, success = float('nan'), True
    else:
        fit_on = test_components if variant == 'duo_oracle' else val_components
        fit_labels = test.labels if variant == 'duo_oracle' else val.labels
        fit = fit_component_scales(fit_on, fit_labels, **fit_kwargs)
        scales, fit_nll, success = fit.scales, fit.nll, fit.success

    out: Dict[str, object] = {
        'variant': variant,
        'n_params': 0 if not _is_fitted(variant) else len(val_components),
        'fit_success': success,
        'fit_nll': fit_nll,
    }
    for i, s in enumerate(scales):
        out[f'scale_{i}'] = float(s)

    for split_name, preds, components in (('val', val, val_components), ('test', test, test_components)):
        probs = _softmax_from_log(fuse(scales, components))
        for metric, value in full_panel(probs, preds.labels).items():
            out[f'{split_name}::{metric}'] = value
    return out


def run_ladder(val: DuoPreds, test: DuoPreds,
               variants: Sequence[str] = LADDER_ORDER, **fit_kwargs) -> List[Dict[str, object]]:
    """Evaluate every ladder variant for one (dataset, seed)."""
    return [evaluate_variant(v, val, test, **fit_kwargs) for v in variants]


# ---------------------------------------------------------------------------
# A2: surrogate sidekicks
# ---------------------------------------------------------------------------

def class_permuted(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute the class axis of every row."""
    n, c = z.shape
    order = np.argsort(rng.random((n, c)), axis=1)
    return np.take_along_axis(z, order, axis=1)


def row_permuted(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle whole rows across examples."""
    return z[rng.permutation(len(z))]


def matched_gaussian(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample from a Gaussian matched to the column mean and covariance."""
    mean = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-8 * np.eye(z.shape[1])
    return rng.multivariate_normal(mean, cov, size=len(z))


def zero_sidekick(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uninformative constant sidekick; reduces the duo to base TS."""
    return np.zeros_like(z)


SURROGATES: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    'class_permuted': class_permuted,
    'row_permuted': row_permuted,
    'matched_gaussian': matched_gaussian,
    'zero': zero_sidekick,
}

# Deterministic surrogates gain nothing from repeated draws.
DETERMINISTIC_SURROGATES = ('zero',)


def run_nulls(val: DuoPreds, test: DuoPreds,
              surrogates: Sequence[str] = tuple(SURROGATES),
              n_draws: int = 100, seed: int = 0, **fit_kwargs) -> List[Dict[str, object]]:
    """Refit the duo against each surrogate sidekick and score it on test.

    The surrogate is applied to validation and test independently, then
    ``(alpha, beta)`` is refitted from scratch on the surrogate validation
    logits -- otherwise the null would inherit scales tuned to the real
    sidekick.

    Returns one row per (surrogate, draw). Rows carry the fitted ``beta`` as
    well as the metrics, because a collapse of ``beta`` toward zero under the
    nulls is itself evidence that a non-zero ``beta`` on real logits encodes
    information rather than smoothing.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    for name in surrogates:
        make = SURROGATES[name]
        draws = 1 if name in DETERMINISTIC_SURROGATES else n_draws
        for draw in range(draws):
            fake_val = make(val.sidekick.logits, rng)
            fake_test = make(test.sidekick.logits, rng)

            fit = fit_component_scales([val.base.logits, fake_val], val.labels, **fit_kwargs)
            probs = _softmax_from_log(fuse(fit.scales, [test.base.logits, fake_test]))

            row: Dict[str, object] = {
                'variant': f'null::{name}',
                'surrogate': name,
                'draw_id': draw,
                'fit_success': fit.success,
                'fit_nll': fit.nll,
                'scale_0': float(fit.scales[0]),
                'scale_1': float(fit.scales[1]),
                'sidekick_weight': float(fit.scales[1] / (fit.scales[0] + fit.scales[1])),
            }
            for metric, value in full_panel(probs, test.labels).items():
                row[f'test::{metric}'] = value
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# A3: informativeness dose-response
# ---------------------------------------------------------------------------

def run_dose_response(val: DuoPreds, test: DuoPreds,
                      rhos: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                      n_draws: int = 10, seed: int = 0, **fit_kwargs) -> List[Dict[str, object]]:
    """Interpolate between the real sidekick and a class-permuted copy.

    ``z_s(rho) = rho * z_s + (1 - rho) * z_s_permuted`` holds the smoothness
    roughly fixed while sweeping how much input-aligned information survives.
    A smoothing-only account predicts a flat curve.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    for rho in rhos:
        draws = 1 if rho == 1.0 else n_draws
        for draw in range(draws):
            perm_val = class_permuted(val.sidekick.logits, rng)
            perm_test = class_permuted(test.sidekick.logits, rng)
            mixed_val = rho * val.sidekick.logits + (1 - rho) * perm_val
            mixed_test = rho * test.sidekick.logits + (1 - rho) * perm_test

            fit = fit_component_scales([val.base.logits, mixed_val], val.labels, **fit_kwargs)
            probs = _softmax_from_log(fuse(fit.scales, [test.base.logits, mixed_test]))

            row: Dict[str, object] = {
                'variant': f'dose::rho_{rho:g}',
                'rho': float(rho),
                'draw_id': draw,
                'fit_success': fit.success,
                'scale_0': float(fit.scales[0]),
                'scale_1': float(fit.scales[1]),
            }
            for metric, value in full_panel(probs, test.labels).items():
                row[f'test::{metric}'] = value
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Refit verification
# ---------------------------------------------------------------------------

def verify_refit(val: DuoPreds, stored: Dict[str, float],
                 **fit_kwargs) -> Optional[Dict[str, float]]:
    """Compare a fresh duo refit against the scales stored during training.

    The analysis code fits in float64 with an analytic gradient, whereas the
    training pipeline ran in float32 with finite differences. This reports the
    discrepancy so it can be checked before any ablation number is quoted
    alongside a published one. Returns ``None`` if nothing was stored.
    """
    if 'duo_base_scale' not in stored or 'duo_sidekick_scale' not in stored:
        return None

    fit = fit_component_scales([val.base.logits, val.sidekick.logits], val.labels, **fit_kwargs)
    alpha_new, beta_new = float(fit.scales[0]), float(fit.scales[1])
    alpha_old, beta_old = float(stored['duo_base_scale']), float(stored['duo_sidekick_scale'])

    def _rel(new: float, old: float) -> float:
        return abs(new - old) / max(abs(old), 1e-12)

    return {
        'alpha_stored': alpha_old,
        'beta_stored': beta_old,
        'alpha_refit': alpha_new,
        'beta_refit': beta_new,
        'alpha_rel_diff': _rel(alpha_new, alpha_old),
        'beta_rel_diff': _rel(beta_new, beta_old),
        'sidekick_weight_stored': beta_old / (alpha_old + beta_old),
        'sidekick_weight_refit': beta_new / (alpha_new + beta_new),
    }
