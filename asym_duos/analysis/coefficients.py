"""Group B: making alpha and beta interpretable.

Review 1 asks how the magnitudes of the learned coefficients can be read as
evidence of relative contribution when they are also absorbing the two models'
native logit scales, and suggests a scale-normalised characterisation.

Two answers:

B1  Report contributions in units that already include the logit scale:
    ``s_b = alpha * sd(z_b)`` and ``s_s = beta * sd(z_s)``, and the scale-free
    ratio ``r = s_s / (s_b + s_s)``. Also decompose the variance of the fused
    logit into base, sidekick and covariance shares.

B2  Refit on per-example standardised logits, which removes the confound at the
    source and makes the coefficients directly comparable. Test metrics should
    be unchanged, since standardising is an affine reparameterisation the fit
    can undo -- which is itself the check that B1 and B2 tell the same story.
"""
from typing import Dict

import numpy as np

from ..utils.duo_fit import fit_component_scales, fuse
from ..utils.optimizer_helpers import _softmax_from_log
from ..utils.preds_io import DuoPreds
from .metrics_panel import full_panel


def _centred(z: np.ndarray) -> np.ndarray:
    """Subtract the per-example mean over classes (softmax is shift-invariant)."""
    return z - z.mean(axis=1, keepdims=True)


def _mean_spread(z: np.ndarray) -> float:
    """Mean over examples of the per-example standard deviation across classes."""
    return float(_centred(z).std(axis=1).mean())


def _standardise(z: np.ndarray) -> np.ndarray:
    """Centre per example and divide by the per-example spread."""
    centred = _centred(z)
    scale = centred.std(axis=1, keepdims=True)
    return centred / np.maximum(scale, 1e-12)


# ---------------------------------------------------------------------------
# B1: scale-normalised reporting
# ---------------------------------------------------------------------------

def scale_normalised_report(alpha: float, beta: float,
                            z_base: np.ndarray, z_side: np.ndarray) -> Dict[str, float]:
    """Express the fitted coefficients in scale-free terms."""
    spread_b, spread_s = _mean_spread(z_base), _mean_spread(z_side)
    contrib_b, contrib_s = alpha * spread_b, beta * spread_s
    total_contrib = contrib_b + contrib_s

    scaled_b = alpha * _centred(z_base)
    scaled_s = beta * _centred(z_side)
    var_b = float(scaled_b.var())
    var_s = float(scaled_s.var())
    cov = float(np.mean(scaled_b * scaled_s) - scaled_b.mean() * scaled_s.mean())
    var_total = var_b + var_s + 2 * cov

    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'temperature_base': 1.0 / alpha if alpha > 0 else float('inf'),
        'temperature_sidekick': 1.0 / beta if beta > 0 else float('inf'),
        'naive_sidekick_weight': float(beta / (alpha + beta)) if (alpha + beta) > 0 else float('nan'),
        'logit_spread_base': spread_b,
        'logit_spread_sidekick': spread_s,
        'effective_contrib_base': float(contrib_b),
        'effective_contrib_sidekick': float(contrib_s),
        'effective_sidekick_share': float(contrib_s / total_contrib) if total_contrib > 0 else float('nan'),
        'var_share_base': float(var_b / var_total) if var_total > 0 else float('nan'),
        'var_share_sidekick': float(var_s / var_total) if var_total > 0 else float('nan'),
        'var_share_cov': float(2 * cov / var_total) if var_total > 0 else float('nan'),
    }


# ---------------------------------------------------------------------------
# B2: standardised refit
# ---------------------------------------------------------------------------

def standardised_refit(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> Dict[str, float]:
    """Refit on standardised logits and confirm the test metrics are unchanged."""
    raw_fit = fit_component_scales([val.base.logits, val.sidekick.logits], val.labels, **fit_kwargs)
    std_fit = fit_component_scales([_standardise(val.base.logits), _standardise(val.sidekick.logits)],
                                   val.labels, **fit_kwargs)

    raw_probs = _softmax_from_log(fuse(raw_fit.scales, [test.base.logits, test.sidekick.logits]))
    std_probs = _softmax_from_log(fuse(std_fit.scales, [_standardise(test.base.logits),
                                                        _standardise(test.sidekick.logits)]))
    raw_panel = full_panel(raw_probs, test.labels)
    std_panel = full_panel(std_probs, test.labels)

    alpha_t, beta_t = float(std_fit.scales[0]), float(std_fit.scales[1])
    return {
        'alpha_standardised': alpha_t,
        'beta_standardised': beta_t,
        'standardised_sidekick_share': beta_t / (alpha_t + beta_t) if (alpha_t + beta_t) > 0 else float('nan'),
        'test_nll_raw_fit': raw_panel['nll'],
        'test_nll_standardised_fit': std_panel['nll'],
        'test_nll_gap': std_panel['nll'] - raw_panel['nll'],
        'test_ece_raw_fit': raw_panel['ece'],
        'test_ece_standardised_fit': std_panel['ece'],
    }


def run_coefficients(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> Dict[str, float]:
    """B1 + B2 for one (dataset, seed), fitted on validation as usual."""
    fit = fit_component_scales([val.base.logits, val.sidekick.logits], val.labels, **fit_kwargs)
    alpha, beta = float(fit.scales[0]), float(fit.scales[1])

    out = scale_normalised_report(alpha, beta, val.base.logits, val.sidekick.logits)
    out.update(standardised_refit(val, test, **fit_kwargs))
    out['fit_success'] = fit.success
    return out
