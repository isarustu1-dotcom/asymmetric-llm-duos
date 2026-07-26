"""Pure fitting of non-negative logit-fusion scales.

This is the numerical core shared by the duo optimizer and every post-hoc
ablation. It does no I/O and prints nothing, so it is safe to call in the tight
loops that permutation nulls and calibration-size sweeps need (thousands of
refits per dataset).

A *component* is an ``(N, C)`` logit matrix. Fitting learns one non-negative
scale per component so that ``softmax(sum_j s_j * C_j)`` minimises validation
NLL. Expressing the ladder of fusion variants this way keeps a single optimiser
behind all of them:

    base + TS            components = [z_base]                -> 1 scale
    symmetric fusion     components = [z_base + z_sidekick]   -> 1 scale
    asymmetric duo       components = [z_base, z_sidekick]    -> 2 scales

Numerics
--------
New code should use the defaults (float64 + analytic gradient). Logits are
persisted as float32, and NumPy does not upcast an array when multiplying it by
a scalar, so a float32 objective is evaluated in float32 throughout; with
finite-difference gradients that can round to exactly zero and strand L-BFGS-B
at its starting point. ``temperature_scaling._nll_and_grad`` documents the same
hazard. ``legacy_kwargs()`` reproduces the original float32 / finite-difference
behaviour for the entry points that produced the published numbers.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .optimizer_helpers import _logsumexp

# Scales are bounded away from exactly zero so that 1/s stays finite when the
# fit is reported as an effective temperature.
MIN_SCALE = 1e-6


@dataclass
class ScaleFit:
    """Result of fitting one non-negative scale per component."""

    scales: np.ndarray
    nll: float
    success: bool
    message: str = ''
    n_iter: int = 0
    stalled: bool = False   # optimiser never moved off its starting point
    extras: Dict[str, float] = field(default_factory=dict)

    def fuse(self, components: Sequence[np.ndarray]) -> np.ndarray:
        """Apply the fitted scales to a (possibly different-split) component list."""
        return fuse(self.scales, components)


def legacy_kwargs() -> Dict[str, object]:
    """Fitting options that reproduce the pre-refactor duo optimizer exactly.

    Warning: with float32 logits -- which is what ``save_predictions`` writes --
    these options make the finite-difference gradient evaluate to exactly zero,
    so L-BFGS-B terminates at its starting point and the "fit" returns
    ``alpha = beta = 1``. Check ``ScaleFit.stalled`` on the result.
    """
    return {'dtype': None, 'use_grad': False}


STALL_WARNING = (
    'Duo fit returned its starting point unchanged: the optimiser saw a zero '
    'gradient and never moved, so these scales are NOT fitted. This is the '
    'float32 finite-difference failure mode described in duo_fit; the logits '
    'reaching the fit are float32 (save_predictions writes np.float32) and the '
    'finite-difference step underflows. Refit in float64 with the analytic '
    'gradient (the asym_duos.analysis default) to get real scales.'
)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def fuse(scales: Sequence[float], components: Sequence[np.ndarray]) -> np.ndarray:
    """Return ``sum_j scales[j] * components[j]``."""
    if len(scales) != len(components):
        raise ValueError(f'Got {len(scales)} scales for {len(components)} components.')
    out = scales[0] * components[0]
    for s, c in zip(scales[1:], components[1:]):
        out = out + s * c
    return out


def nll(scales: Sequence[float], components: Sequence[np.ndarray], y: np.ndarray) -> float:
    """Mean negative log-likelihood of the fused logits."""
    z = fuse(scales, components)
    log_p = z - _logsumexp(z, axis=1, keepdims=True)
    return float(-np.mean(log_p[np.arange(len(y)), y]))


def _nll_and_grad(scales: np.ndarray, components: Sequence[np.ndarray], y: np.ndarray):
    """NLL and its gradient w.r.t. each scale.

        d NLL / d s_j = -(1/N) * sum_i ( C_j[i, y_i] - E_{p_i}[C_j[i, :]] )
    """
    z = fuse(scales, components)
    log_p = z - _logsumexp(z, axis=1, keepdims=True)
    rows = np.arange(len(y))
    loss = float(-np.mean(log_p[rows, y]))

    probs = np.exp(log_p)
    grad = np.empty(len(components), dtype=np.float64)
    for j, comp in enumerate(components):
        true_term = comp[rows, y]
        expected_term = (probs * comp).sum(axis=1)
        grad[j] = float(-np.mean(true_term - expected_term))
    return loss, grad


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_component_scales(components: Sequence[np.ndarray],
                         y: np.ndarray,
                         x0: Optional[Sequence[float]] = None,
                         sum_to_one: bool = False,
                         use_grad: bool = True,
                         dtype: Optional[type] = np.float64,
                         max_restarts: int = 2) -> ScaleFit:
    """Fit one non-negative scale per component by minimising NLL.

    Args:
        components:   list of ``(N, C)`` logit matrices.
        y:            ``(N,)`` integer labels.
        x0:           starting point; defaults to all-ones, or a uniform simplex
                      point when ``sum_to_one``.
        sum_to_one:   constrain the scales to the simplex (SLSQP) instead of
                      independent non-negative scales (L-BFGS-B).
        use_grad:     supply the analytic gradient. Disable only to reproduce
                      the original finite-difference behaviour.
        dtype:        cast components to this dtype first; ``None`` leaves them
                      untouched.
        max_restarts: extra attempts from perturbed starting points if the
                      optimiser reports failure.

    Returns:
        A :class:`ScaleFit`. Unlike the original optimizer this never raises on
        non-convergence — callers that need loud failure should check
        ``.success`` themselves. Sweeps over thousands of refits rely on this.
    """
    if not components:
        raise ValueError('Need at least one component to fit.')
    if dtype is not None:
        components = [np.asarray(c, dtype=dtype) for c in components]
    y = np.asarray(y).astype(int)

    k = len(components)
    if x0 is None:
        x0 = np.full(k, 1.0 / k if sum_to_one else 1.0, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)

    if sum_to_one:
        bounds = [(0.0, 1.0)] * k
        constraints = [{'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)}]
        method = 'SLSQP'
    else:
        bounds = [(MIN_SCALE, None)] * k
        constraints = []
        method = 'L-BFGS-B'

    objective = _nll_and_grad if use_grad else (lambda s, c, t: _nll_and_grad(s, c, t)[0])

    best: Optional[ScaleFit] = None
    rng = np.random.default_rng(0)
    for attempt in range(max_restarts + 1):
        start = x0 if attempt == 0 else np.clip(x0 * rng.uniform(0.5, 1.5, size=k), MIN_SCALE, None)
        res = minimize(objective, start, args=(components, y), method=method,
                       jac=use_grad, bounds=bounds, constraints=constraints)
        candidate = ScaleFit(
            scales=np.asarray(res.x, dtype=np.float64),
            nll=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
            n_iter=int(getattr(res, 'nit', 0)),
            stalled=bool(np.allclose(res.x, start, rtol=0, atol=1e-12)),
        )
        if best is None or (candidate.success and not best.success) or candidate.nll < best.nll:
            best = candidate
        if best.success:
            break

    return best


# ---------------------------------------------------------------------------
# Convenience wrappers for the two-model case
# ---------------------------------------------------------------------------

def fit_duo(z_base: np.ndarray, z_sidekick: np.ndarray, y: np.ndarray,
            constrained: bool = False, **kwargs) -> ScaleFit:
    """Fit ``alpha * z_base + beta * z_sidekick``."""
    return fit_component_scales([z_base, z_sidekick], y, sum_to_one=constrained, **kwargs)


def fit_temperature_scale(z: np.ndarray, y: np.ndarray, **kwargs) -> ScaleFit:
    """Fit a single scale ``alpha * z`` (temperature scaling with ``T = 1/alpha``)."""
    return fit_component_scales([z], y, **kwargs)


def duo_params(fit: ScaleFit) -> Dict[str, float]:
    """Expand a two-component fit into the reporting dict the pipeline stores."""
    alpha, beta = float(fit.scales[0]), float(fit.scales[1])
    total = alpha + beta
    return {
        'scale_base': alpha,
        'scale_sidekick': beta,
        'weight_base': alpha / total if total > 0 else float('nan'),
        'weight_sidekick': beta / total if total > 0 else float('nan'),
        'temperature_base': 1.0 / alpha if alpha > 0 else float('inf'),
        'temperature_sidekick': 1.0 / beta if beta > 0 else float('inf'),
    }
