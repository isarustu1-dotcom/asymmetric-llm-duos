"""Group E3: exact per-example decomposition of the Duo's logit change.

Review 3 asks for the per-example interaction to be decomposed into a
base-rescaling term and a sidekick-contribution term, each correlated with
correctness, in order to tell "complementary information" apart from
"smoothing".

Setup. Work in centred logits, since softmax is shift-invariant. Take **base
temperature scaling** as the reference point, ``z_ts = a_ts * z_b`` -- not the
raw base, or the decomposition would attribute to rescaling a move that TS
already makes for free. Then::

    z_duo - z_ts = (alpha - a_ts) * z_b   +   beta * z_s
                   \\_____ R _____/           \\___ S ___/
                   residual rescaling         sidekick contribution

Attribution. The per-example NLL change ``d_i = l_i(z_duo) - l_i(z_ts)`` is split
between ``R`` and ``S`` by a two-player Shapley value::

    phi_R = 1/2 [l(z_ts + R) - l(z_ts)] + 1/2 [l(z_ts + R + S) - l(z_ts + S)]
    phi_S = 1/2 [l(z_ts + S) - l(z_ts)] + 1/2 [l(z_ts + R + S) - l(z_ts + R)]

With two players this is exact and cheap: ``phi_R + phi_S = d_i`` identically,
four NLL evaluations per example, no approximation and nothing to tune.
Integrated gradients along the straight path is available as a cross-check; it
is also exact and should agree to quadrature error.

What the two accounts predict. Under pure smoothing, ``phi_S`` carries little of
the total improvement, is uncorrelated with correctness, and points toward the
true class only at chance. Every statistic here is therefore also computed under
the class-permuted null, which is the smoothness-matched reference.
"""
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from ..utils.duo_fit import fit_component_scales
from ..utils.preds_io import DuoPreds
from .fusion_variants import class_permuted
from .metrics_panel import per_example_nll


def _centred(z: np.ndarray) -> np.ndarray:
    return z - z.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Exact two-player Shapley attribution
# ---------------------------------------------------------------------------

def shapley_split(z_ref: np.ndarray, term_r: np.ndarray, term_s: np.ndarray,
                  labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Exactly split the per-example NLL change between two additive terms."""
    l_none = per_example_nll(z_ref, labels)
    l_r = per_example_nll(z_ref + term_r, labels)
    l_s = per_example_nll(z_ref + term_s, labels)
    l_both = per_example_nll(z_ref + term_r + term_s, labels)

    phi_r = 0.5 * ((l_r - l_none) + (l_both - l_s))
    phi_s = 0.5 * ((l_s - l_none) + (l_both - l_r))
    return {'phi_r': phi_r, 'phi_s': phi_s, 'delta': l_both - l_none,
            'nll_ref': l_none, 'nll_duo': l_both}


def integrated_gradients_split(z_ref: np.ndarray, term_r: np.ndarray, term_s: np.ndarray,
                               labels: np.ndarray, n_steps: int = 64) -> Dict[str, np.ndarray]:
    """Path-integral attribution along the straight line from ``z_ref`` to the Duo.

    Also exact up to quadrature error; used only to confirm the Shapley split.
    """
    rows = np.arange(len(labels))
    total = term_r + term_s
    acc_r = np.zeros(len(labels))
    acc_s = np.zeros(len(labels))

    for step in range(n_steps):
        t = (step + 0.5) / n_steps
        z = z_ref + t * total
        z = z - z.max(axis=1, keepdims=True)
        probs = np.exp(z)
        probs /= probs.sum(axis=1, keepdims=True)
        # d l_i / d z = p - onehot(y)
        grad = probs.copy()
        grad[rows, labels] -= 1.0
        acc_r += (grad * term_r).sum(axis=1) / n_steps
        acc_s += (grad * term_s).sum(axis=1) / n_steps

    return {'ig_phi_r': acc_r, 'ig_phi_s': acc_s}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _summarise(split: Dict[str, np.ndarray], term_s: np.ndarray,
               labels: np.ndarray, correct: np.ndarray, prefix: str = '') -> Dict[str, float]:
    """Aggregate a per-example split into the numbers that go in the table."""
    phi_r, phi_s, delta = split['phi_r'], split['phi_s'], split['delta']
    total_delta = float(delta.sum())

    # Does the sidekick term push probability toward the true class?
    rows = np.arange(len(labels))
    directional = term_s[rows, labels] - term_s.mean(axis=1)

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        if np.all(a == a[0]) or np.all(b == b[0]):
            return float('nan')
        return float(spearmanr(a, b).statistic)

    return {
        f'{prefix}total_delta_nll': total_delta,
        f'{prefix}phi_r_sum': float(phi_r.sum()),
        f'{prefix}phi_s_sum': float(phi_s.sum()),
        f'{prefix}sidekick_share_of_gain': float(phi_s.sum() / total_delta) if total_delta != 0 else float('nan'),
        f'{prefix}rescaling_share_of_gain': float(phi_r.sum() / total_delta) if total_delta != 0 else float('nan'),
        f'{prefix}phi_s_mean': float(phi_s.mean()),
        f'{prefix}phi_r_mean': float(phi_r.mean()),
        f'{prefix}phi_s_helps_rate': float((phi_s < 0).mean()),
        f'{prefix}directional_correct_rate': float((directional > 0).mean()),
        f'{prefix}directional_mean': float(directional.mean()),
        f'{prefix}spearman_phi_s_correct': _spearman(phi_s, correct.astype(float)),
        f'{prefix}spearman_phi_r_correct': _spearman(phi_r, correct.astype(float)),
        f'{prefix}spearman_directional_correct': _spearman(directional, correct.astype(float)),
    }


def run_decomposition(val: DuoPreds, test: DuoPreds, n_null_draws: int = 20,
                      seed: int = 0, with_integrated_gradients: bool = True,
                      **fit_kwargs) -> Dict[str, float]:
    """Decompose the Duo's NLL change on test, with a class-permuted null."""
    ts_fit = fit_component_scales([val.base.logits], val.labels, **fit_kwargs)
    duo_fit = fit_component_scales([val.base.logits, val.sidekick.logits], val.labels, **fit_kwargs)
    a_ts = float(ts_fit.scales[0])
    alpha, beta = float(duo_fit.scales[0]), float(duo_fit.scales[1])

    labels = test.labels
    z_b = _centred(test.base.logits)
    z_s = _centred(test.sidekick.logits)

    z_ref = a_ts * z_b
    term_r = (alpha - a_ts) * z_b
    term_s = beta * z_s
    correct = (z_ref + term_r + term_s).argmax(axis=1) == labels

    split = shapley_split(z_ref, term_r, term_s, labels)
    out: Dict[str, float] = {
        'alpha': alpha, 'beta': beta, 'alpha_ts': a_ts,
        'residual_rescale': alpha - a_ts,
        'ts_fit_success': ts_fit.success, 'duo_fit_success': duo_fit.success,
    }
    out.update(_summarise(split, term_s, labels, correct))

    if with_integrated_gradients:
        ig = integrated_gradients_split(z_ref, term_r, term_s, labels)
        out['ig_phi_s_sum'] = float(ig['ig_phi_s'].sum())
        out['ig_phi_r_sum'] = float(ig['ig_phi_r'].sum())
        # The Shapley split is exact by construction, so this is a check on the
        # quadrature, not on the attribution. Reported relative to the mean
        # absolute NLL change, since an absolute residual is not interpretable
        # on its own.
        abs_diff = np.abs((ig['ig_phi_s'] + ig['ig_phi_r']) - split['delta'])
        scale = float(np.mean(np.abs(split['delta'])))
        out['ig_vs_shapley_max_abs_diff'] = float(abs_diff.max())
        out['ig_vs_shapley_rel_diff'] = float(abs_diff.max() / scale) if scale > 0 else float('nan')

    # Same decomposition against a smoothness-matched, uninformative sidekick.
    if n_null_draws > 0:
        rng = np.random.default_rng(seed)
        null_rows: List[Dict[str, float]] = []
        for _ in range(n_null_draws):
            fake_val = class_permuted(val.sidekick.logits, rng)
            fake_test = _centred(class_permuted(test.sidekick.logits, rng))
            null_fit = fit_component_scales([val.base.logits, fake_val], val.labels, **fit_kwargs)
            n_alpha, n_beta = float(null_fit.scales[0]), float(null_fit.scales[1])

            null_r = (n_alpha - a_ts) * z_b
            null_s = n_beta * fake_test
            null_correct = (z_ref + null_r + null_s).argmax(axis=1) == labels
            null_split = shapley_split(z_ref, null_r, null_s, labels)
            summary = _summarise(null_split, null_s, labels, null_correct)
            summary['beta'] = n_beta
            null_rows.append(summary)

        # Prefixed with "null_" rather than "null::" on purpose: "::" is the
        # split separator in the tidy schema, and a split literally named
        # "null" round-trips through CSV as NaN.
        for key in null_rows[0]:
            values = np.array([r[key] for r in null_rows], dtype=float)
            out[f'null_{key}_mean'] = float(np.nanmean(values))
            out[f'null_{key}_sd'] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float('nan')

    return out


def per_example_frame(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> Dict[str, np.ndarray]:
    """Per-example decomposition arrays, for callers that want the raw values."""
    ts_fit = fit_component_scales([val.base.logits], val.labels, **fit_kwargs)
    duo_fit = fit_component_scales([val.base.logits, val.sidekick.logits], val.labels, **fit_kwargs)
    a_ts = float(ts_fit.scales[0])
    alpha, beta = float(duo_fit.scales[0]), float(duo_fit.scales[1])

    z_b = _centred(test.base.logits)
    z_s = _centred(test.sidekick.logits)
    z_ref = a_ts * z_b
    term_r = (alpha - a_ts) * z_b
    term_s = beta * z_s

    split = shapley_split(z_ref, term_r, term_s, test.labels)
    rows = np.arange(len(test.labels))
    return {
        'idx': test.base.idx,
        'label': test.labels,
        **split,
        'directional': term_s[rows, test.labels] - term_s.mean(axis=1),
        'base_correct': (test.base.logits.argmax(axis=1) == test.labels),
        'sidekick_correct': (test.sidekick.logits.argmax(axis=1) == test.labels),
    }
