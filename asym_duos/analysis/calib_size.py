"""Group C: sensitivity to the calibration set.

Review 1 asks how sensitive the learned coefficients and the resulting test
metrics are to the number of calibration examples. The implicit worry is that
the Duo's two parameters overfit the validation set where temperature scaling's
single parameter does not -- which is also the strongest place to defend the
method, since holding up at small calibration sizes makes the extra degree of
freedom cheap.

C1  Subset validation at proportions of its full size, refit both base TS and
    the Duo on each subset, and evaluate on the full test set. Paired within
    each draw, so the comparison is not diluted by subset-to-subset variance.

C3  Report validation and test NLL side by side for base TS and the Duo, which
    turns the generalisation gap of the extra parameter into a number.
"""
from typing import Dict, List, Sequence

import numpy as np

from ..utils.duo_fit import fit_component_scales, fuse
from ..utils.optimizer_helpers import _softmax_from_log
from ..utils.preds_io import DuoPreds
from .metrics_panel import full_panel

DEFAULT_PROPORTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_DRAWS = 10


def _fit_and_score(components_fit: List[np.ndarray], labels_fit: np.ndarray,
                   components_eval: List[np.ndarray], labels_eval: np.ndarray,
                   **fit_kwargs):
    fit = fit_component_scales(components_fit, labels_fit, **fit_kwargs)
    probs = _softmax_from_log(fuse(fit.scales, components_eval))
    return fit, full_panel(probs, labels_eval)


# ---------------------------------------------------------------------------
# C1: calibration-size sweep
# ---------------------------------------------------------------------------

def run_size_sweep(val: DuoPreds, test: DuoPreds,
                   proportions: Sequence[float] = DEFAULT_PROPORTIONS,
                   n_draws: int = DEFAULT_DRAWS, seed: int = 0,
                   **fit_kwargs) -> List[Dict[str, object]]:
    """Refit base TS and the Duo on validation subsets, score on the full test set.

    Returns one row per (proportion, draw, variant). The full-size proportion is
    drawn once, since every draw would be the same set.
    """
    rng = np.random.default_rng(seed)
    n_val = val.n
    rows: List[Dict[str, object]] = []

    test_base = [test.base.logits]
    test_duo = [test.base.logits, test.sidekick.logits]

    for proportion in proportions:
        subset_size = max(2, int(round(proportion * n_val)))
        draws = 1 if subset_size >= n_val else n_draws

        for draw in range(draws):
            if subset_size >= n_val:
                sel = np.arange(n_val)
            else:
                sel = rng.choice(n_val, size=subset_size, replace=False)

            labels_fit = val.labels[sel]
            # A subset with a single class present makes NLL degenerate.
            if len(np.unique(labels_fit)) < 2:
                continue

            variants = {
                'base_ts': ([val.base.logits[sel]], test_base),
                'duo': ([val.base.logits[sel], val.sidekick.logits[sel]], test_duo),
            }
            for variant, (components_fit, components_eval) in variants.items():
                fit, panel = _fit_and_score(components_fit, labels_fit,
                                            components_eval, test.labels, **fit_kwargs)
                row: Dict[str, object] = {
                    'variant': variant,
                    'proportion': float(proportion),
                    'n_calibration': int(subset_size),
                    'draw_id': draw,
                    'fit_success': fit.success,
                    'fit_nll': fit.nll,
                }
                for i, s in enumerate(fit.scales):
                    row[f'scale_{i}'] = float(s)
                for metric, value in panel.items():
                    row[f'test::{metric}'] = value
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# C3: fit-vs-generalisation gap
# ---------------------------------------------------------------------------

def run_generalisation_gap(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> List[Dict[str, object]]:
    """Validation and test NLL for base TS and the Duo, fitted on full validation."""
    setups = {
        'base_ts': ([val.base.logits], [test.base.logits]),
        'duo': ([val.base.logits, val.sidekick.logits], [test.base.logits, test.sidekick.logits]),
    }

    rows: List[Dict[str, object]] = []
    for variant, (components_val, components_test) in setups.items():
        fit = fit_component_scales(components_val, val.labels, **fit_kwargs)
        val_panel = full_panel(_softmax_from_log(fuse(fit.scales, components_val)), val.labels)
        test_panel = full_panel(_softmax_from_log(fuse(fit.scales, components_test)), test.labels)

        row: Dict[str, object] = {
            'variant': variant,
            'n_params': len(components_val),
            'fit_success': fit.success,
            'val::nll': val_panel['nll'],
            'test::nll': test_panel['nll'],
            'nll_gap': test_panel['nll'] - val_panel['nll'],
            'val::ece': val_panel['ece'],
            'test::ece': test_panel['ece'],
            'ece_gap': test_panel['ece'] - val_panel['ece'],
        }
        for i, s in enumerate(fit.scales):
            row[f'scale_{i}'] = float(s)
        rows.append(row)

    return rows
