"""Groups D, E1 and E2: stratified analyses of where the Duo acts.

Everything here compares the Duo against **base TS**, not against the raw base.
Temperature scaling is the ``beta = 0`` member of the same family, so measuring
against the raw base would credit the Duo with a temperature move that TS
already performs for free.

D   Split the test set by whether the Duo changes the base's argmax. Separates
    re-ranking from re-confidencing, and reports each stratum's share of the
    total NLL improvement. Note that base TS never changes an argmax -- scaling
    logits by a positive constant preserves their order -- so this partition is
    identical whichever of the two is used as the reference.

E1  Cross base and sidekick correctness. A smoothing-only account predicts
    nothing distinctive in the "only sidekick correct" cell, because a
    content-free smooth signal cannot preferentially rescue the cases the base
    gets wrong and the sidekick gets right.

E2  Review 3's explicit question: on the base's confidently-wrong examples, does
    the sidekick systematically place more mass on the true class (informative
    correction), or does it merely flatten everything (smoothing)? Measured
    against both the chance level and the class-permuted null, which is the
    correctly smoothness-matched reference.
"""
from typing import Dict, List, Sequence

import numpy as np

from ..utils.duo_fit import fit_component_scales, fuse
from ..utils.optimizer_helpers import _softmax_from_log
from ..utils.preds_io import DuoPreds
from .fusion_variants import class_permuted
from .metrics_panel import full_panel, per_example_nll

DEFAULT_TAUS = (0.8, 0.9, 0.95)


def _fit_reference_and_duo(val: DuoPreds, test: DuoPreds, **fit_kwargs):
    """Fit base TS and the Duo on validation, return their test logits."""
    ts_fit = fit_component_scales([val.base.logits], val.labels, **fit_kwargs)
    duo_fit = fit_component_scales([val.base.logits, val.sidekick.logits], val.labels, **fit_kwargs)

    z_ts = fuse(ts_fit.scales, [test.base.logits])
    z_duo = fuse(duo_fit.scales, [test.base.logits, test.sidekick.logits])
    return ts_fit, duo_fit, z_ts, z_duo


def _stratum_row(name: str, mask: np.ndarray, z_ts: np.ndarray, z_duo: np.ndarray,
                 labels: np.ndarray, total_delta_nll: float) -> Dict[str, object]:
    """Metrics for one stratum, plus its share of the aggregate NLL change."""
    n_total = len(labels)
    n = int(mask.sum())
    row: Dict[str, object] = {'stratum': name, 'n': n, 'fraction': n / n_total if n_total else float('nan')}
    if n == 0:
        return row

    nll_ts = per_example_nll(z_ts, labels)
    nll_duo = per_example_nll(z_duo, labels)
    delta_here = float((nll_duo[mask] - nll_ts[mask]).sum())

    ts_panel = full_panel(_softmax_from_log(z_ts[mask]), labels[mask])
    duo_panel = full_panel(_softmax_from_log(z_duo[mask]), labels[mask])

    row.update({
        'base_ts_accuracy': ts_panel['accuracy'],
        'duo_accuracy': duo_panel['accuracy'],
        'base_ts_nll': ts_panel['nll'],
        'duo_nll': duo_panel['nll'],
        'base_ts_ece': ts_panel['ece'],
        'duo_ece': duo_panel['ece'],
        'delta_nll_sum': delta_here,
        'delta_nll_share': delta_here / total_delta_nll if total_delta_nll != 0 else float('nan'),
    })
    return row


# ---------------------------------------------------------------------------
# D: does the Duo re-rank or re-confidence?
# ---------------------------------------------------------------------------

def run_prediction_change(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> List[Dict[str, object]]:
    """Partition test by whether the Duo changes the base prediction."""
    _, _, z_ts, z_duo = _fit_reference_and_duo(val, test, **fit_kwargs)
    labels = test.labels

    pred_base = test.base.logits.argmax(axis=1)
    pred_duo = z_duo.argmax(axis=1)
    changed = pred_duo != pred_base

    base_correct = pred_base == labels
    duo_correct = pred_duo == labels

    total_delta = float((per_example_nll(z_duo, labels) - per_example_nll(z_ts, labels)).sum())

    strata = {
        'all': np.ones(len(labels), dtype=bool),
        'argmax_unchanged': ~changed,
        'argmax_changed': changed,
        'changed_wrong_to_right': changed & ~base_correct & duo_correct,
        'changed_right_to_wrong': changed & base_correct & ~duo_correct,
        'changed_wrong_to_wrong': changed & ~base_correct & ~duo_correct,
    }
    return [_stratum_row(name, mask, z_ts, z_duo, labels, total_delta) for name, mask in strata.items()]


# ---------------------------------------------------------------------------
# E1: agreement- and correctness-conditioned cells
# ---------------------------------------------------------------------------

def run_agreement_cells(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> List[Dict[str, object]]:
    """Cross base and sidekick correctness, and base/sidekick argmax agreement."""
    _, _, z_ts, z_duo = _fit_reference_and_duo(val, test, **fit_kwargs)
    labels = test.labels

    pred_base = test.base.logits.argmax(axis=1)
    pred_side = test.sidekick.logits.argmax(axis=1)
    base_correct = pred_base == labels
    side_correct = pred_side == labels
    agree = pred_base == pred_side

    total_delta = float((per_example_nll(z_duo, labels) - per_example_nll(z_ts, labels)).sum())

    strata = {
        'agree': agree,
        'disagree': ~agree,
        'both_correct': base_correct & side_correct,
        'only_base_correct': base_correct & ~side_correct,
        'only_sidekick_correct': ~base_correct & side_correct,
        'neither_correct': ~base_correct & ~side_correct,
    }
    return [_stratum_row(name, mask, z_ts, z_duo, labels, total_delta) for name, mask in strata.items()]


# ---------------------------------------------------------------------------
# E2: is the sidekick's correction informative, or just flat?
# ---------------------------------------------------------------------------

def run_confident_wrong(val: DuoPreds, test: DuoPreds,
                        taus: Sequence[float] = DEFAULT_TAUS,
                        n_null_draws: int = 100, seed: int = 0,
                        **fit_kwargs) -> List[Dict[str, object]]:
    """Probe the sidekick on the base's confidently-wrong examples.

    For each confidence threshold ``tau`` the set is
    ``O = {i : max_k p_base >= tau and argmax(z_base) != y}``. On ``O`` we ask
    whether the sidekick puts more mass on the true class than chance and than
    it puts on the base's (wrong) pick, and whether it merely flattens -- its
    entropy on ``O`` versus elsewhere.

    ``sidekick_true_mass_null`` is the same quantity under the class-permuted
    sidekick, averaged over draws. It is the reference a smoothing account
    predicts the real sidekick should match.
    """
    _, _, z_ts, z_duo = _fit_reference_and_duo(val, test, **fit_kwargs)
    labels = test.labels
    n_classes = test.base.n_classes

    probs_base = _softmax_from_log(test.base.logits)
    probs_side = _softmax_from_log(test.sidekick.logits)
    pred_base = probs_base.argmax(axis=1)
    rows_idx = np.arange(len(labels))

    # Rank of the true class under the sidekick: 1 = most probable.
    side_rank_of_true = (probs_side > probs_side[rows_idx, labels][:, None]).sum(axis=1) + 1
    entropy_side = -np.sum(probs_side * np.log(probs_side + 1e-12), axis=1)

    rng = np.random.default_rng(seed)
    null_probs = [_softmax_from_log(class_permuted(test.sidekick.logits, rng)) for _ in range(n_null_draws)]

    total_delta = float((per_example_nll(z_duo, labels) - per_example_nll(z_ts, labels)).sum())

    out: List[Dict[str, object]] = []
    for tau in taus:
        mask = (probs_base.max(axis=1) >= tau) & (pred_base != labels)
        row = _stratum_row(f'base_confident_wrong_tau_{tau:g}', mask, z_ts, z_duo, labels, total_delta)
        row['tau'] = float(tau)
        row['chance_mass'] = 1.0 / n_classes
        if int(mask.sum()) == 0:
            out.append(row)
            continue

        sel = np.where(mask)[0]
        true_mass = probs_side[sel, labels[sel]]
        base_pick_mass = probs_side[sel, pred_base[sel]]
        null_true_mass = np.mean([p[sel, labels[sel]].mean() for p in null_probs]) if null_probs else float('nan')

        row.update({
            'sidekick_true_mass': float(true_mass.mean()),
            'sidekick_true_mass_null': float(null_true_mass),
            'sidekick_mass_on_base_pick': float(base_pick_mass.mean()),
            'sidekick_prefers_truth_rate': float((true_mass > base_pick_mass).mean()),
            'sidekick_rank_of_true': float(side_rank_of_true[sel].mean()),
            'sidekick_rank_chance': (n_classes + 1) / 2,
            'sidekick_top1_correct_rate': float((probs_side[sel].argmax(axis=1) == labels[sel]).mean()),
            'sidekick_entropy_in_stratum': float(entropy_side[sel].mean()),
            'sidekick_entropy_elsewhere': float(entropy_side[~mask].mean()) if (~mask).any() else float('nan'),
        })
        out.append(row)

    return out
