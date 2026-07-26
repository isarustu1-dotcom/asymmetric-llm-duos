"""Paired statistics for method comparisons.

The metareview's central factual claim is that temperature scaling "performs
within 1 standard deviation of the proposed method". That standard deviation is
the *marginal* one across seeds, and it is dominated by seed-to-seed variation
in the underlying fine-tune -- variation that TS and the Duo **share**, because
both are fitted post hoc on the same checkpoints.

The comparison that respects the design is therefore the paired within-seed
difference. These helpers compute it, together with a Wilcoxon signed-rank test
and a Holm correction across datasets.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class PairedResult:
    """Summary of a paired comparison between two methods."""

    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float          # a - b
    sd_diff: float
    sd_marginal_a: float
    sd_marginal_b: float
    wilcoxon_stat: float
    p_value: float
    win_rate: float           # fraction of pairs where a < b (lower-is-better metrics)

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def paired_compare(values_a: Sequence[float], values_b: Sequence[float],
                   lower_is_better: bool = True) -> PairedResult:
    """Compare two methods evaluated on the same units (e.g. the same seeds).

    ``values_a`` and ``values_b`` must be aligned: index ``i`` is the same seed
    or the same (dataset, seed) pair in both.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f'Paired inputs must align; got {a.shape} vs {b.shape}.')

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    diff = a - b

    if len(diff) >= 2 and np.any(diff != 0):
        stat, p = wilcoxon(a, b)
    else:
        stat, p = float('nan'), float('nan')

    wins = (diff < 0) if lower_is_better else (diff > 0)
    return PairedResult(
        n_pairs=int(len(diff)),
        mean_a=float(a.mean()) if len(a) else float('nan'),
        mean_b=float(b.mean()) if len(b) else float('nan'),
        mean_diff=float(diff.mean()) if len(diff) else float('nan'),
        sd_diff=float(diff.std(ddof=1)) if len(diff) >= 2 else float('nan'),
        sd_marginal_a=float(a.std(ddof=1)) if len(a) >= 2 else float('nan'),
        sd_marginal_b=float(b.std(ddof=1)) if len(b) >= 2 else float('nan'),
        wilcoxon_stat=float(stat),
        p_value=float(p),
        win_rate=float(wins.mean()) if len(diff) else float('nan'),
    )


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    adjusted = np.full(n, np.nan)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (len(order) - rank) * p[idx]
        running = max(running, candidate)   # enforce monotonicity
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_ci(statistic: Callable[[np.ndarray], float], values: np.ndarray,
                 n_resamples: int = 1000, alpha: float = 0.05,
                 rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Percentile bootstrap CI for a statistic of per-example values."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    n = len(values)
    draws = np.empty(n_resamples)
    for r in range(n_resamples):
        draws[r] = statistic(values[rng.integers(0, n, size=n)])
    return {
        'point': float(statistic(values)),
        'ci_low': float(np.quantile(draws, alpha / 2)),
        'ci_high': float(np.quantile(draws, 1 - alpha / 2)),
    }


def paired_table(records: List[dict], variant_a: str, variant_b: str,
                 metric: str, group_key: str = 'dataset',
                 lower_is_better: bool = True) -> List[dict]:
    """Run :func:`paired_compare` per group, then Holm-correct across groups.

    ``records`` are tidy rows carrying at least ``group_key``, ``seed``,
    ``variant``, ``metric`` and ``value``. Rows are paired on ``seed`` within
    each group, and an ``ALL`` row pools every (group, seed) pair.
    """
    relevant = [r for r in records if r.get('metric') == metric]
    groups = sorted({r[group_key] for r in relevant})

    rows: List[dict] = []
    pooled_a: List[float] = []
    pooled_b: List[float] = []

    for group in groups:
        by_seed_a = {r['seed']: r['value'] for r in relevant
                     if r[group_key] == group and r['variant'] == variant_a}
        by_seed_b = {r['seed']: r['value'] for r in relevant
                     if r[group_key] == group and r['variant'] == variant_b}
        seeds = sorted(set(by_seed_a) & set(by_seed_b))
        if not seeds:
            continue

        a = [by_seed_a[s] for s in seeds]
        b = [by_seed_b[s] for s in seeds]
        pooled_a.extend(a)
        pooled_b.extend(b)

        result = paired_compare(a, b, lower_is_better=lower_is_better)
        rows.append({group_key: group, 'metric': metric,
                     'variant_a': variant_a, 'variant_b': variant_b,
                     **result.as_dict()})

    if pooled_a:
        pooled = paired_compare(pooled_a, pooled_b, lower_is_better=lower_is_better)
        rows.append({group_key: 'ALL', 'metric': metric,
                     'variant_a': variant_a, 'variant_b': variant_b,
                     **pooled.as_dict()})

    # Holm across the per-group tests only; the pooled row is not one of them.
    per_group = [r for r in rows if r[group_key] != 'ALL']
    adjusted = holm_correction([r['p_value'] for r in per_group])
    for row, p_adj in zip(per_group, adjusted):
        row['p_value_holm'] = float(p_adj)
    for row in rows:
        row.setdefault('p_value_holm', float('nan'))

    return rows
