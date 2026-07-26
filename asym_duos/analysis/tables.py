"""Tidy-record helpers and markdown rendering.

Numbers and presentation are kept apart on purpose: ``run_ablations.py`` writes
long-format CSVs, and ``make_ablation_tables.py`` renders them. Tables for the
response letter can then be reshaped without recomputing anything.

The long format is::

    ablation, dataset, seed, variant, split, draw_id, metric, value

Output is markdown only -- the discussion does not allow figures.
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TIDY_COLUMNS = ['ablation', 'dataset', 'seed', 'variant', 'split', 'draw_id', 'metric', 'value']

# Only these prefixes are read as a split in a "<split>::<metric>" key. Anything
# else keeps its colons and stays part of the metric name -- otherwise a metric
# such as "null::sidekick_share" would be filed under a split called "null",
# which pandas then reads back as NaN on load.
KNOWN_SPLITS = ('val', 'test')

# Metrics where a smaller number is the better one.
LOWER_IS_BETTER = {
    'nll', 'ece', 'brier', 'brier_toplabel', 'aurc', 'reliability',
    'murphy_residual', 'mean_uncertainty',
}


def to_tidy(rows: Iterable[Dict[str, object]], ablation: str, dataset: str, seed: int,
            default_split: str = 'test') -> List[Dict[str, object]]:
    """Flatten wide analysis rows into long-format records.

    Keys shaped ``"<split>::<metric>"`` are split into their two parts; any
    other scalar key becomes a metric on ``default_split``. Non-scalar and
    identifier keys are carried along as context on every emitted record.
    """
    out: List[Dict[str, object]] = []
    for row in rows:
        context = {k: v for k, v in row.items()
                   if k in ('variant', 'surrogate', 'stratum', 'rho', 'proportion',
                            'n_calibration', 'tau', 'source', 'n_params', 'n', 'fraction')}
        variant = str(row.get('variant') or row.get('stratum') or 'default')
        draw_id = row.get('draw_id', 0)

        for key, value in row.items():
            if key in ('variant', 'stratum', 'draw_id', 'surrogate', 'source'):
                continue
            if isinstance(value, (bool, np.bool_)):
                value = float(value)
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue

            split, metric = default_split, key
            if '::' in key:
                prefix, rest = key.split('::', 1)
                if prefix in KNOWN_SPLITS:
                    split, metric = prefix, rest

            record = {
                'ablation': ablation, 'dataset': dataset, 'seed': seed,
                'variant': variant, 'split': split, 'draw_id': draw_id,
                'metric': metric, 'value': float(value),
            }
            record.update({k: v for k, v in context.items() if k != 'variant'})
            out.append(record)
    return out


def write_tidy(records: Sequence[Dict[str, object]], path: str) -> pd.DataFrame:
    """Write long-format records to CSV, ordered so diffs stay readable."""
    df = pd.DataFrame(list(records))
    if df.empty:
        df = pd.DataFrame(columns=TIDY_COLUMNS)
    ordered = [c for c in TIDY_COLUMNS if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered].sort_values([c for c in ('ablation', 'dataset', 'variant', 'metric', 'seed', 'draw_id')
                                  if c in df.columns])
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _format_cell(mean: float, sd: float, precision: int = 4) -> str:
    if not np.isfinite(mean):
        return '--'
    if not np.isfinite(sd):
        return f'{mean:.{precision}f}'
    return f'{mean:.{precision}f} ± {sd:.{precision}f}'


def render_markdown(df: pd.DataFrame, index: str = 'variant', columns: str = 'dataset',
                    metric: str = 'nll', split: str = 'test',
                    precision: int = 4, index_order: Optional[Sequence[str]] = None) -> str:
    """Pivot a tidy frame into a mean ± sd markdown table.

    Aggregation runs over seeds and draws, so a cell reports the spread across
    everything that was replicated for that (variant, dataset) pair.
    """
    subset = df[(df['metric'] == metric) & (df['split'] == split)]
    if subset.empty:
        return f'_No rows for metric={metric!r}, split={split!r}._'

    grouped = subset.groupby([index, columns])['value'].agg(['mean', 'std', 'count'])
    row_labels = list(index_order) if index_order else sorted(subset[index].unique())
    row_labels = [r for r in row_labels if r in subset[index].unique()]
    col_labels = sorted(subset[columns].unique())

    lines = ['| ' + ' | '.join([index] + col_labels) + ' |',
             '|' + '|'.join(['---'] * (len(col_labels) + 1)) + '|']
    for row in row_labels:
        cells = []
        for col in col_labels:
            if (row, col) in grouped.index:
                stats = grouped.loc[(row, col)]
                cells.append(_format_cell(stats['mean'], stats['std'], precision))
            else:
                cells.append('--')
        lines.append('| ' + ' | '.join([str(row)] + cells) + ' |')

    arrow = 'lower is better' if metric in LOWER_IS_BETTER else 'higher is better'
    lines.append('')
    lines.append(f'_{metric} on {split}; mean ± sd over seeds and draws ({arrow})._')
    return '\n'.join(lines)


def render_paired(rows: Sequence[Dict[str, object]], precision: int = 4) -> str:
    """Render :func:`stats.paired_table` output as markdown."""
    if not rows:
        return '_No paired comparisons._'

    header = ['dataset', 'n', 'mean A', 'mean B', 'paired Δ (A−B)', 'sd(Δ)',
              'sd marginal A', 'sd marginal B', 'win rate', 'p', 'p (Holm)']
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']

    for row in rows:
        def _num(key: str, digits: int = precision) -> str:
            value = row.get(key, float('nan'))
            return f'{value:.{digits}f}' if isinstance(value, (int, float)) and np.isfinite(value) else '--'

        lines.append('| ' + ' | '.join([
            str(row.get('dataset', '')), str(row.get('n_pairs', '')),
            _num('mean_a'), _num('mean_b'), _num('mean_diff'), _num('sd_diff'),
            _num('sd_marginal_a'), _num('sd_marginal_b'),
            _num('win_rate', 2), _num('p_value', 4), _num('p_value_holm', 4),
        ]) + ' |')

    lines.append('')
    lines.append('_A = variant_a, B = variant_b. The paired sd is the one that respects the design; '
                 'the marginal sds are shown for contrast._')
    return '\n'.join(lines)


def render_null_comparison(df: pd.DataFrame, real_variant: str = 'duo',
                           metric: str = 'nll', split: str = 'test',
                           precision: int = 4) -> str:
    """Place the real Duo inside each surrogate's null distribution.

    Reports the null mean ± sd and the real Duo's percentile within it, per
    dataset and surrogate.
    """
    subset = df[(df['metric'] == metric) & (df['split'] == split)]
    real = subset[subset['variant'] == real_variant]
    nulls = subset[subset['variant'].astype(str).str.startswith('null::')]
    if real.empty or nulls.empty:
        return '_Need both the real duo and the null draws in the same frame._'

    header = ['dataset', 'surrogate', 'real', 'null mean ± sd', 'percentile of real', 'n draws']
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']

    for dataset in sorted(subset['dataset'].unique()):
        real_value = real[real['dataset'] == dataset]['value'].mean()
        for variant in sorted(nulls[nulls['dataset'] == dataset]['variant'].unique()):
            draws = nulls[(nulls['dataset'] == dataset) & (nulls['variant'] == variant)]['value'].values
            if len(draws) == 0:
                continue
            percentile = float((draws < real_value).mean() * 100)
            lines.append('| ' + ' | '.join([
                dataset, str(variant).replace('null::', ''),
                f'{real_value:.{precision}f}',
                _format_cell(float(np.mean(draws)), float(np.std(draws, ddof=1)) if len(draws) > 1 else np.nan,
                             precision),
                f'{percentile:.1f}%', str(len(draws)),
            ]) + ' |')

    lines.append('')
    lines.append('_A low percentile means the real duo beats almost every smoothness-matched surrogate._')
    return '\n'.join(lines)
