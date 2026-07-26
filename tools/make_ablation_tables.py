"""Render the ablation CSVs written by ``run_ablations.py`` as markdown.

Kept separate from the numerical stage so tables for the response letter can be
reshaped without recomputing anything. Markdown only -- the discussion does not
allow figures.

Usage
-----
python tools/make_ablation_tables.py \\
    --in-dir data/results/ablations \\
    --out data/results/ablations/report.md \\
    --metrics nll ece auroc_error
"""
import os.path as osp
from argparse import ArgumentParser
from typing import List, Optional

import pandas as pd

from asym_duos.analysis.fusion_variants import LADDER_ORDER
from asym_duos.analysis.stats import paired_table
from asym_duos.analysis.tables import render_markdown, render_null_comparison, render_paired

DEFAULT_METRICS = ('nll', 'ece', 'auroc_error', 'aurc')

SECTION_TITLES = {
    'A1': 'A1 — Nested fusion ladder',
    'A2': 'A2 — Smoothness-matched sidekick nulls',
    'A3': 'A3 — Sidekick informativeness dose-response',
    'B': 'B — Scale-normalised coefficients',
    'C1': 'C1 — Calibration-set size sweep',
    'C3': 'C3 — Fit-vs-generalisation gap',
    'D': 'D — Re-ranking vs re-confidencing',
    'E1': 'E1 — Agreement- and correctness-conditioned cells',
    'E2': "E2 — The base's confidently-wrong set",
    'E3': 'E3 — Per-example Shapley decomposition',
    'F': 'F — Composition with other methods',
    'VERIFY': 'VERIFY — Refit against the scales stored during training',
}


def parse_args():
    p = ArgumentParser('Render ablation CSVs as markdown.')
    p.add_argument('--in-dir', required=True, help='Directory holding <ablation>.csv files.')
    p.add_argument('--out', required=True, help='Markdown file to write.')
    p.add_argument('--metrics', nargs='+', default=list(DEFAULT_METRICS))
    p.add_argument('--precision', type=int, default=4)
    return p.parse_args()


def _load(in_dir: str, name: str) -> Optional[pd.DataFrame]:
    path = osp.join(in_dir, f'{name}.csv')
    if not osp.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def _section(title: str, body: str) -> str:
    return f'## {title}\n\n{body}\n'


def main():
    args = parse_args()
    parts: List[str] = ['# Ablation results\n',
                        '_All numbers are post-hoc on saved logits; mean ± sd over seeds '
                        'and draws unless stated otherwise._\n']

    # --- A1: the ladder, plus the paired Duo-vs-TS comparison ---------------
    df = _load(args.in_dir, 'A1')
    if df is not None:
        body = []
        for metric in args.metrics:
            body.append(f'### {metric}\n')
            body.append(render_markdown(df, metric=metric, split='test',
                                        precision=args.precision, index_order=LADDER_ORDER))
            body.append('')
        parts.append(_section(SECTION_TITLES['A1'], '\n'.join(body)))

        paired_body = []
        for metric in args.metrics:
            records = df[df['split'] == 'test'].to_dict('records')
            rows = paired_table(records, variant_a='duo', variant_b='base_ts', metric=metric)
            paired_body.append(f'### Duo vs base TS — {metric}\n')
            paired_body.append(render_paired(rows, precision=args.precision))
            paired_body.append('')
            rows = paired_table(records, variant_a='duo', variant_b='symmetric_learned', metric=metric)
            paired_body.append(f'### Duo vs learned symmetric fusion — {metric}\n')
            paired_body.append(render_paired(rows, precision=args.precision))
            paired_body.append('')
        parts.append(_section('Paired comparisons (within seed)', '\n'.join(paired_body)))

    # --- A2: real duo placed inside each null distribution ------------------
    df_a2 = _load(args.in_dir, 'A2')
    if df_a2 is not None:
        combined = pd.concat([df_a2, df]) if df is not None else df_a2
        body = []
        for metric in args.metrics:
            body.append(f'### {metric}\n')
            body.append(render_null_comparison(combined, metric=metric, precision=args.precision))
            body.append('')
        body.append('### Fitted sidekick weight under each null\n')
        body.append(render_markdown(df_a2, metric='sidekick_weight', split='test',
                                    precision=args.precision))
        parts.append(_section(SECTION_TITLES['A2'], '\n'.join(body)))

    # --- Remaining ablations: one pivot per metric --------------------------
    simple = {
        'A3': ('nll', 'test'),
        'B': ('effective_sidekick_share', 'test'),
        'C1': ('nll', 'test'),
        'C3': ('nll_gap', 'test'),
        'D': ('delta_nll_share', 'test'),
        'E1': ('delta_nll_share', 'test'),
        'E2': ('sidekick_true_mass', 'test'),
        'E3': ('sidekick_share_of_gain', 'test'),
        'F': ('nll', 'test'),
        'VERIFY': ('beta_rel_diff', 'test'),
    }
    for name, (metric, split) in simple.items():
        frame = _load(args.in_dir, name)
        if frame is None:
            continue
        body = [render_markdown(frame, metric=metric, split=split, precision=args.precision), '']

        # A couple of ablations have a natural second view worth printing.
        extras = {
            'E2': ['sidekick_true_mass_null', 'sidekick_rank_of_true', 'sidekick_entropy_in_stratum'],
            'E3': ['rescaling_share_of_gain', 'directional_correct_rate',
                   'null_sidekick_share_of_gain_mean', 'null_directional_correct_rate_mean',
                   'spearman_phi_s_correct', 'ig_vs_shapley_rel_diff'],
            'D': ['duo_accuracy', 'fraction'],
            'E1': ['fraction', 'duo_nll'],
            'C1': ['ece'],
            'B': ['var_share_sidekick', 'standardised_sidekick_share'],
            'VERIFY': ['alpha_rel_diff'],
        }
        for extra in extras.get(name, []):
            rendered = render_markdown(frame, metric=extra, split=split, precision=args.precision)
            if not rendered.startswith('_No rows'):
                body.extend([f'**{extra}**\n', rendered, ''])

        parts.append(_section(SECTION_TITLES[name], '\n'.join(body)))

    with open(args.out, 'w') as handle:
        handle.write('\n'.join(parts))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
