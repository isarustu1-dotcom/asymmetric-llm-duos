"""Run the post-hoc rebuttal ablations over saved prediction logits.

Nothing here retrains anything: every ablation reads the val/test logits already
written under an experiment directory, so the whole suite runs on CPU while
training jobs are busy elsewhere.

Usage
-----
python tools/run_ablations.py \\
    --result-dirs data/results/arc_c/qwen2_7b_qwen2_15b/lora_tempscaling_duo_unconstrained \\
                  data/results/csqa/qwen2_7b_qwen2_15b/lora_tempscaling_duo_unconstrained \\
    --seeds 1 2 3 4 5 \\
    --out-dir data/results/ablations \\
    --ablations all

The dataset name is taken from the result directory path
(``<root>/<dataset>/<model-tag>/<method>``) and is also the NPZ stem, unless
``--npz-name`` overrides it.

Ablations
---------
A1  fusion ladder                      A2  smoothing nulls
A3  informativeness dose-response      B   coefficient interpretation
C1  calibration-size sweep             C3  generalisation gap
D   prediction-change strata           E1  agreement cells
E2  base confidently-wrong probe       E3  per-example Shapley decomposition
F   composition with other methods
VERIFY  refit vs the scales stored during training

Each writes ``<out-dir>/<ablation>.csv`` in long format
(``ablation, dataset, seed, variant, split, draw_id, metric, value``). Render
them with ``tools/make_ablation_tables.py``.
"""
import os
import os.path as osp
import sys
import traceback
from argparse import ArgumentParser
from typing import Callable, Dict, List

from asym_duos.analysis import calib_size, coefficients, composition, decomposition
from asym_duos.analysis import fusion_variants, strata
from asym_duos.analysis.tables import to_tidy, write_tidy
from asym_duos.utils.preds_io import load_duo_experiment, load_stored_duo_params

ALL_ABLATIONS = ('A1', 'A2', 'A3', 'B', 'C1', 'C3', 'D', 'E1', 'E2', 'E3', 'F', 'VERIFY')

# Ablations worth running first: they carry the most weight in the response.
PRIORITY_ABLATIONS = ('E2', 'E3', 'A1', 'A2', 'C1')


def parse_args():
    p = ArgumentParser('Run post-hoc ablations on saved logits.')
    p.add_argument('--result-dirs', nargs='+', required=True,
                   help='Experiment directories, each containing base/ and sidekick/.')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--out-dir', required=True, help='Where the tidy CSVs are written.')
    p.add_argument('--ablations', nargs='+', default=['all'],
                   help=f'Subset of {ALL_ABLATIONS}, or "all", or "priority".')
    p.add_argument('--npz-name', default=None,
                   help='Override the NPZ filename (defaults to <dataset>.npz).')
    p.add_argument('--null-draws', type=int, default=100,
                   help='Permutation draws for the A2 nulls (default: 100).')
    p.add_argument('--dose-draws', type=int, default=10,
                   help='Draws per mixing ratio for A3 (default: 10).')
    p.add_argument('--calib-draws', type=int, default=10,
                   help='Subset draws per proportion for C1 (default: 10).')
    p.add_argument('--decomposition-null-draws', type=int, default=20,
                   help='Null draws for the E3 decomposition (default: 20).')
    p.add_argument('--rng-seed', type=int, default=0,
                   help='Base RNG seed; the run seed is mixed in per experiment.')
    p.add_argument('--legacy-numerics', action='store_true',
                   help='Fit in float32 with finite differences, as the training '
                        'pipeline did, instead of the float64 + analytic-gradient default.')
    return p.parse_args()


def dataset_from_dir(result_dir: str) -> str:
    """``<root>/<dataset>/<model-tag>/<method>`` -> ``<dataset>``."""
    parts = osp.normpath(result_dir).split(osp.sep)
    if len(parts) < 3:
        raise ValueError(f'Cannot infer dataset from {result_dir!r}; expected '
                         '<root>/<dataset>/<model-tag>/<method>.')
    return parts[-3]


def _draw_seed(base_seed: int, seed: int, tag: str) -> int:
    """Stable per-(seed, ablation) RNG seed so reruns reproduce."""
    return (base_seed * 1_000_003 + seed * 9_176 + sum(map(ord, tag))) % (2 ** 31)


def build_runners(args, fit_kwargs) -> Dict[str, Callable]:
    """Map each ablation id to ``(result_dir, npz, seed, val, test) -> rows``."""

    def a1(result_dir, npz, seed, val, test):
        return fusion_variants.run_ladder(val, test, **fit_kwargs)

    def a2(result_dir, npz, seed, val, test):
        return fusion_variants.run_nulls(
            val, test, n_draws=args.null_draws,
            seed=_draw_seed(args.rng_seed, seed, 'A2'), **fit_kwargs)

    def a3(result_dir, npz, seed, val, test):
        return fusion_variants.run_dose_response(
            val, test, n_draws=args.dose_draws,
            seed=_draw_seed(args.rng_seed, seed, 'A3'), **fit_kwargs)

    def b(result_dir, npz, seed, val, test):
        return [coefficients.run_coefficients(val, test, **fit_kwargs)]

    def c1(result_dir, npz, seed, val, test):
        return calib_size.run_size_sweep(
            val, test, n_draws=args.calib_draws,
            seed=_draw_seed(args.rng_seed, seed, 'C1'), **fit_kwargs)

    def c3(result_dir, npz, seed, val, test):
        return calib_size.run_generalisation_gap(val, test, **fit_kwargs)

    def d(result_dir, npz, seed, val, test):
        return strata.run_prediction_change(val, test, **fit_kwargs)

    def e1(result_dir, npz, seed, val, test):
        return strata.run_agreement_cells(val, test, **fit_kwargs)

    def e2(result_dir, npz, seed, val, test):
        return strata.run_confident_wrong(
            val, test, seed=_draw_seed(args.rng_seed, seed, 'E2'), **fit_kwargs)

    def e3(result_dir, npz, seed, val, test):
        return [decomposition.run_decomposition(
            val, test, n_null_draws=args.decomposition_null_draws,
            seed=_draw_seed(args.rng_seed, seed, 'E3'), **fit_kwargs)]

    def f(result_dir, npz, seed, val, test):
        return composition.run_composition(result_dir, npz, seed, **fit_kwargs)

    def verify(result_dir, npz, seed, val, test):
        stored = load_stored_duo_params(result_dir, npz, seed)
        row = fusion_variants.verify_refit(val, stored, **fit_kwargs)
        return [row] if row else []

    return {'A1': a1, 'A2': a2, 'A3': a3, 'B': b, 'C1': c1, 'C3': c3,
            'D': d, 'E1': e1, 'E2': e2, 'E3': e3, 'F': f, 'VERIFY': verify}


def main():
    args = parse_args()

    requested = args.ablations
    if 'all' in requested:
        selected = list(ALL_ABLATIONS)
    elif 'priority' in requested:
        selected = list(PRIORITY_ABLATIONS)
    else:
        unknown = [a for a in requested if a not in ALL_ABLATIONS]
        if unknown:
            raise SystemExit(f'Unknown ablation(s): {unknown}. Choose from {ALL_ABLATIONS}.')
        selected = requested

    # The analysis default is float64 with an analytic gradient; the training
    # pipeline ran float32 with finite differences. Run VERIFY to confirm the
    # two agree before quoting an ablation number next to a published one.
    fit_kwargs = {'dtype': None, 'use_grad': False} if args.legacy_numerics else {}

    runners = build_runners(args, fit_kwargs)
    os.makedirs(args.out_dir, exist_ok=True)

    records: Dict[str, List[dict]] = {name: [] for name in selected}
    failures: List[str] = []

    for result_dir in args.result_dirs:
        dataset = dataset_from_dir(result_dir)
        npz_name = args.npz_name or f'{dataset}.npz'

        for seed in args.seeds:
            try:
                val, test = load_duo_experiment(result_dir, npz_name, seed)
            except (FileNotFoundError, KeyError) as exc:
                failures.append(f'{dataset} seed {seed}: {exc}')
                continue

            print(f'[{dataset}] seed {seed}: n_val={val.n}, n_test={test.n}, '
                  f'n_classes={val.base.n_classes}')

            for name in selected:
                try:
                    rows = runners[name](result_dir, npz_name, seed, val, test)
                except Exception as exc:                                   # noqa: BLE001
                    failures.append(f'{name} / {dataset} seed {seed}: {exc}')
                    traceback.print_exc(limit=3)
                    continue
                records[name].extend(to_tidy(rows, ablation=name, dataset=dataset, seed=seed))
                print(f'    {name}: {len(rows)} rows')

    for name in selected:
        path = osp.join(args.out_dir, f'{name}.csv')
        df = write_tidy(records[name], path)
        print(f'wrote {path} ({len(df)} records)')

    if failures:
        print(f'\n{len(failures)} failure(s):', file=sys.stderr)
        for failure in failures:
            print(f'  - {failure}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
