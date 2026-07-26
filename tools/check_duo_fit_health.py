"""Check whether the stored duo scales were actually fitted.

Background
----------
``save_predictions`` writes logits as ``np.float32``. The duo optimizer used to
minimise NLL with L-BFGS-B and finite-difference gradients, with no cast. At the
default step ``eps = 1.49e-8`` a float32 objective changes by less than its own
rounding error, so the finite difference evaluates to exactly zero, the
optimiser sees a zero gradient, and it terminates at its starting point --
returning ``alpha = beta = 1`` without fitting anything.
``temperature_scaling._nll_and_grad`` documents the same hazard and avoids it by
casting to float64 and supplying an analytic gradient.

This script reads the scales stored in ``metrics/<dataset>.npz`` for each seed,
refits them properly in float64 with the analytic gradient, and reports both.

    stored alpha = beta = 1.0000 exactly  ->  the stored fit stalled
    refit NLL clearly below stored NLL    ->  the stalled fit cost real accuracy

Usage
-----
python tools/check_duo_fit_health.py \\
    --result-dirs data/results/*/qwen2_7b_qwen2_15b/lora_tempscaling_duo_unconstrained \\
    --seeds 1 2 3 4 5
"""
import os.path as osp
from argparse import ArgumentParser

import numpy as np

from asym_duos.utils.duo_fit import fit_duo, legacy_kwargs, nll
from asym_duos.utils.preds_io import load_duo_experiment, load_stored_duo_params


def parse_args():
    p = ArgumentParser('Check whether stored duo scales were actually fitted.')
    p.add_argument('--result-dirs', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--npz-name', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    header = (f'{"dataset":>10} {"seed":>5} {"stored a":>9} {"stored b":>9} '
              f'{"refit a":>9} {"refit b":>9} {"legacy stall":>13} '
              f'{"val NLL stored":>15} {"val NLL refit":>14}')
    print(header)
    print('-' * len(header))

    n_stalled = 0
    n_total = 0
    for result_dir in args.result_dirs:
        dataset = osp.normpath(result_dir).split(osp.sep)[-3]
        npz_name = args.npz_name or f'{dataset}.npz'

        for seed in args.seeds:
            try:
                val, _ = load_duo_experiment(result_dir, npz_name, seed)
            except (FileNotFoundError, KeyError):
                continue
            n_total += 1

            stored = load_stored_duo_params(result_dir, npz_name, seed)
            a_stored = stored.get('duo_base_scale', float('nan'))
            b_stored = stored.get('duo_sidekick_scale', float('nan'))

            components = [val.base.logits, val.sidekick.logits]
            good = fit_duo(val.base.logits, val.sidekick.logits, val.labels)
            legacy = fit_duo(val.base.logits.astype(np.float32),
                             val.sidekick.logits.astype(np.float32),
                             val.labels, **legacy_kwargs())

            nll_stored = (nll([a_stored, b_stored], components, val.labels)
                          if np.isfinite(a_stored) and np.isfinite(b_stored) else float('nan'))
            nll_refit = nll(good.scales, components, val.labels)

            if legacy.stalled:
                n_stalled += 1

            print(f'{dataset:>10} {seed:>5} {a_stored:>9.4f} {b_stored:>9.4f} '
                  f'{good.scales[0]:>9.4f} {good.scales[1]:>9.4f} '
                  f'{("STALLED" if legacy.stalled else "ok"):>13} '
                  f'{nll_stored:>15.6f} {nll_refit:>14.6f}')

    print()
    if n_total == 0:
        print('No experiments found -- check --result-dirs and --seeds.')
        return
    print(f'{n_stalled}/{n_total} runs reproduce the stall under the legacy numerics.')
    if n_stalled:
        print('Stored scales of exactly 1.0000 confirm the published duo used an '
              'unweighted logit sum rather than a fitted one.')


if __name__ == '__main__':
    main()
