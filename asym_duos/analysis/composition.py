"""Group F: how much of each method's gain is temperature scaling alone?

Review 2 asks, for the composition figure, "how much of the gains could be
achieved with TS only, i.e. what's the performance of B x TS vs B x Duo".

For every prediction source available in an experiment directory -- the plain
LoRA / IB-EDL predictions under ``base/`` and ``sidekick/``, plus Laplace-LoRA
and BLoB where they were run -- this reports three columns:

    raw          the base model's own logits
    + base TS    one fitted scale on the base
    + Duo        two fitted scales over base and sidekick

There is deliberately no ``+ TS + Duo`` column. Per-model temperature scaling
followed by a duo fit spans exactly the same family as a duo fit on raw logits,
since ``alpha * (z_b / T_b) + beta * (z_s / T_s)`` is a reparameterisation of
``alpha' * z_b + beta' * z_s``; the two are identical, as shown in the paper.
"""
from typing import Dict, List, Sequence

from ..utils.duo_fit import fit_component_scales, fuse
from ..utils.optimizer_helpers import _softmax_from_log
from ..utils.preds_io import SOURCE_SUBDIRS, DuoPreds, load_duo_experiment
from .metrics_panel import full_panel

# 'ts' is excluded on purpose: the duo fitted on temperature-scaled logits is
# the same model as the duo fitted on raw logits.
COMPOSITION_SOURCES = ('raw', 'laplace', 'blob')


def evaluate_source(val: DuoPreds, test: DuoPreds, **fit_kwargs) -> List[Dict[str, object]]:
    """The three composition columns for one prediction source."""
    rows: List[Dict[str, object]] = []

    raw_probs = _softmax_from_log(test.base.logits)
    row: Dict[str, object] = {'variant': 'raw', 'n_params': 0}
    row.update({f'test::{k}': v for k, v in full_panel(raw_probs, test.labels).items()})
    rows.append(row)

    setups = {
        'base_ts': ([val.base.logits], [test.base.logits]),
        'duo': ([val.base.logits, val.sidekick.logits], [test.base.logits, test.sidekick.logits]),
    }
    for variant, (components_val, components_test) in setups.items():
        fit = fit_component_scales(components_val, val.labels, **fit_kwargs)
        probs = _softmax_from_log(fuse(fit.scales, components_test))
        row = {'variant': variant, 'n_params': len(components_val), 'fit_success': fit.success}
        for i, s in enumerate(fit.scales):
            row[f'scale_{i}'] = float(s)
        row.update({f'test::{k}': v for k, v in full_panel(probs, test.labels).items()})
        rows.append(row)

    return rows


def run_composition(result_dir: str, npz_name: str, seed: int,
                    sources: Sequence[str] = COMPOSITION_SOURCES,
                    **fit_kwargs) -> List[Dict[str, object]]:
    """Run the composition table over every source present on disk.

    Sources that were not run for this experiment are skipped silently -- not
    every result directory has Laplace-LoRA or BLoB predictions.
    """
    rows: List[Dict[str, object]] = []
    for source in sources:
        if source not in SOURCE_SUBDIRS:
            raise ValueError(f'Unknown source {source!r}')
        try:
            val, test = load_duo_experiment(result_dir, npz_name, seed, source=source)
        except (FileNotFoundError, KeyError):
            continue
        for row in evaluate_source(val, test, **fit_kwargs):
            row['source'] = source
            row['variant'] = f'{source}::{row["variant"]}'
            rows.append(row)
    return rows
