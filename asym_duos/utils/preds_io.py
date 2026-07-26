"""Loading helpers for saved prediction NPZs.

Every post-hoc analysis reads the same on-disk layout::

    <result_dir>/<subdir>/<split>/<dataset>.npz     # keyed by str(seed)

with each seed entry holding ``idx``, ``input``, ``logits``, ``softmax_probs``,
``preds`` and ``true_labels``. Predictions are stored in whatever order the
evaluation loop produced them, so every consumer has to sort by ``idx`` before
two models can be compared example-by-example.
"""
import os.path as osp
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .optimizer_helpers import _sort_by_idx

# Where each prediction source lives, as (base_subdir, sidekick_subdir).
SOURCE_SUBDIRS: Dict[str, Tuple[str, str]] = {
    'raw': ('base', 'sidekick'),
    'ts': (osp.join('temperature_scaling', 'base'), osp.join('temperature_scaling', 'sidekick')),
    'laplace': (osp.join('laplace_lora', 'base'), osp.join('laplace_lora', 'sidekick')),
    'blob': (osp.join('blob', 'base'), osp.join('blob', 'sidekick')),
}


@dataclass
class Preds:
    """One model's predictions for one split, sorted by example index."""

    idx: np.ndarray            # (N,)
    logits: np.ndarray         # (N, C), float64
    labels: np.ndarray         # (N,), int
    inputs: Optional[np.ndarray] = None

    @property
    def n_classes(self) -> int:
        return int(self.logits.shape[1])

    def subset(self, mask: np.ndarray) -> 'Preds':
        return Preds(
            idx=self.idx[mask],
            logits=self.logits[mask],
            labels=self.labels[mask],
            inputs=None if self.inputs is None else self.inputs[mask],
        )


@dataclass
class DuoPreds:
    """Index-aligned base and sidekick predictions for one split."""

    base: Preds
    sidekick: Preds

    @property
    def labels(self) -> np.ndarray:
        return self.base.labels

    @property
    def n(self) -> int:
        return int(self.base.logits.shape[0])


def load_split(result_dir: str, subdir: str, split: str, npz_name: str, seed: int,
               with_inputs: bool = False) -> Preds:
    """Load one model's predictions for one split.

    Logits are cast to float64: they are written to disk as float32, and every
    downstream optimiser needs the extra precision (see the note in
    ``temperature_scaling._nll_and_grad``).
    """
    path = osp.join(result_dir, subdir, split, npz_name)
    if not osp.exists(path):
        raise FileNotFoundError(f'No predictions at {path}')

    archive = dict(np.load(path, allow_pickle=True))
    seed_key = str(seed)
    if seed_key not in archive:
        raise KeyError(f'Seed {seed} not found in {path} (have: {sorted(archive)})')
    data = archive[seed_key].item()

    order = np.argsort(data['idx'])
    return Preds(
        idx=_sort_by_idx(np.asarray(data['idx']), order),
        logits=_sort_by_idx(np.asarray(data['logits']), order).astype(np.float64),
        labels=_sort_by_idx(np.asarray(data['true_labels']), order).astype(int),
        inputs=_sort_by_idx(np.asarray(data['input']), order) if with_inputs else None,
    )


def load_duo_split(result_dir: str, split: str, npz_name: str, seed: int,
                   source: str = 'raw', with_inputs: bool = False) -> DuoPreds:
    """Load index-aligned base + sidekick predictions for one split.

    Args:
        source: which prediction source to read, one of ``SOURCE_SUBDIRS``.
    """
    if source not in SOURCE_SUBDIRS:
        raise ValueError(f'Unknown source {source!r}; expected one of {sorted(SOURCE_SUBDIRS)}')
    base_sub, side_sub = SOURCE_SUBDIRS[source]

    base = load_split(result_dir, base_sub, split, npz_name, seed, with_inputs=with_inputs)
    side = load_split(result_dir, side_sub, split, npz_name, seed, with_inputs=with_inputs)

    if not np.array_equal(base.idx, side.idx):
        raise ValueError(f'{split} indices of base and sidekick do not match for source={source!r}.')
    if not np.array_equal(base.labels, side.labels):
        raise ValueError(f'{split} labels of base and sidekick do not match for source={source!r}.')

    return DuoPreds(base=base, sidekick=side)


def load_duo_experiment(result_dir: str, npz_name: str, seed: int,
                        source: str = 'raw') -> Tuple[DuoPreds, DuoPreds]:
    """Load ``(val, test)`` duo predictions in one call."""
    return (
        load_duo_split(result_dir, 'val_preds', npz_name, seed, source=source),
        load_duo_split(result_dir, 'test_preds', npz_name, seed, source=source),
    )


def load_stored_duo_params(result_dir: str, npz_name: str, seed: int) -> Dict[str, float]:
    """Read the duo scales that ``optimize_weights`` persisted for this seed.

    Used to check that a fresh refit reproduces the numbers behind the paper's
    tables. Returns an empty dict when the metrics file has no duo entry.
    """
    path = osp.join(result_dir, 'metrics', npz_name)
    if not osp.exists(path):
        return {}
    archive = dict(np.load(path, allow_pickle=True))
    seed_key = str(seed)
    if seed_key not in archive:
        return {}
    entry = archive[seed_key].item()
    keys = ('duo_base_scale', 'duo_sidekick_scale', 'duo_mode',
            'base_ts_temperature', 'base_ts_scale')
    return {k: entry[k] for k in keys if k in entry}
