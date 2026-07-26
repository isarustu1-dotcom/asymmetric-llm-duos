"""Post-hoc ablations and mechanistic analyses on saved prediction logits.

Nothing in this package retrains or reruns a model: every analysis reads the
val/test logits already written under an experiment directory. The groups map
onto the reviewer requests as follows.

A  ``fusion_variants``  nested fusion ladder (incl. the learned symmetric
                        control) and the smoothness-matched sidekick nulls
B  ``coefficients``     scale-normalised reading of alpha and beta
C  ``calib_size``       calibration-set proportion sweep, generalisation gap
D  ``strata``           re-ranking vs re-confidencing
E  ``strata``,          agreement cells, the base's confidently-wrong set, and
   ``decomposition``    the exact per-example Shapley split of the NLL change
F  ``composition``      raw / + base TS / + Duo for each prediction source

``stats`` holds the paired within-seed comparisons, and ``tables`` turns tidy
records into markdown.
"""
from . import (
    calib_size,
    coefficients,
    composition,
    decomposition,
    fusion_variants,
    metrics_panel,
    stats,
    strata,
    tables,
)
from .metrics_panel import full_panel, per_example_nll
from .stats import holm_correction, paired_compare, paired_table
from .tables import render_markdown, render_paired, to_tidy, write_tidy

__all__ = [
    'calib_size',
    'coefficients',
    'composition',
    'decomposition',
    'fusion_variants',
    'metrics_panel',
    'stats',
    'strata',
    'tables',
    'full_panel',
    'per_example_nll',
    'holm_correction',
    'paired_compare',
    'paired_table',
    'render_markdown',
    'render_paired',
    'to_tidy',
    'write_tidy',
]
