"""A non-redundant metric panel (ablation E4).

Review 3 objects that the paper's three reported quantities -- logit margin,
confidence ``max_k p``, and correct-class probability ``p_{i,y}`` -- are
near-monotone transforms of one another, so they measure the mass on the top
class three times over.

The metrics here are chosen to span genuinely different axes:

calibration vs discrimination
    Murphy decomposition of the top-label Brier score into reliability,
    resolution and uncertainty. Separates "better calibrated" from "better
    ordered".

distribution shape
    Predictive entropy and tail mass. For ``C > 2`` neither is a function of the
    top-class probability alone.

error ranking
    AUROC for misclassification detection and AURC. A single global rescale of
    one model's logits barely moves these, so an improvement is hard to explain
    by a smoothing account and easy to explain by added information. This is the
    load-bearing row against the "regularization-by-smoothing" reading.

selective prediction
    Accuracy at fixed coverage -- operational, and orthogonal to average
    calibration.
"""
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from ..utils.uncertainty_metrics import compute_uncertainty_metrics

DEFAULT_COVERAGES = (0.5, 0.7, 0.9)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def murphy_decomposition(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> Dict[str, float]:
    """Reliability / resolution / uncertainty split of the top-label Brier score.

    Applied to the binary problem "is the top-1 prediction correct", scored by
    the model's confidence.

    ``brier = reliability - resolution + uncertainty`` is exact only when the
    forecast is constant within each bin. With uniform bins the leftover is
    ``within-bin forecast variance - 2 * within-bin cov(forecast, outcome)``,
    which can take either sign and is reported as ``murphy_residual``. It should
    be small relative to the Brier score; a large value means the binning is too
    coarse for the forecast distribution.
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    n = len(labels)
    base_rate = float(correct.mean())

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(num_bins):
        mask = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if not np.any(mask):
            continue
        weight = mask.sum() / n
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        reliability += weight * (bin_conf - bin_acc) ** 2
        resolution += weight * (bin_acc - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    brier_top = float(np.mean((confidences - correct) ** 2))
    return {
        'brier_toplabel': brier_top,
        'reliability': float(reliability),
        'resolution': float(resolution),
        'uncertainty_murphy': float(uncertainty),
        'murphy_residual': float(brier_top - (reliability - resolution + uncertainty)),
    }


def auroc_error(probs: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for detecting misclassifications, scored by ``1 - confidence``.

    Undefined (NaN) when every prediction is correct or every one is wrong.
    """
    errors = (probs.argmax(axis=1) != labels).astype(int)
    if errors.min() == errors.max():
        return float('nan')
    return float(roc_auc_score(errors, 1.0 - probs.max(axis=1)))


def aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better).

    Examples are ordered by decreasing confidence; risk at coverage ``k/n`` is
    the error rate over the ``k`` most confident predictions.
    """
    confidences = probs.max(axis=1)
    errors = (probs.argmax(axis=1) != labels).astype(float)
    order = np.argsort(-confidences)
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def accuracy_at_coverage(probs: np.ndarray, labels: np.ndarray,
                         coverages: Sequence[float] = DEFAULT_COVERAGES) -> Dict[str, float]:
    """Accuracy over the most-confident ``coverage`` fraction of examples."""
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    order = np.argsort(-confidences)
    n = len(labels)

    out = {}
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out[f'acc_at_coverage_{int(round(cov * 100))}'] = float(correct[order[:k]].mean())
    return out


def distribution_shape(probs: np.ndarray) -> Dict[str, float]:
    """Entropy and tail-mass summaries of the predictive distribution."""
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    tail_mass = 1.0 - sorted_probs[:, 0]
    # Mass on classes ranked 3rd and below: zero iff the model is effectively
    # choosing between two options, so it is not recoverable from top-1 mass.
    deep_tail = sorted_probs[:, 2:].sum(axis=1) if probs.shape[1] > 2 else np.zeros(len(probs))
    return {
        'entropy': float(entropy.mean()),
        'tail_mass': float(tail_mass.mean()),
        'deep_tail_mass': float(deep_tail.mean()),
        'runner_up_mass': float(sorted_probs[:, 1].mean()) if probs.shape[1] > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Full panel
# ---------------------------------------------------------------------------

def full_panel(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15,
               coverages: Sequence[float] = DEFAULT_COVERAGES) -> Dict[str, float]:
    """All headline metrics plus the E4 panel, as one flat dict."""
    labels = np.asarray(labels).astype(int)
    out: Dict[str, float] = {'accuracy': float((probs.argmax(axis=1) == labels).mean())}
    out.update(compute_uncertainty_metrics(probs, labels, num_bins=num_bins))
    out.update(murphy_decomposition(probs, labels, num_bins=num_bins))
    out.update(distribution_shape(probs))
    out['auroc_error'] = auroc_error(probs, labels)
    out['aurc'] = aurc(probs, labels)
    out.update(accuracy_at_coverage(probs, labels, coverages))
    return out


def per_example_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-example NLL from unnormalised logits."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(z).sum(axis=1))
    return -(z[np.arange(len(labels)), labels] - log_norm)
