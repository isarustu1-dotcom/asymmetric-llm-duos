"""BLoB (Bayesian Low-rank Adaptation by Backpropagation) model wrapper.

Replaces every lora_A ``nn.Linear`` inside a PEFT model with a ``BLoBLoRAA``
module that carries both mean (mu) and standard-deviation (rho) parameters,
implements the reparameterisation trick in ``forward()``, and exposes a
closed-form Gaussian KL helper.

Reference
---------
Wang et al. (2024) "BLoB: Bayesian Low-Rank Adaptation by Backpropagation"
https://arxiv.org/abs/2406.11675
Official implementation: https://github.com/Wang-ML-Lab/bayesian-peft
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Bayesian lora_A layer
# ---------------------------------------------------------------------------

class BLoBLoRAA(nn.Module):
    """Bayesian LoRA-A layer: mean + standard-deviation parameters.

    Replaces a standard ``nn.Linear`` that lives inside the ``lora_A``
    ``ModuleDict`` of a PEFT ``LoraLayer``.  Every call to ``forward`` samples
    a fresh weight matrix via the reparameterisation trick.

    Two variance parameterisations are supported (controlled by *bayes_eps*):

    * ``bayes_eps >= 0``:  sigma = rho²,
      rho initialised ~ Uniform[eps/√2, eps]
    * ``bayes_eps < 0``:   sigma = softplus(rho),
      rho initialised ~ Uniform[eps - 1, eps]

    The default (``bayes_eps = 0.05``) uses the squared parameterisation,
    which is numerically safer under bfloat16.

    Args:
        in_features:  Number of input features (same as the replaced Linear).
        out_features: Number of output features (= LoRA rank r).
        mu_init:      Initial value for the mean — copy of the original weight.
        bayes_eps:    Variance initialisation scale; sign selects the sigma
                      parameterisation.
    """

    def __init__(self, in_features: int, out_features: int,
                 mu_init: Tensor, bayes_eps: float) -> None:
        super().__init__()
        self.bayes_eps = bayes_eps
        self.mu = nn.Parameter(mu_init.clone())

        if bayes_eps < 0.0:
            lo, hi = float(bayes_eps) - 1.0, float(bayes_eps)
        else:
            lo, hi = float(bayes_eps) / math.sqrt(2.0), float(bayes_eps)

        rho_init = torch.empty_like(mu_init).uniform_(lo, hi)
        self.rho = nn.Parameter(rho_init)

    # ------------------------------------------------------------------
    def _sigma(self) -> Tensor:
        """Compute the standard deviation from rho."""
        if self.bayes_eps >= 0.0:
            return torch.clamp(self.rho, min=1e-4) ** 2
        return F.softplus(self.rho)

    def forward(self, x: Tensor) -> Tensor:
        """Sample a weight and apply the linear transformation."""
        sigma = self._sigma()
        w = self.mu + sigma * torch.randn_like(self.mu)
        return F.linear(x, w)

    def kl(self, prior_var: float) -> Tensor:
        """Closed-form KL( N(mu, sigma²) ‖ N(0, prior_var) ).

        KL = 0.5 · Σ[ sigma² / prior_var
                       + mu²   / prior_var
                       − 1
                       + log(prior_var)
                       − 2 · log(sigma) ]
        """
        sigma = self._sigma()
        return 0.5 * (
            sigma.pow(2) / prior_var
            + self.mu.pow(2) / prior_var
            - 1.0
            + math.log(prior_var)
            - 2.0 * torch.log(sigma)
        ).sum()


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------

def inject_blob_params(model: nn.Module, bayes_eps: float) -> None:
    """Walk a PEFT model and replace every lora_A Linear with BLoBLoRAA.

    The replacement is done in-place by assigning to
    ``module.lora_A[adapter_name]`` on each ``LoraLayer``.  Because
    ``lora_A`` is an ``nn.ModuleDict``, the new ``BLoBLoRAA`` is
    automatically registered in the parameter tree.

    Args:
        model:     A ``PeftModel`` (or any model containing PEFT LoRA layers).
        bayes_eps: Variance initialisation scale; sign selects parameterisation.

    Raises:
        ImportError:  If PEFT >= 0.6 is not available.
        RuntimeError: If no lora_A layers are found (model not built with
                      use_peft=True).
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:
        raise ImportError(
            'peft >= 0.6 is required for BLoB '
            '(import peft.tuners.lora.layer.LoraLayer failed).') from exc

    replaced = 0
    for module in model.modules():
        if not (isinstance(module, LoraLayer) and hasattr(module, 'lora_A')):
            continue
        for adapter_name, linear in list(module.lora_A.items()):
            if not isinstance(linear, nn.Linear):
                continue  # already replaced or unexpected type — skip
            blob_layer = BLoBLoRAA(
                in_features=linear.in_features,
                out_features=linear.out_features,
                mu_init=linear.weight.data,
                bayes_eps=bayes_eps,
            )
            blob_layer = blob_layer.to(device=linear.weight.device,
                                       dtype=linear.weight.dtype)
            module.lora_A[adapter_name] = blob_layer
            replaced += 1

    if replaced == 0:
        raise RuntimeError(
            'inject_blob_params: no lora_A nn.Linear layers were found. '
            'Make sure the model was built with use_peft=True and that PEFT '
            'LoRA adapters are active.')
    print(f'[blob] Replaced {replaced} lora_A layers with BLoBLoRAA '
          f'(bayes_eps={bayes_eps}).')


def compute_total_kl(model: nn.Module, prior_var: float) -> Tensor:
    """Sum KL divergences over all BLoBLoRAA layers in *model*.

    Args:
        model:     Model whose BLoBLoRAA layers to aggregate.
        prior_var: Prior variance σ² for all layers (= blob_cfg.bayes_beta).

    Raises:
        RuntimeError: If no BLoBLoRAA layers are found.
    """
    total: Tensor | None = None
    for module in model.modules():
        if isinstance(module, BLoBLoRAA):
            kl = module.kl(prior_var)
            total = kl if total is None else total + kl
    if total is None:
        raise RuntimeError(
            'compute_total_kl: no BLoBLoRAA layers found. '
            'Call inject_blob_params before training.')
    return total


def get_blob_param_groups(
        model: nn.Module) -> Tuple[List[Tensor], List[Tensor]]:
    """Separate trainable parameters into (mu_params, rho_params).

    All trainable parameters that are *not* a ``BLoBLoRAA.rho`` go into
    *mu_params* — this includes lora_B weights and any other fine-tuned
    parameters.  ``rho`` parameters are updated by the KL optimizer at a
    separate (typically higher) learning rate.

    Returns:
        Tuple of (mu_params, rho_params), each a list of ``Tensor``.
    """
    rho_ids: set = set()
    rho_params: List[Tensor] = []

    for module in model.modules():
        if isinstance(module, BLoBLoRAA):
            rho_params.append(module.rho)
            rho_ids.add(id(module.rho))

    mu_params: List[Tensor] = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in rho_ids
    ]
    return mu_params, rho_params
