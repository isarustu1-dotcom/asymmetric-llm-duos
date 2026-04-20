"""BLoB (Bayesian Low-rank Adaptation by Backpropagation) model utilities.

Follows the original implementation closely:
  Wang et al. (2024) "BLoB: Bayesian Low-Rank Adaptation by Backpropagation"
  https://arxiv.org/abs/2406.11675
  Official code: https://github.com/Wang-ML-Lab/bayesian-peft

Architecture
------------
``lora_A`` (an ``nn.Linear``) is left **untouched**.  A ``lora_A_rho``
``nn.ParameterDict`` is added alongside it to hold the variance parameters.
The LoRA layer's ``forward()`` is monkey-patched with ``blob_linear_forward``
so that Bayesian noise is injected via local reparameterisation when
``layer.blobsample is True``.

Two separate ``torch.optim`` instances (mu and rho) are used at different
learning rates, wrapped in ``_DualOptimizer`` for HuggingFace Trainer
compatibility.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Standalone functions — monkey-patched onto each LoRA layer
# ---------------------------------------------------------------------------

def blob_linear_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
    """Replacement ``forward`` for PEFT ``Linear`` LoRA layers.

    Performs the standard LoRA forward (base + lora_B(lora_A(dropout(x)))),
    then — when ``self.blobsample`` is ``True`` — adds Bayesian noise via
    local reparameterisation using the learned ``lora_A_rho`` parameters.
    This matches the original bayesian-peft ``blob_linear_forward`` exactly.
    """
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        # Standard LoRA path — identical to PEFT's original forward.
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A  = self.lora_A[active_adapter]
            lora_B  = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling

    # BLoB noise injection via local reparameterisation.
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        if self.blobsample:
            if self.bayes_eps < 0:
                A_sigma = torch.log1p(
                    torch.exp(self.lora_A_rho[active_adapter]))
            else:
                A_sigma = self.lora_A_rho[active_adapter] ** 2

            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            x = x.to(lora_A.weight.dtype)

            # Random ±1 sign vectors for local reparameterisation.
            if x.dim() == 2:
                r_A = (
                    torch.ones(
                        (x.size(0), self.in_features),
                        device=x.device, dtype=x.dtype,
                    ).uniform_(-1, 1).sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), self.r[active_adapter]),
                        device=x.device, dtype=x.dtype,
                    ).uniform_(-1, 1).sign()
                )
            else:
                r_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.in_features),
                        device=x.device, dtype=x.dtype,
                    ).uniform_(-1, 1).sign()
                )
                s_A = (
                    torch.ones(
                        (x.size(0), x.size(1), self.r[active_adapter]),
                        device=x.device, dtype=x.dtype,
                    ).uniform_(-1, 1).sign()
                )

            x = dropout(x)
            lora_noise_a = A_sigma * torch.randn_like(lora_A.weight)
            noise = (
                ((x * r_A) @ lora_noise_a.transpose(0, 1)) * s_A
            ) @ self.lora_B[active_adapter].weight.transpose(0, 1)
            result = result + noise * scaling

        result = result.to(previous_dtype)

    return result


def div_posterior_prior(self) -> Tensor:
    """KL( N(mu, sigma²) ‖ N(0, bayes_beta²) ) summed over this layer's adapters.

    Uses the numerically stable form from the original repo, computing in
    float64 to avoid bfloat16 underflow.
    """

    def kl_div_stable(
        mu_q: Tensor, sigma_q: Tensor, mu_p: float, sigma_p: float,
    ) -> Tensor:
        eps = 1e-6
        return (
            math.log(sigma_p + eps)
            - torch.log(sigma_q.to(torch.float64) + eps)
            + (
                sigma_q.to(torch.float64) ** 2
                + (mu_q.to(torch.float64) - mu_p) ** 2
            ) / (2.0 * sigma_p ** 2 + eps)
            - 0.5
        ).sum()

    kl: Tensor = torch.tensor(0.0)
    for active_adapter in self.active_adapters:
        if self.bayes_eps < 0:
            sigma_weight = torch.log1p(
                torch.exp(self.lora_A_rho[active_adapter]))
        else:
            sigma_weight = self.lora_A_rho[active_adapter] ** 2
        kl = kl + kl_div_stable(
            self.lora_A[active_adapter].weight,
            sigma_weight,
            0.0,
            self.bayes_beta,
        )
    return kl


def _blob_sample(self, status: bool = True) -> None:
    """Toggle Bayesian noise sampling on this layer."""
    self.blobsample = status


# ---------------------------------------------------------------------------
# Layer wrapping helper (internal)
# ---------------------------------------------------------------------------

def _wrap_lora_layer(
    lora_layer: nn.Module, bayes_eps: float, bayes_beta: float,
) -> None:
    """Add ``lora_A_rho`` and BLoB attributes to a LoRA layer in-place.

    ``lora_A`` (the existing ``nn.Linear``) is **not replaced**.  A new
    ``nn.ParameterDict`` called ``lora_A_rho`` is registered alongside it.
    Initialisation follows the original repo exactly.
    """
    lora_layer.lora_A_rho = nn.ParameterDict({})
    lora_layer.bayes_eps  = bayes_eps
    lora_layer.bayes_beta = bayes_beta
    lora_layer.blobsample = True

    # Allocate rho tensors.
    for adapter_name in lora_layer._active_adapter:
        lora_layer.lora_A_rho[adapter_name] = nn.Parameter(
            lora_layer.lora_A[adapter_name].weight.new_zeros(
                lora_layer.r[adapter_name], lora_layer.in_features,
            )
        )

    # Initialise rho tensors.
    for adapter_name in lora_layer._active_adapter:
        if adapter_name not in lora_layer.lora_A.keys():
            continue
        if bayes_eps < 0:
            nn.init.uniform_(
                lora_layer.lora_A_rho[adapter_name],
                bayes_eps - 1.0,
                bayes_eps,
            )
        else:
            nn.init.uniform_(
                lora_layer.lora_A_rho[adapter_name],
                bayes_eps / math.sqrt(2.0),
                bayes_eps,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_blob_params(
    model: nn.Module, bayes_eps: float, bayes_beta: float,
) -> None:
    """Walk a PEFT model and patch every LoRA Linear layer for BLoB training.

    For each ``peft.tuners.lora.Linear`` layer:

    * Adds ``lora_A_rho`` — an ``nn.ParameterDict`` of variance parameters.
    * Adds scalar attributes ``bayes_eps``, ``bayes_beta``, ``blobsample``.
    * Monkey-patches ``forward`` → ``blob_linear_forward``.
    * Monkey-patches ``div_posterior_prior`` → KL computation.
    * Monkey-patches ``sample`` → blobsample toggle.

    ``lora_A`` (the original ``nn.Linear``) is left completely untouched so
    PEFT internals that read ``lora_A.weight.dtype`` continue to work.

    Args:
        model:      A ``PeftModel`` with LoRA adapters already applied.
        bayes_eps:  Variance init scale; sign selects sigma parameterisation.
        bayes_beta: Prior standard deviation σ_p (not variance).
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError as exc:
        raise ImportError(
            'peft >= 0.6 is required for BLoB '
            '(import peft.tuners.lora failed).') from exc

    patched = 0
    for module in model.modules():
        if not (isinstance(module, LoraLayer) and isinstance(module, LoraLinear)):
            continue
        _wrap_lora_layer(module, bayes_eps, bayes_beta)
        setattr(module, 'forward',
                blob_linear_forward.__get__(module, module.__class__))
        setattr(module, 'div_posterior_prior',
                div_posterior_prior.__get__(module, module.__class__))
        setattr(module, 'sample',
                _blob_sample.__get__(module, module.__class__))
        patched += 1

    if patched == 0:
        raise RuntimeError(
            'inject_blob_params: no peft.tuners.lora.Linear layers found. '
            'Ensure the model was built with use_peft=True.')
    print(f'[blob] Patched {patched} LoRA layers with BLoB '
          f'(bayes_eps={bayes_eps}, bayes_beta={bayes_beta}).')


def compute_total_kl(model: nn.Module) -> Tensor:
    """Sum KL divergences over all BLoB-patched LoRA layers in *model*.

    Args:
        model: A model whose LoRA layers have been patched by
               ``inject_blob_params``.

    Raises:
        RuntimeError: If no patched layers are found.
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:
        raise ImportError('peft >= 0.6 required.') from exc

    total: Tensor | None = None
    for module in model.modules():
        if isinstance(module, LoraLayer) and hasattr(module, 'lora_A_rho'):
            kl = module.div_posterior_prior()
            total = kl if total is None else total + kl

    if total is None:
        raise RuntimeError(
            'compute_total_kl: no BLoB-patched LoRA layers found. '
            'Call inject_blob_params before training.')
    return total


def set_blobsample(model: nn.Module, status: bool) -> None:
    """Toggle Bayesian noise sampling on every BLoB-patched LoRA layer.

    Args:
        model:  Model whose layers to toggle.
        status: ``True`` to enable sampling (training), ``False`` for MAP
                inference.
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:
        raise ImportError('peft >= 0.6 required.') from exc

    for module in model.modules():
        if isinstance(module, LoraLayer) and hasattr(module, 'blobsample'):
            module.blobsample = status


def get_blob_param_groups(
    model: nn.Module,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Separate trainable parameters into ``(mu_params, rho_params)``.

    ``rho_params``:  all ``lora_A_rho`` tensors (variance parameters).
    ``mu_params``:   every other trainable parameter (``lora_A.weight``,
                     ``lora_B.weight``, bias terms, …).

    Returns:
        Tuple ``(mu_params, rho_params)``, each a list of ``Tensor``.
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:
        raise ImportError('peft >= 0.6 required.') from exc

    rho_ids: set = set()
    rho_params: List[Tensor] = []

    for module in model.modules():
        if isinstance(module, LoraLayer) and hasattr(module, 'lora_A_rho'):
            for p in module.lora_A_rho.values():
                rho_params.append(p)
                rho_ids.add(id(p))

    mu_params: List[Tensor] = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in rho_ids
    ]
    return mu_params, rho_params
