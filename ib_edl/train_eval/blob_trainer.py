"""BLoB trainer: dual-optimizer ELBO training for Bayesian LoRA.

Implements the BLoB (Bayesian Low-rank Adaptation by Backpropagation)
training loop inside HuggingFace Trainer.  Two Adam instances are used:

* ``mu_optimizer``  — updates all trainable parameters *except* rho at the
  main learning rate (NLL + KL gradient).
* ``rho_optimizer`` — updates only the BLoBLoRAA.rho variance parameters at
  the dedicated KL learning rate ``blob_cfg.bayes_kllr``.

A ``_DualOptimizer`` wrapper makes both optimizers look like a single one to
the Trainer machinery (schedulers, logging, checkpointing).

Reference
---------
Wang et al. (2024) "BLoB: Bayesian Low-Rank Adaptation by Backpropagation"
https://arxiv.org/abs/2406.11675
"""
from typing import Any, Dict, Optional, Tuple, Union

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer

from ..utils.blob import compute_total_kl, get_blob_param_groups


# ---------------------------------------------------------------------------
# Dual-optimizer wrapper
# ---------------------------------------------------------------------------

class _DualOptimizer:
    """Exposes a single optimizer interface over two Adam instances.

    ``param_groups`` and ``state`` delegate to ``mu_optimizer`` so that
    HuggingFace Trainer's LR-scheduler and logging work without change.
    ``step()`` calls both optimizers sequentially.
    """

    def __init__(self, mu_optimizer: torch.optim.Optimizer,
                 rho_optimizer: torch.optim.Optimizer) -> None:
        self.mu_optimizer = mu_optimizer
        self.rho_optimizer = rho_optimizer
        # HF Trainer reads param_groups[0]['lr'] for logging and for the
        # scheduler.  Delegating to mu_optimizer means the cosine schedule
        # modifies mu LR while rho LR stays constant.
        self.param_groups = mu_optimizer.param_groups
        self.state = mu_optimizer.state
        self.defaults = mu_optimizer.defaults

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.mu_optimizer.zero_grad(set_to_none=set_to_none)
        self.rho_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.mu_optimizer.step(closure)
        self.rho_optimizer.step(closure)
        return loss

    def state_dict(self) -> dict:
        return {
            'mu': self.mu_optimizer.state_dict(),
            'rho': self.rho_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if 'mu' in state_dict and 'rho' in state_dict:
            self.mu_optimizer.load_state_dict(state_dict['mu'])
            self.rho_optimizer.load_state_dict(state_dict['rho'])
        else:
            # Fallback: treat as mu optimizer state (e.g. legacy checkpoint)
            self.mu_optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# BLoBTrainer
# ---------------------------------------------------------------------------

class BLoBTrainer(Trainer):
    """HuggingFace Trainer subclass for BLoB fine-tuning.

    Differences from ``FTTrainer``:
    * Uses a ``_DualOptimizer`` (two Adam instances, different LRs for mu
      and rho parameters).
    * ``compute_loss`` averages logits over ``bayes_train_n_samples`` MC
      draws and adds the KL penalty scaled by π = 1 / (N · γ).
    * ``prediction_step`` averages logits over ``_blob_eval_n_samples`` draws.
      Set ``trainer._blob_eval_n_samples = blob_cfg.bayes_eval_n_samples_final``
      before the final ``trainer.predict()`` calls for accurate calibration.

    Args:
        cfg:                 mmengine Config (tokenizer_run_cfg, etc.).
        target_ids:          Long tensor of answer-token IDs, shape (C,).
        blob_cfg:            Config node with all bayes_* hyperparameters.
        train_dataset_size:  |D_train|, used to compute π.
        **kwargs:            Forwarded to ``Trainer.__init__``.
    """

    def __init__(self, cfg: mmengine.Config, target_ids: torch.LongTensor,
                 blob_cfg, train_dataset_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.target_ids = target_ids.to(self.args.device)
        self.blob_cfg = blob_cfg
        # KL weight: π = 1 / (N · γ)  (smaller for larger datasets)
        self.kl_weight = 1.0 / (train_dataset_size * float(blob_cfg.bayes_gamma))
        # Overridden before final trainer.predict() calls for MC accuracy.
        self._blob_eval_n_samples: int = int(blob_cfg.bayes_eval_n_samples)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def create_optimizer(self) -> _DualOptimizer:
        """Build mu_optimizer and rho_optimizer, wrap in _DualOptimizer."""
        mu_params, rho_params = get_blob_param_groups(self.model)
        mu_opt = torch.optim.Adam(
            mu_params,
            lr=self.args.learning_rate,
            eps=getattr(self.args, 'adam_epsilon', 1e-8),
        )
        rho_opt = torch.optim.Adam(
            rho_params,
            lr=float(self.blob_cfg.bayes_kllr),
        )
        self.optimizer = _DualOptimizer(mu_opt, rho_opt)
        return self.optimizer

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """ELBO = NLL + kl_weight · KL.

        Averages logits over ``bayes_train_n_samples`` MC draws.  Each draw
        samples new weights inside ``BLoBLoRAA.forward``.
        """
        prompts = inputs.pop('prompts')
        labels  = inputs.pop('labels').to(self.args.device)
        tok = self.processing_class(
            prompts, **self.cfg.tokenizer_run_cfg).to(self.args.device)

        n_samples = int(self.blob_cfg.bayes_train_n_samples)
        logits_acc: Optional[Tensor] = None
        for _ in range(n_samples):
            out = model(**tok)
            l = out.logits[:, -1, self.target_ids]
            logits_acc = l if logits_acc is None else logits_acc + l
        logits = logits_acc / n_samples  # type: ignore[operator]

        nll  = F.cross_entropy(logits, labels)
        kl   = compute_total_kl(model, float(self.blob_cfg.bayes_beta))
        loss = nll + self.kl_weight * kl

        outputs = {'loss': loss, 'logits': logits}
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool = False,
        ignore_keys=None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """MC-averaged prediction step.

        Uses ``self._blob_eval_n_samples`` draws.  Set this attribute to
        ``blob_cfg.bayes_eval_n_samples_final`` (default 10) before calling
        ``trainer.predict()`` for accurate final evaluation.
        """
        inputs   = self._prepare_inputs(inputs)
        prompts  = inputs.pop('prompts')
        labels   = inputs.pop('labels').to(self.args.device)
        tok = self.processing_class(
            prompts, **self.cfg.tokenizer_run_cfg).to(self.args.device)

        n_samples = self._blob_eval_n_samples
        model.eval()
        with torch.no_grad():
            logits_list = []
            for _ in range(n_samples):
                out = model(**tok)
                logits_list.append(out.logits[:, -1, self.target_ids])
            logits = torch.stack(logits_list, dim=0).mean(dim=0)

        loss = F.cross_entropy(logits, labels)
        return (loss.detach(), logits.detach(), labels.detach())
