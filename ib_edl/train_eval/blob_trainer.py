"""BLoB trainer: dual-optimizer ELBO training for Bayesian LoRA.

Implements the BLoB (Bayesian Low-rank Adaptation by Backpropagation)
training loop inside HuggingFace Trainer, matching the original
bayesian-peft repo (Wang-ML-Lab) in every detail:

* Two optimizers:
    - ``mu_optimizer``  = ``Adam`` over every trainable parameter at the
      main learning rate (``args.learning_rate``).
    - ``rho_optimizer`` = ``SGD``  over every trainable parameter at
      ``blob_cfg.bayes_kllr``.  Both optimizers include all parameters,
      exactly as in ``bayesian-peft/modelwrappers/blob.py``; NLL has no
      gradient on ``lora_A_rho`` after the first zero_grad, and KL has
      (near-)zero gradient on the rest, so each optimizer effectively
      touches only its own partition.

* Two backward passes per step (matching original ``fit()``):
    1. forward MC-mean logits (``bayes_train_n_samples`` draws) →
       ``nll`` → ``backward(nll)`` → ``mu_optimizer.step()`` → both
       optimizers' ``zero_grad()`` → ``mu_scheduler.step()``.
    2. Compute KL averaged over ``bayes_train_n_samples`` draws →
       ``kl * pi`` → ``backward`` → ``rho_optimizer.step()`` → both
       optimizers' ``zero_grad()`` → ``rho_scheduler.step()``.

  The second zero_grad between passes is essential: Adam on µ must
  never see the KL gradient (its normalisation would blow up the
  tiny KL gradient into an lr-sized step and collapse ``lora_A``).

* ``pi = 1 / M`` where ``M = len(train_dataloader)`` (batches per
  epoch), matching the original default (``klreweighting=False``).

* ``_DualOptimizer.step()`` is a **no-op**: we step the two underlying
  optimizers manually inside ``training_step``, so the HuggingFace
  ``_inner_training_loop``'s subsequent ``self.optimizer.step()`` must
  not repeat the update with stale Adam moments.  The main
  ``lr_scheduler`` continues to work because its param_groups alias
  ``mu_optimizer.param_groups``.

Reference
---------
Wang et al. (2024) "BLoB: Bayesian Low-Rank Adaptation by Backpropagation"
https://arxiv.org/abs/2406.11675
Official code: https://github.com/Wang-ML-Lab/bayesian-peft
"""
from typing import Any, Dict, Optional, Tuple, Union

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer, get_linear_schedule_with_warmup

from ..utils.blob import compute_total_kl, set_blobsample


# ---------------------------------------------------------------------------
# Dual-optimizer wrapper
# ---------------------------------------------------------------------------

class _DualOptimizer(torch.optim.Optimizer):
    """Facade over (mu, rho) Adam+SGD pair for HuggingFace Trainer compat.

    ``step()`` is deliberately a no-op: ``BLoBTrainer.training_step``
    performs the two backward passes and calls ``mu_optimizer.step`` /
    ``rho_optimizer.step`` manually.  HF's ``_inner_training_loop``
    subsequently calls ``self.optimizer.step()``; that call must NOT
    re-apply updates with stale Adam moments, hence the no-op.

    ``param_groups``, ``state``, ``defaults`` delegate to
    ``mu_optimizer`` so the main ``lr_scheduler`` (LambdaLR / cosine)
    continues to modify µ's learning rate unchanged.
    """

    def __init__(
        self,
        mu_optimizer: torch.optim.Optimizer,
        rho_optimizer: torch.optim.Optimizer,
    ) -> None:
        # Dummy init so Optimizer bookkeeping is valid; then alias to mu.
        super().__init__([{'params': []}], {})
        self.mu_optimizer  = mu_optimizer
        self.rho_optimizer = rho_optimizer
        self.param_groups = mu_optimizer.param_groups
        self.state        = mu_optimizer.state
        self.defaults     = mu_optimizer.defaults

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.mu_optimizer.zero_grad(set_to_none=set_to_none)
        self.rho_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Intentional no-op: updates already applied in training_step.
        return None

    def state_dict(self) -> dict:
        return {
            'mu':  self.mu_optimizer.state_dict(),
            'rho': self.rho_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if 'mu' in state_dict and 'rho' in state_dict:
            self.mu_optimizer.load_state_dict(state_dict['mu'])
            self.rho_optimizer.load_state_dict(state_dict['rho'])
        else:
            self.mu_optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# BLoBTrainer
# ---------------------------------------------------------------------------

class BLoBTrainer(Trainer):
    """HuggingFace Trainer subclass for BLoB fine-tuning.

    Training procedure matches the original
    ``bayesian-peft/modelwrappers/blob.py::fit`` exactly: two separate
    backward passes, Adam on µ at ``args.learning_rate``, SGD on ρ at
    ``blob_cfg.bayes_kllr``, KL weight ``pi = 1 / len(train_loader)``.

    Args:
        cfg:                 mmengine Config (tokenizer_run_cfg, etc.).
        target_ids:          Long tensor of answer-token IDs, shape (C,).
        blob_cfg:            Config node with bayes_* hyperparameters.
        train_dataset_size:  |D_train| (kept for backward-compat; not
                             used for π in the original formulation).
        **kwargs:            Forwarded to ``Trainer.__init__``.
    """

    def __init__(
        self,
        cfg: mmengine.Config,
        target_ids: torch.LongTensor,
        blob_cfg,
        train_dataset_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg               = cfg
        self.target_ids        = target_ids.to(self.args.device)
        self.blob_cfg          = blob_cfg
        self.train_dataset_size = train_dataset_size
        # π = 1 / M, computed lazily once the train dataloader exists.
        self._pi: Optional[float] = None
        self._M:  Optional[int]   = None
        # Set to blob_cfg.bayes_eval_n_samples_final before final predict().
        self._blob_eval_n_samples: int = int(blob_cfg.bayes_eval_n_samples)
        # Built in create_scheduler.
        self.rho_scheduler = None

    # ------------------------------------------------------------------
    # Optimizers & schedulers
    # ------------------------------------------------------------------

    def create_optimizer(self) -> _DualOptimizer:
        """Adam over all trainable params (µ) + SGD over all trainable
        params (ρ), wrapped in ``_DualOptimizer``.

        Matches the original:
            opt  = Adam(<all trainable>, lr=args.learning_rate)
            opt2 = SGD ([{"params": params}], lr=args.bayes_kllr)
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        mu_opt = torch.optim.Adam(
            trainable,
            lr=self.args.learning_rate,
            eps=getattr(self.args, 'adam_epsilon', 1e-8),
        )
        rho_opt = torch.optim.SGD(
            trainable,
            lr=float(self.blob_cfg.bayes_kllr),
        )
        self.mu_optimizer  = mu_opt
        self.rho_optimizer = rho_opt
        self.optimizer = _DualOptimizer(mu_opt, rho_opt)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create the main µ scheduler (HF default) plus a matching
        linear-warmup scheduler for ρ."""
        super().create_scheduler(num_training_steps, optimizer)
        warmup = self.args.get_warmup_steps(num_training_steps)
        self.rho_scheduler = get_linear_schedule_with_warmup(
            self.rho_optimizer, warmup, num_training_steps,
        )
        return self.lr_scheduler

    # ------------------------------------------------------------------
    # Training step — two backward passes, matching original fit()
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> Tensor:
        """One optimisation step of BLoB.

        Replicates the original ``fit()`` inner loop:

            logits = forward_logits(batch, sample=True,
                                    n_samples=train_n_samples).mean(1)
            nll = CE(log_softmax(logits), labels)
            backward(nll); opt.step();  opt.zero_grad();  scheduler.step()

            kl = mean_{k=1..n}(div_posterior_prior(base_model))
            kl_div = kl * pi                # pi = 1/M
            backward(kl_div); opt2.step(); opt2.zero_grad(); scheduler2.step()
        """
        model.train()
        inputs  = self._prepare_inputs(inputs)
        prompts = inputs.pop('prompts')
        labels  = inputs.pop('labels').to(self.args.device)
        tok = self.processing_class(
            prompts, **self.cfg.tokenizer_run_cfg).to(self.args.device)

        # Lazy π = 1 / len(train_loader).
        if self._pi is None:
            self._M  = len(self.get_train_dataloader())
            self._pi = 1.0 / float(self._M)

        n_samples = int(self.blob_cfg.bayes_train_n_samples)
        set_blobsample(model, True)

        # ------------- Pass 1: NLL -------------
        logits_acc: Optional[Tensor] = None
        for _ in range(n_samples):
            out = model(**tok)
            l = out.logits[:, -1, self.target_ids]
            logits_acc = l if logits_acc is None else logits_acc + l
        logits = logits_acc / n_samples  # type: ignore[operator]
        # Match original: log_softmax + NLL (equivalent to cross_entropy).
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, labels, reduction='mean')

        self.accelerator.backward(nll)
        self.mu_optimizer.step()
        self.mu_optimizer.zero_grad(set_to_none=True)
        self.rho_optimizer.zero_grad(set_to_none=True)
        # µ LR scheduler is stepped by HF's training loop after this
        # method returns (it calls self.lr_scheduler.step() once).

        # ------------- Pass 2: KL * π -------------
        kl_list = []
        for _ in range(n_samples):
            kl_list.append(compute_total_kl(model))
        kl = torch.stack(kl_list, dim=0).mean()
        kl_div = kl * self._pi

        self.accelerator.backward(kl_div)
        self.rho_optimizer.step()
        self.rho_optimizer.zero_grad(set_to_none=True)
        self.mu_optimizer.zero_grad(set_to_none=True)
        if self.rho_scheduler is not None:
            self.rho_scheduler.step()

        # Return the NLL as the reported training loss.
        return nll.detach()

    # ------------------------------------------------------------------
    # compute_loss — used by evaluation paths only (training bypasses it
    # via the custom training_step above).
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
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
        outputs = {'loss': nll, 'logits': logits}
        return (nll, outputs) if return_outputs else nll

    # ------------------------------------------------------------------
    # Prediction — MC-averaged logits over _blob_eval_n_samples draws
    # ------------------------------------------------------------------

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool = False,
        ignore_keys=None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        inputs  = self._prepare_inputs(inputs)
        prompts = inputs.pop('prompts')
        labels  = inputs.pop('labels').to(self.args.device)
        tok = self.processing_class(
            prompts, **self.cfg.tokenizer_run_cfg).to(self.args.device)

        n_samples = self._blob_eval_n_samples
        set_blobsample(model, True)
        model.eval()
        with torch.no_grad():
            logits_list = []
            for _ in range(n_samples):
                out = model(**tok)
                logits_list.append(out.logits[:, -1, self.target_ids])
            logits = torch.stack(logits_list, dim=0).mean(dim=0)

        loss = F.cross_entropy(logits, labels)
        return loss.detach(), logits.detach(), labels.detach()
