"""First-order meta-learning derived from the original AQEC-S4 trainer.

Only the learning algorithm is included. Tasks, measurements, and loss functions
are supplied by the caller; this module defines no domain-specific state space.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import random
from collections.abc import Sequence
from typing import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .data import Episode, TaskBatch


LossFunction = Callable[[Tensor, Tensor, Tensor, object], tuple[Tensor, float]]


class AdaptiveInnerLoopOptimizer:
    """Compute the original loss/gradient-dependent inner learning rate.

    The historical implementation accepted momentum settings without applying
    momentum. They are deliberately absent here. ``abs().square()`` also gives a
    real, nonnegative norm when a caller uses complex model parameters.
    """

    def __init__(
        self,
        base_lr: float = 0.05,
        min_lr: float = 0.005,
        max_lr: float = 0.1,
    ) -> None:
        for name, value in (
            ("base_lr", base_lr), ("min_lr", min_lr), ("max_lr", max_lr)
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if min_lr > max_lr:
            raise ValueError("min_lr must not exceed max_lr")
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)

    def compute_adaptive_lr(
        self,
        grads: Sequence[Tensor | None],
        step: int,
        prev_loss: float,
        curr_loss: float,
    ) -> float:
        if step < 0:
            raise ValueError("step must be nonnegative")
        norm_squared = sum(
            float(grad.detach().abs().square().sum().item())
            for grad in grads
            if grad is not None
        )
        grad_norm = math.sqrt(norm_squared)
        if not math.isfinite(grad_norm) or not math.isfinite(curr_loss):
            raise FloatingPointError("Inner loss or gradient norm is nonfinite")

        loss_multiplier = 1.0
        if step > 0 and math.isfinite(prev_loss):
            loss_change = curr_loss - prev_loss
            if loss_change < -0.01:
                loss_multiplier = 1.3
            elif loss_change < 0:
                loss_multiplier = 1.1
            elif loss_change < 0.01:
                loss_multiplier = 1.0
            else:
                loss_multiplier = 0.6

        if grad_norm > 10.0:
            grad_multiplier = 0.5
        elif grad_norm > 5.0:
            grad_multiplier = 0.8
        elif grad_norm < 0.1:
            grad_multiplier = 1.5
        else:
            grad_multiplier = 1.0
        rate = self.base_lr * loss_multiplier * grad_multiplier * (0.95 ** step)
        return float(min(self.max_lr, max(self.min_lr, rate)))


class FOMAMLTrainer:
    """Adapt on support batches and average first-order query gradients.

    Adapted parameters are detached at every inner step. Query gradients are
    explicitly copied back to the corresponding initial model parameters before
    the outer AdamW step. Second-order MAML is not implemented.

    The default AdamW decay and gradient clipping match the source trainer. All
    tasks use the configured ``inner_steps``; no task-specific policy or implicit
    epoch schedule is included. ``fit`` uses only the supplied training episodes.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFunction,
        *,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 3,
        adaptive_inner: bool = True,
        device: str | torch.device = "cpu",
        weight_decay: float = 1e-3,
        max_grad_norm: float = 1.0,
    ) -> None:
        for name, value in (
            ("inner_lr", inner_lr), ("meta_lr", meta_lr),
            ("max_grad_norm", max_grad_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(inner_steps, int) or inner_steps < 0:
            raise ValueError("inner_steps must be a nonnegative integer")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable:
            raise ValueError("The model must have at least one trainable parameter")
        self.loss_fn = loss_fn
        self.inner_lr = float(inner_lr)
        self.meta_lr = float(meta_lr)
        self.inner_steps = inner_steps
        self.adaptive_inner = bool(adaptive_inner)
        self.max_grad_norm = float(max_grad_norm)
        self.adaptive_optimizer = AdaptiveInnerLoopOptimizer(
            base_lr=inner_lr, min_lr=inner_lr * 0.1, max_lr=inner_lr * 4.0
        )
        self.meta_optimizer = torch.optim.AdamW(
            trainable, lr=meta_lr, weight_decay=weight_decay
        )
        self.optimizer = self.meta_optimizer
        self.history: list[dict[str, float | int]] = []

    def _evaluate_loss(
        self,
        batch: TaskBatch,
        params: OrderedDict[str, Tensor],
        buffers: OrderedDict[str, Tensor],
    ) -> tuple[Tensor, float]:
        predictions = functional_call(
            self.model, (params, buffers), (batch.inputs, batch.measurements)
        )
        loss, fidelity = self.loss_fn(
            predictions, batch.targets, batch.measurements, batch.metadata
        )
        if not isinstance(loss, Tensor) or loss.ndim != 0 or loss.is_complex():
            raise TypeError("loss_fn must return a real scalar Tensor as its loss")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("loss_fn returned a nonfinite loss")
        fidelity_value = (
            float(fidelity.detach().item())
            if isinstance(fidelity, Tensor) else float(fidelity)
        )
        if not math.isfinite(fidelity_value):
            raise FloatingPointError("loss_fn returned a nonfinite fidelity")
        return loss, fidelity_value

    def _adapt(
        self, support: TaskBatch
    ) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor], list[float], list[float]]:
        support = support.to(self.device)
        if len(support) == 0:
            raise ValueError("An episode support batch must not be empty")
        params = OrderedDict(
            (name, param.detach().clone().requires_grad_(param.requires_grad))
            for name, param in self.model.named_parameters()
        )
        # Isolate mutable buffers such as normalization statistics per episode.
        buffers = OrderedDict(
            (name, buffer.detach().clone())
            for name, buffer in self.model.named_buffers()
        )
        names = [name for name, param in params.items() if param.requires_grad]
        inner_losses: list[float] = []
        fidelities: list[float] = []
        previous_loss = float("inf")
        for step in range(self.inner_steps):
            loss, fidelity = self._evaluate_loss(support, params, buffers)
            current_loss = float(loss.detach().item())
            gradients = torch.autograd.grad(
                loss, tuple(params[name] for name in names),
                create_graph=False, retain_graph=False, allow_unused=True,
            )
            for gradient in gradients:
                if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError("Nonfinite support gradient")
            if self.adaptive_inner:
                rate = self.adaptive_optimizer.compute_adaptive_lr(
                    gradients, step, previous_loss, current_loss
                )
            else:
                rate = self.inner_lr * (0.9 ** step)
            for name, gradient in zip(names, gradients):
                parameter = params[name]
                updated = parameter if gradient is None else parameter - rate * gradient
                params[name] = updated.detach().requires_grad_(True)
            inner_losses.append(current_loss)
            fidelities.append(fidelity)
            previous_loss = current_loss
        return params, buffers, inner_losses, fidelities

    def inner_loop_update(
        self, support: TaskBatch
    ) -> tuple[OrderedDict[str, Tensor], list[float], list[float]]:
        """Return detached adapted parameters and support training diagnostics.

        For models with mutable buffers, ``train_step`` keeps each episode's
        adapted buffers internally. The public AQEC-S4 model has no such buffers.
        """
        params, _, losses, fidelities = self._adapt(support)
        return params, losses, fidelities

    def train_step(self, episodes: Sequence[Episode]) -> dict[str, float | int]:
        """Perform one outer update over a nonempty batch of episodes.

        Errors and nonfinite values are raised, never counted as successful
        episodes. Autograd must be enabled, as for an ordinary training step.
        """
        if not torch.is_grad_enabled():
            raise RuntimeError("train_step requires autograd; leave torch.no_grad() first")
        if len(episodes) == 0:
            raise ValueError("episodes must not be empty")
        self.model.train()
        self.meta_optimizer.zero_grad(set_to_none=True)
        initial = OrderedDict(
            (name, param) for name, param in self.model.named_parameters()
            if param.requires_grad
        )
        sums = OrderedDict((name, torch.zeros_like(param)) for name, param in initial.items())
        used_names: set[str] = set()
        total_loss = 0.0
        total_fidelity = 0.0
        for episode in episodes:
            params, buffers, _, _ = self._adapt(episode.support)
            query = episode.query.to(self.device)
            if len(query) == 0:
                raise ValueError("An episode query batch must not be empty")
            query_loss, fidelity = self._evaluate_loss(query, params, buffers)
            gradients = torch.autograd.grad(
                query_loss, tuple(params[name] for name in initial),
                create_graph=False, retain_graph=False, allow_unused=True,
            )
            for name, gradient in zip(initial, gradients):
                if gradient is None:
                    continue
                if not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError("Nonfinite query gradient")
                sums[name].add_(gradient.detach())
                used_names.add(name)
            total_loss += float(query_loss.detach().item())
            total_fidelity += fidelity

        if not used_names:
            raise RuntimeError("The query loss does not depend on any trainable parameter")
        count = len(episodes)
        for name, parameter in initial.items():
            # An unused parameter should not receive AdamW weight decay either.
            parameter.grad = sums[name].div(float(count)) if name in used_names else None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            initial.values(), max_norm=self.max_grad_norm, error_if_nonfinite=True
        )
        self.meta_optimizer.step()
        return {
            "loss": total_loss / count,
            "fidelity": total_fidelity / count,
            "valid_episodes": count,
            "grad_norm": float(grad_norm.item()),
            "gradient_tensors": len(used_names),
        }

    def fit(
        self,
        episodes: Sequence[Episode],
        *,
        epochs: int = 10,
        episodes_per_epoch: int | None = None,
        seed: int = 0,
    ) -> list[dict[str, float | int]]:
        """Run one meta update per epoch using reproducibly sampled episodes.

        Sampling is without replacement within each epoch. A separate held-out
        evaluation protocol can be supplied by the caller; training fidelity is
        not used to claim out-of-task performance or select a best checkpoint.
        """
        if not isinstance(epochs, int) or epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        pool = list(episodes)
        if not pool:
            raise ValueError("episodes must not be empty")
        count = len(pool) if episodes_per_epoch is None else episodes_per_epoch
        if not isinstance(count, int) or not 1 <= count <= len(pool):
            raise ValueError("episodes_per_epoch must be between 1 and len(episodes)")
        rng = random.Random(seed)
        history: list[dict[str, float | int]] = []
        for epoch in range(epochs):
            selected = rng.sample(pool, count)
            metrics = self.train_step(selected)
            history.append({"epoch": epoch + 1, **metrics})
        self.history.extend(history)
        return history
