"""Groupwise fine-tuning without a physical simulator or checkpoint loader."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .data import TaskBatch


LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, list[dict] | None],
    tuple[torch.Tensor, object],
]


def parameter_groups(
    model: torch.nn.Module,
    base_lr: float,
    freeze_measurement_fusion: bool = True,
) -> list[dict]:
    """Apply the original transfer learning-rate ratios to trainable parameters.

    Projection/backbone/head/other ratios are 0.05/0.10/1.00/0.20.
    Enabling freezing sets measurement-fusion parameters' ``requires_grad`` to
    False. Disabling it preserves existing flags; it does not unfreeze a model
    that the caller has previously frozen. Empty groups are omitted.
    """
    if isinstance(base_lr, bool) or not isinstance(base_lr, (float, int)):
        raise ValueError("base_lr must be a finite positive number")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base_lr must be a finite positive number")
    if not isinstance(freeze_measurement_fusion, bool):
        raise TypeError("freeze_measurement_fusion must be a boolean")
    named_parameters = list(model.named_parameters())
    if freeze_measurement_fusion:
        for name, parameter in named_parameters:
            if "measurement_fusion" in name:
                parameter.requires_grad_(False)
                parameter.grad = None

    groups = [
        {"params": [], "lr": float(base_lr) * 0.05},
        {"params": [], "lr": float(base_lr) * 0.10},
        {"params": [], "lr": float(base_lr)},
        {"params": [], "lr": float(base_lr) * 0.20},
    ]
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if "input_projection" in name:
            group_index = 0
        elif any(part in name for part in ("s4_layers", "blocks", "encoder", "recurrent")):
            group_index = 1
        elif "control_head" in name or ".head." in name or name.startswith("head."):
            group_index = 2
        else:
            group_index = 3
        groups[group_index]["params"].append(parameter)
    return [group for group in groups if group["params"]]


def finetune(
    model: torch.nn.Module,
    batch: TaskBatch,
    loss_fn: LossFunction,
    *,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-4,
    freeze_measurement_fusion: bool = True,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> list[float]:
    """Fine-tune in place and return one mean minibatch loss per epoch.

    ``loss_fn(predictions, targets, measurements, metadata)`` must return
    ``(scalar_loss, fidelity)``; fidelity is available to the callback and is not
    used by the optimizer. Each shuffled index selects both tensors and its
    metadata. All samples, including a final short minibatch, are used.

    This is an optimization method, not reproduction of the paper's physical
    loss or evaluation. The caller supplies the model, data and differentiable
    loss and manages any checkpoint storage. The model finishes in eval mode.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(batch, TaskBatch) or len(batch) == 0:
        raise ValueError("batch must be a nonempty TaskBatch")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if not all(bool(torch.isfinite(value).all()) for value in (batch.inputs, batch.measurements, batch.targets)):
        raise ValueError("batch tensors must contain only finite values")

    model.to(device)
    data = batch.to(device)
    groups = parameter_groups(model, lr, freeze_measurement_fusion)
    if not groups:
        raise ValueError("the model has no trainable parameters after freezing")
    trainable_parameters = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[float] = []
    # The seed also controls dropout while preserving the caller's RNG state.
    cuda_devices = [] if data.inputs.device.type != "cuda" else [data.inputs.device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        model.train()
        for _ in range(epochs):
            permutation = torch.randperm(len(data), generator=generator)
            epoch_loss = 0.0
            minibatches = 0
            for start in range(0, len(data), batch_size):
                minibatch = data.select(permutation[start:start + batch_size])
                predictions = model(minibatch.inputs, minibatch.measurements)
                result = loss_fn(
                    predictions, minibatch.targets, minibatch.measurements, minibatch.metadata,
                )
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError("loss_fn must return (scalar_loss, fidelity)")
                loss, _ = result
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.is_complex():
                    raise ValueError("loss_fn must return a real scalar loss tensor")
                if not bool(torch.isfinite(loss)):
                    raise ValueError("loss_fn returned a nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 0.5, error_if_nonfinite=True)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                minibatches += 1
            scheduler.step()
            history.append(epoch_loss / minibatches)
    model.eval()
    return history
