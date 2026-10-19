"""Tensor tasks and synthetic examples without a physical-state representation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class TaskBatch:
    """A batch of sequences and complex regression targets.

    Shapes are inputs ``[N, T, D]``, measurements ``[N, T]``, and
    targets ``[N, O]``. Metadata, when supplied, follows the sample order.
    """

    inputs: torch.Tensor
    measurements: torch.Tensor
    targets: torch.Tensor
    metadata: list[dict] | None = None

    def __post_init__(self) -> None:
        tensors = (self.inputs, self.measurements, self.targets)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("inputs, measurements and targets must be torch tensors")
        if self.inputs.ndim != 3 or min(self.inputs.shape[1:]) < 1:
            raise ValueError("inputs must have shape [N, T, D] with T and D positive")
        if self.measurements.shape != self.inputs.shape[:2]:
            raise ValueError("measurements must have shape [N, T] matching inputs")
        if self.targets.ndim != 2 or self.targets.shape[0] != len(self):
            raise ValueError("targets must have shape [N, O] matching inputs")
        if self.targets.shape[1] < 1:
            raise ValueError("targets must have at least one output")
        if not self.inputs.is_floating_point() or not self.measurements.is_floating_point():
            raise TypeError("inputs and measurements must have floating point dtypes")
        if not self.targets.is_complex():
            raise TypeError("targets must have a complex dtype")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all batch tensors must be on the same device")
        if self.metadata is not None:
            if not isinstance(self.metadata, list) or len(self.metadata) != len(self):
                raise ValueError("metadata must be a list with one dictionary per sample")
            if not all(isinstance(item, dict) for item in self.metadata):
                raise TypeError("metadata entries must be dictionaries")

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def to(self, device: torch.device | str) -> TaskBatch:
        """Return a batch on ``device`` while preserving metadata order."""
        return TaskBatch(
            self.inputs.to(device),
            self.measurements.to(device),
            self.targets.to(device),
            self.metadata,
        )

    def select(self, indices: torch.Tensor | Sequence[int]) -> TaskBatch:
        """Select tensors and metadata with the same sample indices."""
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("indices must be a one-dimensional integer tensor")
            selected = indices.to(device=self.inputs.device, dtype=torch.long)
            sample_ids = selected.detach().cpu().tolist()
        else:
            sample_ids = list(indices)
            if any(isinstance(index, bool) or not isinstance(index, int) for index in sample_ids):
                raise TypeError("indices must contain integers")
            selected = torch.tensor(sample_ids, dtype=torch.long, device=self.inputs.device)
        if any(index < 0 or index >= len(self) for index in sample_ids):
            raise IndexError("sample index is outside the batch")
        metadata = None if self.metadata is None else [self.metadata[index] for index in sample_ids]
        return TaskBatch(
            self.inputs.index_select(0, selected),
            self.measurements.index_select(0, selected),
            self.targets.index_select(0, selected),
            metadata,
        )


@dataclass
class Episode:
    """Support and query samples from one regression task."""

    support: TaskBatch
    query: TaskBatch


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def make_synthetic_episodes(
    n_tasks: int = 6,
    support_size: int = 8,
    query_size: int = 8,
    seq_len: int = 12,
    input_dim: int = 4,
    output_dim: int = 4,
    seed: int = 42,
    domain_shift: float = 0.0,
) -> list[Episode]:
    """Generate learnable sequence tasks with a shared random teacher.

    Each task has a small teacher perturbation and bias. Support and query use
    that same task teacher. ``domain_shift`` shifts the observed input and
    measurement distributions; it does not introduce any quantum-state data.
    A local CPU generator keeps this function independent of the global RNG.
    These examples demonstrate software behavior, not paper experiments.
    """
    for name, value in (
        ("n_tasks", n_tasks), ("support_size", support_size),
        ("query_size", query_size), ("seq_len", seq_len),
        ("input_dim", input_dim), ("output_dim", output_dim),
    ):
        _positive_integer(value, name)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(domain_shift, bool) or not isinstance(domain_shift, (float, int)):
        raise ValueError("domain_shift must be a finite number")
    if not math.isfinite(domain_shift):
        raise ValueError("domain_shift must be finite")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    feature_dim = 2 * input_dim + 1
    teacher = torch.randn(feature_dim, 2 * output_dim, generator=generator) / math.sqrt(feature_dim)
    measurement_weights = torch.randn(input_dim, generator=generator) / math.sqrt(input_dim)
    episodes: list[Episode] = []
    for _ in range(n_tasks):
        task_teacher = teacher + 0.06 * torch.randn(teacher.shape, generator=generator)
        task_bias = 0.08 * torch.randn(2 * output_dim, generator=generator)
        count = support_size + query_size
        inputs = torch.randn(count, seq_len, input_dim, generator=generator) + float(domain_shift)
        measurements = inputs @ measurement_weights
        measurements = measurements + 0.03 * torch.randn(count, seq_len, generator=generator)
        features = torch.cat(
            (inputs.mean(dim=1), inputs[:, -1], measurements.mean(dim=1, keepdim=True)), dim=-1,
        )
        components = 0.5 * torch.tanh(features @ task_teacher + task_bias)
        targets = torch.complex(components[:, :output_dim], components[:, output_dim:])
        batch = TaskBatch(inputs, measurements, targets)
        support_ids = torch.arange(support_size)
        query_ids = torch.arange(support_size, count)
        episodes.append(Episode(batch.select(support_ids), batch.select(query_ids)))
    return episodes


def _finite_scalar(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite JSON number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{location} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{location} must be finite")
    return number


def read_task_batch(
    path: str | Path,
    max_seq_len: int = 50,
    output_dim: int = 20,
) -> TaskBatch:
    """Read the original sequence/complex-target JSON schema on CPU.

    Accept a nonempty task list, or ``{"tasks": [...]}``. Each task requires
    ``input_data`` (rectangular T by D), ``measurement_sequence`` (T), and
    ``target_output`` (nonempty dictionaries with numeric ``real``/``imag``).
    All tasks must share D. Sequences and outputs are truncated or zero-padded
    to the requested sizes, matching the original tensor preparation.

    Optional ``sample_metadata`` is passed through unchanged; this reader never
    constructs physical parameters, states, or encoding information. The caller
    is responsible for deciding whether existing metadata may be shared.
    """
    _positive_integer(max_seq_len, "max_seq_len")
    _positive_integer(output_dim, "output_dim")
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task JSON must contain a nonempty list or an object with a tasks list")

    inputs: list[torch.Tensor] = []
    measurements: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata: list[dict] = []
    has_metadata = False
    input_width: int | None = None
    for task_index, task in enumerate(tasks):
        prefix = f"task[{task_index}]"
        if not isinstance(task, dict):
            raise ValueError(f"{prefix} must be an object")
        for field in ("input_data", "measurement_sequence", "target_output"):
            if field not in task:
                raise ValueError(f"{prefix} is missing {field}")
        raw_inputs = task["input_data"]
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise ValueError(f"{prefix}.input_data must be a nonempty list of rows")
        if not isinstance(raw_inputs[0], list) or not raw_inputs[0]:
            raise ValueError(f"{prefix}.input_data rows must be nonempty lists")
        width = len(raw_inputs[0])
        if input_width is None:
            input_width = width
        if width != input_width:
            raise ValueError(f"{prefix}.input_data width differs from the other tasks")
        rows: list[list[float]] = []
        for row_index, row in enumerate(raw_inputs):
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"{prefix}.input_data must be rectangular")
            rows.append([
                _finite_scalar(value, f"{prefix}.input_data[{row_index}][{column}]")
                for column, value in enumerate(row)
            ])
        raw_measurements = task["measurement_sequence"]
        if not isinstance(raw_measurements, list) or len(raw_measurements) != len(rows):
            raise ValueError(f"{prefix}.measurement_sequence must match the sequence length")
        measurement_values = [
            _finite_scalar(value, f"{prefix}.measurement_sequence[{index}]")
            for index, value in enumerate(raw_measurements)
        ]
        raw_targets = task["target_output"]
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(f"{prefix}.target_output must be a nonempty list")
        target_values: list[complex] = []
        for index, value in enumerate(raw_targets):
            if not isinstance(value, dict) or "real" not in value or "imag" not in value:
                raise ValueError(f"{prefix}.target_output[{index}] needs real and imag numbers")
            target_values.append(complex(
                _finite_scalar(value["real"], f"{prefix}.target_output[{index}].real"),
                _finite_scalar(value["imag"], f"{prefix}.target_output[{index}].imag"),
            ))

        x = torch.tensor(rows[:max_seq_len], dtype=torch.float32)
        m = torch.tensor(measurement_values[:max_seq_len], dtype=torch.float32)
        y = torch.tensor(target_values[:output_dim], dtype=torch.complex64)
        if not all(bool(torch.isfinite(value).all()) for value in (x, m, y)):
            raise ValueError(f"{prefix} contains values outside the supported float32 range")
        inputs.append(F.pad(x, (0, 0, 0, max_seq_len - x.shape[0])))
        measurements.append(F.pad(m, (0, max_seq_len - m.shape[0])))
        targets.append(F.pad(y, (0, output_dim - y.shape[0])))
        task_metadata = task.get("sample_metadata")
        if "sample_metadata" in task and not isinstance(task_metadata, dict):
            raise ValueError(f"{prefix}.sample_metadata must be an object when present")
        has_metadata = has_metadata or task_metadata is not None
        metadata.append({} if task_metadata is None else task_metadata)

    return TaskBatch(
        torch.stack(inputs), torch.stack(measurements), torch.stack(targets),
        metadata if has_metadata else None,
    )
