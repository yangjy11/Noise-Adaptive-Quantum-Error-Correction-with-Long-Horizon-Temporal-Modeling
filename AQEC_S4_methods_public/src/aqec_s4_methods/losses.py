"""Standalone objectives for the S4, meta-learning and transfer methods.

``FeedbackGRAPELoss`` retains the source project's objective and class name.
Its regularizers are algebraic proxies on predicted complex vectors, not a
complete physical GRAPE calculation or a density-matrix leakage estimate.
``SyntheticComplexMSELoss`` is a separate, nonphysical demonstration loss.
"""

from __future__ import annotations

import numpy as np
import torch


def _validate_vector_pair(pred_controls, target_states) -> None:
    if not isinstance(pred_controls, torch.Tensor) or not isinstance(target_states, torch.Tensor):
        raise TypeError("pred_controls and target_states must be tensors")
    if pred_controls.ndim != 2 or pred_controls.shape != target_states.shape:
        raise ValueError("pred_controls and target_states must have the same [batch, output_dim] shape")
    if pred_controls.shape[0] == 0 or pred_controls.shape[1] == 0:
        raise ValueError("pred_controls and target_states must be nonempty")
    if pred_controls.device != target_states.device:
        raise ValueError("pred_controls and target_states must be on the same device")
    for name, tensor in (("pred_controls", pred_controls), ("target_states", target_states)):
        if not (tensor.is_floating_point() or torch.is_complex(tensor)):
            raise TypeError(f"{name} must contain floating-point or complex values")


class FeedbackGRAPELoss:
    """Original composite objective, extracted without the physical simulator.

    The fidelity metric is normalized complex-vector overlap. The terms named
    leakage and energy are amplitude-based proxies; they neither construct a
    physical code space nor compute leakage from an evolved physical state.
    The energy proxy is dimensionless and scaled by a normalized duration.

    For source compatibility, ``use_grape_gradient`` is stored but does not
    select a separate gradient computation in this implementation. Gradients
    come from ordinary PyTorch differentiation of the expressions below.

    Metadata may identify ``code_type`` and provide ``physical_params`` and
    ``gate_schedule``. As in the source, batch-specific settings are taken
    from the first metadata record; use homogeneous batches when these
    settings differ. The leakage and energy helpers require a nonempty list
    of metadata records, whereas other settings also accept one dictionary.
    """

    def __init__(
        self, lambda_control=0.001, lambda_smooth=0.0001,
        lambda_feedback=0.01, lambda_entropy=0.005, lambda_timing=0.002,
        lambda_leakage=0.015, lambda_energy=0.008, use_grape_gradient=True,
    ):
        self.lambda_control = lambda_control
        self.lambda_smooth = lambda_smooth
        self.lambda_feedback = lambda_feedback
        self.lambda_entropy = lambda_entropy
        self.lambda_timing = lambda_timing
        self.lambda_leakage = lambda_leakage
        self.lambda_energy = lambda_energy
        self.use_grape_gradient = use_grape_gradient

    def _code_adaptive_weights(self, code_type: str, amp: float):
        ct = (code_type or "other").lower()
        amp = float(np.clip(amp, 0.0, 0.40))
        if ct == "gkp":
            lc = self.lambda_control * (0.65 + 0.25 * amp)
            ls = self.lambda_smooth * 0.85
            lf = self.lambda_feedback * 1.15
        elif ct == "cat":
            lc = self.lambda_control * 0.04
            ls = self.lambda_smooth * 0.75
            lf = self.lambda_feedback * 1.35
        elif ct == "binomial":
            lc = self.lambda_control * 0.02
            ls = self.lambda_smooth * 1.0
            lf = self.lambda_feedback * 1.3
        else:
            lc = self.lambda_control
            ls = self.lambda_smooth
            lf = self.lambda_feedback
        return float(lc), float(ls), float(lf)

    @staticmethod
    def _gate_duration(gate) -> float:
        if isinstance(gate, dict):
            return float(gate.get("duration", gate.get("duration_s", 0.0)) or 0.0)
        return float(getattr(gate, "duration", 0.0) or 0.0)

    def compute_code_leakage_penalty(self, pred_controls, task_metadata):
        """Return the source fourth-power amplitude proxy, not physical leakage."""
        if not isinstance(task_metadata, list) or len(task_metadata) == 0:
            return torch.tensor(0.0, device=pred_controls.device)
        metadata_sample = task_metadata[0]
        code_type = metadata_sample.get("code_type", "other").lower()
        if code_type in ["cat", "binomial"]:
            if pred_controls.shape[1] >= 2:
                theta = torch.abs(pred_controls[:, 0])
                alpha = torch.abs(pred_controls[:, 1])
            else:
                theta = torch.abs(pred_controls[:, 0])
                alpha = torch.zeros_like(theta)
            leakage = torch.mean(theta**4 + alpha**4)
        elif code_type == "gkp":
            alpha = torch.abs(pred_controls).mean(dim=-1)
            leakage = torch.mean(alpha**4)
        else:
            leakage = torch.tensor(0.0, device=pred_controls.device)
        return leakage

    def compute_energy_efficiency_penalty(self, pred_controls, task_metadata):
        """Return the squared-amplitude proxy scaled by duration / 1 microsecond."""
        if not isinstance(task_metadata, list) or len(task_metadata) == 0:
            return torch.tensor(0.0, device=pred_controls.device)
        metadata_sample = task_metadata[0]
        code_type = metadata_sample.get("code_type", "other").lower()
        if code_type in ["cat", "binomial"]:
            if pred_controls.shape[1] >= 2:
                theta = torch.abs(pred_controls[:, 0])
                alpha = torch.abs(pred_controls[:, 1])
                energy = torch.mean(theta**2 + alpha**2)
            else:
                energy = torch.mean(torch.abs(pred_controls)**2)
        elif code_type == "gkp":
            alpha = torch.abs(pred_controls)
            energy = torch.mean(alpha**2)
        else:
            energy = torch.mean(torch.abs(pred_controls)**2)
        if "gate_schedule" in metadata_sample:
            gate_schedule = metadata_sample["gate_schedule"]
            total_time = sum(self._gate_duration(g) for g in gate_schedule) if gate_schedule else 1e-6
        else:
            total_time = 1e-6
        energy_time_product = energy * (total_time / 1e-6)
        return energy_time_product

    def __call__(self, pred_controls, target_states, measurement_history=None, task_metadata=None):
        """Return ``(differentiable_loss, mean_vector_overlap_as_float)``."""
        _validate_vector_pair(pred_controls, target_states)
        if task_metadata is None:
            task_metadata = {}
        metadata_sample = task_metadata[0] if isinstance(task_metadata, list) and task_metadata else task_metadata
        code_type = "other"
        noise_amplitude = 0.10
        if isinstance(metadata_sample, dict):
            code_type = metadata_sample.get("code_type", metadata_sample.get("initial_code", "other"))
            if "physical_params" in metadata_sample:
                noise_amplitude = float(metadata_sample["physical_params"].get("noise_amplitude", noise_amplitude))
        ct = code_type.lower()

        if ct == "cat":
            weight_fidelity = 0.72
            weight_amplitude = 0.045
            weight_phase_smooth = 0.02
            weight_feedback = 0.12
            weight_leakage = 0.10
            weight_energy = 0.025
        elif ct == "binomial":
            weight_fidelity = 0.75
            weight_amplitude = 0.03
            weight_phase_smooth = 0.01
            weight_feedback = 0.12
            weight_leakage = 0.06
            weight_energy = 0.03
        elif ct == "gkp":
            weight_fidelity = 0.62
            weight_amplitude = 0.025
            weight_phase_smooth = 0.015
            weight_feedback = 0.13
            weight_leakage = 0.14
            weight_energy = 0.05
        else:
            weight_fidelity = 0.60
            weight_amplitude = 0.10
            weight_phase_smooth = 0.05
            weight_feedback = 0.08
            weight_leakage = 0.10
            weight_energy = 0.04

        pred_normalized = pred_controls / (torch.linalg.norm(pred_controls, dim=-1, keepdim=True) + 1e-8)
        target_normalized = target_states / (torch.linalg.norm(target_states, dim=-1, keepdim=True) + 1e-8)
        overlap = torch.sum(torch.conj(target_normalized) * pred_normalized, dim=-1)
        fidelity = torch.abs(overlap) ** 2
        fidelity = torch.clamp(fidelity, 0.0, 1.0)
        fidelity_loss = 1.0 - torch.mean(fidelity)

        control_amplitude = torch.abs(pred_controls)
        control_phase = torch.angle(pred_controls)
        lc, ls, lf = self._code_adaptive_weights(code_type, noise_amplitude)
        amplitude_penalty = lc * torch.mean(control_amplitude ** 2)
        if pred_controls.shape[1] > 1:
            phase_diff = torch.diff(control_phase, dim=1)
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            phase_smooth_penalty = ls * torch.mean(phase_diff ** 2)
        else:
            phase_smooth_penalty = torch.tensor(0.0, device=pred_controls.device)

        feedback_penalty = torch.tensor(0.0, device=pred_controls.device)
        if measurement_history is not None:
            meas_strength = torch.mean(torch.abs(measurement_history), dim=-1, keepdim=True)
            expected_control_strength = 0.55 * meas_strength
            actual_control_strength = torch.mean(control_amplitude, dim=-1, keepdim=True)
            feedback_penalty = lf * torch.mean((actual_control_strength - expected_control_strength) ** 2)

        entropy_penalty = torch.tensor(0.0, device=pred_controls.device)
        if measurement_history is not None and isinstance(metadata_sample, dict) and "physical_params" in metadata_sample:
            p_error = float(np.clip(noise_amplitude, 1e-6, 1.0 - 1e-6))
            expected_entropy = -p_error * np.log(p_error + 1e-8) - (1 - p_error) * np.log(1 - p_error + 1e-8)
            meas_probs = torch.sigmoid(measurement_history).clamp(1e-6, 1.0 - 1e-6)
            actual_entropy = -torch.mean(meas_probs * torch.log(meas_probs) + (1 - meas_probs) * torch.log(1 - meas_probs))
            entropy_penalty = self.lambda_entropy * (actual_entropy - expected_entropy) ** 2

        timing_penalty = torch.tensor(0.0, device=pred_controls.device)
        if isinstance(metadata_sample, dict) and "gate_schedule" in metadata_sample:
            gate_schedule = metadata_sample["gate_schedule"]
            total_time = sum(self._gate_duration(g) for g in gate_schedule) if gate_schedule else 0.0
            max_cycle_time = 50e-6
            if total_time > max_cycle_time:
                timing_penalty = self.lambda_timing * ((total_time - max_cycle_time) / max_cycle_time) ** 2

        leakage_penalty = self.lambda_leakage * self.compute_code_leakage_penalty(pred_controls, task_metadata)
        energy_penalty = self.lambda_energy * self.compute_energy_efficiency_penalty(pred_controls, task_metadata)
        total_loss = (
            weight_fidelity * fidelity_loss
            + weight_amplitude * amplitude_penalty
            + weight_phase_smooth * phase_smooth_penalty
            + weight_feedback * feedback_penalty
            + 0.02 * entropy_penalty
            + 0.01 * timing_penalty
            + weight_leakage * leakage_penalty
            + weight_energy * energy_penalty
        )
        avg_fidelity = torch.mean(fidelity).item()
        return total_loss, avg_fidelity


class SyntheticComplexMSELoss:
    """Complex-vector MSE for the independent synthetic training example.

    This objective has the same call signature as ``FeedbackGRAPELoss`` but
    intentionally ignores measurements and metadata. Its second return value
    is normalized vector overlap, retained only as a shared training metric;
    it is not evidence of a physical experiment or quantum-error correction.
    """

    def __call__(self, pred_controls, target_states, measurement_history=None, task_metadata=None):
        _validate_vector_pair(pred_controls, target_states)
        loss = torch.mean(torch.abs(pred_controls - target_states) ** 2)
        with torch.no_grad():
            pred_normalized = pred_controls / (torch.linalg.norm(pred_controls, dim=-1, keepdim=True) + 1e-8)
            target_normalized = target_states / (torch.linalg.norm(target_states, dim=-1, keepdim=True) + 1e-8)
            overlap = torch.sum(torch.conj(target_normalized) * pred_normalized, dim=-1)
            fidelity = torch.clamp(torch.abs(overlap) ** 2, 0.0, 1.0).mean().item()
        return loss, fidelity
