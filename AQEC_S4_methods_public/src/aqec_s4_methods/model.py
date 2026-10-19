"""The project's S4 sequence model, independent of its physical simulator.

Extracted from ``abb.py`` in the supplied code archive. Parameter names and
the default architecture are retained. The constructors deliberately do not
reset the random seed; set it once in the calling training program instead.
This is the implementation used by this project, not a claim of equivalence
to every implementation of the S4 architecture.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def hippo_initialization(N: int) -> torch.Tensor:
    """Return the original project initializer for the state matrix."""
    _positive_integer("N", N)
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = (2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
            elif n == k:
                A[n, k] = n + 1
    A_log = torch.log(torch.abs(A) + 1e-8)
    A_log = -torch.abs(A_log)
    return A_log


class S4Layer(nn.Module):
    """Gated state-space convolution layer from the original implementation.

    ``d_state`` retains the original minimum effective state size of 64.
    Inputs and outputs have shape ``[batch, length, d_model]``.
    """

    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        _positive_integer("d_model", d_model)
        _positive_integer("d_state", d_state)
        self.h = d_model
        self.n = max(d_state, 64)

        self.A_log = nn.Parameter(hippo_initialization(self.n))
        self.B = nn.Parameter(torch.randn(self.n, 1) * (2.0 / self.n) ** 0.5)
        self.C = nn.Parameter(torch.randn(1, self.n) * (2.0 / self.n) ** 0.5)
        self.D = nn.Parameter(torch.randn(1) * 0.01)

        self.input_gate = nn.Sequential(nn.Linear(self.h, self.h), nn.Sigmoid())
        self.forget_gate = nn.Sequential(nn.Linear(self.h, self.h), nn.Sigmoid())
        self.fc1 = nn.Linear(self.h, 2 * self.h)
        self.fc2 = nn.Linear(self.h, self.h)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.h)

    def forward(self, u):
        if not isinstance(u, torch.Tensor) or u.ndim != 3:
            raise ValueError("u must be a tensor with shape [batch, length, d_model]")
        if u.shape[0] == 0 or u.shape[1] == 0 or u.shape[2] != self.h:
            raise ValueError(f"u needs nonempty batch/length and last dimension {self.h}")
        if not u.is_floating_point():
            raise TypeError("u must contain real floating-point features")
        u_residual = u

        i_gate = self.input_gate(u)
        f_gate = self.forget_gate(u)
        u = u * i_gate + u_residual * f_gate
        u = u.transpose(1, 2)

        u_gate = self.fc1(u.transpose(1, 2)).transpose(1, 2)
        u1, u2 = u_gate.chunk(2, dim=1)
        u = u1 * torch.sigmoid(u2)

        A = -torch.exp(self.A_log.clamp(max=0))
        dt = 1.0 / max(1, u.shape[-1])
        A_bar = torch.exp(dt * A)
        denom = A + 1e-8
        B_bar = (A_bar - 1) / denom * self.B.squeeze()
        C_bar = self.C.squeeze()

        K = torch.zeros(u.shape[-1], device=u.device, dtype=u.dtype)
        A_power = torch.ones_like(A_bar)
        for i in range(u.shape[-1]):
            K[i] = (C_bar * A_power * B_bar).sum()
            A_power = A_power * A_bar

        y = torch.zeros_like(u)
        K_f = torch.fft.rfft(K, n=2 * u.shape[-1])
        for i in range(u.shape[1]):
            u_ch = u[:, i, :]
            u_f = torch.fft.rfft(u_ch, n=2 * u.shape[-1], dim=-1)
            y_f = u_f * K_f.unsqueeze(0)
            y_ch = torch.fft.irfft(y_f, n=2 * u.shape[-1], dim=-1)[..., :u.shape[-1]]
            y[:, i, :] = y_ch

        y = y + u * self.D
        y = y.transpose(1, 2)
        y = self.fc2(y)
        y = self.dropout(y)
        y = self.norm(y + u_residual)
        return y


class S4Block(nn.Module):
    """The original pre-normalized S4 and feed-forward residual block."""

    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.s4_layer = S4Layer(d_model, d_state, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.s4_layer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class QuantumS4Model(nn.Module):
    """Map real feature sequences and optional measurements to complex vectors.

    ``x`` has shape ``[batch, length, input_dim]``. ``measurement_seq`` may
    have shape ``[batch, measurement_length]``; a different positive length
    is interpolated to the feature length as in the source implementation.
    The return value has complex dtype and shape ``[batch, output_dim]``.
    The output is a learned vector; this module does not apply physical
    controls or evolve physical states.
    """

    def __init__(
        self, input_dim=4, d_model=128, d_state=64, n_layers=2,
        output_dim=20, dropout=0.1,
    ):
        super().__init__()
        _positive_integer("input_dim", input_dim)
        _positive_integer("d_model", d_model)
        _positive_integer("d_state", d_state)
        _positive_integer("output_dim", output_dim)
        if d_model < 4:
            raise ValueError("d_model must be at least 4 for the measurement fusion projection")
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 0:
            raise ValueError("n_layers must be a nonnegative integer")
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.measurement_fusion = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )
        self.s4_layers = nn.ModuleList(
            [S4Block(d_model, d_state, dropout) for _ in range(n_layers)]
        )
        self.control_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim * 2),
        )

    def forward(self, x, measurement_seq=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a tensor with shape [batch, length, input_dim]")
        if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != self.input_dim:
            raise ValueError(f"x needs nonempty batch/length and last dimension {self.input_dim}")
        if not x.is_floating_point():
            raise TypeError("x must contain real floating-point features")
        B, L, _ = x.shape

        if measurement_seq is not None:
            if not isinstance(measurement_seq, torch.Tensor) or measurement_seq.ndim != 2:
                raise ValueError("measurement_seq must have shape [batch, measurement_length]")
            if measurement_seq.shape[0] != B or measurement_seq.shape[1] == 0:
                raise ValueError("measurement_seq needs the same batch size as x and a positive length")
            if not measurement_seq.is_floating_point():
                raise TypeError("measurement_seq must contain real floating-point values")
            if measurement_seq.device != x.device:
                raise ValueError("x and measurement_seq must be on the same device")

        x = self.input_projection(x)
        if measurement_seq is not None:
            meas_features = self.measurement_fusion(measurement_seq.unsqueeze(-1))
            if meas_features.shape[1] != L:
                meas_features = F.interpolate(
                    meas_features.transpose(1, 2), size=L,
                    mode="linear", align_corners=False,
                ).transpose(1, 2)
            attention_weights = torch.softmax(
                torch.sum(x * meas_features, dim=-1, keepdim=True) / np.sqrt(self.d_model),
                dim=1,
            )
            x = x + meas_features * attention_weights

        pos = self._get_positional_encoding(L, self.d_model, x.device)
        x = x + pos.unsqueeze(0).expand(B, -1, -1)
        for blk in self.s4_layers:
            x = blk(x)

        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1)[0]
        x = x_mean + 0.5 * x_max
        control_params = self.control_head(x).view(B, self.output_dim, 2)
        control_complex = torch.complex(control_params[..., 0], control_params[..., 1])
        return control_complex

    def _get_positional_encoding(self, seq_len, d_model, device):
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device)
            * (-(np.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd widths have one fewer cosine column; even-width results are unchanged.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe
