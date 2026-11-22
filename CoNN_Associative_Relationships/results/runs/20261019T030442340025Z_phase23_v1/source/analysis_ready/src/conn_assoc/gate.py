"""Gaussian connectivity gate preserved from Experiment 2.

The shared domain is strictly positive tau. On that domain the arithmetic
matches ``conn_exp.gates.gaussian_gate``: values below the positive floor
are raised to the floor, the exponent is ``-d^2 / (2 tau^2)``, and the
result is clamped to [0, 1]. Underflow of the exponential to zero is kept.

Experiment 2 also maps non-positive tau onto the floor. This module rejects
non-positive tau so a sign error cannot silently become a wide gate.
"""

from __future__ import annotations

import math

import torch


def resolve_tau(tau: float, tau_floor: float) -> tuple[float, bool]:
    tau = float(tau)
    tau_floor = float(tau_floor)
    if not math.isfinite(tau) or not math.isfinite(tau_floor):
        raise ValueError("tau and tau_floor must be finite")
    if tau_floor <= 0:
        raise ValueError("tau_floor must be strictly positive")
    if tau <= 0:
        raise ValueError("tau must be strictly positive")
    if tau < tau_floor:
        return tau_floor, True
    return tau, False


def gaussian_gate(
    distance: torch.Tensor, tau: float, tau_floor: float = 1e-6
) -> tuple[torch.Tensor, bool]:
    """Return G = exp(-d^2 / (2 tau^2)) and whether the floor replaced tau."""
    tau_used, floored = resolve_tau(tau, tau_floor)
    exponent = -(distance * distance) / (2.0 * tau_used * tau_used)
    gate = torch.exp(exponent)
    gate = torch.clamp(gate, 0.0, 1.0)
    return gate, floored


def effective_weight(weight: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Mathematical product W_eff = W ⊙ G. This function does not detach G."""
    if weight.shape != gate.shape:
        raise ValueError(f"weight shape {tuple(weight.shape)} != gate shape {tuple(gate.shape)}")
    return weight * gate


def training_effective_weight(weight: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Training product used by the original GatedLinear: G is detached."""
    if weight.shape != gate.shape:
        raise ValueError(f"weight shape {tuple(weight.shape)} != gate shape {tuple(gate.shape)}")
    applied = gate.detach().to(dtype=weight.dtype, device=weight.device)
    return weight * applied
