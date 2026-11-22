"""Elementwise gates. Permutations are drawn once and stored."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from conn_exp.geometry import Geometry
from conn_exp.seeding import paired_seed


def resolve_tau(tau: float, tau_floor: float) -> tuple[float, bool]:
    if tau < tau_floor:
        return float(tau_floor), True
    return float(tau), False


def gaussian_gate(
    distance: torch.Tensor, tau: float, tau_floor: float = 1e-6
) -> tuple[torch.Tensor, bool]:
    """G = exp(-D^2 / (2 tau^2)), with tau floored so the denominator cannot vanish."""
    tau_used, floored = resolve_tau(tau, tau_floor)
    exponent = -(distance * distance) / (2.0 * tau_used * tau_used)
    gate = torch.exp(exponent)
    gate = torch.clamp(gate, 0.0, 1.0)
    return gate, floored


def apply_row_permutation(gate: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute columns independently inside each row. Values are copied, not recomputed."""
    if perm.shape != gate.shape:
        raise ValueError(f"perm shape {tuple(perm.shape)} != gate shape {tuple(gate.shape)}")
    rows = torch.arange(gate.shape[0])[:, None]
    return gate[rows, perm]


def scalar_gate(reference: torch.Tensor) -> torch.Tensor:
    """Uniform gate whose square equals the reference layer's mean square."""
    energy = torch.mean(reference * reference)
    value = torch.sqrt(energy)
    return torch.full_like(reference, value)


def layer_energy(gate: torch.Tensor) -> torch.Tensor:
    return torch.mean(gate * gate)


@dataclass
class Masks:
    input_perm: torch.Tensor
    row_perm_ih: torch.Tensor
    row_perm_hh: torch.Tensor
    permutation_type: str = "within_row_column_permutation_frozen"


def _row_perms(n_rows: int, n_cols: int, generator: torch.Generator) -> torch.Tensor:
    return torch.stack(
        [torch.randperm(n_cols, generator=generator) for _ in range(n_rows)],
        dim=0,
    )


def draw_masks(n_in: int, n_h1: int, n_h2: int, seed: int) -> Masks:
    """Draw input and within-row permutations from the mask stream only."""
    generator = torch.Generator()
    generator.manual_seed(paired_seed(seed, "mask"))
    input_perm = torch.randperm(n_in, generator=generator)
    return Masks(
        input_perm=input_perm,
        row_perm_ih=_row_perms(n_h1, n_in, generator),
        row_perm_hh=_row_perms(n_h2, n_h1, generator),
    )


@dataclass
class GateStats:
    energy: float
    row_energy_mean: float
    row_energy_std: float
    frac_lt_1e_6: float
    frac_eq_0: float
    row_energy: torch.Tensor


def summarize_gate(gate: torch.Tensor) -> GateStats:
    row = torch.sum(gate * gate, dim=1)
    std = float(row.std(correction=0)) if row.numel() > 1 else 0.0
    return GateStats(
        energy=float(torch.mean(gate * gate)),
        row_energy_mean=float(row.mean()),
        row_energy_std=std,
        frac_lt_1e_6=float((gate < 1e-6).float().mean()),
        frac_eq_0=float((gate == 0).float().mean()),
        row_energy=row.detach().cpu(),
    )


STRUCTURE = {
    "A": "exact_ones",
    "A_prime": "uniform_scalar_sqrt_mean_square_of_D_gate",
    "B_prime": "frozen_within_row_column_permutation_of_D_gate",
    "C": "frozen_input_coordinate_permutation_hh_matches_D",
    "D": "task_aligned_gaussian_or_exact",
    "E": "task_aligned_gaussian_or_exact",
    "S": "task_aligned_gaussian_or_exact",
}


@dataclass
class GatePack:
    ih: torch.Tensor
    hh: torch.Tensor
    tau_floored: bool
    exact_dense: bool
    stats_ih: GateStats
    stats_hh: GateStats
    structure: str


class GateBuilder:
    """Apply one condition's structural rule to a schedule point.

    ``build`` does not draw random numbers. B' and C use masks stored at init.
    """

    def __init__(
        self,
        geometry: Geometry,
        masks: Masks,
        condition_id: str,
        tau_floor: float,
    ) -> None:
        if condition_id not in STRUCTURE:
            raise KeyError(condition_id)
        self.geometry = geometry
        self.masks = masks
        self.condition_id = condition_id
        self.tau_floor = float(tau_floor)
        self.base_dist_ih = geometry.dist_ih
        self.base_dist_hh = geometry.dist_hh

    def _reference(self, tau: float | None, exact_dense: bool) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if exact_dense:
            ones_ih = torch.ones_like(self.base_dist_ih)
            ones_hh = torch.ones_like(self.base_dist_hh)
            return ones_ih, ones_hh, False
        if tau is None:
            raise ValueError("tau is required when the gate is not exact dense")
        gate_ih, floor_ih = gaussian_gate(self.base_dist_ih, tau, self.tau_floor)
        gate_hh, floor_hh = gaussian_gate(self.base_dist_hh, tau, self.tau_floor)
        return gate_ih, gate_hh, floor_ih or floor_hh

    def build(self, tau: float | None, exact_dense: bool) -> GatePack:
        gate_ih, gate_hh, floored = self._reference(tau, exact_dense)
        condition = self.condition_id
        if condition == "A":
            gate_ih = torch.ones_like(self.base_dist_ih)
            gate_hh = torch.ones_like(self.base_dist_hh)
            floored = False
            exact = True
        elif condition == "A_prime":
            gate_ih = scalar_gate(gate_ih)
            gate_hh = scalar_gate(gate_hh)
            exact = exact_dense
        elif condition == "B_prime":
            gate_ih = apply_row_permutation(gate_ih, self.masks.row_perm_ih)
            gate_hh = apply_row_permutation(gate_hh, self.masks.row_perm_hh)
            exact = exact_dense
        elif condition == "C":
            gate_ih = gate_ih[:, self.masks.input_perm]
            exact = exact_dense
        elif condition in {"D", "E", "S"}:
            exact = exact_dense
        else:
            raise KeyError(condition)
        return GatePack(
            ih=gate_ih,
            hh=gate_hh,
            tau_floored=floored,
            exact_dense=exact,
            stats_ih=summarize_gate(gate_ih),
            stats_hh=summarize_gate(gate_hh),
            structure=STRUCTURE[condition],
        )


def d50_exact(tau: float) -> float:
    return float(tau) * math.sqrt(2.0 * math.log(2.0))
