from __future__ import annotations

import math

import torch

from conn_exp.gates import (
    STRUCTURE,
    GateBuilder,
    apply_row_permutation,
    draw_masks,
    gaussian_gate,
    scalar_gate,
)
from conn_exp.geometry import make_task_geometry
from conn_exp.schema import D50_RECORD_FACTOR


def test_record_factor_and_half_gate():
    assert f"{D50_RECORD_FACTOR:.5f}" == "1.17741"
    distance = torch.tensor([[math.sqrt(2.0 * math.log(2.0)) * 0.2]])
    gate, floored = gaussian_gate(distance, tau=0.2, tau_floor=1e-6)
    assert not floored
    assert torch.allclose(gate, torch.tensor([[0.5]]), atol=1e-6)
    zeros, _ = gaussian_gate(torch.zeros(2, 2), tau=0.2)
    assert torch.equal(zeros, torch.ones(2, 2))


def test_tau_floor_is_finite_and_flagged():
    gate, floored = gaussian_gate(torch.tensor([[0.0, 0.2], [0.4, 0.0]]), tau=0.0, tau_floor=1e-6)
    assert floored
    assert torch.isfinite(gate).all()
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


def test_row_permutation_copies_values_and_is_not_global():
    gate = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    perm = torch.tensor([[1, 0, 2], [2, 0, 1]])
    shuffled = apply_row_permutation(gate, perm)
    assert shuffled[0].tolist() == [2.0, 1.0, 3.0]
    assert shuffled[1].tolist() == [6.0, 4.0, 5.0]
    assert not torch.equal(perm[0], perm[1])


def _builders(condition: str, seed: int = 0):
    geometry = make_task_geometry(100, (32, 32))
    masks = draw_masks(100, 32, 32, seed)
    return geometry, masks, GateBuilder(geometry, masks, condition, 1e-6)


def test_masks_are_within_row_and_frozen():
    _, masks, builder = _builders("B_prime", seed=0)
    assert masks.permutation_type == "within_row_column_permutation_frozen"
    assert STRUCTURE["B_prime"].startswith("frozen_within_row")
    assert not torch.equal(masks.row_perm_ih[0], masks.row_perm_ih[1])
    assert not torch.equal(masks.input_perm, torch.arange(100))
    before = masks.row_perm_ih.clone()
    for _ in range(5):
        builder.build(0.08, False)
    assert torch.equal(before, masks.row_perm_ih)


def test_b_prime_preserves_each_row_and_is_not_the_structured_gate():
    _, _, structured = _builders("D")
    _, masks, shuffled = _builders("B_prime")
    base = structured.build(0.07, False)
    perm = shuffled.build(0.07, False)
    assert not torch.equal(base.ih, perm.ih)
    for row in range(base.ih.shape[0]):
        assert torch.equal(base.ih[row].sort().values, perm.ih[row].sort().values)
        assert torch.allclose(base.stats_ih.row_energy[row], perm.stats_ih.row_energy[row], atol=1e-5)
    assert torch.allclose(torch.tensor(base.stats_ih.energy), torch.tensor(perm.stats_ih.energy), atol=1e-5)
    # A single shared column permutation would make every row use masks.input_perm.
    global_like = base.ih[:, masks.input_perm]
    assert not torch.equal(global_like, perm.ih)


def test_c_matches_row_multisets_and_keeps_hidden_gate():
    _, _, structured = _builders("D")
    _, _, permuted = _builders("C")
    base = structured.build(0.07, False)
    other = permuted.build(0.07, False)
    assert torch.equal(base.hh, other.hh)
    assert not torch.equal(base.ih, other.ih)
    for row in range(base.ih.shape[0]):
        assert torch.equal(base.ih[row].sort().values, other.ih[row].sort().values)


def test_scalar_matches_layer_energy_not_the_row_distribution():
    _, _, structured = _builders("D")
    _, _, scalar_builder = _builders("A_prime")
    base = structured.build(0.07, False)
    scalar = scalar_builder.build(0.07, False)
    assert torch.allclose(scalar.ih, scalar.ih.flatten()[0].expand_as(scalar.ih))
    assert abs(scalar.stats_ih.energy - base.stats_ih.energy) < 1e-5
    assert float(base.stats_ih.row_energy_std) > 1e-3
    assert float(scalar.stats_ih.row_energy_std) < 1e-6
    ones = scalar_builder.build(None, True)
    assert torch.equal(ones.ih, torch.ones_like(ones.ih))
    assert ones.exact_dense


def test_gates_stay_inside_the_unit_interval_for_several_tau():
    geometry = make_task_geometry(100, (32, 32))
    for tau in (0.02, 0.07, 0.3, 1.0):
        gate, _ = gaussian_gate(geometry.dist_ih, tau)
        assert torch.isfinite(gate).all()
        assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0 + 1e-6
    # Near neighbors of a hidden unit are not disconnected at the planned start scale.
    from conn_exp.schedules import EXACT_FACTOR

    tau_start = 0.08 / EXACT_FACTOR
    gate, _ = gaussian_gate(geometry.dist_hh, tau_start)
    nearest = gate.max(dim=1).values
    assert torch.all(nearest > 0.5)
    assert int((gate > 0.1).sum(dim=1).min()) >= 3


def test_scalar_gate_square_matches_reference_energy():
    reference = torch.tensor([[0.2, 0.0], [1.0, 0.5]])
    gate = scalar_gate(reference)
    assert torch.allclose(gate.square().mean(), reference.square().mean(), atol=1e-6)
