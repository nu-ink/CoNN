"""Gaussian-gate invariants and equivalence with the imported Experiment 2 gate."""

import math

import pytest
import torch

from conn_assoc.gate import effective_weight, gaussian_gate, resolve_tau, training_effective_weight
from conn_assoc.geometry import (
    engineered_input_coordinates,
    pairwise_abs_distance,
    temporal_input_coordinates,
    uniform_coordinates,
)
from conn_exp.gates import gaussian_gate as original_gaussian_gate
from conn_exp.geometry import pairwise_abs_distance as original_distance
from conn_exp.geometry import uniform_coordinates as original_uniform
from conn_exp.models import GatedLinear


def test_gate_is_in_unit_interval_and_underflow_may_be_zero():
    distance = torch.linspace(0, 2, 21)
    gate, floored = gaussian_gate(distance, 0.2)
    assert floored is False
    assert torch.all(gate > 0) and torch.all(gate <= 1)
    underflow, floored = gaussian_gate(torch.tensor([1.0]), 1e-8)
    assert floored is True
    assert underflow.item() == 0.0
    assert float(underflow.min()) >= 0.0


def test_gate_equals_one_at_zero_distance():
    gate, floored = gaussian_gate(torch.zeros(3, 4), 1e-8)
    assert floored is True
    assert torch.equal(gate, torch.ones(3, 4))


def test_increasing_tau_does_not_decrease_gate():
    distance = torch.tensor([0.0, 0.3, 1.0, 4.0])
    previous = None
    for tau in (1e-4, 1e-3, 0.01, 0.1, 0.5, 2.0):
        gate, floored = gaussian_gate(distance, tau)
        assert floored is False
        if previous is not None:
            assert torch.all(gate >= previous)
        previous = gate


def test_positive_tau_matches_imported_implementation():
    distance = torch.tensor([[0.0, 0.1, 1.0], [0.5, 2.0, 10.0]], dtype=torch.float64)
    for tau in (1e-8, 1e-6, 0.06794574402304153, 0.25, 1.0, 3.0):
        actual, actual_flag = gaussian_gate(distance, tau, tau_floor=1e-6)
        expected, expected_flag = original_gaussian_gate(distance, tau, tau_floor=1e-6)
        assert actual_flag is expected_flag
        assert torch.equal(actual, expected)


def test_tau_floor_is_configurable_and_non_positive_tau_fails():
    distance = torch.tensor([0.5])
    gate, floored = gaussian_gate(distance, 1e-4, tau_floor=1e-2)
    expected, expected_flag = original_gaussian_gate(distance, 1e-4, tau_floor=1e-2)
    assert floored is True and expected_flag is True
    assert torch.equal(gate, expected)
    with pytest.raises(ValueError, match="strictly positive"):
        resolve_tau(0.0, 1e-6)
    with pytest.raises(ValueError, match="strictly positive"):
        gaussian_gate(distance, -0.2)
    with pytest.raises(ValueError, match="tau_floor"):
        gaussian_gate(distance, 0.2, tau_floor=0.0)
    with pytest.raises(ValueError):
        gaussian_gate(distance, math.nan)


def test_effective_weight_and_original_module_detach_the_gate():
    weight = torch.nn.Parameter(torch.randn(4, 5))
    gate = torch.randn(4, 5).abs().clamp(max=1).requires_grad_()
    product = effective_weight(weight, gate)
    assert torch.equal(product, weight * gate)
    layer = GatedLinear(5, 4)
    with torch.no_grad():
        layer.weight.copy_(weight.detach())
        layer.bias.zero_()
    _, applied = layer(torch.randn(2, 5), gate)
    assert torch.equal(applied, training_effective_weight(weight, gate))
    applied.sum().backward()
    assert gate.grad is None
    assert layer.weight.grad is not None


def test_geometry_matches_imported_coordinates():
    for width in (1, 2, 64):
        assert torch.equal(uniform_coordinates(width), original_uniform(width))
    hidden = uniform_coordinates(8)
    source = temporal_input_coordinates()
    distance = pairwise_abs_distance(hidden, source)
    assert distance.shape == (8, 128 * 9)
    assert torch.equal(distance, original_distance(hidden, source))
    engineered = engineered_input_coordinates(561)
    assert engineered[0].item() == 0.0
    assert torch.allclose(engineered[1], torch.tensor(1 / 560))
    assert torch.equal(pairwise_abs_distance(hidden, engineered), original_distance(hidden, engineered))
    same_time = temporal_input_coordinates().reshape(128, 9)
    assert torch.all(same_time == same_time[:, :1])
