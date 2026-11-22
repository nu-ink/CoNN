from __future__ import annotations

import inspect

import numpy as np
import torch

from conn_exp.diagnostics import bin_mean_abs, far_suppression, fit_and_score_probe, prestep_gradients
from conn_exp.gates import GateBuilder, draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.models import CoNNMLP, initialize_model
from conn_exp.train import evaluate_split


def test_gradient_names_match_the_chain_rule():
    model = CoNNMLP(100, (8, 8))
    initialize_model(model, torch.Generator().manual_seed(3))
    geometry = make_task_geometry(100, (8, 8))
    pack = GateBuilder(geometry, draw_masks(100, 8, 8, 1), "D", 1e-6).build(0.07, False)
    features = torch.randn(16, 100)
    labels = torch.randint(0, 2, (16,)).float()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    grads = prestep_gradients(model, features, labels, pack.ih, pack.hh)
    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key])
    for layer, gate in (("ih", pack.ih), ("hh", pack.hh)):
        attenuated = grads[layer]["dL_dW"]
        data_gradient = grads[layer]["dL_dW_eff"]
        assert torch.allclose(attenuated, data_gradient * gate, atol=1e-5, rtol=1e-4)
        assert torch.all(attenuated.abs() <= data_gradient.abs() + 1e-5)
    rows = bin_mean_abs(grads["ih"]["dL_dW"], geometry.dist_ih, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert sum(row["n_weights"] for row in rows) == grads["ih"]["dL_dW"].numel()
    assert "dL_dW" in grads["ih"] and "dL_dW_eff" in grads["ih"]


def test_far_ablation_uses_base_distance_and_does_not_write_weights():
    model = CoNNMLP(100, (8, 8))
    initialize_model(model, torch.Generator().manual_seed(4))
    geometry = make_task_geometry(100, (8, 8))
    pack = GateBuilder(geometry, draw_masks(100, 8, 8, 2), "B_prime", 1e-6).build(0.07, False)
    features = torch.randn(20, 100)
    labels = torch.randint(0, 2, (20,)).float()
    before = model.fc1.weight.detach().clone()
    suppressed = far_suppression(pack.ih, pack.hh, geometry.dist_ih, geometry.dist_hh, 0.75)
    evaluate_split(model, features, labels, suppressed["gate_ih"], suppressed["gate_hh"])
    assert torch.equal(before, model.fc1.weight)
    assert 0.05 < suppressed["fraction_suppressed_ih"] < 0.95
    assert suppressed["n_suppressed_ih"] > 0


def test_probe_fits_train_only():
    signature = inspect.signature(fit_and_score_probe)
    assert "test" not in signature.parameters
    generator = np.random.default_rng(0)
    train_y = generator.integers(0, 2, size=40).astype(np.float64)
    val_y = generator.integers(0, 2, size=24).astype(np.float64)
    train_acc, val_acc, status = fit_and_score_probe(
        train_y.reshape(-1, 1), train_y, val_y.reshape(-1, 1), val_y, 1e-3
    )
    assert status == "ok"
    assert train_acc == 1.0
    assert val_acc == 1.0
