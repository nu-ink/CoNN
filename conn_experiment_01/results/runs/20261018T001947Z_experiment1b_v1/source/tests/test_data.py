from __future__ import annotations

import inspect

import numpy as np
import torch
from torch import nn

from conn_exp.config import load_config
from conn_exp.data import (
    assert_dataset_invariants,
    best_single_feature_accuracy,
    binary_predictor_accuracy,
    generate_dataset,
)
from conn_exp.geometry import make_task_geometry
from tests.helpers import REPO


def _pilot():
    return load_config(REPO / "configs" / "pilot.yaml")


def test_pilot_config_freezes_the_protocol():
    config = _pilot()
    assert config.task == "and"
    assert config.optimizer == "sgd"
    assert config.momentum == 0.0 and config.weight_decay == 0.0
    assert config.release_fraction == 0.7
    assert config.n_features == 100
    assert config.seed_list == (0, 1, 2, 3, 4)
    assert config.primary_control == "B_prime"
    assert config.primary_control_name == "row_shuffle"
    assert config.lr_selection == "shared_initial"
    assert config.lr == 0.1
    assert config.noise_std == 0.15
    assert config.distractor_amplitude == 0.5
    assert config.lr_grid == (0.01, 0.05, 0.1)
    assert f"{config.d50_start:.2f}" == "0.08"
    assert config.d50_end == 1.0


def test_config_rejects_optimizer_changes():
    config = _pilot()
    try:
        config.override(momentum=0.9)
    except ValueError as exc:
        assert "SGD" in str(exc)
    else:
        raise AssertionError("nonzero momentum was accepted")


def test_and_balance_cells_regions_and_determinism():
    config = _pilot()
    first = generate_dataset(config, seed=0)
    second = generate_dataset(config, seed=0)
    assert_dataset_invariants(first, config)
    assert np.array_equal(first.train.x, second.train.x)
    assert np.array_equal(first.test.y, second.test.y)
    assert first.fingerprint() == second.fingerprint()
    other = generate_dataset(config, seed=1)
    assert first.fingerprint() != other.fingerprint()
    assert float(first.train.y[first.train.motif_a == 0].mean()) == 0.0
    present = float(first.train.y[first.train.motif_a == 1].mean())
    assert 0.65 < present < 0.85
    motif_accuracy = binary_predictor_accuracy(first.train.motif_a, first.train.y)
    assert 0.75 < motif_accuracy < 0.90
    assert best_single_feature_accuracy(first.train.x, first.train.y) < 0.92
    assert (first.train.distractor_pos >= 0).all()


def test_xor_is_balanced_and_not_one_motif():
    config = _pilot().override(task="xor")
    data = generate_dataset(config, seed=2)
    assert_dataset_invariants(data, config)
    present = float(data.train.y[data.train.motif_a == 1].mean())
    assert 0.4 < present < 0.6
    assert binary_predictor_accuracy(data.train.motif_a, data.train.y) < 0.7


def test_noiseless_motif_is_written_at_the_recorded_index():
    config = _pilot().override(noise_std=0.0, n_distractors=0)
    data = generate_dataset(config, seed=0)
    template = np.asarray(config.template_a, dtype=np.float32) * np.float32(config.motif_amplitude)
    for row in range(data.train.x.shape[0]):
        if data.train.motif_a[row]:
            start = int(data.train.motif_a_pos[row])
            np.testing.assert_allclose(data.train.x[row, start : start + config.motif_length], template)


def test_splits_are_reused_objects_and_coordinates_have_no_label_argument():
    signature = inspect.signature(make_task_geometry)
    joined = " ".join(signature.parameters)
    assert "label" not in joined and "test" not in joined and "y" not in joined
    geometry = make_task_geometry(100, (32, 32))
    again = make_task_geometry(100, (32, 32))
    assert torch.equal(geometry.input_coords, again.input_coords)
    assert float(geometry.input_coords[0]) == 0.0
    assert float(geometry.input_coords[-1]) == 1.0
    gap = int(_pilot().region_b[0] - _pilot().region_a[1])
    assert gap >= 20


def test_dense_mlp_beats_chance_on_and():
    config = _pilot().override(n_train=256, n_val=64, n_test=64, noise_std=0.2, n_distractors=1, distractor_amplitude=0.5)
    data = generate_dataset(config, seed=0)
    state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(config.n_features, 32), nn.ReLU(), nn.Linear(32, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
        x = torch.from_numpy(data.train.x)
        y = torch.from_numpy(data.train.y)
        order = np.random.default_rng(0).permutation(len(y))
        for step in range(400):
            batch = order[np.arange(step * 32, step * 32 + 32) % len(y)]
            logits = model(x[batch]).reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = (model(torch.from_numpy(data.val.x)).reshape(-1) > 0).numpy()
        accuracy = float((pred == data.val.y.astype(bool)).mean())
    finally:
        torch.set_rng_state(state)
    assert accuracy > 0.65, accuracy
