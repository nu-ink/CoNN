from __future__ import annotations

import torch

from conn_exp.data import generate_dataset
from conn_exp.gates import GateBuilder, draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.models import CoNNMLP, build_optimizer, initial_state_dict, trainable_count
from conn_exp.schema import CONDITIONS
from conn_exp.train import HeldOutSplit, minibatch_order, train_condition
from conn_exp.config import load_config
from tests.helpers import REPO, tiny_config


def _prepare(config, seed: int = 0):
    data = generate_dataset(config, seed)
    geometry = make_task_geometry(config.n_features, config.hidden_sizes)
    masks = draw_masks(config.n_features, config.hidden_sizes[0], config.hidden_sizes[1], seed)
    init = initial_state_dict(config.n_features, config.hidden_sizes, seed)
    order = minibatch_order(config.n_train, seed)
    device = torch.device("cpu")
    tensors = {
        "x_train": torch.from_numpy(data.train.x),
        "y_train": torch.from_numpy(data.train.y),
        "x_val": torch.from_numpy(data.val.x),
        "y_val": torch.from_numpy(data.val.y),
        "x_test": torch.from_numpy(data.test.x),
        "y_test": torch.from_numpy(data.test.y),
        "motif_a_train": data.train.motif_a,
        "motif_b_train": data.train.motif_b,
        "motif_a_val": data.val.motif_a,
        "motif_b_val": data.val.motif_b,
    }
    return data, geometry, masks, init, order, device, tensors


def _train(config, condition, diagnostics, start=0, stop=None, model=None, optimizer=None, init=None, bundle=None):
    bundle = bundle or _prepare(config)
    _, geometry, masks, default_init, order, device, tensors = bundle
    init = default_init if init is None else init
    if model is None:
        model = CoNNMLP(config.n_features, config.hidden_sizes)
        model.load_state_dict(init)
        optimizer, hparams = build_optimizer(model, config.lr)
    else:
        _, hparams = build_optimizer(model, config.lr)
        if optimizer is None:
            raise AssertionError("continuing a run requires the existing optimizer")
    access = HeldOutSplit(tensors["x_test"], tensors["y_test"])
    result = train_condition(
        model=model,
        optimizer=optimizer,
        optimizer_hparams=hparams,
        config=config,
        gate_builder=GateBuilder(geometry, masks, condition, config.tau_floor),
        order=order,
        x_train=tensors["x_train"],
        y_train=tensors["y_train"],
        x_val=tensors["x_val"],
        y_val=tensors["y_val"],
        motif_a_train=tensors["motif_a_train"],
        motif_b_train=tensors["motif_b_train"],
        motif_a_val=tensors["motif_a_val"],
        motif_b_val=tensors["motif_b_val"],
        test_access=access,
        base_dist_ih=geometry.dist_ih,
        base_dist_hh=geometry.dist_hh,
        seed=0,
        condition_id=condition,
        run_id="unit",
        data_fingerprint="data",
        init_fingerprint="init",
        start_step=start,
        stop_step=stop,
        diagnostics=diagnostics,
    )
    return model, optimizer, result


def test_forward_does_not_write_w_and_bias_is_ungated():
    model = CoNNMLP(100, (8, 8))
    generator = torch.Generator().manual_seed(5)
    from conn_exp.models import initialize_model

    initialize_model(model, generator)
    before = model.fc1.weight.detach().clone()
    geometry = make_task_geometry(100, (8, 8))
    builder = GateBuilder(geometry, draw_masks(100, 8, 8, 0), "D", 1e-6)
    pack = builder.build(0.08, False)
    features = torch.randn(4, 100)
    _, cache = model(features, pack.ih, pack.hh)
    assert torch.equal(before, model.fc1.weight)
    assert cache.w_eff_ih.data_ptr() != model.fc1.weight.data_ptr()
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc2.weight.zero_()
        model.out_weight.zero_()
        model.fc1.bias.fill_(0.4)
        model.fc2.bias.fill_(0.3)
        model.out_bias.fill_(-0.2)
    ones = torch.ones_like(pack.ih), torch.ones_like(pack.hh)
    zeros = torch.zeros_like(pack.ih), torch.zeros_like(pack.hh)
    first, _ = model(features, *ones)
    second, _ = model(features, *zeros)
    assert torch.allclose(first, second)


def test_conditions_share_initialization_and_parameter_count():
    config = tiny_config()
    _, _, _, init, _, _, _ = _prepare(config)
    counts = set()
    for condition in CONDITIONS:
        model = CoNNMLP(config.n_features, config.hidden_sizes)
        model.load_state_dict(init)
        counts.add(trainable_count(model))
        again = initial_state_dict(config.n_features, config.hidden_sizes, 0)
        for key, value in again.items():
            assert torch.equal(value, init[key])
        assert "row_perm" not in "".join(model.state_dict())
    assert len(counts) == 1
    model = CoNNMLP(config.n_features, config.hidden_sizes)
    optimizer, hparams = build_optimizer(model, 0.05)
    assert hparams["name"] == "sgd"
    assert hparams["momentum"] == 0.0 and hparams["weight_decay"] == 0.0
    assert type(optimizer).__name__ == "SGD"


def test_all_conditions_stay_finite_and_repeat():
    config = tiny_config()
    bundle = _prepare(config)
    for condition in CONDITIONS:
        _, _, result = _train(config, condition.id, diagnostics=False, bundle=bundle)
        assert result.non_finite is False
        assert result.test_evaluations == 1
        losses = [row["loss"] for row in result.metrics if row["split"] == "val"]
        assert losses and all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    model_a, _, _ = _train(config, "D", diagnostics=False, bundle=bundle)
    model_b, _, _ = _train(config, "D", diagnostics=False, bundle=bundle)
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(left, right)


def test_diagnostics_do_not_change_the_update_and_resume_matches():
    config = tiny_config()
    bundle = _prepare(config)
    with_diag, _, _ = _train(config, "D", diagnostics=True, bundle=bundle)
    without, _, _ = _train(config, "D", diagnostics=False, bundle=bundle)
    for left, right in zip(with_diag.parameters(), without.parameters()):
        assert torch.equal(left, right)
    partial, optimizer, partial_result = _train(config, "B_prime", diagnostics=True, stop=4, bundle=bundle)
    assert partial_result.test_evaluations == 0
    continued = CoNNMLP(config.n_features, config.hidden_sizes)
    continued.load_state_dict(partial.state_dict())
    continued_opt, _ = build_optimizer(continued, config.lr)
    continued_opt.load_state_dict(optimizer.state_dict())
    continued, _, _ = _train(
        config,
        "B_prime",
        diagnostics=True,
        start=4,
        stop=config.total_steps,
        model=continued,
        optimizer=continued_opt,
        bundle=bundle,
    )
    full, _, _ = _train(config, "B_prime", diagnostics=True, bundle=bundle)
    for left, right in zip(continued.parameters(), full.parameters()):
        assert torch.allclose(left, right, atol=0, rtol=0)


def test_dense_condition_beats_chance_on_and():
    config = load_config(REPO / "configs" / "pilot.yaml").override(seed_list=(0,), eval_interval=100)
    _, _, result = _train(config, "A", diagnostics=False)
    final = [
        row
        for row in result.metrics
        if row["split"] == "val"
        and row["measure_timing"] == "post_step"
        and row["optimizer_steps_completed"] == config.total_steps
    ]
    assert len(final) == 1
    assert final[0]["accuracy"] > 0.75, final[0]["accuracy"]
