from __future__ import annotations

import inspect
import json
import subprocess
import sys

import torch

from conn_exp.data import generate_dataset
from conn_exp.runner import execute, regenerate
from conn_exp.schema import CONDITIONS, DISTANCE_DEFINITION
from conn_exp.seeding import paired_seed
from tests.helpers import REPO, tiny_config


def test_streams_are_not_condition_specific_and_do_not_interfere():
    signature = inspect.signature(paired_seed)
    assert "condition" not in signature.parameters
    assert paired_seed(0, "data") != paired_seed(0, "mask")
    assert paired_seed(0, "init") != paired_seed(0, "order")
    from conn_exp.gates import draw_masks
    from conn_exp.models import initial_state_dict

    first_init = initial_state_dict(100, (8, 8), 0)
    draw_masks(100, 8, 8, 0)
    second_init = initial_state_dict(100, (8, 8), 0)
    for key in first_init:
        assert torch.equal(first_init[key], second_init[key])


def test_dataset_generator_is_called_once_per_seed(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = generate_dataset

    def wrapped(config, seed):
        calls["n"] += 1
        return real(config, seed)

    monkeypatch.setattr("conn_exp.runner.generate_dataset", wrapped)
    code = execute(tiny_config(), tmp_path / "once", "unit", False)
    assert code == 0
    assert calls["n"] == 1


def test_tiny_run_writes_the_contract_and_release_checkpoints(tmp_path):
    run_dir = tmp_path / "tiny"
    assert execute(tiny_config(), run_dir, "unit-tiny", False) == 0
    for name in (
        "config.resolved.yaml",
        "config_sha256.txt",
        "environment.json",
        "protocol.json",
        "metrics.csv",
        "gradients.csv",
        "ablation.csv",
        "probes.csv",
        "row_energy.csv",
        "scores.csv",
        "paired_contrast.csv",
        "summary.json",
        "report.md",
        "data/seed_0.npz",
        "masks/seed_0.pt",
    ):
        assert (run_dir / name).is_file(), name
    for condition in CONDITIONS:
        folder = run_dir / "checkpoints" / "seed_0" / condition.id
        pre = torch.load(folder / "gate_pre_release.pt", weights_only=False)
        post = torch.load(folder / "gate_post_release.pt", weights_only=False)
        final = torch.load(folder / "final.pt", weights_only=False)
        for key, value in pre["model"].items():
            assert torch.equal(value, post["model"][key])
        assert pre["exact_dense"] is False or condition.id in {"A", "S"}
        if condition.id == "D":
            assert pre["exact_dense"] is False
            assert post["exact_dense"] is True
            assert not torch.equal(pre["gate_ih"], post["gate_ih"])
        if condition.id == "S":
            assert final["exact_dense"] is False
        if condition.id in {"A", "D", "E", "A_prime", "B_prime", "C"}:
            assert final["exact_dense"] is True
        assert final["permutation_rule"] == "within_row_column_permutation_frozen"
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["primary_control_name"] == "row_shuffle"
    assert protocol["optimizer"] == "sgd"
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "not a result" in report
    assert DISTANCE_DEFINITION
    regenerate(run_dir)
    for plot in (
        "loss_accuracy.png",
        "gate_energy.png",
        "tau_d50_schedule.png",
        "row_energy.png",
        "gradient_distance.png",
        "ablation_delta.png",
        "seed_scatter_D_minus_B_prime.png",
    ):
        assert (run_dir / "plots" / plot).is_file()


def test_cli_help_lists_the_pilot_commands():
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "conn_exp.runner", "--help"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "configs/pilot.yaml --smoke" in completed.stdout
    assert "configs/pilot.yaml --seeds 5" in completed.stdout
