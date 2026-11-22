import numpy as np
import pandas as pd
import pytest
import torch

from conn_exp.experiment1b import ValidationOnly, select_validation, seal, checkpoint_ablation, run_phase
from conn_exp.experiment1b_report import summarize
from conn_exp.geometry import make_task_geometry
from conn_exp.models import CoNNMLP
from tests.helpers import tiny_config


def candidates():
    return pd.DataFrame([dict(condition_id=c, total_steps=s, lr=lr, seed=seed,
        val_accuracy=.7 + (.05 if lr == .1 else 0) + (.02 if s == 800 else 0),
        val_loss=.5, test_accuracy=1 if lr == .01 else 0)
        for c in ["D", "B_prime"] for s in [400, 800] for lr in [.01, .1] for seed in [5, 6]])


def test_selection_uses_validation_and_complete_equal_budgets():
    frame = candidates()
    _, _, _, steps, rates = select_validation(frame)
    assert steps == 800 and rates == {"D": .1, "B_prime": .1}
    frame.test_accuracy = np.nan
    assert select_validation(frame)[3:] == (steps, rates)
    with pytest.raises(ValueError, match="Unequal"):
        select_validation(frame.iloc[1:])
    with pytest.raises(ValueError, match="Incomplete"):
        select_validation(frame[~((frame.condition_id == "D") & (frame.lr == .01) & (frame.total_steps == 400))])


def test_validation_ties_use_loss_then_lr_then_duration():
    frame = candidates()
    frame.val_accuracy = .75
    frame.val_loss = .5
    assert select_validation(frame)[3:] == (400, {"D": .01, "B_prime": .01})
    frame.loc[frame.lr == .1, "val_loss"] = .4
    assert select_validation(frame)[4] == {"D": .1, "B_prime": .1}


def test_test_access_forbidden_and_freeze_exclusive(tmp_path):
    with pytest.raises(AssertionError, match="forbidden"):
        ValidationOnly().evaluate()
    path = tmp_path / "freeze.json"
    seal(path, {"steps": 400})
    with pytest.raises(FileExistsError):
        seal(path, {"steps": 3200})


def test_paired_stats_and_missing_data_rejection():
    result = summarize([.01, .02, .03])
    assert result["mean"] == pytest.approx(.02)
    assert result["sd"] == pytest.approx(.01)
    assert result["cohen_dz"] == pytest.approx(2)
    assert result["ci_low"] < result["mean"] < result["ci_high"]
    with pytest.raises(ValueError):
        summarize([.01, np.nan])


def test_checkpoint_ablation_reloads_without_mutation(tmp_path):
    config = tiny_config()
    model = CoNNMLP(config.n_features, config.hidden_sizes)
    geometry = make_task_geometry(config.n_features, config.hidden_sizes)
    folder = tmp_path / "checkpoints/seed_0/D"
    folder.mkdir(parents=True)
    path = folder / "final.pt"
    torch.save(dict(model=model.state_dict(), gate_ih=torch.ones_like(geometry.dist_ih),
        gate_hh=torch.ones_like(geometry.dist_hh), seed=0, condition_id="D",
        optimizer_steps_completed=config.total_steps, measure_timing="post_step", phase="exact_dense"), path)
    before = path.read_bytes()
    x = torch.randn(16, config.n_features)
    y = torch.randint(0, 2, (16,)).float()
    checkpoint_ablation(tmp_path, config, geometry, x, y)
    assert path.read_bytes() == before
    frame = pd.read_csv(tmp_path / "checkpoint_ablation.csv")
    assert len(frame) == 3 and set(frame.intervention) == {"ih", "hh", "both"}
    assert set(frame.split) == {"val"}
    assert np.allclose(frame.val_loss_delta, frame.suppressed_val_loss-frame.val_loss)


def test_validation_only_training_all_seven_conditions(tmp_path):
    from conn_exp.runtime import configure_runtime
    from conn_exp.schema import CONDITIONS
    configure_runtime()
    config = tiny_config()
    scores = run_phase(tmp_path / "tuning", config, [5],
        [(config.total_steps, .1, c.id) for c in CONDITIONS])
    assert len(scores) == 7 and (scores.test_evaluations == 0).all()
    assert "test_accuracy" not in scores.columns
    for job in scores.job:
        metrics = pd.read_csv(tmp_path / "tuning" / job / "metrics.csv")
        assert "test" not in set(metrics.split)
        assert len(pd.read_csv(tmp_path / "tuning" / job / "checkpoint_ablation.csv")) == 9
