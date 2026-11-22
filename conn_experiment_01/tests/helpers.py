from __future__ import annotations

from pathlib import Path

from conn_exp.config import load_config

REPO = Path(__file__).resolve().parents[1]


def tiny_config():
    config = load_config(REPO / "configs" / "pilot.yaml")
    return config.override(
        seed_list=(0,),
        n_train=32,
        n_val=16,
        n_test=16,
        total_steps=12,
        eval_interval=4,
        hidden_sizes=(8, 8),
        batch_size=8,
    )
