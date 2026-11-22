"""Paths inside this experiment. Nothing here writes into Experiment 2."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = EXPERIMENT_ROOT / "PROTOCOL_v2.md"
MANIFEST_PATH = EXPERIMENT_ROOT / "MANIFEST.json"
CONFIG_PATH = EXPERIMENT_ROOT / "config" / "phase01.yaml"


def experiment2_root() -> Path:
    return (EXPERIMENT_ROOT / ".." / "conn_experiment_01").resolve()


def experiment2_run() -> Path:
    return experiment2_root() / "results" / "runs" / "20261019T004346Z_experiment2_har_v1"


def archive_path() -> Path:
    return experiment2_root() / "data" / "raw" / "uci_har_original.zip"
