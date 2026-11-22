"""Regression: the script must not shadow the stdlib inspect module."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def launch(*args):
    return subprocess.run([sys.executable, "scripts/inspect.py", *args], cwd=ROOT,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                          text=True, capture_output=True, timeout=30)


def test_direct_script_import_and_help():
    result = launch("--help")
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


def test_explicit_output_rejects_overwrite_and_experiment2_writes():
    original = ROOT / "audit/inspection.json"
    before = original.read_bytes()
    result = launch("--output", str(original))
    assert result.returncode == 2 and "already exists" in result.stderr
    result = launch("--output", str(ROOT.parent / "conn_experiment_01/forbidden.json"))
    assert result.returncode == 2 and "inside this experiment" in result.stderr
    assert original.read_bytes() == before
