"""Experiment 04: fixed relationship geometry, isolated from historical experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
# Import original modules without installing or editing the historical package.
sys.path.insert(0, str(ROOT / 'conn_experiment_01' / 'src'))
