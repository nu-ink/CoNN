"""CLI without editable installation; keeps legacy scripts/inspect.py off sys.path."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=ROOT/'scripts']
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'conn_experiment_01/src')]
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from conn_assoc.phase23 import main

if __name__=='__main__':main()
