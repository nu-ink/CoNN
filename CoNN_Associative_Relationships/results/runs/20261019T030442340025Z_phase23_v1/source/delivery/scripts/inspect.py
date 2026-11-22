"""CLI entry: python scripts/inspect.py, from this experiment directory."""

from __future__ import annotations

import sys
from pathlib import Path

# A direct script launch puts scripts/ first on sys.path. This file must not
# shadow Python's standard-library inspect module when NumPy/PyTorch import it.
script_dir = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != script_dir]
sys.path.insert(0, str(script_dir.parent / "src"))

from conn_assoc.inspect import main

if __name__ == "__main__":
    raise SystemExit(main())
