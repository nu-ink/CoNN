"""CLI entry: python scripts/inspect.py, from this experiment directory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conn_assoc.inspect import main

if __name__ == "__main__":
    raise SystemExit(main())
