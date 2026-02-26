from __future__ import annotations

import os
import pathlib
import sys

# Ensure headless-safe plotting in local/CI test runs.
os.environ.setdefault("MPLBACKEND", "Agg")
MPL_DIR = pathlib.Path(__file__).resolve().parents[1] / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
