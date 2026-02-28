"""Compatibility wrapper. Canonical script lives in examples/adaptive_example.py."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "examples" / "adaptive_example.py"),
    run_name="__main__",
)
