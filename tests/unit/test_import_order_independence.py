"""Public-API modules import cleanly as the *first* import of a fresh interpreter.

`import batcher` walks the package in one fixed order, which hides a cycle that only bites
when something else is imported first. `batcher.api.groupby` was one: it imported
`api.dataset.compat.guidance`, whose package initializer imports `frame`, which imports
`groupby` while it is half-initialized. Under `pytest -n 8` that made a docs test pass or
fail depending on which module a worker happened to import first. A fresh process per
module is the only way to see it, since one interpreter caches the first successful import.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "batcher.api.groupby",
    "batcher.api.multi_group",
    "batcher.api.dataset",
    "batcher.api.dataset.frame",
    "batcher.api.functions",
    "batcher.api.terminal",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"`import {module}` as the first import failed:\n{proc.stderr}"
