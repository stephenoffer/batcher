"""Anti-drift gate: ``MAP.md`` still describes the tree it indexes.

``MAP.md`` is the file-level index agents and contributors grep instead of searching
690 modules, so a stale entry is worse than a missing one — it sends the reader to a
module that moved or misreports what one does. The map is generated from each
module's own docstring and each crate's manifest (``tools/gen_map.py``), which makes
staleness purely a question of whether someone re-ran the generator.

This test answers that question in milliseconds, rather than waiting for a human to
notice the map is wrong. If it fails, run ``just map`` and commit the result.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


def test_map_is_current() -> None:
    """``MAP.md`` matches what ``tools/gen_map.py`` would write today."""
    result = subprocess.run(
        [sys.executable, "tools/gen_map.py", "--check"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"MAP.md is out of date with the source tree.\n{result.stderr}"
