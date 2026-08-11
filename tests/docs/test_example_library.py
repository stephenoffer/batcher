"""The docs example library lists every script under ``examples/``.

A library that silently stops covering a script is worse than no library: the reader
concludes the surface is smaller than it is. ``tools/example_library.py`` regenerates the
tables from the tree, and this runs it in check mode so a new script that no page claims
fails the suite rather than going undocumented.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "docs" / "examples"


@pytest.mark.docs
def test_example_library_is_current() -> None:
    """Every example is in a table, and every table matches the tree."""
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "example_library.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.docs
def test_every_library_page_is_reachable() -> None:
    """Each library page is listed in the section's toctree."""
    index = (LIBRARY / "index.md").read_text()
    pages = sorted(p.stem for p in LIBRARY.glob("*.md") if p.stem != "index")
    missing = [name for name in pages if f"\n{name}\n" not in index]
    assert not missing, f"pages missing from the toctree in docs/examples/index.md: {missing}"
