"""Execute the standalone scripts under ``examples/``.

Every ``examples/*.py`` script is run end to end against the built engine, so a
demonstrated workflow that references a removed or renamed API fails the suite
instead of silently rotting. This is the usage-coverage counterpart to
``tests/docs/test_doc_examples.py`` (which runs the fenced blocks embedded in the
docs).

Contract for example authors:

- A script runs by default. A script whose first non-blank, non-shebang line is
  ``# examples: skip`` is collected but not executed; use it only for scripts that
  genuinely need external infrastructure (a live warehouse endpoint, a real GPU, a
  cloud bucket). Such scripts must still import cleanly and show the real API shape.
- Scripts should be self-contained: build their own in-memory or temp-dir data and
  assert on their own output, so the suite needs no fixtures.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"
_SKIP = "# examples: skip"


def _example_files() -> list[Path]:
    if not EXAMPLES_ROOT.is_dir():
        return []
    # Recursive: the per-API-family scripts live in subdirectories (`expressions/`,
    # `metrics/`, ...) so the tree stays navigable as it grows. A flat glob would
    # collect the handful at the root and silently skip every one of those.
    #
    # Underscore-prefixed parts are support code, not examples: `_common/` holds the
    # shared S3 dataset loader and device-selection helpers that the scripts import.
    # Running those as scripts would prove nothing, and `__pycache__` is not source.
    return sorted(
        p
        for p in EXAMPLES_ROOT.rglob("*.py")
        if not any(part.startswith("_") for part in p.relative_to(EXAMPLES_ROOT).parts)
    )


def _is_skipped(text: str) -> bool:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#!"):
            continue
        return line.startswith(_SKIP)
    return False


def test_examples_exist() -> None:
    """The example tree is non-empty.

    `test_example_runs` is parametrized over `_example_files()`, so an empty result is not
    a failure — it collects **zero** tests and the run still prints green. Every guard in
    this file (does the script still import, does the API it demonstrates still exist,
    does it still produce the right answer) would then be switched off by a moved
    directory or a renamed glob, with nothing anywhere going red. This is the assertion
    that cannot be satisfied vacuously.
    """
    assert _example_files(), f"no example scripts found under {EXAMPLES_ROOT}"


@pytest.mark.docs
@pytest.mark.integration
@pytest.mark.parametrize(
    "path", _example_files(), ids=lambda p: p.name if isinstance(p, Path) else str(p)
)
def test_example_runs(path: Path) -> None:
    """Run one example script to completion as ``__main__``."""
    if _is_skipped(path.read_text()):
        pytest.skip(f"{path.name} needs external infrastructure (# examples: skip)")
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:  # a clean sys.exit(0) is success
        if exc.code not in (0, None):
            pytest.fail(f"{path.name} exited with code {exc.code}")
