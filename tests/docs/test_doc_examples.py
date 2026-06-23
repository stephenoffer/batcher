"""Execute the code examples embedded in the documentation.

Every fenced ``python`` block under ``docs/`` is extracted and run against the
built engine, so a documented example that references a removed or renamed API
fails the test suite instead of silently rotting (which is how the v1 docs
drifted out of sync with the code).

Contract for doc authors:

- Blocks run in document order, sharing one namespace per file, so a page may
  open with a setup block (imports plus a small ``from_pydict`` dataset) and then
  build on it in later blocks.
- In the user-facing guide sections, every python block runs by default. A block
  whose first line is ``# docs: skip`` is shown in the docs but not executed; use
  it for examples that need external resources (cloud object stores, a Ray
  cluster, a GPU, or a real model).
- The design sections (``architecture/`` and ``internals/``) carry illustrative
  pseudo-code, so their blocks are not executed unless the first line is
  ``# docs: run``.
- Examples should be self-contained within a page and use in-memory data
  (``bt.from_pydict``), so the suite needs no fixtures on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

# ```python ... ``` fenced blocks, capturing the body.
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_SKIP = "# docs: skip"
_RUN = "# docs: run"

# Sections of illustrative design docs: blocks are opt-in via ``# docs: run``.
_DESIGN_SECTIONS = {"architecture", "internals"}


def _doc_files() -> list[Path]:
    return sorted(p for p in DOCS_ROOT.rglob("*.md") if "_build" not in p.parts)


def _runnable_blocks(text: str, *, opt_in: bool) -> list[str]:
    blocks = []
    for match in _BLOCK.finditer(text):
        body = match.group(1)
        first = body.lstrip()
        if opt_in:
            if first.startswith(_RUN):
                blocks.append(body)
        elif not first.startswith(_SKIP):
            blocks.append(body)
    return blocks


@pytest.mark.docs
@pytest.mark.integration
@pytest.mark.parametrize("path", _doc_files(), ids=lambda p: str(p.relative_to(DOCS_ROOT)))
def test_doc_examples(path: Path) -> None:
    """Run every non-skipped python block in one doc page, in order."""
    rel = path.relative_to(DOCS_ROOT)
    opt_in = bool(rel.parts) and rel.parts[0] in _DESIGN_SECTIONS
    blocks = _runnable_blocks(path.read_text(), opt_in=opt_in)
    if not blocks:
        pytest.skip("no runnable python blocks")
    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, f"{path}#block{index}", "exec"), namespace)
        except Exception as exc:  # surface the failing block to the test report
            pytest.fail(
                f"{path.relative_to(DOCS_ROOT)} block {index} failed: "
                f"{type(exc).__name__}: {exc}\n--- block ---\n{block}"
            )
