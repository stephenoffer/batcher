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
- A page is executed in its own empty temporary directory, so a block that *does*
  write a relative path (``write.parquet("events/a.parquet")``, a key file for a
  ``file:`` secret reference) leaves the repository clean. Running the suite used to
  drop those artifacts in the repo root, where an untracked ``aes.key`` sat waiting
  for the next ``git add -A``.
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
def test_doc_examples(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every non-skipped python block in one doc page, in order."""
    rel = path.relative_to(DOCS_ROOT)
    opt_in = bool(rel.parts) and rel.parts[0] in _DESIGN_SECTIONS
    blocks = _runnable_blocks(path.read_text(), opt_in=opt_in)
    if not blocks:
        pytest.skip("no runnable python blocks")
    # Each page gets its own empty directory as cwd. Examples are documented as
    # self-contained and in-memory, so nothing here reads a repo-relative path — but the
    # ones that *write* one (a parquet glob, a `file:` secret key) would otherwise land in
    # the repository root and be swept up by the next `git add -A`.
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, f"{path}#block{index}", "exec"), namespace)
        except Exception as exc:  # surface the failing block to the test report
            if _is_optional_backend_absence(exc):
                # A block that uses an optional backend (`iter_torch_batches`, a Ray
                # cluster) skips when that backend is absent, matching ci.yml's design:
                # it runs `.[dev]` without torch/ray, so these examples cannot execute and
                # must not fail the gate. When the backend *is* installed the block runs in
                # full, so a real breakage in the example still fails.
                pytest.skip(f"{path.relative_to(DOCS_ROOT)} block {index}: {exc}")
            pytest.fail(
                f"{path.relative_to(DOCS_ROOT)} block {index} failed: "
                f"{type(exc).__name__}: {exc}\n--- block ---\n{block}"
            )


# Optional backends ci.yml does not install; a block needing one skips rather than fails.
_OPTIONAL_BACKENDS = ("ray", "torch", "tensorflow", "vllm", "cuda")


def _is_optional_backend_absence(exc: BaseException) -> bool:
    """True if `exc` means an optional backend the block needs is not installed.

    ``MissingDependencyError`` is batcher's own typed error for a missing optional extra,
    so it is *never* a documentation typo — always safe to treat as skip. A bare ``import
    torch`` raises ``ModuleNotFoundError`` whose ``name`` the import system sets; a real
    typo (``import batcher.nonexistent``) names a non-backend module and still fails.
    """
    if type(exc).__name__ == "MissingDependencyError":
        return True
    if isinstance(exc, ModuleNotFoundError):
        return (exc.name or "").split(".")[0] in _OPTIONAL_BACKENDS
    return False
