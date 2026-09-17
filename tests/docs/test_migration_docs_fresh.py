"""Anti-drift gate: the generated migration pages still match the migration registry.

The per-engine pages under ``docs/getting-started/migration/{spark,polars,daft,ray-data}/``
are rendered from the registry rows by ``tools/gen_migration_docs.py``, the same rows the
``AttributeError`` guidance reads. A page edited by hand, or a registry row changed without
regenerating, would publish a mapping the engine no longer agrees with, and nothing else
executes a table cell. So this test renders every page in memory and compares it with the
committed file. If it fails, run ``just migration-docs`` and commit the result.
"""

from __future__ import annotations

import difflib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO / "tools" / "gen_migration_docs.py"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_migration_docs", _GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_every_engine_renders_pages() -> None:
    """Each engine gets an index, its pages and a way back, so the gate below is not vacuous."""
    gen = _generator()
    pages = gen.render()
    for engine in gen.ENGINES:
        root = gen.MIGRATION_DOCS / engine.directory
        assert root / "index.md" in pages
        assert root / "leaving-batcher.md" in pages
        assert len([p for p in pages if p.parent == root]) == len(engine.pages) + 2


def test_migration_docs_are_current() -> None:
    """Every generated page matches today's registry, and no stray file sits beside them."""
    gen = _generator()
    pages = gen.render()
    problems: list[str] = []
    for path, text in sorted(pages.items()):
        rel = path.relative_to(_REPO)
        if not path.exists():
            problems.append(f"missing: {rel}")
            continue
        committed = path.read_text()
        if committed != text:
            diff = difflib.unified_diff(
                committed.splitlines(), text.splitlines(), str(rel), "regenerated", lineterm=""
            )
            problems.append("\n".join(list(diff)[:20]))
    problems += [f"not generated: {p.relative_to(_REPO)}" for p in gen.unexpected(pages)]
    assert not problems, (
        "The migration reference pages are out of date with the registry. Run "
        "`just migration-docs` and commit the result.\n\n" + "\n\n".join(problems)
    )
