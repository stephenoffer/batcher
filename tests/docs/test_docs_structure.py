"""Structural anti-drift gate: every docs page is reachable, and none is oversized.

Sphinx's ``-W`` build already fails on an orphan page (``toc.not_included``), but that
signal only arrives at the end of a full HTML build. This test catches the same
problem in seconds so a page added without a toctree entry fails fast — the recurring
failure mode when a new design note or guide lands under ``docs/`` but nobody wires it
into a ``{toctree}``.

The contract: every Markdown file under ``docs/`` is either
- the root document (``index.md``),
- listed as an entry in some ``{toctree}`` block, or
- named in ``exclude_patterns`` in ``conf.py`` (a deliberate non-page: a build
  helper, a contributor RFC, a PDF-only paper).

Anything else is an orphan.

The second contract is size. A page that grows past ``_MAX_PAGE_LINES`` has stopped
being one topic: it needs a bullet preview to navigate, it buries its own sections, and
it is the state every one of these pages reached before being split. The limit is
mechanical for the same reason the Python and Rust size limits are — a reviewer's
patience is not a gate. Genuine exceptions go in ``OVERSIZED_ALLOW`` with a reason.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_TOCTREE = re.compile(r"```\{toctree\}(.*?)```", re.DOTALL)

# One topic per page, and a topic fits in this many lines. Pages that outgrew it were
# split into a section with an index rather than allowlisted.
_MAX_PAGE_LINES = 500

# Published pages allowed past the limit, each with the reason. Drain toward empty.
OVERSIZED_ALLOW: dict[str, str] = {}


def _all_pages() -> set[str]:
    """Every doc name (posix, no suffix, relative to docs/) except the build tree."""
    return {
        p.relative_to(_DOCS).with_suffix("").as_posix()
        for p in _DOCS.rglob("*.md")
        if "_build" not in p.parts and "_static" not in p.parts
    }


def _toctree_entries() -> set[str]:
    """Every doc a ``{toctree}`` references, resolved relative to its own page."""
    entries: set[str] = set()
    for page in _DOCS.rglob("*.md"):
        if "_build" in page.parts:
            continue
        base = page.parent
        for block in _TOCTREE.findall(page.read_text(encoding="utf-8")):
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                # A caption entry may be `Title <path>`; keep the path.
                target = line.split("<")[-1].rstrip(">") if "<" in line else line
                entries.add((base / target).resolve().relative_to(_DOCS).as_posix())
    return entries


def _excluded_pages() -> set[str]:
    """Doc names that ``conf.py`` deliberately excludes from the build."""
    conf = (_DOCS / "conf.py").read_text(encoding="utf-8")
    tree = ast.parse(conf)
    patterns: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "exclude_patterns" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            patterns = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    excluded: set[str] = set()
    for pat in patterns:
        if pat.endswith(".md"):
            excluded.add(pat[:-3])
    return excluded


def test_no_orphan_pages():
    pages = _all_pages()
    reachable = _toctree_entries() | {"index"} | _excluded_pages()
    orphans = sorted(pages - reachable)
    assert not orphans, (
        f"{len(orphans)} docs page(s) are in no toctree and not excluded: {orphans}\n"
        "Add each to a `{toctree}` on its section index, or to `exclude_patterns` in "
        "docs/conf.py if it is a contributor note rather than a published page."
    )


def test_toctree_entries_all_exist():
    pages = _all_pages()
    # autosummary emits generated stubs under api/generated/ at build time; a toctree
    # may legitimately point at pages created by other directives, so only flag entries
    # that look like hand-written docs (no `generated/` segment) yet have no source file.
    missing = sorted(e for e in _toctree_entries() if "generated/" not in e and e not in pages)
    assert not missing, (
        f"{len(missing)} toctree entr(y/ies) point at a nonexistent page: {missing}\n"
        "Fix the path or remove the stale entry."
    )


def test_no_oversized_pages():
    excluded = _excluded_pages()
    oversized: list[tuple[str, int]] = []
    for page in sorted(_DOCS.rglob("*.md")):
        if "_build" in page.parts or "_static" in page.parts:
            continue
        name = page.relative_to(_DOCS).with_suffix("").as_posix()
        if name in excluded or name in OVERSIZED_ALLOW:
            continue
        lines = len(page.read_text(encoding="utf-8").splitlines())
        if lines > _MAX_PAGE_LINES:
            oversized.append((name, lines))

    assert not oversized, (
        f"{len(oversized)} docs page(s) over {_MAX_PAGE_LINES} lines: "
        + ", ".join(f"{n} ({c})" for n, c in oversized)
        + "\nSplit each into a section with its own index and toctree, or add it to "
        "OVERSIZED_ALLOW with a one-line reason."
    )

    stale = sorted(
        name
        for name in OVERSIZED_ALLOW
        if (_DOCS / f"{name}.md").exists()
        and len((_DOCS / f"{name}.md").read_text(encoding="utf-8").splitlines()) <= _MAX_PAGE_LINES
    )
    assert not stale, f"OVERSIZED_ALLOW lists pages that now fit (remove them): {stale}"
