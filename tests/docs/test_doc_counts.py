"""Anti-drift gate: a number the docs state about themselves is the number on disk.

The cookbook opens with "the cookbook is N runnable recipes", the landing page repeats it,
the quick reference repeats it again, and a generated diagram prints it a fourth time with a
per-domain breakdown. None of that was checked, so when a session added one recipe the site
said 145 in four places while the tree held 146, and the diagram's own caption disagreed
with the page it sat on.

A self-referential count is the easiest kind of claim to falsify and the easiest to miss in
review: it reads as a rounded, decorative figure rather than an assertion. It is an
assertion. This module holds the ones the docs make about their own contents to what a
directory walk returns.

Scope: counts of *documentation objects* (recipe pages, example scripts), which this test
can verify by walking the tree. Counts about the engine -- how many Polars names are mapped,
how many functions a namespace has -- are checked where they are generated, by
``tests/docs/test_migration_docs_fresh.py`` and ``tests/docs/test_api_coverage.py``.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_COOKBOOK = _DOCS / "cookbook"


def _recipe_pages() -> list[Path]:
    """Every cookbook page that is a recipe: any page that is not a section index."""
    return sorted(p for p in _COOKBOOK.rglob("*.md") if p.name != "index.md")


def _claimed(text: str, pattern: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(pattern, text)]


def test_the_cookbook_states_its_own_size_correctly() -> None:
    """Every page claiming a cookbook recipe count states the number on disk."""
    actual = len(_recipe_pages())
    claims = {
        "cookbook/index.md": r"The cookbook is (\d+) runnable recipes",
        "index.md": r"\| (\d+) runnable pages, from a one-method recipe",
        "api/reference.md": r"(\d+) runnable recipes, when the signature is not enough",
    }
    wrong = {}
    for rel, pattern in claims.items():
        found = _claimed((_DOCS / rel).read_text(encoding="utf-8"), pattern)
        assert found, f"{rel}: no recipe-count claim matched {pattern!r} (did the wording change?)"
        if any(n != actual for n in found):
            wrong[rel] = found
    assert not wrong, (
        f"{len(_recipe_pages())} recipe pages on disk, but these pages claim otherwise: "
        f"{wrong}. Update the prose, `tools/diagrams/cookbook_map.py`, and rerun that script."
    )


def test_the_cookbook_index_table_matches_the_domains() -> None:
    """The per-domain row counts on the cookbook index match the pages in each domain."""
    actual = collections.Counter(p.relative_to(_COOKBOOK).parts[0] for p in _recipe_pages())
    text = (_COOKBOOK / "index.md").read_text(encoding="utf-8")
    claimed = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"\| \{doc\}`/cookbook/([\w-]+)/index` \| (\d+) \|", text)
    }
    assert claimed, "no per-domain rows found on the cookbook index (did the table change?)"
    wrong = {d: (n, actual[d]) for d, n in claimed.items() if n != actual[d]}
    assert not wrong, (
        f"cookbook index rows disagree with the tree, as {{domain: (claimed, actual)}}: {wrong}"
    )


def test_the_example_library_page_totals_match_its_tables() -> None:
    """Each example page's own count matches the scripts its generated table covers.

    The tables are regenerated from the tree, so they are right by construction. The bar
    chart beside them, and its alt text, are hand-maintained numbers that were eight
    scripts stale when this test was written.
    """
    examples = _ROOT / "examples"
    marker = re.compile(r"<!-- library-table: ([^>]*?) -->")
    total = 0
    for page in sorted((_DOCS / "examples").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        listed = [m.group(1) for m in marker.finditer(text)]
        dirs = [d.strip() for group in listed for d in group.split(",") if d.strip()]
        for directory in dirs:
            root = examples if directory == "." else examples / directory
            total += sum(1 for q in root.glob("*.py") if not q.name.startswith("_"))
    index = (_DOCS / "examples" / "index.md").read_text(encoding="utf-8")
    stated = _claimed(index, r"Batcher ships (\d+) runnable example scripts")
    assert stated, "examples/index.md no longer states a script count"
    assert all(n == total for n in stated), (
        f"the example tables cover {total} scripts, examples/index.md claims {stated}. "
        "Update the prose, the chart alt text, and `tools/diagrams/example_library_map.py`."
    )


def test_the_example_library_states_its_own_size_correctly() -> None:
    """The tour and the landing page state the number of example scripts on disk.

    ``tools/example_library.py`` already regenerates the per-page tables from the tree, so
    the tables cannot drift. The *totals* quoted in prose elsewhere are hand-written and can.
    """
    actual = sum(
        1
        for p in (_ROOT / "examples").rglob("*.py")
        if not p.name.startswith("_") and "_common" not in p.parts
    )
    text = (_DOCS / "index.md").read_text(encoding="utf-8")
    found = _claimed(text, r"\| (\d+) standalone scripts, indexed by what each one shows")
    assert found, "the landing page no longer states an example-script count"
    assert all(n == actual for n in found), (
        f"{actual} example scripts on disk, landing page claims {found}"
    )
