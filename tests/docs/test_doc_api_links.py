"""Anti-drift gate: an API name in a published page is a working cross-reference.

A reference that doesn't link is the failure this catches, and it hides well. Sphinx runs
with ``nitpicky`` off, so ``{py:meth}`Dataset.collapse``` at a target nobody registered
renders as plain grey text and the ``-W`` build stays green: a dead reference and a name
that was never linked at all are indistinguishable in the output. Nothing in the existing
gates looks at either, which is how 3,592 code spans naming a public symbol came to sit in
the docs as unclickable text, 2,000 of them in the generated migration tables where the
Batcher spelling is the only thing on the row a reader needs to look up.

``tools/doc_api_links.py`` closes both halves at once. It derives the set of targets from
the same ``autosummary``/``autoclass`` directives that publish them, so a reference it
emits is one Sphinx registers, and it reports the spans that name a published symbol
without linking to it.

The two tests here are that gate and its foundation: that the index still resolves the
names the site is built around, and that no published hand-written page has an unlinked
one left.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "tools" / "doc_api_links.py"


def _tool() -> ModuleType:
    """Load the linter by path, so the suite needs no ``tools`` package on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("doc_api_links", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["doc_api_links"] = module
    spec.loader.exec_module(module)
    return module


#: Names the site is organized around. If the index stops resolving these, it has stopped
#: reading the directives rather than found a genuinely empty docs tree, and the
#: no-unlinked-spans test below would pass vacuously.
_ANCHORS = {
    "Dataset.collect": "batcher.Dataset.collect",
    "Dataset.select": "batcher.Dataset.select",
    "GroupBy.agg": "batcher.GroupBy.agg",
    "Expr.cast": "batcher.plan.expr_ir.core.Expr.cast",
    "bt.col": "batcher.col",
    "bt.read": "batcher.read",
}


def test_the_index_resolves_the_names_the_site_is_built_on() -> None:
    """The positive control: the resolver finds real targets, not an empty table."""
    tool = _tool()
    resolved = {name: tool.resolve(name) for name in _ANCHORS}
    assert resolved == _ANCHORS


def test_a_chained_expression_is_not_treated_as_one_reference() -> None:
    """``bt.col("a") * bt.col("b")`` is illustrative code, not a lookup of ``col``.

    Without this, the ``--fix`` pass wraps a whole expression in a link to whichever name
    happens to come first in it, which is how the first run of the tool linked
    ``bt.numeric().name.prefix("n_").round(2)`` to ``batcher.numeric``.
    """
    tool = _tool()
    assert tool.resolve("bt.from_pydict(mapping)") == "batcher.from_pydict"
    assert tool.resolve('bt.col("price") * bt.col("qty")') is None
    assert tool.resolve('bt.numeric().name.prefix("n_").round(2)') is None


def test_no_page_points_at_a_document_that_is_not_there() -> None:
    """Every ``{doc}`` reference and card ``:link:`` resolves to a page that exists.

    Sphinx fails on these under ``-W``, but only after a full build, and a relative
    reference left behind by a moved page is the most common way to earn that failure.
    Five of them survived one restructure here and were found by the build rather than by
    anything faster.
    """
    tool = _tool()
    names = tool._page_names()
    broken = {
        page.relative_to(_REPO).as_posix(): dead
        for page in sorted((_REPO / "docs").rglob("*.md"))
        if "_build" not in page.parts and (dead := tool.dead_doc_refs(page, names))
    }
    assert not broken, (
        f"{sum(len(v) for v in broken.values())} dead document reference(s) on "
        f"{len(broken)} page(s): "
        + "; ".join(f"{page}: {', '.join(dead)}" for page, dead in broken.items())
    )


def test_no_published_page_leaves_an_api_name_unlinked() -> None:
    """Every dotted API spelling in a hand-written published page is a cross-reference."""
    tool = _tool()
    offenders = {
        page.relative_to(_REPO).as_posix(): names
        for page in sorted((_REPO / "docs").rglob("*.md"))
        if tool.in_scope(page) and (names := tool.unlinked(page))
    }
    assert not offenders, (
        f"{sum(len(v) for v in offenders.values())} API name(s) on "
        f"{len(offenders)} page(s) render as plain text instead of a link: "
        + "; ".join(
            f"{page}: {', '.join(sorted(set(names))[:4])}" for page, names in offenders.items()
        )
        + "\nRun `python tools/doc_api_links.py --fix`."
    )
