"""`transform_expr_up` must descend into every expression node that has children.

`_EXPR_KIDS` is an exact-type dispatch table. A node **missing** from it is silently
treated as a leaf, so a rewrite that rebuilds the tree beneath it — projection fusion
collapsing ``select(a=…).select(f(col("a")))`` — never substitutes `a`, and leaves a
`Col("a")` pointing at a column the fused projection just removed. The plan then fails
validation with "references unknown column(s) ['a']", or (worse) a rewrite silently
does nothing.

The table is easy to forget when adding a node, and nothing else catches it — so the
first test enumerates the node classes reflectively and asserts none is missing. The
rest are end-to-end regressions for the shapes that were actually broken.
"""

from __future__ import annotations

import inspect

import pytest

import batcher as bt
from batcher.plan import expr_rewrite
from batcher.plan.expr_ir import func_nodes

pytestmark = pytest.mark.unit

# Nodes that genuinely have no sub-expression children (true leaves).
_LEAVES: set[str] = set()


def _nodes_with_children() -> list[type]:
    """Every `func_nodes` IR node that declares an `Expr`-typed field."""
    from batcher.plan.expr_ir.core import Expr

    found = []
    for name, cls in vars(func_nodes).items():
        if not inspect.isclass(cls) or not hasattr(cls, "tag") or name in _LEAVES:
            continue
        annotations = getattr(cls, "__annotations__", {})
        if any(a is Expr or a == "Expr" for a in annotations.values()):
            found.append(cls)
    return found


def test_every_node_with_children_is_in_the_rewrite_tables():
    """A node absent here is a silent leaf: rewrites stop at it and produce a bad plan."""
    missing_kids = [c.__name__ for c in _nodes_with_children() if c not in expr_rewrite._EXPR_KIDS]
    assert not missing_kids, (
        f"nodes missing from _EXPR_KIDS (treated as leaves by transform_expr_up): {missing_kids}"
    )


def test_every_node_in_kids_can_be_rebuilt():
    missing = [c.__name__ for c in expr_rewrite._EXPR_KIDS if c not in expr_rewrite._EXPR_REBUILD]
    assert not missing, f"nodes in _EXPR_KIDS with no _EXPR_REBUILD entry: {missing}"


# --- end-to-end regressions: each of these raised "unknown column(s)" before the fix ---


def test_list_binary_over_derived_columns():
    """`.list.jaccard` / `.list.cosine_similarity` compare two columns — the natural
    spelling derives both in the previous projection."""
    ds = bt.from_pydict({"v": [[1.0, 0.0]]})
    pairs = ds.select(a=bt.col("v"), b=bt.col("v"))
    assert pairs.select(same=bt.col("a").list.jaccard(bt.col("b"))).to_pydict()["same"] == [1.0]
    got = pairs.select(cos=bt.col("a").list.cosine_similarity(bt.col("b"))).to_pydict()["cos"]
    assert got == [1.0]


def test_simhash_signatures_derived_then_compared():
    """The exact shape `similarity_join` documents: signature, then agreement."""
    ds = bt.from_pydict({"v": [[1.0, 0.0], [0.0, 1.0]]})
    sigs = ds.select(a=bt.col("v").list.simhash(64), b=bt.col("v").list.simhash(64))
    assert sigs.select(same=bt.col("a").list.jaccard(bt.col("b"))).to_pydict()["same"] == [1.0, 1.0]


def test_list_set_over_derived_columns():
    ds = bt.from_pydict({"x": [[1, 2, 3]], "y": [[2, 3, 4]]})
    both = ds.select(a=bt.col("x"), b=bt.col("y"))
    got = both.select(i=bt.col("a").list.intersect(bt.col("b"))).to_pydict()["i"]
    assert sorted(got[0]) == [2, 3]


def test_list_position_over_a_derived_column():
    ds = bt.from_pydict({"x": [[10, 20, 30]]})
    got = ds.select(a=bt.col("x")).select(p=bt.col("a").list.position(20)).to_pydict()["p"]
    assert got == [2], "list.position is 1-based (SQL semantics)"


def test_strftime_over_a_derived_column():
    import datetime as dt

    ds = bt.from_pydict({"t": [dt.datetime(2024, 3, 5)]})
    got = ds.select(a=bt.col("t")).select(s=bt.col("a").dt.strftime("%Y-%m")).to_pydict()["s"]
    assert got == ["2024-03"]


def test_convert_timezone_over_a_derived_column():
    import datetime as dt

    ds = bt.from_pydict({"t": [dt.datetime(2024, 3, 5, 12)]})
    out = ds.select(a=bt.col("t")).select(z=bt.col("a").dt.convert_timezone("UTC", "UTC"))
    assert out.to_pydict()["z"] == [dt.datetime(2024, 3, 5, 12)]


def test_date_offset_over_a_derived_column():
    import datetime as dt

    ds = bt.from_pydict({"t": [dt.date(2024, 1, 31)]})
    got = ds.select(a=bt.col("t")).select(o=bt.col("a").dt.offset_by("1d")).to_pydict()["o"]
    assert got == [dt.date(2024, 2, 1)]


def test_list_transform_over_a_derived_column():
    ds = bt.from_pydict({"x": [[1, 2, 3]]})
    out = ds.select(a=bt.col("x")).select(t=bt.col("a").list.transform(bt.element() * 2))
    assert out.to_pydict()["t"] == [[2, 4, 6]]


def test_list_filter_over_a_derived_column():
    ds = bt.from_pydict({"x": [[1, 2, 3, 4]]})
    out = ds.select(a=bt.col("x")).select(f=bt.col("a").list.filter(bt.element() > 2))
    assert out.to_pydict()["f"] == [[3, 4]]
