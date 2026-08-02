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
from batcher.plan.expr_rewrite import traverse as expr_rewrite

pytestmark = pytest.mark.unit

# Nodes that genuinely have no sub-expression children (true leaves).
_LEAVES: set[str] = set()

# Every module that defines `IRNode` subclasses. Scanning only `func_nodes` was the first
# hole in this guard: `Array` and `MakeStruct` live in `nodes`, and a node added there
# would have been missed by the very test written to catch a missed node.
_NODE_MODULES = ("core", "nodes", "func_nodes", "image", "audio", "video")


def _nodes_with_children() -> list[type]:
    """Every IR node class that declares at least one sub-expression field.

    Read from the `child`/`children` field metadata rather than from annotation text.
    Matching the annotation was the second hole: it recognized `input: Expr` but not
    `args: list[Expr]`, so `GeoFunc` — 113 functions' worth of node — was invisible to
    this test while being invisible to the optimizer for the same reason. The metadata is
    the same declaration `to_ir` reads, so a node cannot carry children without it.
    """
    import dataclasses
    import importlib

    from batcher.plan.expr_ir.node_base import _META, IRNode, _Kind

    found = []
    for mod_name in _NODE_MODULES:
        mod = importlib.import_module(f"batcher.plan.expr_ir.{mod_name}")
        for name, cls in vars(mod).items():
            if not inspect.isclass(cls) or name in _LEAVES:
                continue
            if not issubclass(cls, IRNode) or cls is IRNode or cls in found:
                continue
            if not dataclasses.is_dataclass(cls):
                continue
            has_child = any(
                (spec := f.metadata.get(_META)) is not None
                and spec.kind in (_Kind.CHILD, _Kind.CHILDREN)
                for f in dataclasses.fields(cls)
            )
            if has_child:
                found.append(cls)
    return found


def test_the_guard_itself_sees_the_variadic_and_non_func_nodes_shapes():
    """The two shapes this test used to miss, pinned so the hole cannot reopen.

    `Array` proves the scan reaches beyond `func_nodes`; `GeoFunc` and `MakeTemporal`
    prove it recognizes a `list[Expr]` field. All three were absent from the discovered
    set before this guard was widened, and `GeoFunc` and `MakeTemporal` really were
    missing from `_EXPR_KIDS` as a result.

    `Case` and `MakeStruct` are deliberately *not* asserted here: they nest their
    sub-expressions inside tuples rather than declaring them as fields, so no metadata
    scan can find them and they are registered in the table by hand.
    """
    names = {c.__name__ for c in _nodes_with_children()}
    assert {"Array", "GeoFunc", "MakeTemporal"} <= names, sorted(names)


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


def test_geo_function_over_a_derived_column():
    """The shape that exposed the gap: build a geometry, then measure it.

    `with_columns(geom=st_point(...)).select(st_as_text(col("geom")))` fuses the two
    projections, and with `GeoFunc` absent from the tables the fusion left a `Col("geom")`
    pointing at a column it had just removed.
    """
    ds = bt.from_pydict({"lon": [-122.4194], "lat": [37.7749]})
    located = ds.with_columns(geom=bt.st_point(bt.col("lon"), bt.col("lat")))
    got = located.select(wkt=bt.st_as_text(bt.col("geom"))).to_pydict()["wkt"]
    assert got == ["POINT(-122.4194 37.7749)"]


def test_geo_predicate_over_two_derived_columns():
    ds = bt.from_pydict({"x": [0.0], "y": [0.0]})
    pair = ds.select(a=bt.st_point(bt.col("x"), bt.col("y")), b=bt.lit("POINT(3 4)"))
    got = pair.select(d=bt.st_distance(bt.col("a"), bt.col("b"))).to_pydict()["d"]
    assert got == [5.0]


def test_make_temporal_over_derived_columns():
    """`MakeTemporal` was missing from the tables for the same reason `GeoFunc` was.

    Both declare their children as `args: list[Expr]`, which the old reflective scan did
    not recognize, so neither was ever checked.
    """
    import datetime as dt

    ds = bt.from_pydict({"y": [2024], "m": [3], "d": [5]})
    parts = ds.select(yy=bt.col("y"), mm=bt.col("m"), dd=bt.col("d"))
    out = parts.select(when=bt.make_date(bt.col("yy"), bt.col("mm"), bt.col("dd")))
    assert out.to_pydict()["when"] == [dt.date(2024, 3, 5)]
