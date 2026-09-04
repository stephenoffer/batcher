"""The two front-ends must plan a multi-level GROUP BY the same way.

`ROLLUP`/`CUBE`/`GROUPING SETS` is one query per grouping level, stacked. Both surfaces
build the levels that way -- `ds.rollup(...)` through `api.multi_group`, SQL through
`_sql.parser.grouping_sets` -- and the levels are the *same* relations throughout, so they
have to land on one source list. Binding per level instead is not a cosmetic difference: it
gives each level's subtree a different structural key, which is exactly what stops
`kyber.common_subplan` recognizing them as one shared subtree, so every level re-reads and
re-joins the whole input.

The SQL side had that defect while the DataFrame side did not, because the sharing was
written into `api.multi_group` alone. These tests pin both surfaces to it.
"""

from __future__ import annotations

import batcher as bt
from batcher.kyber.common_subplan import structural_key
from batcher.plan.logical import Union
from batcher.plan.visitor import walk


def _frames():
    fact = bt.from_pydict({"k": [1, 2, 3, 1], "r": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    dim = bt.from_pydict({"k": [1, 2, 3], "c": ["x", "y", "x"]})
    return fact, dim


_SQL = """
SELECT d.c, f.r, sum(f.v) AS total
FROM fact f JOIN dim d ON f.k = d.k
GROUP BY ROLLUP(d.c, f.r)
"""


def _branches(ds) -> tuple:
    """The stacked grouping levels of `ds`'s plan."""
    root = ds._plan
    assert isinstance(root, Union), f"expected the levels to be stacked, got {type(root).__name__}"
    return root.inputs


def test_sql_rollup_binds_each_relation_once() -> None:
    """Three levels over two relations bind two sources, not six."""
    fact, dim = _frames()
    ds = bt.sql(_SQL, fact=fact, dim=dim)
    assert len(_branches(ds)) == 3  # the two-key rollup really does have three levels
    assert len(ds._sources) == 2


def test_dataframe_rollup_binds_each_relation_once() -> None:
    """`ds.rollup` agrees, over the same two relations."""
    fact, dim = _frames()
    ds = fact.join(dim, on="k").rollup("c", "r").agg(total=bt.col("v").sum())
    assert len(_branches(ds)) == 3
    assert len(ds._sources) == 2


def test_an_ordinary_union_still_binds_per_branch() -> None:
    """The positive control for the two above: `Dataset.union` is deliberately unchanged.

    It takes arbitrary datasets, where two sources that compare equal may be two unrelated
    relations, so it concatenates the source lists. Without this case the assertions above
    would also pass if source lists were merged everywhere, which is a change measured to
    cost more than it gains (see `Dataset.union`).
    """
    fact, dim = _frames()
    joined = fact.join(dim, on="k")
    stacked = joined.union(joined, distinct=False)
    assert len(stacked._sources) == 4


def test_the_grouping_levels_share_one_subtree() -> None:
    """Every level's input below the aggregate is structurally the same tree, on both surfaces.

    This is the property the source list buys, stated the way `common_subplan` reads it:
    one structural key across the levels means one repeated subtree to compute once.
    """
    fact, dim = _frames()
    for ds in (
        bt.sql(_SQL, fact=fact, dim=dim),
        fact.join(dim, on="k").rollup("c", "r").agg(total=bt.col("v").sum()),
    ):
        joins = [
            structural_key(node)
            for branch in _branches(ds)
            for node in walk(branch)
            if type(node).__name__ == "Join"
        ]
        assert len(joins) == 3, "expected one join per grouping level"
        assert len(set(joins)) == 1, "the levels' joins must be one shared subtree"


def _level_count(ds) -> int:
    """The grouping levels of `ds`, counted through any nesting of the stack.

    Flattening rather than reading `_branches` keeps this test about the level *vocabulary*
    -- how many levels `CUBE(a, b, c)` has -- and lets the stacking shape be the subject of
    the tests above it, where it belongs.
    """
    root = ds._plan
    if not isinstance(root, Union):
        return 1
    return sum(_level_count(ds._derive(branch)) for branch in root.inputs)


def test_both_surfaces_expand_the_same_levels() -> None:
    """SQL and `ds.rollup`/`cube`/`grouping_sets` agree on how many levels a shape has.

    They expand through the same `rollup_levels`/`cube_levels`, so a change to either
    front-end's idea of what `CUBE(a, b, c)` means shows up here.

    The expected count is spelled out rather than only compared across the surfaces. Sharing
    the expansion means a change to it moves *both* sides together, so an equality between
    them cannot see it -- which is a test that passes while the thing it names is wrong.
    """
    ds = bt.from_pydict({"a": [1, 1], "b": [2, 3], "c": [4, 4], "v": [1, 2]})
    total = bt.col("v").sum()
    cases = [
        ("ROLLUP(a, b, c)", ds.rollup("a", "b", "c").agg(total=total), 4),  # every prefix
        ("CUBE(a, b, c)", ds.cube("a", "b", "c").agg(total=total), 8),  # every subset
        (
            "GROUPING SETS ((a, b), (c), ())",
            ds.grouping_sets(("a", "b"), ("c",), ()).agg(total=total),
            3,  # the members as written
        ),
    ]
    for clause, frame, levels in cases:
        sql = bt.sql(f"SELECT a, b, c, sum(v) AS total FROM t GROUP BY {clause}", t=ds)
        assert _level_count(frame) == levels, clause
        assert _level_count(sql) == levels, clause
