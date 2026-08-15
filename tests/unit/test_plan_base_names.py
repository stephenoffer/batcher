"""`plan.visitor.walk_with_base_names` — plan column names traced back to source columns.

Two analyses read column names off a plan and then look them up in a **source's schema**:
`ndv_columns` (which distinct counts to sketch before planning) and `column_bounds_needed`
(which min/max to fetch). A plan's names are not a source's names — the SQL front-end
disambiguates `date_dim d1, date_dim d2` by projecting every column to `d1__<name>` — so
without this resolution both matched nothing for any aliased table and went *silently*
blind: no error, no missing statistic reported, just a worse plan forever.

These pin the resolution itself; `test_diff_*` pins the answers the plans produce.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.api.source_stats import column_bounds_needed
from batcher.api.terminal._metadata import ndv_columns
from batcher.plan.visitor import walk_with_base_names

pytestmark = pytest.mark.unit


def _named(plan):
    """`{node type: rename map}` for the nodes whose map is worth asserting."""
    return {type(node).__name__: base for node, base in walk_with_base_names(plan)}


def test_a_rename_resolves_to_its_base_column():
    plan = bt.from_pydict({"x": [1, 2]}).select(col("x").alias("y")).filter(col("y") > 0)._plan
    assert _named(plan)["Filter"]["y"] == "x"


def test_a_chain_of_renames_resolves_to_the_first_column():
    ds = bt.from_pydict({"x": [1, 2]})
    plan = ds.select(col("x").alias("a")).select(col("a").alias("b")).filter(col("b") > 0)._plan
    assert _named(plan)["Filter"]["b"] == "x"


def test_a_computed_column_has_no_base_column():
    plan = (
        bt.from_pydict({"x": [1, 2]}).select((col("x") + 1).alias("y")).filter(col("y") > 0)._plan
    )
    assert "y" not in _named(plan)["Filter"]


def test_a_join_carries_each_side_back_to_its_own_source():
    left = bt.from_pydict({"k": [1, 2], "lv": [1, 2]}).select(
        col("k").alias("l__k"), col("lv").alias("l__lv")
    )
    right = bt.from_pydict({"k": [1, 2], "rv": [1, 2]}).select(
        col("k").alias("r__k"), col("rv").alias("r__rv")
    )
    plan = left.join(right, left_on="l__k", right_on="r__k").filter(col("l__lv") > 0)._plan
    base = _named(plan)["Filter"]
    assert base["l__lv"] == "lv"
    assert base["r__rv"] == "rv"


def test_the_aliased_self_join_shape_reaches_the_source_columns():
    """The shape that broke: one table joined to itself under two aliases.

    Both aliases project to prefixed names, so the un-resolved analysis produced
    `d1__key` / `d2__key` — names `date_dim` does not have — and asked for nothing.
    """
    dim = bt.from_pydict({"key": [1, 2, 3], "quarter": ["a", "b", "c"]})
    d1 = dim.select(col("key").alias("d1__key"), col("quarter").alias("d1__quarter"))
    d2 = dim.select(col("key").alias("d2__key"), col("quarter").alias("d2__quarter"))
    ds = d1.join(d2, left_on="d1__key", right_on="d2__key").filter(col("d1__quarter") == "a")

    assert "quarter" in ndv_columns(ds._plan)
    assert "key" in ndv_columns(ds._plan)
    assert "quarter" in column_bounds_needed(ds._plan)
    # And nothing prefixed survives into either answer.
    assert not any(n.startswith("d1__") or n.startswith("d2__") for n in ndv_columns(ds._plan))


def test_a_group_key_resolves_through_its_alias():
    ds = (
        bt.from_pydict({"g": [1, 1, 2], "v": [1, 2, 3]})
        .select(col("g").alias("gg"), col("v").alias("vv"))
        .group_by("gg")
        .agg(s=col("vv").sum())
    )
    assert "g" in ndv_columns(ds._plan)


def test_an_unrenamed_plan_is_unchanged():
    """A plan with no renaming reports exactly the names it carries — no regression."""
    ds = (
        bt.from_pydict({"a": [1], "b": [2]})
        .filter(col("a") == 1)
        .group_by("b")
        .agg(s=col("a").sum())
    )
    assert ndv_columns(ds._plan) == {"a", "b"}
