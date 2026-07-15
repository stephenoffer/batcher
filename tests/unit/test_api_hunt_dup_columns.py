"""Duplicate output-column names must be rejected, not silently collapsed.

A relation whose output has two columns of the same name loses one of them the
moment the result is materialized to a name-keyed structure (``to_pydict``) — silent
data loss. The public API rejects the duplicate at plan-build time (as Polars does)
with an actionable `PlanError`. These pin that rejection across `select`,
`with_columns`, `rename`, and `group_by().agg()`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.mark.unit
def test_select_duplicate_positional_names_rejected():
    ds = bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(PlanError, match="duplicate output column"):
        ds.select("x", "x")


@pytest.mark.unit
def test_select_positional_and_keyword_collide_rejected():
    # `select("x", x=col("y"))` used to silently keep only the keyword's `x`,
    # dropping the original column entirely.
    ds = bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(PlanError, match="duplicate output column"):
        ds.select("x", x=bt.col("y"))


@pytest.mark.unit
def test_rename_onto_existing_column_rejected():
    ds = bt.from_pydict({"x": [1], "y": [2]})
    with pytest.raises(PlanError, match="duplicate output column"):
        ds.rename(x="y")


@pytest.mark.unit
def test_agg_alias_colliding_with_group_key_rejected():
    # `group_by("g").agg(g=...)` used to return a single `g` column holding the
    # aggregate, silently discarding the group-key values.
    ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    with pytest.raises(PlanError, match="duplicate output column"):
        ds.group_by("g").agg(g=bt.col("v").sum())


@pytest.mark.unit
def test_group_by_named_key_colliding_with_positional_key_rejected():
    ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    with pytest.raises(PlanError, match="duplicate output column"):
        ds.group_by("g", g=bt.col("v")).agg(s=bt.col("v").sum())


@pytest.mark.unit
def test_two_positional_aggregates_over_same_column_rejected():
    # Both would be named after the source column; the second used to silently win.
    ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    with pytest.raises(PlanError, match="positional aggregates over column"):
        ds.group_by("g").agg(bt.col("v").sum(), bt.col("v").mean())


@pytest.mark.unit
def test_full_outer_join_temp_key_collision_rejected():
    # A full outer join coalesces each key side through internal `__fk_l_i` temps.
    # A user column literally named `__fk_l_0` used to be silently dropped by the
    # final projection; a join must never silently lose a column.
    left = bt.from_pydict({"k": [1, 2], "__fk_l_0": [7, 8]})
    right = bt.from_pydict({"k": [2, 3], "w": [9, 10]})
    with pytest.raises(PlanError, match="duplicate output column"):
        left.join(right, on="k", how="full")


@pytest.mark.unit
def test_join_suffix_disambiguates_without_error():
    # Colliding non-key names are disambiguated by `suffix`, not rejected.
    left = bt.from_pydict({"k": [1], "v": [1]})
    right = bt.from_pydict({"k": [1], "v": [2]})
    assert left.join(right, on="k").columns == ["k", "v", "v_right"]


@pytest.mark.unit
def test_distinct_names_still_allowed():
    # The valid shapes must keep working — the check only fires on a real collision.
    ds = bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert ds.select("x", y2=bt.col("y")).columns == ["x", "y2"]
    g = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    out = g.group_by("g").agg(s=bt.col("v").sum(), m=bt.col("v").mean())
    assert out.columns == ["g", "s", "m"]
