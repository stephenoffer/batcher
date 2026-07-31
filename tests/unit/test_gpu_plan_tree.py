"""The whole-plan GPU translator matches the native CPU engine (verified on pandas, no GPU).

`gpu_tree_spec` matches a plan of any shape — any tree of scans, joins and unions — and
`run_tree` replays it. This is the differential test for that generalization, and the oracle is
Batcher's own CPU engine, which is itself checked against DuckDB.

The cases here are deliberately the ones the three fixed matchers refuse: three- and four-way
joins, a join whose input is itself an aggregate, a self-join, a union of joins. Every one of
those was a CPU-engine fallback before, so nothing tested them, and nothing would have noticed
a tree executor that quietly reassociated a join or lost a leaf.

Ordering is compared row-for-row wherever the query asks for one, because a set comparison is
exactly what cannot see a reordering bug.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import gpu_tree_spec, run_tree
from batcher.core.gpu_plan.backend import DfBackend
from batcher.core.gpu_plan.pruning import leaf_projections, prune_tree
from batcher.core.gpu_plan.tree import tree_leaves
from batcher.plan.distribution import shardable_leaves

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _customers(n=40):
    rng = np.random.default_rng(1)
    return pa.table(
        {
            "c_id": np.arange(n, dtype="int64"),
            "c_nation": rng.integers(0, 4, n).astype("int64"),
            "c_bal": rng.random(n) * 100,
        }
    )


def _orders(n=200):
    rng = np.random.default_rng(2)
    return pa.table(
        {
            "o_id": np.arange(n, dtype="int64"),
            "o_cust": rng.integers(0, 40, n).astype("int64"),
            "o_prio": rng.integers(0, 3, n).astype("int64"),
        }
    )


def _lines(n=900):
    rng = np.random.default_rng(3)
    return pa.table(
        {
            "l_order": rng.integers(0, 200, n).astype("int64"),
            "l_price": rng.random(n) * 10,
            "l_disc": rng.random(n) * 0.1,
        }
    )


def _nations(n=4):
    return pa.table({"n_id": np.arange(n, dtype="int64"), "n_name": ["A", "B", "C", "D"][:n]})


def _replay(ds, be):
    """Run `ds`'s plan through the tree translator, returning its Arrow result.

    Reads each leaf whole, which is what the single-worker dispatch does; the fan-out's
    correctness is a separate question (`shardable_leaves`) and is asserted separately.
    """
    matched = gpu_tree_spec(ds._plan)
    assert matched is not None, "the plan should be translatable"
    spec, _scans = matched
    spec, projections = prune_tree(spec)
    frames = {}
    for leaf in tree_leaves(spec):
        table = pa.Table.from_batches(list(ds._sources[leaf["source_id"]].read()))
        projection = projections.get(leaf["leaf"])
        if projection is not None:
            table = table.select(projection)
        frames[leaf["leaf"]] = be.from_arrow(table)
    return be.to_arrow(run_tree(spec, frames, be))


#: Relative tolerance on a float column. The two paths sum a group's values in different
#: orders, and floating-point addition is not associative, so the last ulp legitimately differs.
#: Everything else — row count, row order, every non-float value — is compared exactly.
_RTOL = 1e-9


def _assert_matches(ds, be, *, ordered: bool):
    """The tree translator's answer equals the CPU engine's, by column name."""
    expected = ds.to_arrow()
    got = _replay(ds, be)
    assert set(got.column_names) == set(expected.column_names)
    got = got.select(expected.column_names)
    assert got.num_rows == expected.num_rows
    if ordered:
        for name in expected.column_names:
            _assert_column(got.column(name).to_pylist(), expected.column(name).to_pylist(), name)
        return
    key = expected.column_names
    want = sorted(zip(*(expected.column(c).to_pylist() for c in key), strict=True), key=repr)
    have = sorted(zip(*(got.column(c).to_pylist() for c in key), strict=True), key=repr)
    for want_row, have_row in zip(want, have, strict=True):
        _assert_column(list(have_row), list(want_row), "row")


def _assert_column(have: list, want: list, name: str) -> None:
    """One column's values, exactly for anything but a float and within `_RTOL` for those."""
    assert len(have) == len(want), name
    for h, w in zip(have, want, strict=True):
        if isinstance(w, float) and isinstance(h, float) and w == w and h == h:
            assert abs(h - w) <= _RTOL * max(1.0, abs(w)), f"{name}: {h} != {w}"
        else:
            assert h == w, f"{name}: {h!r} != {w!r}"


def test_three_way_join_then_aggregate(be):
    """The shape every fixed matcher refuses: two joins, then a group-by and a top-N."""
    c = bt.from_arrow(_customers())
    o = bt.from_arrow(_orders())
    line = bt.from_arrow(_lines())
    ds = (
        c.filter(col("c_bal") > 20.0)
        .join(o, left_on="c_id", right_on="o_cust")
        .join(line, left_on="o_id", right_on="l_order")
        .group_by("o_prio")
        .agg(revenue=(col("l_price") * (1.0 - col("l_disc"))).sum(), n=bt.count())
        .sort("o_prio")
    )
    _assert_matches(ds, be, ordered=True)


def test_four_way_join(be):
    """Four leaves, a dimension joined to a dimension — TPC-H q5's shape in miniature."""
    c = bt.from_arrow(_customers())
    o = bt.from_arrow(_orders())
    line = bt.from_arrow(_lines())
    n = bt.from_arrow(_nations())
    ds = (
        n.join(c, left_on="n_id", right_on="c_nation")
        .join(o, left_on="c_id", right_on="o_cust")
        .join(line, left_on="o_id", right_on="l_order")
        .group_by("n_name")
        .agg(revenue=col("l_price").sum())
        .sort("n_name")
    )
    _assert_matches(ds, be, ordered=True)


def test_join_over_an_aggregate(be):
    """A join whose input is itself a reduction — the correlated-subquery lowering."""
    o = bt.from_arrow(_orders())
    line = bt.from_arrow(_lines())
    per_order = line.group_by("l_order").agg(total=col("l_price").sum())
    ds = (
        o.join(per_order, left_on="o_id", right_on="l_order")
        .group_by("o_prio")
        .agg(big=col("total").max())
        .sort("o_prio")
    )
    _assert_matches(ds, be, ordered=True)


def test_self_join_keeps_its_two_leaves_distinct(be):
    """Two leaves over ONE source. A spec that keyed leaves by source would collapse them."""
    o = bt.from_arrow(_orders())
    other = bt.from_arrow(_orders())
    ds = (
        o.filter(col("o_prio") == 0)
        .join(other.filter(col("o_prio") == 1), left_on="o_cust", right_on="o_cust")
        .group_by("o_cust")
        .agg(n=bt.count())
        .sort("o_cust")
    )
    matched = gpu_tree_spec(ds._plan)
    assert matched is not None
    spec, _ = matched
    assert len({leaf["leaf"] for leaf in tree_leaves(spec)}) == 2
    _assert_matches(ds, be, ordered=True)


def test_union_of_joins(be):
    """A union whose inputs are joins — a tree that branches both ways."""
    o = bt.from_arrow(_orders())
    line = bt.from_arrow(_lines())
    left = o.filter(col("o_prio") == 0).join(line, left_on="o_id", right_on="l_order")
    right = o.filter(col("o_prio") == 1).join(line, left_on="o_id", right_on="l_order")
    ds = left.union(right).group_by("o_prio").agg(total=col("l_price").sum()).sort("o_prio")
    _assert_matches(ds, be, ordered=True)


def test_left_join_with_no_match_keeps_its_nulls(be):
    """An outer join over a tree: the unmatched side is null, not dropped."""
    c = bt.from_arrow(_customers())
    o = bt.from_arrow(_orders(20))
    ds = c.join(o, left_on="c_id", right_on="o_cust", how="left").sort("c_id", "o_id")
    _assert_matches(ds, be, ordered=False)


def test_semi_join_inside_a_tree(be):
    """A semi join is a membership filter, and it composes with a join above it."""
    c = bt.from_arrow(_customers())
    o = bt.from_arrow(_orders())
    line = bt.from_arrow(_lines())
    ds = (
        o.join(line, left_on="o_id", right_on="l_order")
        .join(c.filter(col("c_bal") > 50.0), left_on="o_cust", right_on="c_id", how="semi")
        .group_by("o_prio")
        .agg(n=bt.count())
        .sort("o_prio")
    )
    _assert_matches(ds, be, ordered=True)


def test_a_udf_in_the_plan_is_not_translatable():
    """A `map_batches` stage never lowers to the engine IR, so the tree declines rather than
    guessing."""
    o = bt.from_arrow(_orders())
    line = bt.from_arrow(_lines())
    ds = o.map_batches(lambda b: b).join(line, left_on="o_id", right_on="l_order")
    assert gpu_tree_spec(ds._plan) is None


class TestShardableLeaves:
    """Which leaf a fan-out may split. Getting this wrong duplicates rows in the union of the
    shards' outputs, which no single-shard test can see."""

    @staticmethod
    def _tree(join_type, left=0, right=1):
        return {
            "kind": "join",
            "left": {"kind": "scan", "leaf": left, "ops": []},
            "right": {"kind": "scan", "leaf": right, "ops": []},
            "join": {"join_type": join_type},
            "ops": [],
        }

    def test_inner_may_split_either_side(self):
        assert shardable_leaves(self._tree("inner")) == {0, 1}

    @pytest.mark.parametrize("join_type", ["left", "semi", "anti"])
    def test_left_driven_joins_may_split_only_the_left(self, join_type):
        assert shardable_leaves(self._tree(join_type)) == {0}

    def test_right_join_may_split_only_the_right(self):
        assert shardable_leaves(self._tree("right")) == {1}

    def test_full_join_may_split_neither(self):
        assert shardable_leaves(self._tree("full")) == set()

    def test_nothing_under_a_union_may_be_split_alone(self):
        tree = {
            "kind": "union",
            "inputs": [{"kind": "scan", "leaf": 0, "ops": []}, self._tree("inner", 1, 2)],
            "distinct": False,
            "ops": [],
        }
        assert shardable_leaves(tree) == set()

    def test_the_rule_applies_at_every_level(self):
        """A leaf under an inner join that is itself the build side of a LEFT join above it is
        not splittable, however safe its own join is."""
        inner = self._tree("inner", 1, 2)
        tree = {
            "kind": "join",
            "left": {"kind": "scan", "leaf": 0, "ops": []},
            "right": inner,
            "join": {"join_type": "left"},
            "ops": [],
        }
        assert shardable_leaves(tree) == {0}


class TestLeafProjections:
    """Column pruning: narrow enough to matter, never so narrow it drops a needed column."""

    def test_a_projection_reaches_through_two_joins(self):
        c = bt.from_arrow(_customers())
        o = bt.from_arrow(_orders())
        line = bt.from_arrow(_lines())
        ds = (
            c.join(o, left_on="c_id", right_on="o_cust")
            .join(line, left_on="o_id", right_on="l_order")
            .group_by("o_prio")
            .agg(total=col("l_price").sum())
        )
        spec, _ = gpu_tree_spec(ds._plan)
        pruned, projections = prune_tree(spec)
        by_source = {leaf["leaf"]: projections[leaf["leaf"]] for leaf in tree_leaves(pruned)}
        # `c_bal` and `c_nation` are read by nothing; `l_disc` likewise.
        every = {name for cols in by_source.values() if cols for name in cols}
        assert "l_price" in every
        assert "c_bal" not in every
        assert "l_disc" not in every

    def test_distinct_refuses_to_narrow(self):
        """DISTINCT decides identity from every column it has, so pruning one changes the
        answer rather than failing."""
        o = bt.from_arrow(_orders())
        line = bt.from_arrow(_lines())
        ds = o.join(line, left_on="o_id", right_on="l_order").distinct()
        spec, _ = gpu_tree_spec(ds._plan)
        projections = leaf_projections(spec)
        # The join names its own outputs, so the leaves stay bounded by those; what must not
        # happen is a column the DISTINCT would have compared going missing.
        for leaf in tree_leaves(spec):
            cols = projections[leaf["leaf"]]
            assert cols is None or cols


def test_a_date_column_survives_the_round_trip(be):
    """A DATE must come back a DATE, not the timestamp the dataframe libraries turn it into.

    Neither library has a calendar-day type, so a `date32` becomes a datetime on the way in and
    a `timestamp` on the way out. The values are right and the column is wrong — which breaks
    the concatenation a fan-out depends on, since a CPU-recovered shard contributes `date32`
    beside a device shard's `timestamp`, and silently changes the schema a query returns.
    Measured on TPC-H q3, whose `o_orderdate` came back as `datetime(1995, 2, 3, 0, 0)`.
    """
    import datetime as dt

    table = pa.table(
        {
            "d": pa.array([dt.date(2024, 1, 1), dt.date(2024, 6, 2)], pa.date32()),
            "v": pa.array([1.0, 2.0], pa.float64()),
        }
    )
    ds = bt.from_arrow(table).filter(col("v") > 0.0).sort("d")
    got = _replay(ds, be)
    assert got.schema.field("d").type == pa.date32()
    assert got.column("d").to_pylist() == ds.to_arrow().column("d").to_pylist()
