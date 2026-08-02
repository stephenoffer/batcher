"""A reduced UNION distributes through the shuffle, not through the driver.

`UNION` and the two set operators are the shapes where "distributed" used to mean "both
inputs, whole, on one node": `_distributed_union` runs each branch to a driver table and
concatenates there, and `UNION (distinct)` then deduplicated that concatenation
single-node. Every query that *reduces* a union paid it — `union(...).group_by(...)`, and
`intersect`/`except_`, which lower to an aggregate over a union of tagged branches.

They are mergeable like any other aggregate: a union's branches map into one bucket space,
identical keys hash to one reducer wherever they came from, and `combine` is associative and
commutative — so the reducers see exactly what they would have seen had the branches been
concatenated first. These tests hold that equality against DuckDB *and* assert the routing,
because the funnel and the shuffle return the same rows and only the routing tells them
apart: a correctness-only test passes just as happily on the path this replaces.

The multi-branch map stage is what the equality rests on, so the fixtures carry the data
that breaks a naive one: duplicates spanning both branches (a group split across branches is
the failure this shape invites), nulls in a key, `-0.0` against `0.0`, and branches whose
row counts differ so no accidental symmetry hides a mis-routed bucket.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
pytest.importorskip("ray", reason="the distributed path needs Ray")

#: Two branches sharing keys, so every reducer sees rows from both. `None` and `-0.0` are
#: here because they are the two values that have split one group in two before.
LEFT = pa.table(
    {
        "k": pa.array([1, 2, 2, 3, None, 4], pa.int64()),
        "v": pa.array([0.0, 1.5, 1.5, -0.0, 2.0, 3.0], pa.float64()),
        "g": pa.array(["a", "b", "b", "c", "d", "e"]),
    }
)
RIGHT = pa.table(
    {
        "k": pa.array([2, 3, 5, None], pa.int64()),
        "v": pa.array([1.5, 0.0, 4.0, 2.0], pa.float64()),
        "g": pa.array(["b", "c", "f", "d"]),
    }
)

_W = 2


def _dist(ds) -> pa.Table:
    return ds.collect(distributed=True, num_workers=_W)


@pytest.fixture
def no_driver_dedup(monkeypatch):
    """Fail the test if the union's driver-side concatenate-and-dedup path runs.

    The fused path and the funnel it replaces return identical rows, so this is the only
    thing that can tell them apart. Patched at the definition site, so a caller that
    imported the name earlier still reaches the guard.
    """
    from batcher.dist.executors import union as union_mod

    def forbidden(*_a, **_k):
        raise AssertionError(
            "the union was concatenated and deduplicated on the driver; the aggregate "
            "shuffle should have absorbed its branches"
        )

    monkeypatch.setattr(union_mod, "_dedup", forbidden)


@pytest.mark.parametrize("distinct", [False, True])
def test_grouped_union_matches_duckdb(duck, distinct, no_driver_dedup):
    """`union(...).group_by(...)` — the aggregate absorbs the union's branches."""
    duck.register("l", LEFT)
    duck.register("r", RIGHT)
    op = "UNION" if distinct else "UNION ALL"
    # `count(k)`, not `count(*)`: one group's key IS null, and the two spellings
    # legitimately differ there. Comparing them would fail on SQL semantics, not on routing.
    sql = (
        f"SELECT k, sum(v) AS s, count(k) AS n "
        f"FROM (SELECT * FROM l {op} SELECT * FROM r) GROUP BY k"
    )
    ds = (
        bt.from_arrow(LEFT)
        .union(bt.from_arrow(RIGHT), distinct=distinct)
        .group_by("k")
        .agg(s=bt.col("v").sum(), n=bt.col("k").count())
    )
    assert_same(_dist(ds), duck.sql(sql))


def test_union_distinct_matches_duckdb(duck, no_driver_dedup):
    """`UNION` itself — a DISTINCT over UNION ALL, deduplicated by the reducers."""
    duck.register("l", LEFT)
    duck.register("r", RIGHT)
    ds = bt.from_arrow(LEFT).union(bt.from_arrow(RIGHT), distinct=True)
    assert_same(_dist(ds), duck.sql("SELECT * FROM l UNION SELECT * FROM r"))


@pytest.mark.parametrize("op", ["intersect", "except_"])
def test_set_operators_match_duckdb(duck, op, no_driver_dedup):
    """INTERSECT / EXCEPT lower to an aggregate over a union of tagged branches."""
    duck.register("l", LEFT)
    duck.register("r", RIGHT)
    sql = "SELECT * FROM l INTERSECT SELECT * FROM r"
    if op == "except_":
        sql = "SELECT * FROM l EXCEPT SELECT * FROM r"
    ds = getattr(bt.from_arrow(LEFT), op)(bt.from_arrow(RIGHT))
    assert_same(_dist(ds), duck.sql(sql))


@pytest.mark.parametrize("distinct", [False, True])
def test_distributed_equals_single_node(distinct):
    """The mergeable-algebra invariant, stated directly against the single-node result."""

    def build():
        return (
            bt.from_arrow(LEFT)
            .union(bt.from_arrow(RIGHT), distinct=distinct)
            .group_by("g")
            .agg(s=bt.col("v").sum(), n=bt.col("g").count())
        )

    assert_tables_equal(_dist(build()), build().collect())


def test_three_branch_union_reduces_through_one_shuffle(no_driver_dedup):
    """More than two branches: every one of them maps into the same bucket space."""
    third = pa.table(
        {
            "k": pa.array([1, 9], pa.int64()),
            "v": pa.array([5.0, 6.0], pa.float64()),
            "g": pa.array(["a", "z"]),
        }
    )

    def build():
        return (
            bt.from_arrow(LEFT)
            .union(bt.from_arrow(RIGHT), bt.from_arrow(third))
            .group_by("k")
            .agg(s=bt.col("v").sum())
        )

    assert_tables_equal(_dist(build()), build().collect())


def test_branch_with_its_own_breaker_still_runs():
    """A branch the shuffle cannot absorb keeps the per-branch path, and still agrees.

    `shuffle_branches` refuses a branch carrying its own breaker, because a map task would
    evaluate that breaker once per partition. The fused path must decline rather than
    compute, and the query must still return the single-node answer — so no `no_driver_dedup`
    guard here: taking the driver path is the correct behavior under test.
    """

    def build():
        return (
            bt.from_arrow(LEFT)
            .group_by("k")
            .agg(v=bt.col("v").sum())
            .union(bt.from_arrow(RIGHT).select("k", "v"), distinct=True)
        )

    assert_tables_equal(_dist(build()), build().collect())


def test_a_type_mismatched_union_is_not_fused():
    """Branches whose key types differ are refused, and the query still agrees.

    `Union` promotes an `Int64` branch against a `Float64` one; independent mappers skip that
    promotion, so `1` and `1.0` would hash to different reducers and one group would come
    back as two. `shuffle_branches` must decline the shape rather than split the group.
    """
    from batcher.dist.executors.plan_analysis import shuffle_branches

    ints = pa.table({"k": pa.array([1, 2], pa.int64()), "v": pa.array([1.0, 2.0])})
    floats = pa.table({"k": pa.array([1.0, 3.0], pa.float64()), "v": pa.array([3.0, 4.0])})

    def build():
        return (
            bt.from_arrow(ints).union(bt.from_arrow(floats)).group_by("k").agg(s=bt.col("v").sum())
        )

    assert shuffle_branches(build()._plan.input) is None
    assert_tables_equal(_dist(build()), build().collect())


def test_a_promoted_union_concatenates_distributed_as_it_does_single_node(duck):
    """A UNION of `Int64` against `Float64` returns the promoted relation, distributed too.

    A union's branches must agree on column *names*, not types — `Union.available_schema`
    widens the pair and the single-node engine returns the widened relation. The per-branch
    distributed path concatenated the branch results without that promotion, so this exact
    query raised `ArrowInvalid: Schema at index 1 was different` under `distributed=True`
    while succeeding single-node: a distributed-only failure on a query with a good answer.
    """
    ints = pa.table({"k": pa.array([1, 2], pa.int64())})
    floats = pa.table({"k": pa.array([2.5, 3.0], pa.float64())})
    duck.register("i", ints)
    duck.register("f", floats)
    ds = bt.from_arrow(ints).union(bt.from_arrow(floats))
    assert_same(_dist(ds), duck.sql("SELECT k FROM i UNION ALL SELECT k FROM f"))
