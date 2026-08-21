"""Differential tests for the out-of-core (`spill=True`) path, vs DuckDB.

The spilling operators are a *second consumer* of the mergeable primitives, not a second
implementation of them — so every shape that works in memory must work spilled, with the
identical result. These tests pin that.

They exist because it was not pinned: `collect(spill=True)` on a `descending` sort used to
return **unsorted** data (nulls emitted mid-result), because the out-of-core sort hand-rolled
its own range partitioner instead of calling the shared `range_partition_batches`, and got the
null-bucket end wrong for the reversed emission order. Nothing caught it — the ordinary
differential harness (`assert_same`) is order-*independent*, so it is structurally blind to a
sort bug, and no test combined `spill=True` with `descending`. Hence: ordered assertions, and
the ordering flags in the parameter matrix.
"""

from __future__ import annotations

import itertools

import pyarrow as pa
import pytest

from _harness import assert_same, assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

ORDERINGS = list(itertools.product([False, True], [False, True]))  # (descending, nulls_first)


@pytest.fixture
def spill_table() -> pa.Table:
    """Nulls, duplicates, negative values and a string column — one table, many shapes."""
    return pa.table(
        {
            "k": pa.array([3, 1, None, 10, 7, None, 0, 5, 9, None, 2, 8, 4, None, 6], pa.int64()),
            "g": pa.array(
                ["a", "b", "a", "c", None, "b", "a", "c", "b", None, "a", "c", "b", "a", "c"]
            ),
            "v": pa.array([5, 3, 9, 1, 4, 8, 2, 7, 6, 0, 5, 3, 8, 1, 2], pa.int64()),
        }
    )


def _register(duck, table: pa.Table, name: str = "t") -> None:
    duck.register(name, table)


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
def test_spilling_sort_matches_duckdb(duck, spill_table, descending, nulls_first):
    """A spilled sort is byte-for-byte the sort DuckDB produces — every ordering flag.

    The regression: `descending=True` used to interleave nulls into the middle of the
    result, because the null bucket was chosen from `nulls_first` alone while the buckets
    were *emitted* in reverse. Ordered assertion — an unordered one cannot see this.
    """
    _register(duck, spill_table)
    out = (
        bt.from_arrow(spill_table)
        .sort(bt.col("k"), descending=descending, nulls_first=nulls_first)
        .collect(spill=True)
    )
    direction = "DESC" if descending else "ASC"
    nulls = "FIRST" if nulls_first else "LAST"
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {direction} NULLS {nulls}"))


@pytest.mark.parametrize(("descending", "nulls_first"), ORDERINGS)
def test_spilling_sort_equals_in_memory_sort(spill_table, descending, nulls_first):
    """The spilled sort equals the in-memory sort — one operator, two schedulings."""
    plan = bt.from_arrow(spill_table).sort(
        bt.col("k"), descending=descending, nulls_first=nulls_first
    )
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)


@pytest.mark.parametrize("descending", [False, True])
def test_spilling_top_n_matches_duckdb(duck, spill_table, descending):
    """A spilled top-N (sort + limit) stops early but still emits in key order."""
    _register(duck, spill_table)
    out = (
        bt.from_arrow(spill_table)
        .sort(bt.col("k"), descending=descending, nulls_first=False)
        .limit(5)
        .collect(spill=True)
    )
    direction = "DESC" if descending else "ASC"
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {direction} NULLS LAST LIMIT 5"))


def test_spilling_sort_on_string_key_matches_in_memory(spill_table):
    """A string sort key spills out-of-core and orders exactly as the in-memory sort does.

    Twice-changed history, so the assertion outlived both spellings of the behaviour. The gate
    once admitted any plain column and then died inside the partitioner (`TypeError` on a
    string subtraction); it was then narrowed to numeric keys, making this shape *fall back*.
    `column_string_quantiles` / `range_partition_batches_str` since gave string keys real
    boundaries, so it spills again -- see `SPILLING_SHAPES` in
    tests/integration/test_spill_route_is_taken.py, which pins which route runs. What this
    test asserts is unchanged either way: same rows, same order, whichever route is taken.
    """
    plan = bt.from_arrow(spill_table).sort(bt.col("g"))
    assert_tables_equal(plan.collect(spill=True), plan.collect(), ordered=True)


def test_spilling_aggregate_matches_duckdb(duck, spill_table):
    """A spilled group-by equals DuckDB's, nulls in the key included."""
    _register(duck, spill_table)
    out = (
        bt.from_arrow(spill_table)
        .group_by("g")
        .agg(s=bt.col("v").sum(), n=bt.col("v").count())
        .collect(spill=True)
    )
    assert_same(out, duck.sql("SELECT g, SUM(v) AS s, COUNT(v) AS n FROM t GROUP BY g"))


def test_spilling_distinct_matches_duckdb(duck, spill_table):
    """A spilled DISTINCT equals DuckDB's — nulls compare equal, so they collapse to one."""
    _register(duck, spill_table)
    out = bt.from_arrow(spill_table).select(bt.col("g")).distinct().collect(spill=True)
    assert_same(out, duck.sql("SELECT DISTINCT g FROM t"))


def test_streaming_distinct_under_a_budget_matches_duckdb(duck):
    """A high-cardinality DISTINCT run through the *normal* `collect()` (not `spill=True`) under
    a tight memory budget must never OOM and must match DuckDB.

    This exercises the streaming executor's deferred-breaker path: DISTINCT is handed to the
    oracle, but under a budget the streaming driver bounds it — over budget it yields to the
    spilling parallel executor. The regression it guards: that deferral used to run the oracle
    *unbounded*, so a DISTINCT whose key set exceeded RAM crashed where the parallel path spills.
    """
    import numpy as np

    from batcher.config import Config, MemoryConfig, config_context

    rng = np.random.default_rng(5)
    n = 300_000
    t = pa.table(
        {
            "k": rng.integers(0, 250_000, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )
    _register(duck, t)
    # A 2 MB envelope is far smaller than the ~175k-group DISTINCT state, so the streaming
    # breaker goes over budget and must spill rather than OOM.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=2_000_000))
    with config_context(cfg):
        distinct = bt.from_arrow(t).select(bt.col("k")).distinct().collect()
        union = (
            bt.from_arrow(t)
            .select(bt.col("k"))
            .union(bt.from_arrow(t).select(bt.col("k")), distinct=True)
            .collect()
        )
    assert_same(distinct, duck.sql("SELECT DISTINCT k FROM t"))
    assert_same(union, duck.sql("SELECT DISTINCT k FROM t"))  # union-distinct of t with itself


def test_spilling_join_matches_duckdb(duck, spill_table):
    """A spilled hash join equals DuckDB's, including the null keys that never match."""
    right = pa.table(
        {"k": pa.array([1, 3, 5, 7, 9, None], pa.int64()), "w": ["p", "q", "r", "s", "u", "z"]}
    )
    _register(duck, spill_table)
    duck.register("r", right)
    out = (
        bt.from_arrow(spill_table)
        .join(bt.from_arrow(right), left_on="k", right_on="k")
        .collect(spill=True)
    )
    assert_same(out, duck.sql("SELECT * FROM t JOIN r USING (k)"))


#: shape -> builder, for every breaker that a `LIMIT` can legally sit above. Each pairs an
#: order-destroying out-of-core breaker with an order-sensitive `LIMIT`.
LIMIT_OVER_BREAKER = {
    "distinct": lambda d: d.distinct().limit(5),
    "distinct_then_filter": lambda d: d.distinct().filter(bt.col("k").is_not_null()).limit(5),
    "distinct_then_project": lambda d: d.distinct().limit(5).select(bt.col("k")),
    "aggregate": lambda d: d.group_by("g").agg(s=bt.col("k").sum()).limit(2),
    "window": lambda d: d.with_columns(
        r=bt.col("k").rank().over(partition_by="g", order_by="k")
    ).limit(5),
}


@pytest.mark.parametrize("shape", sorted(LIMIT_OVER_BREAKER))
def test_a_limit_above_a_spilled_breaker_keeps_the_same_rows(spill_table, shape):
    """`LIMIT k` over a spilled breaker must keep the rows `collect()` keeps.

    Out-of-core, `Distinct`/`Aggregate`/`Join`/`Window` emit in hash-partition order while the
    in-memory path emits in input order, so re-applying the `LIMIT` above the spilled breaker
    selected a *different k rows* -- a wrong answer, not a slower one.
    `bc_ir::RelOp::Distinct` states the contract it broke: "the rows kept are the first k in
    input order", chosen so that one node and many agree. `spill_collect` now declines the
    shape rather than reordering it.

    Order-*independent* comparison would be blind to this (both paths return k rows of the
    same schema), so this asserts the rows themselves, in order.
    """
    build = LIMIT_OVER_BREAKER[shape]
    assert_tables_equal(
        build(bt.from_arrow(spill_table)).collect(spill=True),
        build(bt.from_arrow(spill_table)).collect(),
        ordered=True,
    )


def test_a_limit_above_a_spilled_join_keeps_the_same_rows(spill_table):
    """The two-input case of the shape above."""
    right = pa.table({"k": pa.array([1, 3, 5, 7, 9, None], pa.int64()), "w": [*"pqrsuz"]})
    build = lambda d: d.join(bt.from_arrow(right), left_on="k", right_on="k").limit(4)  # noqa: E731
    assert_tables_equal(
        build(bt.from_arrow(spill_table)).collect(spill=True),
        build(bt.from_arrow(spill_table)).collect(),
        ordered=True,
    )


def test_a_limit_above_a_spilled_sort_still_runs_out_of_core(spill_table):
    """`Sort` *does* define the order, so top-N must keep its out-of-core path.

    Guards the fix above from being over-applied: declining every `LIMIT` would silently
    take the one order-defining breaker off the spilling route too.
    """
    from batcher import core, kyber
    from batcher.api.orchestration import auto_num_partitions
    from batcher.dist.spill import spill_collect

    ds = bt.from_arrow(spill_table).sort(bt.col("k")).limit(5)
    hub = core.default_hub()
    optimized = kyber.optimize_logical(ds._plan, sources=ds._sources, hub=hub)
    partitions = auto_num_partitions(ds._plan, ds._sources, hub)
    assert spill_collect(optimized, ds._sources, partitions) is not None


#: shape -> builder, over an input that names TWO sources (a join or a union underneath).
#: Every spillable breaker is represented, because the gate is per-breaker.
MULTI_SOURCE_BREAKERS = {
    "sort": lambda d, r: d.join(r, on="k").sort(bt.col("v")),
    "aggregate": lambda d, r: d.join(r, on="k").group_by("k").agg(s=bt.col("v").sum()),
    "global_aggregate": lambda d, r: d.join(r, on="k").agg(s=bt.col("v").sum()),
    "distinct": lambda d, r: d.join(r, on="k").distinct(),
    "window_partitioned": lambda d, r: d.join(r, on="k").with_columns(
        x=bt.col("v").sum().over(partition_by="k")
    ),
    "window_global": lambda d, r: d.join(r, on="k").with_columns(
        x=bt.col("v").sum().over(order_by="v")
    ),
    "window_over_union": lambda d, r: d.union(d).with_columns(
        x=bt.col("v").sum().over(partition_by="k")
    ),
}


@pytest.mark.parametrize("shape", sorted(MULTI_SOURCE_BREAKERS))
def test_a_breaker_over_two_sources_declines_rather_than_raising(shape):
    """`collect(spill=True)` must never raise on a shape `collect()` answers.

    The out-of-core executors relabel their input's single scan to source 0, so a breaker
    whose input spans two sources cannot ride them. Every gate is supposed to *decline* that
    (the caller then falls back in memory, costing memory and never correctness) -- but the
    two window gates never got the check, so a window above a join or a union collected fine
    and died under spill with `PlanError: expected a single-source subplan to relabel`.
    `supports_spilling_join`'s docstring asserted "Sort and Window already gate this way";
    only Sort did.
    """
    left = pa.table(
        {
            "k": pa.array([0, 1, 2, 3, 4, 4], pa.int64()),
            "v": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
        }
    )
    right = pa.table(
        {
            "k": pa.array([0, 1, 2, 3, 4, 4], pa.int64()),
            "w": pa.array([9, 8, 7, 6, 5, 4], pa.int64()),
        }
    )
    build = MULTI_SOURCE_BREAKERS[shape]

    def make():
        return build(bt.from_arrow(left), bt.from_arrow(right))

    assert_tables_equal(make().collect(spill=True), make().collect())


# --- A global window whose bucket still does not fit ---------------------------------------


def _wide_ordered(n: int) -> pa.Table:
    """Fat rows on a distinct float order key — one bucket overflows a small envelope."""
    import numpy as np

    rng = np.random.default_rng(7)
    return pa.table(
        {
            "v": pa.array(rng.random(n).tolist(), pa.float64()),
            "s": pa.array(["z" * (i % 2000) for i in range(n)], pa.large_string()),
        }
    )


@pytest.mark.parametrize(
    "functions",
    [
        pytest.param({"r": ("sum", "v")}, id="sum"),
        pytest.param({"r": ("avg", "v")}, id="avg"),
        pytest.param({"r": ("count", "v")}, id="count"),
        pytest.param({"lo": ("min", "v"), "hi": ("max", "v")}, id="min_max"),
    ],
)
@pytest.mark.parametrize("descending", [False, True])
def test_a_global_window_re_splits_a_bucket_that_still_does_not_fit(functions, descending):
    """`stage_and_partition` bounds the *average* bucket, not the largest.

    The range boundaries come from a sampled grid, so one bucket lands over the envelope; it
    was then read back whole and the engine refused it (`MemoryBudgetExceededError: window
    without PARTITION BY cannot spill`) -- the query dying under memory pressure on the path
    that exists to survive it. The partitioned window, the aggregate and the join all answer
    this with a re-split before the read; the ordered path now does too, splitting by *range*
    on the order key rather than by a hash salt, because a global window's buckets have to
    stay globally ordered for the offset algebra to mean anything.

    Rows are matched on `v` -- the ORDER BY column, carried through untouched and distinct per
    row -- rather than by position, because the result is an unordered relation. The derived
    columns compare up to reassociation: a running float reduction depends on summation order,
    and the ordered-bucket algebra already shows that at HEAD whenever it uses more than one
    bucket, so it is the documented exception rather than anything this path introduced.
    """
    import dataclasses
    import math

    from batcher.config import Config, config_context

    table = _wide_ordered(20_000)
    build = lambda d: d.window(  # noqa: E731
        order_by=[("v", descending)], functions=functions
    )
    # The budgeted run goes FIRST. The engine's buffer pool is process-global and sized on
    # first use, so computing the unbudgeted baseline first leaves a pool the budgeted run
    # then reuses -- which is why this shape passes in file order and fails in isolation.
    # Count the streamed-window invocations: without this the test passes whenever the plan
    # never reaches the out-of-core route at all, which is the one way it could not fail.
    # Patch the *package* attribute: `dist.spill.aggregate` imports the name from
    # `batcher.dist.global_window`, so patching the `stream` submodule is a no-op and the
    # counter would silently see nothing (`.claude/rules/concurrent-agents.md`, "monkeypatch
    # targets follow the name").
    import batcher.dist.global_window as gws

    original = gws.stream_spilling_global_window
    runs = []

    def counted(*args, **kwargs):
        runs.append(1)
        yield from original(*args, **kwargs)

    cfg = Config()
    tiny = cfg.replace(memory=dataclasses.replace(cfg.memory, max_memory_bytes=1 << 20))
    gws.stream_spilling_global_window = counted
    try:
        with config_context(tiny):
            spilled = {r["v"]: r for r in build(bt.from_arrow(table)).collect().to_pylist()}
    finally:
        gws.stream_spilling_global_window = original

    assert runs, (
        "the streamed global window never ran, so this proves nothing — the plan stayed "
        "in memory instead of taking the out-of-core route the test is about"
    )
    baseline = {r["v"]: r for r in build(bt.from_arrow(table)).collect().to_pylist()}
    assert set(spilled) == set(baseline), "the spilled window lost or invented rows"
    for key, want in baseline.items():
        got = spilled[key]
        assert set(got) == set(want)
        for col, expected in want.items():
            actual = got[col]
            if isinstance(expected, float) and isinstance(actual, float):
                assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (
                    f"{col} at v={key}: {actual!r} != {expected!r}"
                )
            else:
                assert actual == expected, f"{col} at v={key}"
