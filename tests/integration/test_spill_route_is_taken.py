"""The out-of-core path must actually run — not merely produce the right answer.

`tests/integration/test_spilling.py` is ~30 tests of the form::

    assert _norm(ds.collect(spill=True, num_partitions=16)) == _norm(ds.collect())

Every one of them is trivially true when nothing spills. `spill_collect` returns `None`
for a shape with no out-of-core path and the caller *silently falls through to the
in-memory path*, so the "spilled" result is the in-memory result, compared against
itself. If the entire out-of-core engine regressed to `return None`, that whole file
would stay green and the operator-by-operator spill implementations would be untested
while reporting full coverage.

This file is the missing half: it asserts **which route ran**. For every shape it pins
whether the forced-spill path handles it, and the fallback set is pinned just as
explicitly — a shape that quietly *loses* its out-of-core path fails here, which is the
regression the result comparison cannot see.

Correctness is asserted alongside, so a shape that takes the route and gets the answer
wrong fails too. Route without correctness would be as hollow as correctness without
route.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core, kyber

pytestmark = pytest.mark.integration

#: Enough batches to be worth partitioning, small enough to keep the matrix quick.
_BATCHES, _ROWS, _GROUPS = 40, 1_000, 500


def _table() -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.Table.from_batches(
        [
            pa.record_batch(
                {
                    "k": rng.integers(0, _GROUPS, _ROWS).astype("int64"),
                    "v": rng.integers(0, 100, _ROWS).astype("int64"),
                    "s": pa.array([f"g{i % 97}" for i in range(_ROWS)]),
                }
            )
            for _ in range(_BATCHES)
        ]
    )


TABLE = _table()
RIGHT = pa.table({"k": np.arange(_GROUPS, dtype="int64"), "w": np.arange(_GROUPS, dtype="int64")})

#: shape -> builder. Every stateful operator that claims an out-of-core implementation.
SPILLING_SHAPES = {
    "group_by_agg": lambda d: d.group_by("k").agg(s=col("v").sum()),
    "group_by_agg_string_key": lambda d: d.group_by("s").agg(t=col("v").sum()),
    "group_by_multi_key": lambda d: d.group_by("k", "s").agg(t=col("v").sum()),
    "global_agg": lambda d: d.agg(s=col("v").sum()),
    "distinct": lambda d: d.select(col("k")).distinct(),
    "count_distinct": lambda d: d.group_by("k").agg(n=col("v").n_unique()),
    "stddev": lambda d: d.group_by("k").agg(sd=col("v").std()),
    "median": lambda d: d.group_by("k").agg(m=col("v").median()),
    "sort_numeric": lambda d: d.sort(col("k")),
    "sort_descending": lambda d: d.sort(col("k"), descending=True),
    "sort_then_limit": lambda d: d.sort(col("k")).limit(10),
    # Was a documented fallback: the range partitioner was numeric-only, so a string key had
    # no boundaries and fell back in memory. `column_string_quantiles` /
    # `range_partition_batches_str` gave it one, but this list was never updated -- the shape
    # had been spilling correctly (all four ordering flags) while pinned as not spilling.
    "sort_string_key": lambda d: d.sort(col("s")),
    "window_sum": lambda d: d.with_columns(x=col("v").sum().over(partition_by="k")),
    "join_inner": lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="inner"),
    "join_left": lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k", how="left"),
}

#: shape -> why it legitimately has no out-of-core path. Pinned so that a *new* silent
#: fallback cannot be mistaken for one of these, and so that any of these gaining a spill
#: implementation shows up here as a failure to be celebrated and removed.
NON_SPILLING_SHAPES = {
    "filter_only": (
        lambda d: d.filter(col("v") > 50),
        "a filter is streaming — it holds no state, so there is nothing to spill",
    ),
    "project_only": (
        lambda d: d.select(col("k"), (col("v") * 2).alias("d")),
        "a projection is streaming — same reason",
    ),
    "limit_over_distinct": (
        lambda d: d.select(col("k")).distinct().limit(10),
        "a LIMIT re-applied above a spilled breaker takes the first k rows of hash-partition "
        "order, where the in-memory path takes them in input order -- different rows, so a "
        "wrong answer rather than a slower one (see "
        "test_diff_spill_paths.py::test_a_limit_above_a_spilled_breaker_keeps_the_same_rows)",
    ),
    "limit_over_group_by": (
        lambda d: d.group_by("k").agg(s=col("v").sum()).limit(10),
        "same reason as limit_over_distinct: the spilled aggregate emits in partition order",
    ),
    "limit_over_join": (
        lambda d: d.join(bt.from_arrow(RIGHT), left_on="k", right_on="k").limit(10),
        "same reason as limit_over_distinct: the grace-partitioned join emits in bucket order",
    ),
    "limit_over_window": (
        lambda d: d.with_columns(x=col("v").sum().over(partition_by="k")).limit(10),
        "same reason as limit_over_distinct: the spilled window emits in partition order",
    ),
}


def _spill_route(ds) -> pa.Table | None:
    """Run `ds` through the forced-spill route exactly as `collect(spill=True)` does.

    Returns the out-of-core result, or `None` when the shape has no spilling path — which
    is the signal every assertion in this file is about.
    """
    from batcher.api.orchestration import auto_num_partitions
    from batcher.dist.spill import spill_collect

    hub = core.default_hub()
    plan, sources = ds._plan, ds._sources
    partitions = auto_num_partitions(plan, sources, hub)
    optimized = kyber.optimize_logical(plan, sources=sources, hub=hub)
    return spill_collect(optimized, sources, partitions)


def _norm(table: pa.Table) -> list[tuple]:
    return sorted(
        tuple(round(v, 6) if isinstance(v, float) else v for v in row.values())
        for row in table.to_pylist()
    )


@pytest.mark.parametrize("shape", sorted(SPILLING_SHAPES))
def test_the_out_of_core_route_actually_runs(shape):
    """The shape is handled out-of-core, rather than silently falling back."""
    spilled = _spill_route(SPILLING_SHAPES[shape](bt.from_arrow(TABLE)))
    assert spilled is not None, (
        f"{shape} fell through to the in-memory path: `spill_collect` returned None. "
        f"Either this operator lost its out-of-core implementation, or the shape belongs "
        f"in NON_SPILLING_SHAPES with the reason why."
    )


@pytest.mark.parametrize("shape", sorted(SPILLING_SHAPES))
def test_the_out_of_core_result_equals_the_in_memory_result(shape):
    """...and having actually run out-of-core, it computes the same answer."""
    build = SPILLING_SHAPES[shape]
    spilled = _spill_route(build(bt.from_arrow(TABLE)))
    assert spilled is not None, (
        "covered by the route test; guarded so this one cannot pass vacuously"
    )
    assert _norm(spilled) == _norm(build(bt.from_arrow(TABLE)).collect())


@pytest.mark.parametrize("shape", sorted(NON_SPILLING_SHAPES))
def test_the_documented_fallbacks_are_still_exactly_these(shape):
    """The fallback set is a contract, not an accident.

    Pinning it is what lets `test_the_out_of_core_route_actually_runs` mean something: a
    shape drifting into the fallback set is then a failure rather than a silent change of
    behaviour. If one of these gains an out-of-core path, this test fails — move it into
    `SPILLING_SHAPES` and delete the entry.
    """
    build, reason = NON_SPILLING_SHAPES[shape]
    assert _spill_route(build(bt.from_arrow(TABLE))) is None, (
        f"{shape} now HAS an out-of-core path, but is listed as a fallback ({reason}). "
        f"Move it to SPILLING_SHAPES."
    )


@pytest.mark.parametrize("shape", sorted(NON_SPILLING_SHAPES))
def test_a_fallback_shape_still_returns_the_right_answer(shape):
    """Falling back is a routing decision, never a correctness one."""
    build, _ = NON_SPILLING_SHAPES[shape]
    with_spill = build(bt.from_arrow(TABLE)).collect(spill=True)
    assert _norm(with_spill) == _norm(build(bt.from_arrow(TABLE)).collect())


def test_collect_spill_true_reaches_the_out_of_core_route():
    """The public flag must reach the engine, not just be accepted.

    Everything above drives `spill_collect` directly. This pins the wiring in between —
    `collect(spill=True)` routing through `batcher.dist.spill` — so the flag becoming a
    no-op at the API layer fails here rather than silently returning in-memory results.
    """
    import batcher.api.terminal.core as terminal_core

    calls: list[int] = []
    real = terminal_core.__dict__.get("spill_collect")
    assert real is None, (
        "spill_collect is imported inside the function; patch the module it comes from"
    )

    import batcher.dist.spill as spill_module

    original = spill_module.spill_collect

    def _counting(logical, sources, partitions):
        calls.append(partitions)
        return original(logical, sources, partitions)

    spill_module.spill_collect = _counting
    try:
        out = bt.from_arrow(TABLE).group_by("k").agg(s=col("v").sum()).collect(spill=True)
    finally:
        spill_module.spill_collect = original

    assert calls, "collect(spill=True) never called the out-of-core executor"
    assert calls[0] > 0, f"the route ran with a non-positive partition count: {calls[0]}"
    assert out.num_rows == _GROUPS


def test_the_route_probe_can_distinguish_the_two_outcomes():
    """The probe these tests are built on must actually discriminate.

    If `_spill_route` returned non-`None` unconditionally, every route assertion above
    would pass forever. Running one shape from each set through it and requiring different
    outcomes is what rules that out.
    """
    spilling = _spill_route(SPILLING_SHAPES["group_by_agg"](bt.from_arrow(TABLE)))
    falling_back = _spill_route(NON_SPILLING_SHAPES["filter_only"][0](bt.from_arrow(TABLE)))
    assert spilling is not None and falling_back is None, (
        f"the probe does not discriminate: spilling={type(spilling).__name__}, "
        f"fallback={type(falling_back).__name__}"
    )
