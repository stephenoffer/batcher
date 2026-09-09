"""A breaker beneath a spilled sort/window must be combined, not run per morsel.

The out-of-core sort, window and join stage their inputs by running each input's
sub-plan **once per input morsel** (`spill_breakers.sort::stage_and_partition`,
`spill_breakers.window`, and the join's per-side map all call
`execute_plan(map_ir, [[batch]])` in a loop). That is exactly right for the linear
scan/filter/project chain the paths were written for, and silently wrong for a *breaker*:
each morsel yields a partial, the partials are partitioned and processed, and nothing ever
combines them.

Nothing in the predicate stack caught it. `supports_spilling_sort` asks whether the leading
key can be range-partitioned and whether the input names a single source;
`supports_spilling_join` asks whether each side names a single source. `Sort(Aggregate(Scan(0)))`
and `Join(Aggregate(Scan(0)), Scan(1))` answer yes to everything asked. The `Aggregate`
branch of `spill_collect` had carried the guard (`_peel_to_breaker`) since it was written;
the ordering and binary branches had not.

TPC-H q1 is that shape, and it is the shape most of TPC-H ends in (`GROUP BY ... ORDER BY
...`). At sf10 under a 2 GiB envelope it returned **1,224 rows where the answer is 4**, with
no error, while the same query uncapped returned 4. A query whose answer changes because it
was short of memory is the worst failure this path can have: the envelope is the thing that
differs between a laptop and a cluster node.

**These fixtures must exceed 8 MiB, and that is the whole reason they are large.**
`_iter_spill_morsels` coalesces a source's batches into ~`_SPILL_INPUT_CHUNK_BYTES`
(8 MiB) chunks, so a small fixture is mapped as *one* chunk — one partial, which is
trivially the complete answer, and the bug is invisible. Measured: the same shape over a
40 x 1,000-row table passed unfixed, and over a 30 MiB one returned 28 rows for 7 —
both for the sort and for the join.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from _harness import assert_same

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

#: Rows enough to exceed `_SPILL_INPUT_CHUNK_BYTES` several times over at 16 bytes/row
#: (~30 MiB, ~4 chunks), which is what makes the per-morsel map observable at all.
_ROWS = 2_000_000

#: Few enough groups that an uncombined result is obviously wrong (chunks x groups) rather
#: than merely large, and that the expected answer is small enough to read.
_GROUPS = 7


@pytest.fixture(scope="module")
def wide_table() -> pa.Table:
    """A multi-chunk relation with a low-cardinality group key."""
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "g": rng.integers(0, _GROUPS, _ROWS).astype("int64"),
            "v": rng.integers(0, 100, _ROWS).astype("int64"),
        }
    )


def _spill_route(ds) -> pa.Table | None:
    """Run `ds` through the forced out-of-core route, as `collect(spill=True)` does.

    Driven directly rather than through `collect(spill=True)` so the test cannot pass by
    silently taking the in-memory path: `None` here means the shape declined, and every
    assertion below rejects that explicitly.
    """
    from batcher import core, kyber
    from batcher.api.orchestration import auto_num_partitions
    from batcher.dist.spill import spill_collect

    hub = core.default_hub()
    plan, sources = ds._plan, ds._sources
    optimized = kyber.optimize_logical(plan, sources=sources, hub=hub)
    return spill_collect(optimized, sources, auto_num_partitions(plan, sources, hub))


def test_a_sort_over_a_spilled_aggregate_matches_duckdb(duck, wide_table):
    """`GROUP BY ... ORDER BY ...` out-of-core: the groups are combined, once each."""
    duck.register("t", wide_table)
    ds = bt.from_arrow(wide_table).group_by("g").agg(s=bt.col("v").sum()).sort("g")

    spilled = _spill_route(ds)
    assert spilled is not None, (
        "the shape declined the out-of-core route, so this test would compare the "
        "in-memory result against DuckDB and prove nothing about spilling"
    )
    # The count is asserted first and separately, because it is the symptom: one row per
    # (chunk, group) instead of one per group. `assert_same` would catch it too, but the
    # count says *what* went wrong.
    assert spilled.num_rows == _GROUPS
    assert_same(spilled, duck.sql("SELECT g, sum(v) AS s FROM t GROUP BY g"))


def test_a_sort_over_a_spilled_distinct_matches_duckdb(duck, wide_table):
    """The same shape with `DISTINCT` as the inner breaker rather than an aggregate."""
    duck.register("t", wide_table)
    ds = bt.from_arrow(wide_table).select(bt.col("g")).distinct().sort("g")

    spilled = _spill_route(ds)
    assert spilled is not None, "the shape declined the out-of-core route"
    assert spilled.num_rows == _GROUPS
    assert_same(spilled, duck.sql("SELECT DISTINCT g FROM t"))


def test_a_window_over_a_spilled_aggregate_matches_duckdb(duck, wide_table):
    """A window function reads the aggregate's combined output, not its partials."""
    duck.register("t", wide_table)
    ds = (
        bt.from_arrow(wide_table)
        .group_by("g")
        .agg(s=bt.col("v").sum())
        .with_columns(r=bt.col("s").rank().over(order_by="s"))
    )

    spilled = _spill_route(ds)
    assert spilled is not None, "the shape declined the out-of-core route"
    assert spilled.num_rows == _GROUPS
    assert_same(
        spilled,
        duck.sql(
            "SELECT g, s, rank() OVER (ORDER BY s) AS r "
            "FROM (SELECT g, sum(v) AS s FROM t GROUP BY g)"
        ),
    )


def test_a_join_over_a_spilled_aggregate_matches_duckdb(duck, wide_table):
    """A breaker under a join *side*: `supports_spilling_join` sees a single source and says yes.

    The binary case is separate from the unary one above because the guard it defeats is a
    different guard, and because staging one side leaves the other side's source indices to
    stay valid — the reason `_stage_breaker_inputs` appends to a copy of the list.
    """
    dim = pa.table(
        {
            "g": np.arange(_GROUPS, dtype="int64"),
            "label": [f"L{i}" for i in range(_GROUPS)],
        }
    )
    duck.register("t", wide_table)
    duck.register("d", dim)
    ds = (
        bt.from_arrow(wide_table)
        .group_by("g")
        .agg(s=bt.col("v").sum())
        .join(bt.from_arrow(dim), left_on="g", right_on="g", how="inner")
    )

    spilled = _spill_route(ds)
    assert spilled is not None, "the shape declined the out-of-core route"
    assert spilled.num_rows == _GROUPS
    assert_same(
        spilled,
        duck.sql(
            "SELECT a.g, a.s, d.label FROM (SELECT g, sum(v) AS s FROM t GROUP BY g) a "
            "JOIN d ON a.g = d.g"
        ),
    )


def test_a_sort_over_a_linear_input_still_takes_the_fast_staging_path(duck, wide_table):
    """The positive control: a sort with *no* breaker below it is unaffected.

    The fix routes a sort whose input peels to a breaker through `spill_collect` on that
    inner node. A sort over a plain filter must not be diverted — it is the shape
    `stage_and_partition`'s per-morsel map is correct for, and the one it exists to serve.
    """
    duck.register("t", wide_table)
    ds = bt.from_arrow(wide_table).filter(bt.col("v") > 50).sort("g")

    spilled = _spill_route(ds)
    assert spilled is not None, "a sort over a filter must keep its out-of-core route"
    assert_same(spilled, duck.sql("SELECT * FROM t WHERE v > 50"))


def test_the_spilled_sort_is_actually_ordered(wide_table):
    """`assert_same` is order-independent by design, so the ordering needs its own check.

    Without this the tests above would pass on a spilled sort that emitted its buckets in
    the wrong order — the exact blind spot that let a `descending` out-of-core sort return
    unsorted data once before (see `test_diff_spill_paths`).
    """
    ds = bt.from_arrow(wide_table).group_by("g").agg(s=bt.col("v").sum()).sort("g")

    spilled = _spill_route(ds)
    assert spilled is not None, "the shape declined the out-of-core route"
    keys = spilled.column("g").to_pylist()
    assert keys == sorted(keys)
