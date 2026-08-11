"""A positional window function under a whole-partition frame still reads the ORDER BY.

`FIRST_VALUE` / `LAST_VALUE` / `NTH_VALUE` accept an explicit frame, and over
``ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`` they name a row of that frame
*by position*. That makes them the exception to an otherwise sound optimizer shortcut: an
aggregate over the whole partition cannot see the ordering, so the sort is dead work — but a
positional function can, and dropping the order keys answered it from the scan order instead.
The failure is silent, order-dependent, and disappears the moment the input happens to arrive
sorted, which is why it is pinned here against the oracle rather than as a plan-shape test.

The aggregates are included in the same parametrization on purpose: the fix must not stop the
rewrite from firing where it *is* sound.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.window_extra  # registers the rules into DEFAULT_REGISTRY
from _harness import assert_same

WHOLE = "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"


@pytest.fixture
def t(duck):
    # Deliberately *not* in `o` order, so a dropped sort changes the answer.
    tbl = pa.table(
        {
            "k": [1, 1, 1, 1, 1, 2, 2, 2],
            "o": [5, 1, 4, 2, 3, 2, None, 2],
            "v": [50, 10, 40, 20, 30, 9, 7, None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.differential
@pytest.mark.parametrize(
    "fn",
    [
        "first_value(v)",
        "last_value(v)",
        "nth_value(v, 2)",
        "nth_value(v, 9)",
        # Still sound for an aggregate — the rewrite must keep firing here.
        "sum(v)",
        "min(v)",
        "max(v)",
        "count(v)",
        "avg(v)",
    ],
)
@pytest.mark.parametrize(
    "frame", [WHOLE, "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"]
)
def test_whole_partition_frame_matches_duckdb(t, duck, fn, frame):
    query = f"SELECT k, o, v, {fn} OVER (PARTITION BY k ORDER BY o, v {frame}) AS r FROM t"
    assert_same(bt.sql(query, t=bt.from_arrow(t)).collect(), duck.sql(query))


@pytest.mark.differential
def test_the_sort_is_still_dropped_for_a_pure_aggregate_window(t):
    """The rewrite is a performance one, so its plan effect is asserted, not only its result."""
    from batcher.kyber.optimizer import optimize_logical
    from batcher.plan.visitor import walk

    ds = bt.from_arrow(t)
    agg_only = bt.sql(f"SELECT sum(v) OVER (ORDER BY o {WHOLE}) AS r FROM t", t=ds)
    positional = bt.sql(f"SELECT first_value(v) OVER (ORDER BY o {WHOLE}) AS r FROM t", t=ds)

    def order_keys(dataset):
        from batcher.plan.logical import Window

        plan = optimize_logical(dataset._plan)
        return [len(n.order_keys) for n in walk(plan) if isinstance(n, Window)]

    assert order_keys(agg_only) == [0]
    assert order_keys(positional) == [1]
