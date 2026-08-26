"""The operator shapes both engagement suites ask their question of.

`tests/unit/test_out_of_core_engagement.py` asks which *bounded-memory* mechanism each shape
engages (`collect(spill=True)`, `iter_batches()`); `tests/integration/test_distributed_
engagement.py` asks which *distributed executor* it reaches. Two questions, one operator
table — and the table has to be one object rather than two copies, because the whole point of
both files is that an operator cannot quietly go unasked. A duplicated list is exactly how one
of them would.

It lives at `tests/` root for the same reason `_harness.py` and `_ray_cluster.py` do: that is
the directory both suites can import from.

`EXPECTED_DISTRIBUTED` lives here rather than beside the tests that use it for one reason: the
guard that says *every* operator is asked the distributed question must not itself need a
cluster. Kept in the integration file it would have run only where Ray does, which is exactly
the job it exists to do everywhere.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher._internal.errors import PlanError

_ROWS = 600
_T = pa.table(
    {
        "k": pa.array([i % 17 for i in range(_ROWS)], pa.int64()),
        "t": pa.array([(i * 7) % 101 for i in range(_ROWS)], pa.int64()),
        "s": pa.array([f"g{i % 13}" for i in range(_ROWS)]),
    }
)
_R = pa.table(
    {
        "k": pa.array(list(range(17)), pa.int64()),
        # `t` so the same fixture serves the ASOF shapes; ascending, as an ASOF right side is.
        "t": pa.array([i * 6 for i in range(17)], pa.int64()),
        "w": pa.array([f"w{i}" for i in range(17)]),
    }
)


#: Each shape, as a builder over the two bound datasets. One home for the names — the tables
#: below and the coverage test all read this dict rather than restating it.
_BUILDERS = {
    "sort_plain": lambda left, right: left.sort("k"),
    "sort_string_key": lambda left, right: left.sort("s"),
    "sort_computed_key": lambda left, right: left.sort(bt.col("k") + bt.col("t")),
    "window_partitioned": lambda left, right: left.window(
        partition_by=["k"], order_by=["t"], functions={"r": "row_number"}
    ),
    "window_computed_key": lambda left, right: left.window(
        partition_by=[bt.col("k") % 4], order_by=["t"], functions={"r": "row_number"}
    ),
    "window_global_ordered": lambda left, right: left.window(
        order_by=["t"], functions={"r": "row_number"}
    ),
    "window_global_fold": lambda left, right: left.window(
        order_by=["t"], functions={"w": ("bit_or", bt.col("k"))}
    ),
    "window_global_lag": lambda left, right: left.window(
        order_by=["t"], functions={"w": ("lag", bt.col("k"))}
    ),
    "aggregate": lambda left, right: left.group_by("k").agg(n=bt.col("t").count()),
    "distinct": lambda left, right: left.select("k").distinct(),
    "join": lambda left, right: left.join(right, on="k"),
    "asof_join_by": lambda left, right: left.join_asof(right, on="t", by="k"),
    "asof_join_keyless": lambda left, right: left.join_asof(right, on="t"),
}


def _build(name):
    return _BUILDERS[name](bt.from_arrow(_T), bt.from_arrow(_R))


#: shape -> the executor that must run it, a set when the transport decides between two, or
#: `PlanError` when the contract is that the dispatcher refuses rather than quietly using one
#: node. `map` covers the breaker-free shapes, which fan the *read* out rather than shuffling.
EXPECTED_DISTRIBUTED: dict[str, object] = {
    "sort_plain": "sort",
    "sort_string_key": "sort",
    "sort_computed_key": "sort",
    "window_partitioned": "window",
    "window_computed_key": "window",
    # Which of the two global-window drivers runs is resolved from the cluster shape, so
    # pinning one would make this a topology assay rather than a routing check.
    "window_global_ordered": {"global_window", "global_window_flight"},
    "window_global_fold": {"global_window", "global_window_flight"},
    # A global `lag` reads rows its own ordered bucket does not hold, so no offset recovers
    # the global value and there is no decomposition. The dispatcher **raises** rather than
    # routing the whole query to one node behind the user's back; that is the policy
    # `_unsupported` states, so the raise is the contract and is asserted as one.
    "window_global_lag": PlanError,
    "aggregate": "aggregate",
    "distinct": "distinct",
    "join": "join",
    "asof_join_by": "asof_by",
    "asof_join_keyless": "asof_keyless",
}
