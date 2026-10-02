"""Aggregate *values* that stress arithmetic must come out the same spilled as in memory.

The spill matrices (`test_diff_operator_matrix.py`, `test_diff_mergeable_algebra_matrix.py`,
`test_diff_spill2_signed_zero.py`) pin `-0.0`/NaN as *keys*: whether a group survives being
hashed, partitioned and merged. This file pins the values a partial carries through that
merge, where the hazard is the arithmetic rather than the grouping:

* **cancellation** -- `1e16 + 1 - 1e16` loses the `1` in naive summation, and a merge that
  combines partials in a different order loses it differently;
* **large integer magnitudes** -- sums of `±2^62` must stay exact through every merge;
* **integer overflow** -- a sum that does not fit `i64` must raise in every scheduling, never
  come back wrapped from one of them. DuckDB widens to `HUGEINT` here instead; Batcher
  refuses with an error telling the caller to cast, which is a stated difference rather than
  a result to compare. A group whose *total* fits but one of whose partials does not must
  return that total in every scheduling: a partial carries its exact 128-bit total and only
  `finalize` narrows it (`bc-runtime/src/agg/int_sum`), so where the partials were cut
  cannot decide whether the query raises;
* **NaN and `-0.0` values** -- NaN must propagate through `sum`/`max`, and a group of zeros
  must not come back as a different number.

Every scheduling is compared against DuckDB -- `fsum` (exactly rounded) is the reference for
the float sums -- and the spilled ones against the in-memory result. A control shows the
partition-count knob is not inert: under a tight envelope the aggregate reports that it
spilled, and still returns the same answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _engine_run import run_plan
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

_BIG = 2**62
_N = 60_000
#: Enough groups that the aggregate state itself exceeds the control's envelope.
_GROUPS = 6_000


def _table() -> pa.Table:
    """Per group: a cancelling float triple, an overflow-prone integer run that nets out, and
    NaN or signed zeros confined to chosen groups."""
    g = [i % _GROUPS for i in range(_N)]
    cancel = [(1e16, 1.0, -1e16)[(i // _GROUPS) % 3] for i in range(_N)]
    # +2^62, -2^62 alternating per group, plus a residue that survives: every contiguous run
    # of a group's rows sums inside `i64`, so no partial can overflow however it is cut, and
    # the exact total is what every merge order must reach.
    inter = [(_BIG, -_BIG)[(i // _GROUPS) % 2] + (i % 7) for i in range(_N)]

    def special(i: int) -> float | None:
        grp = i % _GROUPS
        if grp == 0:
            return float("nan") if (i // _GROUPS) % 50 == 0 else 1.5
        if grp == 1:
            return -0.0 if i % 2 else 0.0
        if grp == 2:
            return -0.0
        return None if i % 11 == 0 else float(i % 13) - 6.0

    return pa.table(
        {
            "g": pa.array(g, pa.int64()),
            "cancel": pa.array(cancel, pa.float64()),
            "inter": pa.array(inter, pa.int64()),
            "special": pa.array([special(i) for i in range(_N)], pa.float64()),
        }
    )


def _query(ds: bt.Dataset) -> bt.Dataset:
    return ds.group_by("g").agg(
        cancel_sum=bt.col("cancel").sum(),
        inter_sum=bt.col("inter").sum(),
        special_sum=bt.col("special").sum(),
        special_min=bt.col("special").min(),
        special_max=bt.col("special").max(),
        special_avg=bt.col("special").mean(),
    )


_SQL = (
    "SELECT g, fsum(cancel) AS cancel_sum, CAST(sum(inter) AS BIGINT) AS inter_sum, "
    "fsum(special) AS special_sum, min(special) AS special_min, max(special) AS special_max, "
    "avg(special) AS special_avg FROM t GROUP BY g"
)

_SCHEDULINGS = {
    "in_memory": lambda ds: ds.collect(),
    "spill_1": lambda ds: ds.collect(spill=True, num_partitions=1),
    "spill_3": lambda ds: ds.collect(spill=True, num_partitions=3),
    "spill_8": lambda ds: ds.collect(spill=True, num_partitions=8),
}


@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
def test_arithmetic_edge_values_match_duckdb_in_every_scheduling(duck, scheduling):
    table = _table()
    duck.register("t", table)
    out = _SCHEDULINGS[scheduling](_query(bt.from_arrow(table)))
    assert_same(out, duck.sql(_SQL))


@pytest.mark.parametrize("scheduling", ["spill_1", "spill_3", "spill_8"])
def test_spilled_results_equal_the_in_memory_one(scheduling):
    table = _table()
    reference = _query(bt.from_arrow(table)).collect()
    assert_tables_equal(_SCHEDULINGS[scheduling](_query(bt.from_arrow(table))), reference)


def test_the_nan_and_signed_zero_groups_are_what_they_should_be():
    """Spelled out, because a float tolerance could hide a NaN turned into a number."""
    rows = {r["g"]: r for r in _query(bt.from_arrow(_table())).collect(spill=True).to_pylist()}
    nan_group, zeros, neg_zeros = rows[0], rows[1], rows[2]
    assert nan_group["special_sum"] != nan_group["special_sum"]  # NaN propagates
    assert nan_group["special_max"] != nan_group["special_max"]  # NaN is the greatest
    assert nan_group["special_min"] == 1.5
    for r in (zeros, neg_zeros):
        assert r["special_sum"] == 0.0 and r["special_min"] == 0.0 and r["special_max"] == 0.0


@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
def test_a_genuine_integer_overflow_raises_in_every_scheduling(scheduling):
    """Never a wrapped value from one scheduling and an error from another."""
    table = pa.table({"g": [1, 1, 2], "i": pa.array([2**63 - 1, 1, 5], pa.int64())})
    ds = bt.from_arrow(table).group_by("g").agg(s=bt.col("i").sum())
    with pytest.raises(Exception, match="overflow"):
        _SCHEDULINGS[scheduling](ds)


@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
def test_a_total_that_fits_is_exact_in_every_scheduling(duck, scheduling):
    """`[M, M, -M, -M]` with `M = 2^62`: the total fits, a partial holding both positives
    does not. Every scheduling returns the exact total, 0 -- none may refuse it (F212)."""
    m = _BIG
    table = pa.table({"g": pa.array([1] * 4000, pa.int64()), "i": [m, m, -m, -m] * 1000})
    duck.register("t", table)
    (expected,) = duck.sql("SELECT CAST(sum(i) AS BIGINT) FROM t").fetchone()
    ds = bt.from_arrow(table).group_by("g").agg(s=bt.col("i").sum())
    out = _SCHEDULINGS[scheduling](ds)
    assert out.column("s").to_pylist() == [expected]


def test_control_the_aggregate_really_goes_out_of_core(tmp_path):
    """Under an envelope a fraction of its input, the aggregate reports a spill and agrees.

    Without this, every comparison above could be between in-memory runs.
    """
    col = lambda name: {"e": "col", "name": name}  # noqa: E731
    plan = {
        "op": "aggregate",
        "input": {"op": "scan", "source_id": 0},
        "group_keys": [{"expr": col("g"), "alias": "g"}],
        "aggregates": [
            {"func": "sum", "alias": "cancel_sum", "input": col("cancel")},
            {"func": "sum", "alias": "inter_sum", "input": col("inter")},
            {"func": "sum", "alias": "special_sum", "input": col("special")},
            {"func": "min", "alias": "special_min", "input": col("special")},
            {"func": "max", "alias": "special_max", "input": col("special")},
            {"func": "mean", "alias": "special_avg", "input": col("special")},
        ],
    }
    table = _table()
    report = run_plan(tmp_path, plan, [table], 100_000)
    assert report.get("spilled", {}).get("aggregate"), report
    reference = _query(bt.from_arrow(table)).collect()
    assert_tables_equal(report["table"], reference)
