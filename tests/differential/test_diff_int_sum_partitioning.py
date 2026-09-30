"""An integer `SUM` raises exactly when its *true* total leaves `Int64`, however it is split.

Finding F212. Each partition used to narrow its partial total to `Int64` before the partitions
were merged, so a relation whose total fits but one of whose partitions does not -- one morsel
holding `2**63 - 1` and `1` for a group, a later one holding `-2` -- raised when the engine
partitioned it (sharded streaming, the morsel-parallel materializing executor, the spilling
aggregate) and answered when it did not. The partial state is now the exact 128-bit total,
and `finalize` is the only place it is narrowed.

**The dtype contract this pins.** DuckDB's `SUM(BIGINT)` returns `HUGEINT`; Batcher's returns
`Int64` and raises when the true total does not fit, telling the caller to cast. So values are
compared against DuckDB, the Batcher column is asserted to be `int64`, and for a total past
`Int64` the test asserts both halves of the stated difference: DuckDB returns a number
outside `Int64`, Batcher raises.

**Why this fixture is large.** Below `MIN_ROWS_TO_SHARD` nothing shards and every scheduling
computes one partial, which is the one case that was always right. `_ROWS` clears the
threshold read from the Rust source (`test_the_fixture_shards`), the two overflow-relevant
rows sit in the first morsel and the cancelling one in the last, and a subprocess run under a
tight envelope proves the spilled scheduling really goes out of core.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from test_diff_execution_mode_matrix import _collect, _shard_threshold_rows, _stream

import batcher as bt
from _engine_run import run_plan

pytestmark = pytest.mark.differential

_MAX = 2**63 - 1
_ROWS = 100_000
#: Enough groups that the aggregate's state exceeds the control's envelope.
_GROUPS = 5_000
#: The cancelling row's position: group 0, in the last morsel.
_LAST = _ROWS - _GROUPS
#: The engine's morsel. The first one holds no negative filler, so its *global* partial is
#: `_MAX + 1` too, not just group 0's.
_MORSEL = 16_384


def _table(last: int) -> pa.Table:
    """Group 0: `_MAX` at row 0 and `1` at row `_GROUPS` (both in the first morsel), `last` at
    `_LAST`, zeros elsewhere. Group 1: all NULL. Every other group: `-(i % 7)` past the first
    morsel (zero inside it, so the global partial of that morsel overflows as well), so the global
    total is group 0's less a few hundred thousand, and fits whenever group 0's does."""

    def v(i: int) -> int | None:
        if i == 0:
            return _MAX
        if i == _GROUPS:
            return 1
        if i == _LAST:
            return last
        g = i % _GROUPS
        if g == 0 or i < _MORSEL:
            return 0 if g != 1 else None
        return None if g == 1 else -(i % 7)

    return pa.table(
        {
            "g": pa.array([i % _GROUPS for i in range(_ROWS)], pa.int64()),
            "v": pa.array([v(i) for i in range(_ROWS)], pa.int64()),
        }
    )


_FITS = _table(-2)
_OVERFLOWS = _table(10**9)


def _grouped(ds: bt.Dataset) -> bt.Dataset:
    return ds.group_by("g").agg(s=bt.col("v").sum())


def _global(ds: bt.Dataset) -> bt.Dataset:
    return ds.agg(s=bt.col("v").sum())


_SHAPES = {
    "grouped": (_grouped, "SELECT g, sum(v) AS s FROM t GROUP BY g"),
    "global": (_global, "SELECT sum(v) AS s FROM t"),
}

#: Every way the engine can split the relation into partials.
_SCHEDULINGS = {
    "streaming-sharded": lambda b, t: _collect(b, t, streaming=True),
    "materializing": lambda b, t: _collect(b, t, streaming=False),
    "streaming-spill": lambda b, t: _collect(b, t, streaming=True, spill=True),
    "materializing-spill": lambda b, t: _collect(b, t, streaming=False, spill=True),
    "iter_batches": lambda b, t: _stream(b, t, streaming=True),
    "spill_partitions_8": lambda b, t: b(bt.from_arrow(t)).collect(spill=True, num_partitions=8),
}


def _rows(table: pa.Table) -> list[tuple]:
    """Rows as exact Python ints, sorted -- no float tolerance, and DuckDB's HUGEINT (which
    arrives as a `decimal128(38, 0)`) compares as the integer it is."""
    cols = [
        [None if x is None else int(x) for x in table.column(n).to_pylist()]
        for n in table.column_names
    ]
    return sorted(zip(*cols, strict=True), key=repr)


def test_the_fixture_shards():
    assert _shard_threshold_rows() < _ROWS, "below the shard threshold nothing is partitioned"


@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_a_total_that_fits_matches_duckdb_in_every_scheduling(duck, shape, scheduling):
    build, sql = _SHAPES[shape]
    duck.register("t", _FITS)
    out = _SCHEDULINGS[scheduling](build, _FITS)
    assert out.schema.field("s").type == pa.int64(), "SUM(BIGINT) returns BIGINT"
    want = duck.sql(sql).to_arrow_table()
    assert _rows(out) == _rows(want)
    if shape == "grouped":
        by_group = dict(_rows(out))
        assert by_group[0] == _MAX - 1, "the group whose first partial overflows"
        assert by_group[1] is None, "an all-NULL group sums to NULL, not 0"


@pytest.mark.parametrize("scheduling", sorted(_SCHEDULINGS))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_a_true_overflow_raises_in_every_scheduling(duck, shape, scheduling):
    build, sql = _SHAPES[shape]
    duck.register("t", _OVERFLOWS)
    # The stated difference, both halves: DuckDB widens to HUGEINT and answers...
    totals = [r[-1] for r in duck.sql(sql).fetchall() if r[-1] is not None]
    assert max(totals) > _MAX
    # ...and Batcher refuses, on every path, rather than wrapping on any of them.
    with pytest.raises(bt.ExecutionError, match="overflow"):
        _SCHEDULINGS[scheduling](build, _OVERFLOWS)


def test_control_the_spilled_aggregate_really_goes_out_of_core(tmp_path, duck):
    """Under an envelope far below its state, the aggregate reports a spill and still agrees."""
    col = lambda name: {"e": "col", "name": name}  # noqa: E731
    plan = {
        "op": "aggregate",
        "input": {"op": "scan", "source_id": 0},
        "group_keys": [{"expr": col("g"), "alias": "g"}],
        "aggregates": [{"func": "sum", "alias": "s", "input": col("v")}],
    }
    report = run_plan(tmp_path, plan, [_FITS], 64_000)
    assert report.get("spilled", {}).get("aggregate"), report
    duck.register("t", _FITS)
    assert _rows(report["table"]) == _rows(duck.sql(_SHAPES["grouped"][1]).to_arrow_table())
