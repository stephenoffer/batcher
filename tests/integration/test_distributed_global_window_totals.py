"""Global window functions that need the relation's *total* — distributed, and equal.

`percent_rank` and `cume_dist` divide by the row count of the whole relation, `ntile` cuts
that same total into tiles, and `last_value` reads the partition's final row. None of those
is a number an ordered bucket knows, so the ordered-bucket algebra used to decline all four —
and a global window is not a `_split_at` pass-through, so declining did not mean "run it
slower", it meant `PlanError` on distributed data. ``ORDER BY <col>`` with a `percent_rank`
simply could not be distributed.

They are offsettable after all, one pass later. Each rides a helper the kernel computes
beside it — a running `rank`, a running row count, a running `row_number` — which the
ordinary per-bucket offset shifts during the walk; when the walk ends, the total row count is
just how many rows went past, and `OrderedBucketOffsets.finalize` closes the four out over
the assembled result. A driver that concatenates its buckets can do that (both distributed
ones); the single-node streaming driver yields as it goes and cannot, which is what
`supports_ordered_bucket_offsets(..., assembled=...)` distinguishes.

`var` and `stddev` are here too, for a different reason: they need no second pass but no
constant shift either, combining by Chan's formula over `(count, mean, M2)`.

The source is a real multi-file Parquet directory rather than an in-memory table on purpose.
`dist.executor._unsupported` runs an in-memory source on one node by design — correct, since
there is no distributed data — so a missing route over `bt.from_arrow` is indistinguishable
from the right answer. On a splittable source it raises, and the test would fail rather than
quietly pass.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import ray_session_fixture

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_N = 400
_WORKERS = 2


_ray_session = ray_session_fixture(4)


@pytest.fixture(scope="module")
def splittable(cluster_scratch) -> str:
    """Three Parquet files of the same rows — genuinely splittable, with duplicate keys.

    The duplicates matter: `percent_rank` and `cume_dist` are defined over peer groups, and a
    key that never repeats would let a per-row approximation pass.
    """
    table = pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "x": pa.array([(i * 37) % 91 for i in range(_N)], pa.int64()),
            "v": pa.array(
                [None if i % 13 == 0 else float(i % 17) for i in range(_N)], pa.float64()
            ),
        }
    )
    directory = cluster_scratch("global_window_totals")
    for part in range(3):
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


#: One entry per function this route gained. `mixed` is not redundant: the four finalized
#: functions share one `finalize` pass and one helper namespace, so a window carrying several
#: at once is the case where a helper alias collision or a dropped column would show.
_CASES: dict[str, dict] = {
    "percent_rank": {"r": "percent_rank"},
    "cume_dist": {"r": "cume_dist"},
    "ntile_3": {"r": ("ntile", 3)},
    "ntile_7": {"r": ("ntile", 7)},
    "last_value": {"r": ("last_value", bt.col("v"))},
    "var": {"r": ("var", bt.col("v"))},
    "stddev": {"r": ("stddev", bt.col("v"))},
    "mixed": {
        "a": "row_number",
        "b": "percent_rank",
        "c": ("var", bt.col("v")),
        "d": ("last_value", bt.col("v")),
        "e": ("avg", bt.col("v")),
        "f": ("ntile", 4),
        "g": "cume_dist",
        "h": ("stddev", bt.col("v")),
    },
}


def _by_rid(table: pa.Table) -> dict:
    """`table` in `rid` order — a window result is an unordered relation, so sort it here."""
    return table.sort_by([("rid", "ascending")]).to_pydict()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_distributed_answer_is_the_single_node_answer(splittable, case):
    functions = _CASES[case]
    windowed = bt.read.parquet(splittable).window(order_by=["x"], functions=functions)
    single = _by_rid(windowed.collect())
    distributed = _by_rid(
        bt.read.parquet(splittable)
        .window(order_by=["x"], functions=functions)
        .collect(distributed=True, num_workers=_WORKERS)
    )

    assert sorted(distributed) == sorted(single)
    for column in single:
        got, want = distributed[column], single[column]
        assert [x is None for x in got] == [x is None for x in want], column
        pairs = [(a, b) for a, b in zip(got, want, strict=True) if b is not None]
        if pairs and isinstance(pairs[0][1], float):
            # `var`/`stddev` are float reductions the two paths associate differently; the
            # rest are exact and pass this comparison at any tolerance.
            assert [a for a, _ in pairs] == pytest.approx([b for _, b in pairs], rel=1e-12), column
        else:
            assert [a for a, _ in pairs] == [b for _, b in pairs], column


def test_the_streaming_driver_still_declines_the_finalized_four(splittable):
    """The control. `finalize` is what makes these four correct, and the streaming driver
    yields each bucket before the total is known — so it must keep the materializing kernel.

    Without this, a change that dropped the `assembled` distinction would leave every
    assertion above green while `collect(spill=True)` returned a `percent_rank` divided by
    one bucket's row count: a number in `[0, 1]`, on the right rows, and wrong.
    """
    from batcher.dist.global_window import supports_ordered_bucket_offsets

    plan = bt.read.parquet(splittable).window(order_by=["x"], functions={"r": "percent_rank"})._plan
    assert supports_ordered_bucket_offsets(plan) is False
    assert supports_ordered_bucket_offsets(plan, assembled=True) is True


@pytest.fixture(scope="module")
def unique_keys(cluster_scratch) -> str:
    """Three Parquet files whose order key never repeats, so `nth_value` has one answer.

    Under a tie at the k-th position, *which* tied row is the k-th is left open by the
    `ORDER BY` and may differ between the paths (the window-tie exception in
    `.claude/rules/python-control-plane.md`), so the value comparison needs no ties at all.
    """
    directory = cluster_scratch("global_window_nth")
    for part in range(3):
        keys = range(part * _N, (part + 1) * _N)
        # Interleave the keys across files so no file is one contiguous key range.
        table = pa.table(
            {
                "rid": pa.array([(k * 7919) % (3 * _N) for k in keys], pa.int64()),
                "v": pa.array([None if k % 11 == 0 else float(k) for k in keys], pa.float64()),
            }
        )
        pq.write_table(table, directory / f"p{part}.parquet")
    return str(directory)


@pytest.mark.parametrize("k", [2, 5, 400, 1199, 1200, 1201])
def test_nth_value_past_the_first_distributes(unique_keys, k):
    """`nth_value(v, k)` for `k > 1` -- the one spelling Kyber does not turn into `first_value`.

    The value is the relation's k-th row once a row's frame reaches it and NULL before, so it
    is decided in whichever bucket holds that row; `k` sweeps early, mid-relation, the last
    row, and one past it (NULL everywhere). Compared per `rid` against single-node.
    """
    functions = {"r": ("nth_value", bt.col("v"), k)}
    single = _by_rid(
        bt.read.parquet(unique_keys).window(order_by=["rid"], functions=functions).collect()
    )
    distributed = _by_rid(
        bt.read.parquet(unique_keys)
        .window(order_by=["rid"], functions=functions)
        .collect(distributed=True, num_workers=_WORKERS)
    )
    assert distributed == single
    # Not two all-NULL columns agreeing: the last row's frame is the whole relation, so it
    # holds the k-th row's `v` (NULL only where the fixture put one) -- the single value any
    # row takes -- and past the relation it is NULL too.
    last = single["r"][-1]
    assert {x for x in single["r"] if x is not None} == ({last} - {None})
    if k > 3 * _N:
        assert last is None
    elif k in (2, 5, 400):
        assert last is not None, "the control needs a non-NULL k-th value"


#: A leading ORDER BY key that is an *expression* (F092). The range partitioner reads the cut
#: key from a column, so before `hoist_window_keys` learned the global case these raised
#: `PlanError` on this splittable source -- which is what makes the comparison below a test
#: of the distributed route rather than of a fallback: the unfixed system fails it outright.
_COMPUTED_ORDER = {
    # The fixture repeats every `rid` once per file, so a `row_number` would break ties the
    # query leaves open; `dense_rank` gives peers one value and has a single answer.
    "dense_rank_negated": ({"r": "dense_rank"}, [bt.col("rid") * -1]),
    "rank_dup": ({"r": "rank"}, [bt.col("x") % 13]),
    "sum_then_column": ({"r": ("sum", bt.col("v"))}, [bt.col("x") * 2 + 1, bt.col("rid")]),
    "percent_rank_dup": ({"r": "percent_rank"}, [bt.col("x") - 40]),
}


@pytest.mark.parametrize("case", sorted(_COMPUTED_ORDER))
def test_a_computed_order_key_distributes(splittable, case):
    functions, order = _COMPUTED_ORDER[case]
    windowed = bt.read.parquet(splittable).window(order_by=order, functions=functions)
    single = _by_rid(windowed.collect(distributed=False))
    distributed = windowed.collect(distributed=True, num_workers=_WORKERS)
    assert distributed.column_names == windowed.collect(distributed=False).column_names
    got = _by_rid(distributed)
    assert got.keys() == single.keys()
    for column in single:
        want = single[column]
        if want and isinstance(next((w for w in want if w is not None), None), float):
            assert [g is None for g in got[column]] == [w is None for w in want], column
            pairs = [(g, w) for g, w in zip(got[column], want, strict=True) if w is not None]
            assert [g for g, _ in pairs] == pytest.approx([w for _, w in pairs], rel=1e-12)
        else:
            assert got[column] == want, column


#: A top-N bound over a global ranking, which Kyber fuses into the window as `rank_limit`
#: (F094). Fused, a bucket knows only its local rank, so these raised `PlanError` here. Each is
#: checked against DuckDB directly as well as against single-node: `row_number` over a key
#: with no ties (so the tie exception cannot pick different rows), `rank`/`dense_rank` over a
#: heavily duplicated key, where the bound must keep every row tied at the cut, and the
#: degenerate bounds -- zero rows, and more rows than the relation holds.
_RANK_BOUNDS = {
    "row_number_le_25": ("row_number", "rid", "<=", 25),
    "row_number_eq_1_desc": ("row_number", ("rid", True), "=", 1),
    "row_number_le_0": ("row_number", "rid", "<=", 0),
    "row_number_past_the_end": ("row_number", "rid", "<=", 10 * _N),
    "rank_le_40_dups": ("rank", "x", "<=", 40),
    "dense_rank_le_3_dups": ("dense_rank", "x", "<=", 3),
}


def _bounded(path: str, case: str):
    func, order, op, k = _RANK_BOUNDS[case]
    ranked = bt.read.parquet(path).window(order_by=[order], functions={"r": func})
    predicate = bt.col("r") <= k if op == "<=" else bt.col("r") == k
    return ranked.filter(predicate)


@pytest.mark.parametrize("case", sorted(_RANK_BOUNDS))
def test_a_global_rank_bound_distributes_and_matches_duckdb(splittable, unique_keys, case):
    import duckdb

    from _harness import assert_same

    func, order, op, k = _RANK_BOUNDS[case]
    path = unique_keys if func == "row_number" else splittable
    single = _bounded(path, case).collect(distributed=False)
    distributed = _bounded(path, case).collect(distributed=True, num_workers=_WORKERS)
    assert distributed.column_names == single.column_names
    assert _by_rid(distributed) == _by_rid(single)

    key, desc = order if isinstance(order, tuple) else (order, False)
    over = f"{func}() OVER (ORDER BY {key}{' DESC' if desc else ''})"
    con = duckdb.connect()
    con.register("t", bt.read.parquet(path).collect(distributed=False))
    columns = ", ".join(single.column_names[:-1])
    duck = con.sql(f"SELECT * FROM (SELECT {columns}, {over} AS r FROM t) WHERE r {op} {k}")
    assert_same(distributed, duck)
    if k in (25, 40, 3):
        assert distributed.num_rows > 0, "the control needs a non-empty bound"


#: The two ROWS frames every cumulative and rolling helper builds (F093). An explicit frame on
#: a global window was refused, so each raised `PlanError` on this splittable source. Over a
#: unique key, so a ROWS frame has one answer. `rolling_count(1500)` is wider than a bucket at
#: this size, so its frame reaches more than one bucket back.
_ROWS_FRAMES = {
    "rolling_sum_4": lambda c: c("v").rolling_sum(4, order_by="rid"),
    "rolling_mean_30": lambda c: c("v").rolling_mean(30, order_by="rid"),
    "rolling_max_5": lambda c: c("v").rolling_max(5, order_by="rid"),
    "rolling_count_1500": lambda c: c("v").rolling_count(1500, order_by="rid"),
    "cum_sum": lambda c: c("v").cum_sum(order_by="rid"),
    "cum_min": lambda c: c("v").cum_min(order_by="rid"),
}


@pytest.mark.parametrize("case", sorted(_ROWS_FRAMES))
def test_a_rows_framed_global_window_distributes(unique_keys, case):
    framed = bt.read.parquet(unique_keys).with_columns(r=_ROWS_FRAMES[case](bt.col))
    single = _by_rid(framed.collect(distributed=False))
    distributed = framed.collect(distributed=True, num_workers=_WORKERS)
    got = _by_rid(distributed)
    assert got.keys() == single.keys()
    assert [g is None for g in got["r"]] == [w is None for w in single["r"]]
    pairs = [(g, w) for g, w in zip(got["r"], single["r"], strict=True) if w is not None]
    assert pairs, "the control needs non-NULL values"
    assert [g for g, _ in pairs] == pytest.approx([w for _, w in pairs], rel=1e-12)
