"""A global window over a ``ROWS`` frame must split into ordered buckets, and agree.

`cum_sum` and its siblings build ``ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`` and
`rolling_sum(n)` builds ``ROWS BETWEEN n - 1 PRECEDING AND CURRENT ROW``. The ordered-bucket
algebra refused every explicit frame on a global window, so both shapes materialized under
`collect(spill=True)` and `iter_batches()` and raised `PlanError` under
`collect(distributed=True)` (audit finding F093). `dist/global_window/frames.py` now carries
both: the running frame takes the default frame's offset, and the trailing one a boundary
exchange of the prior buckets' last `p` values.

The trailing frame is the one whose correctness depends on the cut, the way `lag`'s does: a
correction wrong by a boundary is wrong on exactly `p` rows per bucket. So the frame widths
straddle the cases that matter -- `1` row back, a few, and **more rows than a bucket holds**,
where a row's frame reaches two or more buckets back -- at several bucket counts, ascending
and descending, over integers with nulls and floats with NaN (the kernel orders NaN greatest,
which `max` must propagate and `min` must skip).

Every order key is unique, so a ``ROWS`` frame has one answer (over tied keys it would
depend on the tie order, which the window-tie exception leaves open). DuckDB is the oracle for
the integer cases; the NaN cases are held to the in-memory kernel, whose NaN ordering is the
engine's own contract.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

_N = 3000


def _rows(n: int) -> pa.Table:
    return pa.table(
        {
            "rid": pa.array(range(n), pa.int64()),
            "t": pa.array([(i * 7919) % max(n, 1) for i in range(n)], pa.int64()),
            "x": pa.array([None if i % 7 == 0 else (i * 13) % 101 - 50 for i in range(n)]),
            "f": pa.array(
                [
                    None if i % 5 == 0 else (float("nan") if i % 31 == 0 else float(i % 17))
                    for i in range(n)
                ],
                pa.float64(),
            ),
        }
    )


#: (window expression over `x`, the DuckDB aggregate and frame it must equal).
_INT_CASES = {
    "rolling_sum_1": (lambda c: c("x").rolling_sum(1, order_by="t"), "sum(x)", "0 PRECEDING"),
    "rolling_sum_3": (lambda c: c("x").rolling_sum(3, order_by="t"), "sum(x)", "2 PRECEDING"),
    "rolling_min_50": (lambda c: c("x").rolling_min(50, order_by="t"), "min(x)", "49 PRECEDING"),
    "rolling_max_7": (lambda c: c("x").rolling_max(7, order_by="t"), "max(x)", "6 PRECEDING"),
    "rolling_mean_4": (lambda c: c("x").rolling_mean(4, order_by="t"), "avg(x)", "3 PRECEDING"),
    # Wider than a bucket at 7 and 13 buckets: the frame reaches several buckets back.
    "rolling_count_900": (
        lambda c: c("x").rolling_count(900, order_by="t"),
        "count(x)",
        "899 PRECEDING",
    ),
    "cum_sum": (lambda c: c("x").cum_sum(order_by="t"), "sum(x)", "UNBOUNDED PRECEDING"),
    "cum_max": (lambda c: c("x").cum_max(order_by="t"), "max(x)", "UNBOUNDED PRECEDING"),
    "cum_count": (lambda c: c("x").cum_count(order_by="t"), "count(x)", "UNBOUNDED PRECEDING"),
}

_PATHS = ["collect", "spill_2", "spill_7", "spill_13", "iter_batches"]


def _run(ds, path: str) -> pa.Table:
    if path == "collect":
        return ds.collect()
    if path.startswith("spill_"):
        return ds.collect(spill=True, num_partitions=int(path.split("_")[1]))
    batches = list(ds.iter_batches())
    return pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)


def _duck(rows: pa.Table, agg: str, start: str, *, descending: bool = False):
    con = duckdb.connect()
    con.register("t_", rows)
    order = "t DESC" if descending else "t"
    over = f"OVER (ORDER BY {order} ROWS BETWEEN {start} AND CURRENT ROW)"
    return con.sql(f"SELECT rid, t, x, f, {agg} {over} AS w FROM t_")


@pytest.mark.parametrize("case", sorted(_INT_CASES))
@pytest.mark.parametrize("path", _PATHS)
def test_a_rows_framed_global_window_matches_duckdb(case, path):
    rows = _rows(_N)
    build, agg, start = _INT_CASES[case]
    got = _run(bt.from_arrow(rows).with_columns(w=build(bt.col)), path)
    assert got.column_names == ["rid", "t", "x", "f", "w"]
    assert_same(got, _duck(rows, agg, start))


@pytest.mark.parametrize("width", [2, 40, 1200])
@pytest.mark.parametrize("parts", [3, 11])
def test_a_descending_trailing_frame_matches_duckdb(width, parts):
    """Bucket 0 holds the lowest keys, so a descending window is walked highest-first; a tail
    carried in visit order but read in key order would pass every ascending case above."""
    rows = _rows(_N)
    expr = bt.col("x").sum().over(order_by=[("t", True)], frame=(-(width - 1), 0))
    got = bt.from_arrow(rows).with_columns(w=expr).collect(spill=True, num_partitions=parts)
    assert_same(got, _duck(rows, "sum(x)", f"{width - 1} PRECEDING", descending=True))


@pytest.mark.parametrize(
    "build",
    [
        lambda c: c("f").rolling_sum(3, order_by="t"),
        lambda c: c("f").rolling_max(5, order_by="t"),
        lambda c: c("f").rolling_min(5, order_by="t"),
        lambda c: c("f").rolling_mean(9, order_by="t"),
        lambda c: c("f").cum_max(order_by="t"),
    ],
    ids=["sum", "max", "min", "mean", "cum_max"],
)
@pytest.mark.parametrize("parts", [2, 13])
def test_nan_follows_the_kernels_order(build, parts):
    ds = bt.from_arrow(_rows(_N)).with_columns(w=build(bt.col))
    by_rid = [("rid", "ascending")]
    assert_tables_equal(
        ds.collect(spill=True, num_partitions=parts).sort_by(by_rid),
        ds.collect().sort_by(by_rid),
        ordered=True,
    )


@pytest.mark.parametrize("n", [0, 1, 2])
def test_a_relation_shorter_than_the_frame(n):
    rows = _rows(n)
    build, agg, start = _INT_CASES["rolling_min_50"]
    ds = bt.from_arrow(rows).with_columns(w=build(bt.col))
    for got in (ds.collect(), ds.collect(spill=True, num_partitions=3)):
        assert got.column_names == ["rid", "t", "x", "f", "w"]
        assert_same(got, _duck(rows, agg, start))


def test_the_frames_reach_the_ordered_bucket_walk(monkeypatch):
    """The positive control: the in-memory kernel answers all of the above correctly, so the
    answers alone cannot show the split ran. Count the buckets the offset walk corrects."""
    from batcher.dist.global_window import offsets

    walked: list[int] = []
    original = offsets.OrderedBucketOffsets.apply

    def spy(self, wt):
        walked.append(wt.num_rows)
        return original(self, wt)

    monkeypatch.setattr(offsets.OrderedBucketOffsets, "apply", spy)
    for build, _, _ in (_INT_CASES["rolling_mean_4"], _INT_CASES["cum_sum"]):
        walked.clear()
        ds = bt.from_arrow(_rows(_N)).with_columns(w=build(bt.col))
        ds.collect(spill=True, num_partitions=7)
        assert len(walked) == 7
        assert sum(walked) == _N


def test_frames_the_split_cannot_carry_still_decline():
    """A FOLLOWING edge reads the bucket the walk has not reached; a decimal trailing sum would
    fold through a cast. Both must stay on the materializing kernel rather than be split."""
    from batcher.dist.global_window import supports_ordered_bucket_offsets

    ds = bt.from_arrow(_rows(20))
    following = ds.with_columns(w=bt.col("x").sum().over(order_by="t", frame=(-1, 1)))
    assert supports_ordered_bucket_offsets(following._plan) is False
    decimal = ds.with_columns(d=bt.col("x").cast(pa.decimal128(12, 2)))
    framed = decimal.with_columns(w=bt.col("d").sum().over(order_by="t", frame=(-2, 0)))
    assert supports_ordered_bucket_offsets(framed._plan) is False


@pytest.mark.parametrize("path", _PATHS)
def test_a_running_rows_variance_matches_duckdb(path):
    """The moments are rebuilt from helper columns, which must span the same ROWS frame.

    Compared with a relative tolerance rather than through `assert_same`: its fixed
    significant-digit rounding splits values whose last bits differ between two correct
    Welford-style accumulations (851.317437824 against 851.317437823), which says nothing
    about the frame. A helper computed over the wrong frame is wrong in the leading digits.
    """
    rows = _rows(_N)
    expr = bt.col("x").var().over(order_by="t", frame=(None, 0))
    got = _run(bt.from_arrow(rows).with_columns(w=expr), path).sort_by([("rid", "ascending")])
    want = _duck(rows, "var_samp(x)", "UNBOUNDED PRECEDING").order("rid").to_arrow_table()
    pairs = list(zip(got.column("w").to_pylist(), want.column("w").to_pylist(), strict=True))
    assert [g is None for g, _ in pairs] == [w is None for _, w in pairs]
    defined = [(g, w) for g, w in pairs if w is not None]
    assert defined, "the control needs defined variances"
    assert [g for g, _ in defined] == pytest.approx([w for _, w in defined], rel=1e-9)
