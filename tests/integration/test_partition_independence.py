"""Partition-independence: results must not depend on input chunking.

This is the single most important invariant for distribution. If a query produces
the same answer whether the input is one morsel or many, then splitting that input
across partitions/actors (the essence of distributed execution) is safe. Every
operator we add is checked here against a 1-chunk vs many-chunk baseline.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

pytest.importorskip("batcher._native", reason="native engine not built")


def _chunks(table: pa.Table, n: int) -> list[pa.RecordBatch]:
    """Re-chunk a table into ~n record batches (≥1)."""
    rows = table.num_rows
    size = max(1, (rows + n - 1) // n)
    return table.combine_chunks().to_batches(max_chunksize=size)


def _rowset(table: pa.Table) -> list[tuple]:
    """Order-independent multiset view of a result (rows sorted as tuples)."""
    cols = table.column_names
    rows = [tuple(r[c] for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(v)) for v in t))


def _assert_chunk_invariant(build, table: pa.Table):
    """`build(ds)` must give the same multiset of rows for 1 chunk and many chunks.

    Compared order-independently: parallel/distributed execution of hash-based
    operators (aggregate/distinct/join) does not fix output order, but the *set*
    of result rows must be identical — that is the distribution-safety guarantee.
    """
    one = _rowset(build(bt.from_arrow(table.combine_chunks().to_batches())).collect())
    many = _rowset(build(bt.from_arrow(_chunks(table, 5))).collect())
    assert one == many, f"\n1-chunk:  {one}\nN-chunk:  {many}"


def test_filter_project_partition_independent():
    t = pa.table({"x": list(range(20)), "y": list(range(100, 120))})
    _assert_chunk_invariant(
        lambda ds: ds.filter(col("x") % 2 == 0).select("x", s=col("x") + col("y")),
        t,
    )


def test_aggregation_partition_independent():
    t = pa.table(
        {
            "g": [i % 4 for i in range(40)],
            "v": [i * 3 % 7 for i in range(40)],
        }
    )
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(
            s=col("v").sum(), n=count(), a=col("v").mean(), lo=col("v").min(), hi=col("v").max()
        ),
        t,
    )


def test_bool_aggregates_partition_independent():
    t = pa.table(
        {
            "g": [i % 4 for i in range(40)],
            "flag": [None if i % 7 == 0 else (i % 3 == 0) for i in range(40)],
        }
    )
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(ba=col("flag").bool_and(), bo=col("flag").bool_or()),
        t,
    )


def test_approx_distinct_partition_independent():
    # HLL registers are order-independent and merge register-wise, so the *estimate*
    # is bit-identical regardless of chunking — partition-independence holds exactly.
    t = pa.table({"g": [i % 4 for i in range(200)], "v": [(i * 7) % 60 for i in range(200)]})
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(nd=col("v").approx_n_unique()),
        t,
    )


def test_approx_quantile_partition_independent():
    # KLL merge is deterministic and order-independent, so the approx quantile is
    # identical regardless of chunking — partition-independence holds exactly.
    t = pa.table(
        {"g": [i % 3 for i in range(300)], "v": [float((i * 13) % 97) for i in range(300)]}
    )
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(m=col("v").approx_median()),
        t,
    )


def test_mode_partition_independent():
    # mode breaks frequency ties by the smallest value, so it is deterministic and
    # identical regardless of chunking — even when ties are present.
    t = pa.table({"g": [i % 3 for i in range(60)], "v": [(i * 5) % 7 for i in range(60)]})
    _assert_chunk_invariant(lambda ds: ds.group_by("g").agg(m=col("v").mode()), t)


def test_arg_extreme_partition_independent():
    # arg_min/arg_max break key ties by the smallest value, so the (key, value)
    # winner is deterministic regardless of chunking.
    t = pa.table(
        {
            "g": [i % 4 for i in range(80)],
            "val": [(i * 11) % 50 for i in range(80)],
            "key": [(i * 7) % 13 for i in range(80)],
        }
    )
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(
            hi=col("val").arg_max(by=col("key")), lo=col("val").arg_min(by=col("key"))
        ),
        t,
    )


def test_sorted_result_partition_independent():
    t = pa.table({"x": [7, 3, 9, 1, 5, 2, 8, 4, 6, 0]})
    # The fully-ordered result is identical regardless of input chunking.
    _assert_chunk_invariant(lambda ds: ds.sort("x"), t)


def test_distinct_partition_independent():
    t = pa.table({"a": [1, 1, 2, 3, 3, 2, 1], "b": ["x", "x", "y", "z", "z", "y", "x"]})
    _assert_chunk_invariant(lambda ds: ds.distinct().sort("a", "b"), t)


def test_join_partition_independent():
    """A join's result is identical whether either side is one chunk or many.

    This is the distribution-safety guarantee for joins: hash-partitioning each
    side across actors cannot change the answer.
    """
    left = pa.table({"k": [i % 5 for i in range(30)], "lv": list(range(30))})
    right = pa.table({"k": [0, 1, 2, 3, 4, 2, 3], "rv": [10, 11, 12, 13, 14, 22, 33]})

    def run(lds, rds):
        return _rowset(
            lds.join(rds, on="k").group_by("k").agg(n=count(), s=col("lv").sum()).collect()
        )

    one = run(
        bt.from_arrow(left.combine_chunks().to_batches()),
        bt.from_arrow(right.combine_chunks().to_batches()),
    )
    many = run(bt.from_arrow(_chunks(left, 4)), bt.from_arrow(_chunks(right, 3)))
    assert one == many, f"\n1-chunk: {one}\nN-chunk: {many}"


def _stat_table() -> pa.Table:
    rng = __import__("numpy").random.default_rng(0)
    return pa.table(
        {
            "g": [i % 5 for i in range(120)],
            "v": rng.integers(0, 100, 120).astype("int64"),
            "f": rng.normal(0, 1, 120),
        }
    )


def _assert_chunk_invariant_approx(build, table: pa.Table):
    """Like `_assert_chunk_invariant`, but tolerant of float rounding.

    Float addition is not associative, so a float aggregate (e.g. variance of a
    float column) summed in a different chunk order can differ in the last bit.
    The distribution-safety invariant for floats is therefore "equal up to
    rounding", which is what this checks (values agree to ~9 significant digits).
    """

    def rounded(table):
        cols = table.column_names
        rows = [
            tuple(round(r[c], 9) if isinstance(r[c], float) else r[c] for c in cols)
            for r in table.to_pylist()
        ]
        return sorted(rows, key=lambda t: tuple((v is None, str(v)) for v in t))

    one = rounded(build(bt.from_arrow(table.combine_chunks().to_batches())).collect())
    many = rounded(build(bt.from_arrow(_chunks(table, 5))).collect())
    assert one == many, f"\n1-chunk:  {one}\nN-chunk:  {many}"


def test_statistical_aggregates_partition_independent():
    """var/stddev (3-column state) and median/n_unique (list state) merge the same
    whether the input is one chunk or many — the invariant for distributing them.
    (Float aggregates are compared up to rounding; see `_assert_chunk_invariant_approx`.)"""
    _assert_chunk_invariant_approx(
        lambda ds: ds.group_by("g").agg(
            vv=col("v").var(),
            sd=col("v").std(),
            m=col("v").median(),
            nd=col("v").n_unique(),
            fv=col("f").var(),
        ),
        _stat_table(),
    )


def test_global_statistical_aggregates_partition_independent():
    _assert_chunk_invariant_approx(
        lambda ds: ds.group_by().agg(
            m=col("v").median(), nd=col("v").n_unique(), sd=col("v").std()
        ),
        _stat_table(),
    )


def test_decimal_aggregate_partition_independent():
    import decimal as D

    prices = pa.array(
        [D.Decimal(f"{(i * 7) % 50}.{i % 100:02d}") for i in range(60)],
        pa.decimal128(12, 2),
    )
    t = pa.table({"g": [i % 4 for i in range(60)], "p": prices})
    _assert_chunk_invariant(
        lambda ds: ds.group_by("g").agg(s=col("p").sum(), lo=col("p").min(), hi=col("p").max()),
        t,
    )


def test_window_partition_independent():
    """Window output (whole-partition aggregates and running aggregates) must not
    depend on how the input was chunked before the window operator."""
    t = pa.table({"p": [i % 3 for i in range(30)], "v": [(i * 7) % 11 for i in range(30)]})
    _assert_chunk_invariant(
        lambda ds: ds.window(
            partition_by=["p"],
            functions={"tot": ("sum", "v"), "mx": ("max", "v")},
        ),
        t,
    )
    _assert_chunk_invariant(
        lambda ds: ds.window(
            partition_by=["p"],
            order_by=[("v", False)],
            functions={"rn": "row_number", "run": ("sum", "v")},
        ),
        t,
    )
