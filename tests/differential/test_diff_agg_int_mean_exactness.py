"""Differential coverage for an integer ``AVG``'s exact 128-bit accumulator.

``AVG`` needs a sum wider than its input, and the way Batcher gets one is a wider
*accumulator* (``Decimal128(38, 0)``, i.e. an ``i128``) rather than a widened copy of the
column. Two things about that are worth pinning against DuckDB rather than against Batcher's
own oracle:

* **A running sum in f64 is silently wrong on ordinary columns.** Once a partial sum passes
  2^53 every later addend loses its low bits, so ``AVG`` over IDs, nanosecond timestamps or
  cents returns a plausible number that is not the mean. DuckDB sums into a HUGEINT for the
  same reason, which makes it the right oracle here — the values below are chosen so an f64
  accumulator is off by whole units, far outside the harness's 1e-9 rounding tolerance.
* **The accumulator is reached by three different code paths**, and they are easy to drift
  apart: the fused multi-aggregate scan (two or more aggregates in one query), the per-call
  scatter (one aggregate), and the whole-column reduction (no ``GROUP BY``). A query picks
  one by shape alone, so each is exercised below rather than assumed equivalent.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

#: Magnitudes either side of 2^53 that cancel: the mean is small and exact, while an f64
#: running sum swallows the small addends and reports a different one. Nulls are present in
#: both groups, and group ``"c"`` is all-null — an ``AVG`` over no values is NULL, which is
#: the arm a no-null fast path skips over.
_BIG = 1 << 62
BASE = pa.table(
    {
        "g": ["a", "a", "a", "b", "b", "b", "c", "c"],
        "v": pa.array([_BIG, -_BIG, 3, _BIG + 1, -_BIG, None, None, None], pa.int64()),
        "w": pa.array([1, 2, 3, 4, 5, 6, 7, 8], pa.int64()),
    }
)


def test_grouped_int_avg_matches_duckdb(duck):
    """One ``AVG`` per group — the per-call scatter, with nulls and an all-null group."""
    duck.register("t", BASE)
    got = bt.from_arrow(BASE).group_by("g").agg(m=bt.col("v").mean()).collect()
    assert_same(got, duck.sql("SELECT g, AVG(v) AS m FROM t GROUP BY g"))


def test_fused_int_avg_matches_duckdb(duck):
    """Several aggregates over one scan — the fused path, which has its own accumulator."""
    duck.register("t", BASE)
    got = (
        bt.from_arrow(BASE)
        .group_by("g")
        .agg(m=bt.col("v").mean(), mw=bt.col("w").mean(), s=bt.col("w").sum())
        .collect()
    )
    assert_same(
        got,
        duck.sql("SELECT g, AVG(v) AS m, AVG(w) AS mw, SUM(w) AS s FROM t GROUP BY g"),
    )


def test_global_int_avg_matches_duckdb(duck):
    """No ``GROUP BY`` — the whole-column reduction, which reads no group ids at all."""
    duck.register("t", BASE)
    got = bt.from_arrow(BASE).agg(m=bt.col("v").mean()).collect()
    assert_same(got, duck.sql("SELECT AVG(v) AS m FROM t"))


def test_int_avg_agrees_across_execution_paths():
    """The same mean whether the accumulator is folded once, spilled, or streamed.

    Enough rows to morselize, so ``partial`` runs many times and ``combine`` folds the
    128-bit states rather than the single-partial identity shortcut — which is where a state
    type that disagreed between the scan and the merge would first show.
    """
    n = 200_000
    table = pa.table(
        {
            "g": pa.array([f"g{i % 7}" for i in range(n)]),
            # Alternating ±2^62 with a small residue: every partial sum is near the f64
            # precision cliff, and the exact mean is a small number an error cannot hide in.
            "v": pa.array(
                [(_BIG if i % 2 == 0 else -_BIG) + (i % 5) for i in range(n)], pa.int64()
            ),
        }
    )
    build = lambda ds: ds.group_by("g").agg(m=bt.col("v").mean())  # noqa: E731
    oracle = build(bt.from_arrow(table)).collect()
    assert_tables_equal(build(bt.from_arrow(table)).collect(spill=True), oracle)
    assert_tables_equal(
        pa.Table.from_batches(list(build(bt.from_arrow(table)).iter_batches())), oracle
    )
