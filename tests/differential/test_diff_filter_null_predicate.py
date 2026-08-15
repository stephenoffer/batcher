"""Differential tests for WHERE / filter three-valued-logic on NULL predicates.

A SQL filter keeps a row only when the predicate is TRUE; a NULL predicate drops
the row (three-valued logic). These guard a fixed zone-map pruning defect: over a
column whose min/max prove an inner predicate empty, ``NOT (that predicate)`` was
folded to *always-true* and the whole filter dropped — which wrongly KEPT the NULL-
predicate rows the filter must drop. See docs/architecture/internals/bug_hunt_ledger.md.

Root cause: `kyber/rules/zonemap_pruning.py::_predicate_status` negated its tri-state
with a plain `not inner`, turning an "always-empty" (`_FALSE`, which conflates FALSE
and NULL rows) into "always-true" (`_TRUE`). A null row negates to null (still
dropped), so `Not(_FALSE)` is NOT provably always-true. The negation is sound only in
the `_TRUE -> _FALSE` direction, which `_not` now enforces.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def test_not_between_out_of_range_keeps_null_dropped(duck):
    """`i NOT BETWEEN -2 AND 0` over a column whose only non-null value is out of range.

    ``i BETWEEN -2 AND 0`` is provably empty from min/max ([-3,-3] ∩ [-2,0] = ∅), but the
    NULL row makes ``NOT BETWEEN`` NULL, not TRUE — so the filter must drop it. The plan
    must not fold the filter away.
    """
    t = pa.table({"i": pa.array([None, -3], pa.int64()), "row": pa.array([0, 1], pa.int64())})
    out = bt.from_arrow(t).filter(~bt.col("i").between(-2, 0)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT BETWEEN -2 AND 0"))


def test_not_in_out_of_range_keeps_null_dropped(duck):
    """`i NOT IN (1, 2)` where the only non-null value (-3) is outside the list, plus a NULL."""
    t = pa.table({"i": pa.array([None, -3], pa.int64()), "row": pa.array([0, 1], pa.int64())})
    out = bt.from_arrow(t).filter(~bt.col("i").is_in([1, 2])).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT IN (1, 2)"))


def test_not_between_with_nulls_and_passing_rows(duck):
    """Mixed nulls and out-of-range values: only the genuinely-passing rows survive."""
    t = pa.table(
        {
            "i": pa.array([None, None, -3, 5], pa.int64()),
            "row": pa.array([0, 1, 2, 3], pa.int64()),
        }
    )
    out = bt.from_arrow(t).filter(~bt.col("i").between(-2, 0)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT BETWEEN -2 AND 0"))


def test_not_between_always_true_no_nulls_still_prunes(duck):
    """The sound direction is preserved: with no nulls, `NOT BETWEEN` a range covering all
    rows is genuinely always-false → empty result (matches DuckDB)."""
    t = pa.table({"i": pa.array([-3, -2], pa.int64()), "row": pa.array([0, 1], pa.int64())})
    out = bt.from_arrow(t).filter(~bt.col("i").between(-100, 100)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i NOT BETWEEN -100 AND 100"))


def test_between_out_of_range_prune_still_correct(duck):
    """The always-empty pruning direction is unaffected: `i BETWEEN 50 AND 60` over [-3,-2]
    is empty, and the NULL-safe path must still prune it (no rows, no NULL kept)."""
    t = pa.table(
        {"i": pa.array([None, -3, -2], pa.int64()), "row": pa.array([0, 1, 2], pa.int64())}
    )
    out = bt.from_arrow(t).filter(bt.col("i").between(50, 60)).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT * FROM t WHERE i BETWEEN 50 AND 60"))


# --- all-null columns -------------------------------------------------------------
#
# The other end of the null-count range. A column whose EXACT null count equals the
# relation's row count decides both spellings outright: `IS NULL` keeps every row and
# `IS NOT NULL` keeps none. Both are plan rewrites that change which rows come back — one
# deletes the whole relation, the other deletes the filter — so both are held against
# DuckDB rather than against a plan shape alone.


def _all_null_parquet(tmp_path):
    """A Parquet file (so the null count is an EXACT footer figure) with an all-null column."""
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "note": pa.array([None, None, None], pa.string()),
        }
    )
    path = str(tmp_path / "an.parquet")
    pq.write_table(table, path)
    return path, table


def test_is_not_null_over_all_null_column(duck, tmp_path):
    """Provably empty — and the rewrite must not disagree with executing the filter."""
    path, table = _all_null_parquet(tmp_path)
    duck.register("an1", table)
    out = bt.read.parquet(path).filter(bt.col("note").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM an1 WHERE note IS NOT NULL"))


def test_is_null_over_all_null_column(duck, tmp_path):
    """Provably always-true — dropping the filter must keep exactly the rows DuckDB keeps."""
    path, table = _all_null_parquet(tmp_path)
    duck.register("an2", table)
    out = bt.read.parquet(path).filter(bt.col("note").is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM an2 WHERE note IS NULL"))


def test_all_null_column_in_a_conjunction(duck, tmp_path):
    """The decided conjunct must compose, not short-circuit the sibling it sits beside."""
    path, table = _all_null_parquet(tmp_path)
    duck.register("an3", table)
    out = bt.read.parquet(path).filter(bt.col("note").is_null() & (bt.col("id") > 1)).collect()
    assert_same(out, duck.sql("SELECT * FROM an3 WHERE note IS NULL AND id > 1"))


def test_left_join_over_an_all_null_key_keeps_the_preserved_side(duck, tmp_path):
    """The all-null decision reaches a join side, and must not cost the preserved rows.

    `filter_null_join_keys` pushes `IS NOT NULL` onto a left join's non-preserved side, and
    that filter is now provably empty when the side's key is all null — so the right input
    is dropped entirely. The result must still be every left row padded with nulls, which
    is a claim about rows and belongs here rather than in a plan-shape assertion.
    """
    import pyarrow.parquet as pq

    fact = pa.table({"k": pa.array([1, 2, 2, 3], pa.int64()), "v": pa.array([10, 20, 30, 40])})
    dim = pa.table({"k": pa.array([None, None, None], pa.int64()), "w": pa.array([5, 6, 7])})
    fp, dp = str(tmp_path / "f.parquet"), str(tmp_path / "d.parquet")
    pq.write_table(fact, fp)
    pq.write_table(dim, dp)
    duck.register("jf", fact)
    duck.register("jd", dim)

    out = bt.read.parquet(fp).join(bt.read.parquet(dp), on="k", how="left").collect()
    assert_same(out, duck.sql("SELECT f.k, f.v, d.w FROM jf f LEFT JOIN jd d ON f.k = d.k"))
