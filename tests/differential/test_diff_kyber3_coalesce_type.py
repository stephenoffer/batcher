"""Differential guard: a COALESCE rewrite must not narrow the result *type*.

A ``COALESCE``'s type is the *join* of its arguments' types, so dropping an argument is
sound only when a surviving one already carries that type. ``coalesce_simplify`` used to
truncate everything after the first non-null literal checking only that the dropped tail
could not *error* — never that it did not carry the result type. So
``coalesce(5, CAST(-1 AS DOUBLE))`` (a DOUBLE that yields ``5.0``) was rewritten to the
INT literal ``5``: a silently wrong dtype *and*, once cast to text, a wrong value
(``'5'`` vs DuckDB's ``'5.0'``). The type-guarded truncation lives in
``coalesce_drop_nulls_after_first_non_null``; ``coalesce_simplify`` must leave it alone.

These cases fail (``'5'`` vs ``'5.0'``, or an int64 vs float64 column) before the fix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
duckdb = pytest.importorskip("batcher")
_duck = pytest.importorskip("duckdb")

from _harness import assert_same  # noqa: E402
from batcher import coalesce, col, lit  # noqa: E402


def test_coalesce_int_then_double_cast_keeps_double_value() -> None:
    """``CAST(coalesce(5, CAST(-1 AS DOUBLE)) AS VARCHAR)`` is ``'5.0'``, not ``'5'``.

    The string cast makes the type-narrowing observable as a *value* difference the
    type-tolerant multiset compare cannot mask (``'5'`` != ``'5.0'`` as text).
    """
    t = pa.table({"k": pa.array([0, 1], pa.int64())})
    ds = bt.from_arrow(t.to_batches()).with_columns(
        c=coalesce(lit(5), lit(-1).cast("float64")).cast("string")
    )
    con = _duck.connect()
    con.register("t", t)
    duck = con.sql("SELECT k, CAST(coalesce(5, CAST(-1 AS DOUBLE)) AS VARCHAR) AS c FROM t")
    assert_same(ds.collect(), duck)


def test_coalesce_int_literal_over_float_column_is_double() -> None:
    """``coalesce(5, f)`` over a DOUBLE column is a DOUBLE column of ``5.0`` — dtype pinned."""
    t = pa.table({"f": pa.array([1.5, None], pa.float64())})
    ds = bt.from_arrow(t.to_batches()).with_columns(c=coalesce(lit(5), col("f")))
    out = ds.collect()
    assert out.schema.field("c").type == pa.float64(), (
        f"coalesce(int, double) must stay DOUBLE, got {out.schema.field('c').type}"
    )
    con = _duck.connect()
    con.register("t", t)
    assert_same(out, con.sql("SELECT f, coalesce(5, f) AS c FROM t"))
