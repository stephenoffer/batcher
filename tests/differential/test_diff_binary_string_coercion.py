"""Binary-typed string columns behave like VARCHAR vs DuckDB.

Some Parquet/Arrow sources (e.g. ClickBench's ``hits``) deliver string columns as
``Binary`` (a ``BYTE_ARRAY`` with no UTF-8 logical annotation). Batcher coerces a
``Binary`` column against a ``Utf8`` string literal (and into string functions) so the
query runs with the same result DuckDB gives its VARCHAR column, rather than erroring on
the physical type. These cases pin that equivalence.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _t(duck):
    # A Binary-typed "string" column, the shape that used to error on comparison.
    t = pa.table(
        {
            "s": pa.array([b"apple", b"", b"banana", b"apple", None], type=pa.binary()),
            "n": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
        }
    )
    duck.register("t", t)
    return bt.from_arrow(t)


def test_binary_neq_empty_string(duck):
    from conftest import assert_same

    ds = _t(duck).filter(col("s") != "")
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE s <> ''"))


def test_binary_eq_literal(duck):
    from conftest import assert_same

    ds = _t(duck).filter(col("s") == "apple")
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE s = 'apple'"))


def test_binary_group_by_and_count(duck):
    from conftest import assert_same

    ds = _t(duck).filter(col("s") != "").group_by("s").agg(c=col("n").count())
    assert_same(
        ds.collect(),
        duck.sql("SELECT s, COUNT(n) AS c FROM t WHERE s <> '' GROUP BY s"),
    )
