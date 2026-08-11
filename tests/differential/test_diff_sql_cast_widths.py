"""SQL casts preserve the width the type name asks for, checked against DuckDB.

The SQL translator used to flatten all twelve integer widths onto ``int64``, so a
narrowing cast was a no-op. Two things followed, both silent:

``CAST(32768 AS TINYINT)`` returned 32768 rather than raising, and — the damaging one —
``TRY_CAST(32768 AS TINYINT)`` returned it too. ``TRY_CAST`` exists to yield NULL for
values that do not fit, which makes it the standard way to *filter* out-of-range input;
filtering nothing is worse than failing, because the row survives with a value the
declared type cannot hold.

The engine was never the problem. Its dtype registry already spells every SQL name at
its true width (``tinyint`` is int8, ``usmallint`` is uint16, ``decimal(10,2)`` is
decimal128), and the DataFrame path has always narrowed and range-checked correctly.
Only the translator's own lookup table stood in between, so the fix deleted it.

Both the *values* and the *result dtype* are asserted here. A value comparison alone
cannot see this bug: `assert_same` is type-tolerant by design, so a cast that returns
the right numbers in the wrong width passes it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    """Values straddling every signed and unsigned integer boundary."""
    return pa.table(
        {
            "i": pa.array(
                [0, 1, 127, 128, 255, 256, 32767, 32768, -128, -129, -1, None],
                type=pa.int64(),
            ),
            "f": pa.array([0.0, 1.0, 1.5, 2.5, -2.5, 1e18, -1e18, 0.1, 127.9, 1e9, -1e9, None]),
        }
    )


@pytest.mark.parametrize(
    "sql_type",
    ["TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"],
)
def test_try_cast_nulls_what_does_not_fit(duck, sql_type):
    """The whole point of TRY_CAST: out of range becomes NULL, not a surviving value."""
    table = _table()
    sql = f"SELECT TRY_CAST(i AS {sql_type}) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    ("sql_type", "arrow_type"),
    [
        ("TINYINT", pa.int8()),
        ("SMALLINT", pa.int16()),
        ("INTEGER", pa.int32()),
        ("BIGINT", pa.int64()),
        ("UTINYINT", pa.uint8()),
        ("USMALLINT", pa.uint16()),
        ("UINTEGER", pa.uint32()),
        ("UBIGINT", pa.uint64()),
    ],
)
def test_the_result_carries_the_width_that_was_asked_for(sql_type, arrow_type):
    """The assertion `assert_same` cannot make — it tolerates int width by design."""
    table = _table()
    got = bt.sql(f"SELECT TRY_CAST(i AS {sql_type}) AS r FROM t", t=table)
    assert got.schema.field("r").type == arrow_type


@pytest.mark.parametrize(
    ("sql_type", "too_big"),
    [("TINYINT", 128), ("SMALLINT", 32768), ("INTEGER", 2**31), ("UTINYINT", -1)],
)
def test_a_plain_cast_refuses_instead_of_returning_an_impossible_value(sql_type, too_big):
    """`CAST` is not `TRY_CAST`: a value that does not fit is an error, as in DuckDB.

    Asserted separately from the TRY_CAST cases because returning the out-of-range value
    would satisfy every one of them if the two spellings were ever collapsed.
    """
    table = pa.table({"i": pa.array([1, too_big], type=pa.int64())})
    with pytest.raises(Exception, match=r"(?i)cast"):
        bt.sql(f"SELECT CAST(i AS {sql_type}) AS r FROM t", t=table).collect()


def test_float_to_narrow_int_matches_duckdb_on_rounding_and_overflow(duck):
    """Two behaviours at once: how a fraction rounds, and what a 1e18 float does."""
    table = _table()
    sql = "SELECT TRY_CAST(f AS INTEGER) AS a, TRY_CAST(f AS SMALLINT) AS b FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_parametrized_decimal_stays_a_decimal():
    """`decimal(p,s)` resolves in the registry, so it must not degrade to float64."""
    table = pa.table({"f": pa.array([1.25, -3.5, None])})
    got = bt.sql("SELECT CAST(f AS DECIMAL(10, 2)) AS r FROM t", t=table)
    assert pa.types.is_decimal(got.schema.field("r").type)


@pytest.mark.parametrize("sql_type", ["DECIMAL", "NUMERIC"])
def test_an_unqualified_decimal_keeps_its_exactness(duck, sql_type):
    """The default dialect fills in DECIMAL's precision, so it reaches a real decimal.

    This used to be flattened to float64, which is the one thing a DECIMAL cast exists to
    avoid. sqlglot's duckdb dialect renders a bare ``DECIMAL`` as ``DECIMAL(18, 3)``, which
    the registry resolves, so Batcher and DuckDB now agree on the value *and* on its type.
    Other dialects emit the name unqualified and land on the float64 alias, because there
    is no single correct precision to invent for them.
    """
    table = pa.table({"f": pa.array([1.25, -3.5, None])})
    sql = f"SELECT CAST(f AS {sql_type}) AS r FROM t"
    duck.register("t", table)
    got = bt.sql(sql, t=table)
    assert pa.types.is_decimal(got.schema.field("r").type)
    assert_same(got.collect(), duck.sql(sql))


def test_an_unparametrized_varchar_and_a_sized_one_agree(duck):
    """`VARCHAR(10)` does not resolve verbatim; the head word has to carry it."""
    table = _table()
    sql = "SELECT CAST(i AS VARCHAR) AS a, CAST(i AS VARCHAR(10)) AS b FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_type_name_the_dialect_rewrites_still_reaches_its_engine_dtype(duck):
    """`UINTEGER` and `HUGEINT` never arrive spelled that way.

    sqlglot's duckdb dialect normalizes them to `UINT` and `INT128` before the translator
    sees them, and neither spelling is in the dtype registry. They used to land on the
    `string` fall-through, so `CAST(i AS UINTEGER)` returned `['1', '300']` — the wrong
    values in the wrong type, from a query that still returned rows.
    """
    table = pa.table({"i": pa.array([0, 1, 300, 70000], type=pa.int64())})
    duck.register("t", table)
    got = bt.sql("SELECT CAST(i AS UINTEGER) AS r FROM t", t=table)
    assert got.schema.field("r").type == pa.uint32()
    assert_same(got.collect(), duck.sql("SELECT CAST(i AS UINTEGER) AS r FROM t"))
    assert (
        bt.sql("SELECT CAST(i AS HUGEINT) AS r FROM t", t=table).schema.field("r").type
        == pa.int64()
    )


@pytest.mark.parametrize("sql_type", ["TIMESTAMP", "TIMESTAMPTZ", "DATETIME", "DATE", "TIME"])
def test_the_temporal_names_survive_the_dialect_rewrite(sql_type):
    """The regression the width fix nearly shipped, and the reason this file exists.

    Replacing the old longest-prefix lookup with an exact one quietly broke
    `CAST(x AS TIMESTAMP)` — the most common temporal cast there is — because sqlglot
    renders it `TIMESTAMPNTZ`, which the registry does not carry. The prefix match had
    been absorbing that, so nothing in the type name a user writes reveals it.
    """
    table = pa.table({"s": pa.array(["2024-01-05 10:30:00", "2024-02-06 00:00:00"])})
    got = bt.sql(f"SELECT TRY_CAST(s AS {sql_type}) AS r FROM t", t=table)
    assert not pa.types.is_string(got.schema.field("r").type), sql_type
    got.collect()


@pytest.mark.parametrize("sql_type", ["BIT", "ARRAY<INT>", "STRUCT<a INT>"])
def test_an_unsupported_type_name_says_so_rather_than_returning_text(sql_type):
    """The fall-through this replaced: an unknown type silently became a string cast."""
    table = pa.table({"i": pa.array([1, 2], type=pa.int64())})
    with pytest.raises(Exception, match=r"(?i)not supported|no dtype"):
        bt.sql(f"SELECT CAST(i AS {sql_type}) AS r FROM t", t=table).collect()


def test_narrowing_survives_a_partitioned_collect(duck):
    """The cast dtype is part of the schema, and a schema that differs per partition
    is how a distributed run produces a result the single-node run cannot."""
    table = _table()
    sql = "SELECT TRY_CAST(i AS SMALLINT) AS r FROM t"
    duck.register("t", table)
    ds = bt.sql(sql, t=table)
    one = ds.collect()
    many = ds.repartition(4).collect()
    assert one.schema.field("r").type == many.schema.field("r").type == pa.int16()
    assert_same(many, duck.sql(sql))
