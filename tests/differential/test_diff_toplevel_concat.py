"""`bt.concat` against DuckDB's ``UNION ALL`` / ``UNION`` / positional join.

`concat` builds no new IR — it composes `union`, a null-filling projection, and a
positional full join — so the oracle here is checking that the *composition* means
what the SQL spelling of it means, including the null padding a diagonal or ragged
horizontal concatenation introduces.
"""

from __future__ import annotations

import datetime
import decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential

_A = {"k": [1, 2, 3], "v": ["a", "b", "c"]}
_B = {"k": [3, 4], "v": ["c", "d"]}


def _register(duck, name, data):
    duck.register(name, bt.from_pydict(data).to_arrow())


def test_concat_vertical_matches_union_all(duck):
    _register(duck, "a", _A)
    _register(duck, "b", _B)
    got = bt.concat([bt.from_pydict(_A), bt.from_pydict(_B)])
    assert_same(got.to_arrow(), duck.sql("SELECT * FROM a UNION ALL SELECT * FROM b"))


def test_concat_vertical_relaxed_matches_union(duck):
    _register(duck, "a", _A)
    _register(duck, "b", _B)
    got = bt.concat([bt.from_pydict(_A), bt.from_pydict(_B)], how="vertical_relaxed")
    assert_same(got.to_arrow(), duck.sql("SELECT * FROM a UNION SELECT * FROM b"))


def test_concat_of_three_datasets_matches_a_chained_union_all(duck):
    _register(duck, "a", _A)
    _register(duck, "b", _B)
    got = bt.concat([bt.from_pydict(_A), bt.from_pydict(_B), bt.from_pydict(_A)])
    assert_same(
        got.to_arrow(),
        duck.sql("SELECT * FROM a UNION ALL SELECT * FROM b UNION ALL SELECT * FROM a"),
    )


def test_concat_diagonal_matches_a_null_padded_union_all(duck):
    left = {"x": [1, 2]}
    right = {"y": ["p", "q"]}
    duck.register("l", bt.from_pydict(left).to_arrow())
    duck.register("r", bt.from_pydict(right).to_arrow())
    got = bt.concat([bt.from_pydict(left), bt.from_pydict(right)], how="diagonal")
    expected = duck.sql(
        "SELECT x, NULL::VARCHAR AS y FROM l UNION ALL SELECT NULL::BIGINT AS x, y FROM r"
    )
    assert_same(got.to_arrow(), expected)


def test_concat_horizontal_matches_a_positional_full_join(duck):
    left = {"a": [1, 2, 3]}
    right = {"b": ["p"]}
    duck.register("l", bt.from_pydict(left).to_arrow())
    duck.register("r", bt.from_pydict(right).to_arrow())
    got = bt.concat([bt.from_pydict(left), bt.from_pydict(right)], how="horizontal")
    expected = duck.sql(
        """
        SELECT a, b FROM (
            SELECT row_number() OVER () - 1 AS i, a FROM l
        ) li FULL JOIN (
            SELECT row_number() OVER () - 1 AS i, b FROM r
        ) ri USING (i)
        ORDER BY i
        """
    )
    assert_same_ordered(got.to_arrow(), expected)


def test_concat_str_matches_duckdb_concat(duck):
    data = {"a": ["x", "y", None], "b": ["1", None, "2"]}
    duck.register("t", bt.from_pydict(data).to_arrow())
    got = bt.from_pydict(data).select(c=bt.concat_str(bt.col("a"), bt.col("b")))
    assert_same(got.to_arrow(), duck.sql("SELECT concat(a, b) AS c FROM t"))


# A diagonal concat pads each input's missing columns with nulls of the column's own type.
# A decimal and a time-zone-aware timestamp used to be refused outright, and a
# millisecond timestamp came back retyped to microseconds.
_PADDED_TYPES = [
    (pa.decimal128(12, 2), [decimal.Decimal("1.25")], "DECIMAL(12, 2)"),
    (pa.timestamp("us", "UTC"), [datetime.datetime(2020, 1, 1)], "TIMESTAMPTZ"),
    (pa.timestamp("ms"), [datetime.datetime(2020, 1, 1)], "TIMESTAMP_MS"),
    (pa.date64(), [datetime.date(2020, 1, 1)], "DATE"),
]


@pytest.mark.parametrize(("dtype", "values", "duck_type"), _PADDED_TYPES, ids=lambda v: str(v)[:24])
def test_concat_diagonal_pads_with_a_null_of_the_exact_type(duck, dtype, values, duck_type):
    duck.execute("SET TimeZone = 'UTC'")
    left = pa.table({"k": pa.array([1], pa.int64()), "c": pa.array(values, dtype)})
    right = pa.table({"k": pa.array([2], pa.int64())})
    duck.register("l", left)
    duck.register("r", right)
    got = bt.concat([bt.from_arrow(left), bt.from_arrow(right)], how="diagonal").to_arrow()
    assert got.schema.field("c").type == dtype
    expected = duck.sql(f"SELECT k, c FROM l UNION ALL SELECT k, NULL::{duck_type} AS c FROM r")
    assert_same(got, expected)
