"""`UNNEST` in a FROM clause must return DuckDB's *relation*, not just its values.

An unnest in the FROM clause **adds** a column; it does not consume the list. DuckDB's
`SELECT * FROM t, UNNEST(arr)` returns `id, arr (still the list), unnest (the element)`, and
naming the element with `AS u(x)` changes only that last name.

Batcher returned `id, arr (the element)` for the unaliased form — one column fewer, with
`arr` holding a different type. Nothing caught it: every existing test named the columns it
wanted instead of starring, so the shape was never compared. These tests star deliberately.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table():
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "arr": pa.array([[10, 20], [30], []], type=pa.list_(pa.int64())),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t, UNNEST(arr)",
        "SELECT * FROM t CROSS JOIN UNNEST(arr)",
        "SELECT * FROM t, UNNEST(arr) AS u(elem)",
        "SELECT * FROM t LEFT JOIN UNNEST(arr) ON TRUE",
        "SELECT * FROM t LEFT JOIN UNNEST(arr) AS u(elem) ON TRUE",
    ],
    ids=["comma", "cross", "aliased", "outer", "outer_aliased"],
)
def test_the_unnested_relation_has_duckdbs_shape(duck, sql):
    table = _table()
    duck.register("t", table)
    got = bt.sql(sql, t=table)
    want = duck.sql(sql)
    assert list(got.columns) == list(want.columns), f"{sql}\n{got.columns} != {want.columns}"
    assert_same(got.collect(), want)


def test_the_list_column_survives_the_unnest(duck):
    """The property the old behaviour lost: `arr` is still the list, beside the element."""
    table = _table()
    sql = "SELECT id, arr, unnest FROM t, UNNEST(arr)"
    duck.register("t", table)
    got = bt.sql(sql, t=table).sort("id", "unnest").to_pydict()
    assert got["arr"] == [[10, 20], [10, 20], [30]], "the list repeats once per element"
    assert got["unnest"] == [10, 20, 30]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_an_element_name_that_collides_is_refused_by_name():
    """`unnest` as an existing column name would otherwise silently shadow or duplicate."""
    table = pa.table(
        {
            "unnest": pa.array([1], type=pa.int64()),
            "arr": pa.array([[1, 2]], type=pa.list_(pa.int64())),
        }
    )
    with pytest.raises(Exception, match="collides"):
        bt.sql("SELECT * FROM t, UNNEST(arr)", t=table).collect()
    # Naming it explicitly is the documented way out, and it works.
    out = bt.sql("SELECT * FROM t, UNNEST(arr) AS u(e)", t=table).to_pydict()
    assert out["e"] == [1, 2]
