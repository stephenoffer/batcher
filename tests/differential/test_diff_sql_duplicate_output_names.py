"""Two SELECT items sharing an output name must both survive, as they do in DuckDB.

The projection map was keyed on the output name, so the second of two items sharing one
silently overwrote the first and the query returned *fewer columns than it selected*:

    SELECT t.id, u.id FROM t JOIN u ON t.id = u.id   -- DuckDB: 2 columns. Batcher: 1.
    SELECT t.*, u.*   FROM t JOIN u ON t.id = u.id   -- DuckDB: 4 columns. Batcher: 3.
    SELECT sum(v) AS s, sum(w) AS s FROM t           -- DuckDB: 2 columns. Batcher: 1, and
                                                     -- the value was sum(w), not sum(v).

Silent column loss, on a shape (`a.k, b.k` across a join) that is ordinary SQL rather than
a corner. `tests/differential/test_diff_derived_table_column_collision.py` documented it as
a known separate defect and deliberately aliased around it; this is the file that closes it.

A `Dataset` is name-keyed and rejects duplicates by design
(`tests/unit/test_api_hunt_dup_columns.py` pins that for the DataFrame API), so the SQL
frontend disambiguates instead of collapsing:
the second `id` becomes `id_1`. That is the same name DuckDB produces whenever it has to
make result names unique, so the *names* are asserted here and not only the values — an
order-independent value comparison cannot see a column that is missing from both sides.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential


def _oracle(duck, query: str) -> pa.Table:
    """DuckDB's result, under the unique column names DuckDB itself assigns it.

    An Arrow table may carry two columns of the same name and DuckDB's does; a `Dataset`
    may not. `assert_same` compares name *sets*, so it cannot be the oracle here — it
    would reject `['id', 'id_1']` against `['id', 'id']` while both hold the same data,
    and it also could not tell a missing column from a duplicated name. DuckDB's own
    pandas conversion resolves the duplicates to `id`, `id_1`, which is exactly the
    convention Batcher adopts, so those names are applied to the Arrow result and the
    comparison is positional from there.
    """
    return duck.sql(query).to_arrow_table().rename_columns(list(duck.sql(query).df().columns))


def _t() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "g": pa.array(["a", "a", "b"]),
            "v": pa.array([10, 20, 30], pa.int64()),
            "w": pa.array([4, 5, 6], pa.int64()),
        }
    )


def _u() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "x": pa.array([7, 8, 9], pa.int64()),
        }
    )


def _both(duck) -> dict[str, bt.Dataset]:
    duck.register("t", _t())
    duck.register("u", _u())
    return {"t": bt.from_arrow(_t()), "u": bt.from_arrow(_u())}


# Each query selects an output name twice. `duck.sql(...).df()` is DuckDB's own
# unique-naming of the same result, which is where the `_1` convention comes from.
_QUERIES = [
    "SELECT t.id, u.id FROM t JOIN u ON t.id = u.id",
    "SELECT t.id, u.id, t.v FROM t JOIN u ON t.id = u.id",
    "SELECT t.*, u.* FROM t JOIN u ON t.id = u.id",
    "SELECT id AS a, v AS a FROM t",
    "SELECT v, v FROM t",
    "SELECT g, g FROM t GROUP BY g",
    "SELECT sum(v) AS s, sum(w) AS s FROM t",
    "SELECT g, count(*), count(*) FROM t GROUP BY g",
    "SELECT g, sum(v) AS n, count(*) AS n FROM t GROUP BY g",
    "SELECT t.id, u.id FROM t LEFT JOIN u ON t.id = u.id ORDER BY 1",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_duplicate_output_names_keep_every_column(duck, query):
    tables = _both(duck)
    got = bt.sql(query, **tables).collect()
    expected = _oracle(duck, query)
    assert got.column_names == expected.column_names, query
    assert_tables_equal(got, expected)


def test_the_first_of_two_colliding_aggregates_is_not_overwritten(duck):
    """The value half of the bug: the surviving column used to hold the *second* item."""
    tables = _both(duck)
    got = bt.sql("SELECT sum(v) AS s, sum(w) AS s FROM t", **tables).collect().to_pydict()
    assert got == {"s": [60], "s_1": [15]}


def test_star_across_a_join_keeps_both_key_columns(duck):
    tables = _both(duck)
    got = bt.sql("SELECT t.*, u.* FROM t JOIN u ON t.id = u.id", **tables).collect()
    assert got.column_names == ["id", "g", "v", "w", "id_1", "x"]


def test_suffix_skips_a_name_the_query_already_uses(duck):
    """`id_1` is taken by the user, so the collision must land on `id_2`."""
    tables = _both(duck)
    query = "SELECT t.id, u.id AS id_1, t.v AS id FROM t JOIN u ON t.id = u.id"
    got = bt.sql(query, **tables).collect()
    assert got.column_names == list(duck.sql(query).df().columns)
    assert sorted(got.column_names) == ["id", "id_1", "id_2"]
