"""``SELECT *`` over an ``ON``-form join must list the columns in relation order.

SQL's ``ON`` form keeps both sides' key columns, so the translator copies each key to a
shadow column that the join cannot coalesce and restores the names afterwards. Restoring
*appends*, so the right side's key came out last instead of in its own relation's
position:

    SELECT * FROM (VALUES (1),(2)) v(a) JOIN o ON o.id = v.a
    -- DuckDB: a, id, g      Batcher, before the fix: a, g, id

The values were right and every column was present, which is why an order-independent
value comparison never saw it — and why the assertions here are on `column_names`.
Column order is part of what ``SELECT *`` promises: it is what a positional consumer
(`to_pandas().values`, a CSV writer, an unpacking loop) reads.

The ``USING`` and ``NATURAL`` forms merge their key into one column and never took this
path; they are here so the fix cannot quietly change them.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _o() -> pa.Table:
    return pa.table({"id": pa.array([1, 2, 3], pa.int64()), "g": pa.array(["a", "b", "c"])})


def _u() -> pa.Table:
    return pa.table({"k": pa.array([1, 2, 9], pa.int64()), "w": pa.array([7, 8, 9], pa.int64())})


def _s() -> pa.Table:
    """A third relation sharing `o`'s key name, for the merged-key (USING/NATURAL) forms."""
    return pa.table({"id": pa.array([1, 2, 9], pa.int64()), "z": pa.array([5, 6, 7], pa.int64())})


@pytest.fixture
def tables(duck):
    duck.register("o", _o())
    duck.register("u", _u())
    duck.register("s", _s())
    return {"o": bt.from_arrow(_o()), "u": bt.from_arrow(_u()), "s": bt.from_arrow(_s())}


@pytest.mark.parametrize(
    "query",
    [
        # Differently-named keys, both orders — the shape that reordered.
        "SELECT * FROM o JOIN u ON o.id = u.k",
        "SELECT * FROM u JOIN o ON u.k = o.id",
        "SELECT * FROM o LEFT JOIN u ON o.id = u.k",
        "SELECT * FROM o RIGHT JOIN u ON o.id = u.k",
        "SELECT * FROM o FULL JOIN u ON o.id = u.k",
        # An inline relation on the left, which is how the divergence surfaced.
        "SELECT * FROM (VALUES (1),(2)) v(a) JOIN o ON o.id = v.a",
        "SELECT * FROM o JOIN (VALUES (1),(2)) v(a) ON o.id = v.a",
        # A comma join with the predicate in WHERE takes a different path.
        "SELECT * FROM (VALUES (1),(2)) v(a), o WHERE o.id = v.a",
        # Merged-key forms, which never took the shadow-column path. (A *self*-join is
        # deliberately not used: two references to one table are disambiguated to
        # `o__g`/`o2__g`, a separate pre-existing naming difference from DuckDB's
        # `g`/`g_1` that has nothing to do with column order.)
        "SELECT * FROM o JOIN s USING (id)",
        "SELECT * FROM o NATURAL JOIN s",
        # A residual alongside the equi-key.
        "SELECT * FROM o JOIN u ON o.id = u.k AND u.w > 0",
    ],
)
def test_star_column_order_matches_duckdb(tables, duck, query):
    got = bt.sql(query, **tables).collect()
    assert got.column_names == list(duck.sql(query).df().columns), query
    assert_same(got, duck.sql(query))


def test_both_key_columns_survive_in_their_own_positions(tables):
    got = bt.sql("SELECT * FROM o JOIN u ON o.id = u.k", **tables).collect()
    assert got.column_names == ["id", "g", "k", "w"]


def test_an_outer_join_still_null_extends_the_right_key(tables):
    """The property the shadow columns exist for; the reorder must not disturb it."""
    got = bt.sql("SELECT * FROM o LEFT JOIN u ON o.id = u.k", **tables).collect().to_pydict()
    by_id = dict(zip(got["id"], got["k"], strict=True))
    assert by_id[3] is None  # `o.id = 3` matched nothing, so `u.k` is NULL, not 3
