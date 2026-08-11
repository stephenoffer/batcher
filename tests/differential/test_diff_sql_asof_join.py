"""SQL ``ASOF JOIN`` must match the nearest right row, not every row satisfying the ON.

sqlglot carries ``ASOF`` in a `Join`'s ``method`` slot — the slot the translator read only
for ``NATURAL`` — so an ``ASOF JOIN`` fell through to the ordinary ON path and ran as a
theta join. ``t ASOF JOIN u ON t.id >= u.id`` returned **21** rows where DuckDB returns 8,
each left row carrying every right row below it instead of the single nearest one.

That is a wrong row multiset rather than an error, so these assertions are what catch it:
`assert_same` is order-independent but not count-blind. `tests/differential/test_diff_asof.py`
covers the same operator through `Dataset.join_asof`, and passed throughout — the defect was
only ever on the SQL spelling, which nothing exercised.

The inner/left distinction is pinned separately because it is the half a witness column
implements: SQL ``ASOF JOIN`` drops a left row that matched nothing, while
``ASOF LEFT JOIN`` keeps it null-extended, and the plan node is left-style either way.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

pytestmark = pytest.mark.differential


def _left() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5, 6, 7, 8], pa.int64()),
            "g": pa.array(["a", "b", "a", "c", "b", "a", "x", "c"]),
        }
    )


def _right() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 9, 10], pa.int64()),
            "v": pa.array([100, 200, 300, 400, 500], pa.int64()),
            "g": pa.array(["a", "b", "a", "d", "e"]),
        }
    )


def _both(duck) -> dict[str, bt.Dataset]:
    duck.register("t", _left())
    duck.register("u", _right())
    return {"t": bt.from_arrow(_left()), "u": bt.from_arrow(_right())}


@pytest.mark.parametrize(
    "query",
    [
        # Backward (>=): the largest right key at or below the left one. The shape that
        # returned the full inequality join.
        "SELECT t.id, u.v FROM t ASOF JOIN u ON t.id >= u.id",
        # Forward (<=).
        "SELECT t.id, u.v FROM t ASOF JOIN u ON t.id <= u.id",
        # An equality conjunct is an exact-match `by` key, not a second nearest-match key.
        "SELECT t.id, u.v FROM t ASOF JOIN u ON t.g = u.g AND t.id >= u.id",
        # ASOF LEFT keeps the rows that matched nothing; with a `by` key several do.
        "SELECT t.id, u.v FROM t ASOF LEFT JOIN u ON t.g = u.g AND t.id >= u.id",
        "SELECT t.id, u.v FROM t ASOF LEFT JOIN u ON t.id >= u.id",
        # The inequality written right-side-first is the same join mirrored, so the
        # direction must be re-derived from the oriented pair rather than the operator.
        "SELECT t.id, u.v FROM t ASOF JOIN u ON u.id <= t.id",
    ],
)
def test_asof_matches_duckdb(duck, query):
    assert_same(bt.sql(query, **_both(duck)).collect(), duck.sql(query))


def test_inner_asof_drops_unmatched_left_rows(duck):
    """The witness column, isolated: inner keeps 5 of 8 left rows, ASOF LEFT keeps all 8."""
    tables = _both(duck)
    on = "ON t.g = u.g AND t.id >= u.id"
    inner = bt.sql(f"SELECT t.id, u.v FROM t ASOF JOIN u {on}", **tables).collect()
    outer = bt.sql(f"SELECT t.id, u.v FROM t ASOF LEFT JOIN u {on}", **tables).collect()
    assert inner.num_rows == 5
    assert outer.num_rows == 8
    # The witness must not leak into the output.
    assert inner.column_names == ["id", "v"]


def test_both_sides_keys_survive_the_join(duck):
    """SQL's ON form keeps both keys; `join_asof` coalesces them, so they are shadowed.

    Without that, `SELECT t.id, u.id` over an ASOF join died with
    ``projection 'id_1' references unknown column(s) ['u__id']`` — the right key had been
    merged away before the projection could read it. Found only by combining ASOF with a
    qualified reference to both keys, which no single-feature test does.
    """
    tables = _both(duck)
    for query in (
        "SELECT t.id, u.id, u.v FROM t ASOF JOIN u ON t.id >= u.id",
        "SELECT t.id, u.id, u.v FROM t ASOF LEFT JOIN u ON t.id >= u.id",
        # A `by` key is coalesced the same way and must survive too.
        "SELECT t.id, t.g, u.id, u.g, u.v FROM t ASOF JOIN u ON t.g = u.g AND t.id >= u.id",
    ):
        got = bt.sql(query, **tables).collect()
        # DuckDB's Arrow result repeats the name; its pandas conversion disambiguates the
        # way Batcher must, so it is the oracle for the names.
        expected = duck.sql(query).to_arrow_table()
        expected = expected.rename_columns(list(duck.sql(query).df().columns))
        assert got.column_names == expected.column_names, query
        assert_tables_equal(got, expected)


def test_asof_never_emits_more_rows_than_the_left_side(duck):
    """The property the theta-join lowering broke: ASOF matches at most one right row."""
    tables = _both(duck)
    for query in (
        "SELECT t.id FROM t ASOF JOIN u ON t.id >= u.id",
        "SELECT t.id FROM t ASOF LEFT JOIN u ON t.id >= u.id",
    ):
        assert bt.sql(query, **tables).collect().num_rows <= _left().num_rows


def test_strict_inequality_is_rejected_not_approximated(duck):
    """`>` has no inclusive-node representation, so it must raise rather than answer."""
    tables = _both(duck)
    with pytest.raises(Exception, match="strict inequality"):
        bt.sql("SELECT t.id, u.v FROM t ASOF JOIN u ON t.id > u.id", **tables).collect()


def test_asof_without_an_inequality_is_rejected(duck):
    tables = _both(duck)
    with pytest.raises(Exception, match="nearest-match inequality"):
        bt.sql("SELECT t.id, u.v FROM t ASOF JOIN u ON t.g = u.g", **tables).collect()
