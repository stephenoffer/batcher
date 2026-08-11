"""`SELECT DISTINCT col AS alias … ORDER BY col`, checked against DuckDB.

With `DISTINCT` the sort has to run *after* the projection: deduping is a hash operation and
does not preserve order, so sorting first would silently discard the `ORDER BY`. Correct —
but it means the sort sees the projection's *output* names, and this query names the input
one. SQL resolves it perfectly well, because `f` does appear in the SELECT list; the relation
simply no longer has a column called `f` by then, so it failed with "sort key references
unknown column(s)".

Every neighbouring form worked, which is what kept the gap invisible: without `DISTINCT`,
without the alias, or ordering by the alias, all resolved. Only the combination broke.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential


def _table():
    return pa.table(
        {
            "f": pa.array([3.0, 1.0, 2.0, 1.0, None]),
            "g": pa.array(["a", "b", "a", "b", "c"]),
            "n": pa.array([1, 2, 3, 2, 5], type=pa.int64()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT DISTINCT f AS r FROM t ORDER BY f",
        "SELECT DISTINCT f AS r FROM t ORDER BY f DESC",
        "SELECT DISTINCT f AS r FROM t ORDER BY f NULLS FIRST",
        "SELECT DISTINCT f AS r FROM t ORDER BY r",
        "SELECT DISTINCT f AS r, g AS h FROM t ORDER BY g, f",
        "SELECT DISTINCT f AS r, g AS h FROM t ORDER BY h DESC, r",
        "SELECT DISTINCT g AS h FROM t ORDER BY g",
        "SELECT DISTINCT n AS k FROM t ORDER BY n DESC",
        # The forms that already worked, so the fix cannot have broken them.
        "SELECT DISTINCT f FROM t ORDER BY f",
        "SELECT f AS r FROM t ORDER BY f",
        "SELECT DISTINCT f AS r FROM t ORDER BY 1",
    ],
    ids=[
        "asc",
        "desc",
        "nulls_first",
        "by_alias",
        "two_keys",
        "mixed_alias",
        "string",
        "int_desc",
        "no_alias",
        "no_distinct",
        "positional",
    ],
)
def test_ordering_by_the_source_column_matches_duckdb(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same_ordered(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_computed_projection_is_still_refused():
    """`ORDER BY f` beside `SELECT DISTINCT f * 2` is genuinely ill-defined.

    Two different `f` can dedup to one `f * 2`, so there is no single `f` to sort that row
    by. DuckDB is lenient here; Batcher refuses, and the refusal is the point — retargeting
    only ever rewrites a *bare rename*, where the two names denote the same values.
    """
    with pytest.raises(Exception, match="unknown column"):
        bt.sql("SELECT DISTINCT f * 2 AS r FROM t ORDER BY f", t=_table()).collect()


def test_the_alias_is_not_confused_with_a_different_source_column(duck):
    """A projection that *swaps* two names must not retarget one onto the other."""
    table = _table()
    sql = "SELECT DISTINCT g AS f, f AS g FROM t ORDER BY f"
    duck.register("t", table)
    assert_same_ordered(bt.sql(sql, t=table).collect(), duck.sql(sql))
