"""Correlated subqueries whose row count per key is decided by a clause other than WHERE.

Decorrelation pulls the correlation out of a subquery's WHERE and joins what is left on the
correlation key. Four families of clause do not survive that move unchanged, and every case
below returned a wrong answer, not an error, before `subquery.shape` and
`subquery.scalar_sub` existed:

- ``ORDER BY … LIMIT`` / ``OFFSET`` was applied to the whole inner table, so a per-key
  "latest value" kept one row for the entire table.
- ``EXISTS`` over an ungrouped aggregate is always TRUE (one row per outer row), and a
  ``HAVING`` decides it per key; both were answered as "does an inner row match".
- The empty group of an unmatched key is not NULL for ``count(*) + 1`` or
  ``coalesce(sum(w), 0)``; only a bare ``count`` was patched.
- More than one row per key must raise, as DuckDB does, rather than duplicate outer rows.

It also covers the error messages that used to name a sqlglot node instead of the SQL
construct (``EXISTS`` in a SELECT list, ``HAVING count(*) IN (…)``, ``STRING_AGG … OVER``).
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_for_query
from batcher._internal.errors import ExecutionError

pytestmark = pytest.mark.differential

_T = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5, 6, None], pa.int64()),
        "g": pa.array(["a", "a", "b", "b", "b", "c", None]),
        "v": pa.array([10, 20, None, 40, 50, 60, 70], pa.int64()),
    }
)
_U = pa.table(
    {
        "k": pa.array([1, 2, 2, None, 8], pa.int64()),
        "g": pa.array(["a", "a", "b", "b", None]),
        "w": pa.array([100, 200, None, 300, 5], pa.int64()),
    }
)
_EMPTY = pa.table({"k": pa.array([], pa.int64()), "w": pa.array([], pa.int64())})


def _session() -> bt.Session:
    s = bt.Session()
    s.register("t", _T)
    s.register("u", _U)
    s.register("e", _EMPTY)
    return s


def _duck(con) -> duckdb.DuckDBPyConnection:
    con.register("t", _T)
    con.register("u", _U)
    con.register("e", _EMPTY)
    return con


def _check(duck, query: str) -> None:
    got = _session().sql(query).collect()
    assert_same_for_query(got, _duck(duck).sql(query), query)


# --- 1. ORDER BY / LIMIT / OFFSET are per key --------------------------------------------

_PAGED = [
    # The audit's two headline shapes: DuckDB returns one row per group, Batcher returned 1.
    "SELECT id FROM t t1 WHERE v = (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v DESC LIMIT 1)",
    "SELECT id, (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v DESC LIMIT 1) AS m FROM t t1",
    "SELECT id, (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v LIMIT 1 OFFSET 1) AS m FROM t t1",
    "SELECT id, (SELECT v AS x FROM t t2 WHERE t2.g = t1.g ORDER BY x DESC LIMIT 1) m FROM t t1",
    "SELECT id, (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY 1 DESC LIMIT 1) m FROM t t1",
    # EXISTS: LIMIT n >= 1 changes nothing per key, LIMIT 0 makes it FALSE everywhere.
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g AND u.w > 150 LIMIT 1)",
    "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.g = t.g LIMIT 1)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g LIMIT 0)",
    "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.g = t.g LIMIT 0)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g ORDER BY w LIMIT 5)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g OFFSET 1)",
    "SELECT id FROM t WHERE v > 45 OR EXISTS (SELECT 1 FROM u WHERE u.g = t.g LIMIT 1)",
    # IN: the probed set is each key's top-N.
    "SELECT id FROM t t1 WHERE v IN (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v LIMIT 1)",
    "SELECT id FROM t t1 WHERE v IN (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v DESC LIMIT 2)",
    "SELECT id FROM t t1 WHERE v NOT IN (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v LIMIT 1)",
    "SELECT id FROM t t1 WHERE v IN (SELECT max(v) FROM t t2 WHERE t2.g = t1.g LIMIT 1)",
]


@pytest.mark.parametrize("query", _PAGED)
def test_paging_inside_a_correlated_subquery_is_per_key(duck, query):
    """A LIMIT/OFFSET/ORDER BY inside the subquery ranks rows within one correlation key."""
    _check(duck, query)


# --- 2. EXISTS over aggregates and HAVING -------------------------------------------------

_EXISTS_AGG = [
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING count(*) > 5)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING count(*) > 1)",
    "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING count(*) > 1)",
    # Over the empty group of an unmatched key `count(*) = 0` holds, so those rows qualify.
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING count(*) = 0)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING sum(w) > 150)",
    # An ungrouped aggregate is one row per outer row, matched or not: always TRUE.
    "SELECT id FROM t WHERE EXISTS (SELECT count(*) FROM u WHERE u.g = t.g)",
    "SELECT id FROM t WHERE NOT EXISTS (SELECT max(w) FROM u WHERE u.g = t.g)",
    "SELECT id FROM t WHERE EXISTS (SELECT count(*) FROM u WHERE u.g = t.g OFFSET 1)",
    # Grouped: a key qualifies when one of its groups passes HAVING.
    "SELECT id FROM t WHERE EXISTS "
    "(SELECT k FROM u WHERE u.g = t.g GROUP BY k HAVING count(*) > 1)",
    "SELECT id FROM t WHERE EXISTS (SELECT k FROM u WHERE u.g = t.g GROUP BY k)",
    "SELECT id FROM t WHERE v < 25 OR EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING count(*) > 1)",
]


@pytest.mark.parametrize("query", _EXISTS_AGG)
def test_exists_over_an_aggregate_matches_duckdb(duck, query):
    """EXISTS asks whether the subquery yields a row, which an aggregate always does."""
    _check(duck, query)


# --- 3. The empty group has a value --------------------------------------------------------

_EMPTY_GROUP = [
    "SELECT id, (SELECT count(*) + 1 FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT coalesce(sum(w), 0) FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT count(w) FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT count(*) FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT count(*) * 2.5 FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT coalesce(avg(w), 0) FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT CASE WHEN count(*) = 0 THEN 'none' ELSE 'some' END FROM u "
    "WHERE u.k = t.id) AS m FROM t",
    "SELECT id FROM t WHERE (SELECT count(*) + 0 FROM u WHERE u.k = t.id) = 0",
    "SELECT id, (SELECT count(*) FROM u WHERE u.k = t.id HAVING count(*) > 1) AS m FROM t",
    "SELECT id, (SELECT count(*) FROM u WHERE u.k = t.id HAVING count(*) = 0) AS m FROM t",
    "SELECT id, (SELECT max(w) FROM u WHERE u.k = t.id LIMIT 0) AS m FROM t",
]


@pytest.mark.parametrize("query", _EMPTY_GROUP)
def test_unmatched_key_sees_the_aggregate_over_no_rows(duck, query):
    """An outer row with no match reads the aggregate evaluated over an empty group."""
    _check(duck, query)


# --- 4. More than one row is an error, not duplicated outer rows ---------------------------

_MULTI_ROW = [
    "SELECT id, (SELECT w FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id FROM t WHERE v < (SELECT w FROM u WHERE u.k = t.id)",
    # Two identical rows are still two rows.
    "SELECT id, (SELECT g FROM u WHERE u.k = t.id) AS m FROM t",
    "SELECT id, (SELECT k FROM u) AS m FROM t",
    "SELECT id FROM t WHERE v > (SELECT w FROM u WHERE w IS NOT NULL)",
    "SELECT id, (SELECT v FROM t t2 WHERE t2.g = t1.g ORDER BY v LIMIT 2) AS m FROM t t1",
]


@pytest.mark.parametrize("query", _MULTI_ROW)
def test_scalar_subquery_returning_several_rows_raises(duck, query):
    """DuckDB raises "More than one row returned"; so does Batcher, with the same words."""
    with pytest.raises(duckdb.Error, match="More than one row"):
        _duck(duck).sql(query).fetchall()
    with pytest.raises(ExecutionError, match="More than one row returned by a subquery"):
        _session().sql(query).collect()


_ONE_ROW = [
    # The positive control for the check above: the same shapes over one row per key.
    "SELECT id, (SELECT w FROM u WHERE u.k = t.id AND u.w > 150) AS m FROM t",
    "SELECT id, (SELECT k FROM u WHERE w = 300) AS m FROM t",
    "SELECT id, (SELECT k FROM u WHERE w = 999) AS m FROM t",
    "SELECT id FROM t WHERE v < (SELECT min(w) FROM u)",
    "SELECT id, (SELECT max(w) FROM e) AS m FROM t",
    "SELECT id FROM t WHERE v > (SELECT avg(v) FROM t)",
    "SELECT id, v - (SELECT min(v) FROM t) AS d FROM t ORDER BY id NULLS LAST",
]


@pytest.mark.parametrize("query", _ONE_ROW)
def test_scalar_subquery_with_at_most_one_row_matches_duckdb(duck, query):
    """The cardinality check does not fire where each key has at most one row."""
    _check(duck, query)


# --- 9. Constructs that used to report a sqlglot node --------------------------------------

_NOW_SUPPORTED = [
    "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.k = t.id) AS f FROM t",
    "SELECT id, NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.id) AS f FROM t",
    "SELECT id, EXISTS (SELECT 1 FROM e) AS f FROM t",
    "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.g = t.g HAVING count(*) > 1) AS f FROM t",
    "SELECT g FROM t GROUP BY g HAVING count(*) IN (SELECT k FROM u)",
    "SELECT g FROM t GROUP BY g HAVING count(*) NOT IN (SELECT k FROM u WHERE k IS NOT NULL)",
]


@pytest.mark.parametrize("query", _NOW_SUPPORTED)
def test_exists_in_select_and_having_in_match_duckdb(duck, query):
    """EXISTS read as a value, and an aggregate probed against a subquery set."""
    _check(duck, query)


def test_string_agg_window_names_the_function():
    """A window STRING_AGG is refused by its SQL name, not sqlglot's `groupconcat`."""
    query = "SELECT id, string_agg(g, ',') OVER (ORDER BY id) AS l FROM t"
    with pytest.raises(NotImplementedError, match="STRING_AGG is not supported as a window"):
        _session().sql(query)


def test_exists_in_an_aggregating_select_list_says_why():
    """Refused with the reason and the rewrite, not "unsupported SQL expression: Exists"."""
    query = "SELECT g, EXISTS (SELECT 1 FROM u WHERE u.g = t.g) FROM t GROUP BY g"
    with pytest.raises(NotImplementedError, match="aggregating query"):
        _session().sql(query)


# --- NOT IN keeps its three-valued answer, uncorrelated and correlated ---------------------

_NOT_IN = [
    "SELECT id FROM t WHERE id NOT IN (SELECT k FROM u)",
    "SELECT id FROM t WHERE id NOT IN (SELECT k FROM u WHERE k IS NOT NULL)",
    "SELECT id FROM t WHERE id NOT IN (SELECT k FROM e)",
    "SELECT id FROM t WHERE id NOT IN (SELECT id FROM t WHERE id > 3)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM e)",
    "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM e)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE w > 250)",
    "SELECT id FROM t t1 WHERE v NOT IN (SELECT v FROM t t2 WHERE t2.g = t1.g AND t2.v > 15)",
    "SELECT id FROM t t1 WHERE id NOT IN (SELECT k FROM u WHERE u.g = t1.g)",
]


@pytest.mark.parametrize("query", _NOT_IN)
def test_uncorrelated_set_predicates_match_duckdb(duck, query):
    """NOT IN / EXISTS set predicates, including a correlated NOT IN over NULLs."""
    _check(duck, query)


# --- 6. A comma join beside a correlated scalar subquery keeps its join key ----------------

_COMMA_JOIN_WITH_SCALAR = [
    # TPC-H q17's shape: the join equality and the subquery comparison share one WHERE.
    "SELECT sum(t.v) AS s FROM t, u WHERE t.id = u.k AND u.g = 'a' "
    "AND t.v < (SELECT 2 * avg(w) FROM u u2 WHERE u2.k = t.id)",
    "SELECT t.id, u.w FROM t, u WHERE t.id = u.k "
    "AND u.w > (SELECT min(v) FROM t t2 WHERE t2.g = t.g)",
    # An empty group reads the aggregate over no rows, which is what forces the LEFT join.
    "SELECT t.id FROM t, u WHERE t.id = u.k AND (SELECT count(*) FROM u u2 WHERE u2.g = t.g) < 3",
    # Both sources have a `g`, so the join renames `t.g`; the correlated reference must follow.
    "SELECT t.id, u.w FROM t, u WHERE t.id = u.k AND u.w > (SELECT min(v) FROM t t2 "
    "WHERE t2.g = t.g) - 100",
    # ...unless the subquery rebinds the alias `t` itself, when `t.g` is its own column.
    "SELECT t.id FROM t, u WHERE t.id = u.k AND u.w > (SELECT min(v) FROM t WHERE t.g = 'b')",
]


@pytest.mark.parametrize("query", _COMMA_JOIN_WITH_SCALAR)
def test_comma_join_beside_a_scalar_subquery_is_an_equi_join(duck, query):
    """The comma join's equality becomes a join key rather than a filter over a cross join.

    Decorrelation projects the subquery's value through a `CASE`, and the filter above that
    projection used to be refused whole, so the equality never reached `derive_join_keys` and
    the relation was a cartesian product -- TPC-H q17 at sf1 went past 34 GB. At these sizes a
    cartesian product still returns the right rows, so the plan is checked as well as the answer.
    """
    _check(duck, query)
    assert "__cross_key" not in _session().sql(query).explain()
