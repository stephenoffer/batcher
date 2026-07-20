"""SQL front-end depth: correlated/scalar subqueries, comma joins, aggregates — vs DuckDB.

Regression coverage for defects found by the wave-2 SQL-depth bug hunt:

* a correlated subquery whose inner aliases the *same base table* as an unaliased
  outer (``FROM emp`` outer, ``FROM emp e2`` inner referencing ``emp.dept``) — the
  outer reference was misclassified as local, so the correlation was dropped;
* a scalar subquery returning **zero rows** must be NULL, not an error;
* a comma / cross join of two tables that **share a column name**
  (``emp e, dept d`` both with ``dept``) — the equi-condition in WHERE degenerated
  to ``dept = dept`` and produced a cartesian product;
* a correlated ``IN`` whose subquery **aggregates**
  (``sal IN (SELECT max(sal) … WHERE e2.dept = e.dept)``) leaked an internal error.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "name": ["al", "bo", "cy", "di", "ed", "fi", None],
            "dept": [10, 10, 20, 20, 30, None, 30],
            "sal": [100, 200, 150, None, 300, 50, 300],
            "mgr": [None, 1, 1, 2, 2, 3, 3],
        }
    )
    dept = pa.table(
        {
            "dept": [10, 20, 30, 40],
            "dname": ["eng", "sales", "hr", "ops"],
            "budget": [1000, 2000, None, 500],
        }
    )
    duck.register("emp", emp)
    duck.register("dept", dept)
    return emp, dept


def _check(duck, tables, q):
    emp, dept = tables
    assert_same(bt.sql(q, emp=emp, dept=dept).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        # Correlated scalar/agg subquery whose inner aliases the same base table as
        # the unaliased outer — `emp.dept` is an OUTER reference, not a local one.
        "SELECT id, (SELECT count(*) FROM emp e2 WHERE e2.dept = emp.dept) c FROM emp",
        "SELECT id, (SELECT avg(sal) FROM emp e2 WHERE e2.dept = emp.dept) a FROM emp",
        "SELECT id FROM emp WHERE sal = (SELECT max(sal) FROM emp e2 WHERE e2.dept = emp.dept)",
    ],
)
def test_correlated_same_base_table_unaliased_outer(duck, tables, q):
    _check(duck, tables, q)


@pytest.mark.parametrize(
    "q",
    [
        # 0-row scalar subquery → NULL (per row), not an error.
        "SELECT id, (SELECT sal FROM emp WHERE id = 999) s FROM emp",
        "SELECT id, coalesce((SELECT sal FROM emp WHERE id = 999), -1) s FROM emp",
    ],
)
def test_scalar_subquery_zero_rows_is_null(duck, tables, q):
    _check(duck, tables, q)


@pytest.mark.parametrize(
    "q",
    [
        # Comma / cross join of two DIFFERENT tables sharing a column name (`dept`).
        # The equi-condition in WHERE must actually join, not degenerate to a
        # cartesian product.
        "SELECT e.id, d.dname FROM emp e, dept d WHERE e.dept = d.dept",
        "SELECT e.id, d.dname FROM emp e, dept d WHERE e.dept = d.dept AND d.budget > 800",
        "SELECT e.id, d.dept, d.budget FROM emp e, dept d WHERE e.dept = d.dept",
        "SELECT e.id FROM emp e, dept d, dept d2 WHERE e.dept = d.dept AND d.dept = d2.dept",
        # Explicit CROSS JOIN with the shared name then a qualified filter.
        "SELECT e.id FROM emp e CROSS JOIN dept d WHERE e.dept = d.dept AND d.budget IS NOT NULL",
    ],
)
def test_comma_join_shared_column_name(duck, tables, q):
    _check(duck, tables, q)


@pytest.mark.parametrize(
    "q",
    [
        # Explicit JOIN paths sharing a column name must keep working (no regression).
        "SELECT e.dept AS ed, d.dept AS dd FROM emp e JOIN dept d ON e.dept = d.dept",
        "SELECT dept, id, dname FROM emp JOIN dept USING (dept)",
        "SELECT dept, id, dname FROM emp LEFT JOIN dept USING (dept)",
        "SELECT id, dname FROM emp NATURAL JOIN dept",
    ],
)
def test_shared_column_explicit_joins_unregressed(duck, tables, q):
    _check(duck, tables, q)


@pytest.mark.parametrize(
    "q",
    [
        # Correlated IN whose subquery aggregates → per-key aggregate semi-join.
        "SELECT id FROM emp e WHERE sal IN (SELECT max(sal) FROM emp e2 WHERE e2.dept = e.dept)",
        "SELECT id FROM emp e WHERE sal IN "
        "(SELECT min(sal) FROM emp e2 WHERE e2.dept = e.dept AND e2.sal IS NOT NULL)",
    ],
)
def test_correlated_in_with_aggregate(duck, tables, q):
    _check(duck, tables, q)


@pytest.mark.parametrize(
    "q",
    [
        # MIN/MAX(DISTINCT) == MIN/MAX (dedup is a no-op for the extrema).
        "SELECT min(DISTINCT sal) m, max(DISTINCT dept) d FROM emp",
        "SELECT dept, min(DISTINCT sal) m, max(DISTINCT mgr) x FROM emp GROUP BY dept",
    ],
)
def test_min_max_distinct(duck, tables, q):
    _check(duck, tables, q)


def test_sum_distinct(duck, tables):
    # SUM(DISTINCT) is now supported: it is SUM over rows deduped on the group keys plus
    # the aggregated expression, so the dedup happens once up front. This test previously
    # asserted a clean *error*; the limitation it pinned is gone, so it now pins the
    # result instead. Full coverage lives in `test_diff_distinct_agg.py`.
    #
    # Aliased explicitly because Batcher's auto-generated alias lowercases the DISTINCT
    # keyword (`sum(distinct sal)`) where DuckDB preserves it — a pre-existing naming
    # difference that `count(DISTINCT ...)` shares, unrelated to the aggregate's value.
    _check(duck, tables, "SELECT sum(DISTINCT sal) AS s FROM emp")


def test_sum_distinct_mixed_with_plain_agg(duck, tables):
    # A plain aggregate alongside a DISTINCT one is now supported via a two-level
    # aggregate (level 1 groups by the keys plus the distinct expression, deduping it
    # implicitly while pre-aggregating the plain one; level 2 combines the partials).
    # Full coverage lives in `test_diff_distinct_agg.py`.
    _check(duck, tables, "SELECT sum(DISTINCT sal) AS s, count(sal) AS c FROM emp")


def test_sum_distinct_with_non_mergeable_agg_clean_error(tables):
    # `avg` has no single-column mergeable partial, so it cannot be pre-aggregated
    # alongside the dedup. That must stay an explicit error, never a plausible wrong number.
    emp, dept = tables
    with pytest.raises(NotImplementedError, match="mergeable partial"):
        bt.sql("SELECT sum(DISTINCT sal) s, avg(sal) a FROM emp", emp=emp, dept=dept).collect()
