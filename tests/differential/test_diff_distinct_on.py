"""SQL ``DISTINCT ON`` differential tests vs DuckDB.

``SELECT DISTINCT ON (keys) ... ORDER BY ...`` keeps one row per key set — the first
row in ORDER BY order — then orders the survivors by the same ORDER BY (Postgres/DuckDB
semantics). This was previously rejected with ``NotImplementedError``; every case here
returned nothing (raised) before the translator learned to lower it to a
``row_number()`` per-key window + filter.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered


@pytest.fixture
def emp():
    t = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "dept": ["a", "a", "b", "b", "c", None, "a"],
            "sal": [100, 200, 150, 300, 250, None, 120],
            "bonus": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        # Single key: the highest-sal row per dept, incl. the NULL-dept group.
        "SELECT DISTINCT ON (dept) id, dept, sal FROM emp ORDER BY dept, sal DESC",
        # Ascending pick — the lowest-sal row per dept.
        "SELECT DISTINCT ON (dept) id, dept, sal FROM emp ORDER BY dept, sal",
        # NULLS FIRST must both group the NULL dept and place it first in the output.
        "SELECT DISTINCT ON (dept) id, dept, sal FROM emp ORDER BY dept NULLS FIRST, sal",
        # Multiple DISTINCT ON keys.
        "SELECT DISTINCT ON (dept, sal) id, dept, sal FROM emp ORDER BY dept, sal, id",
        # ORDER BY column absent from the SELECT list still drives the row choice.
        "SELECT DISTINCT ON (dept) id FROM emp ORDER BY dept, sal DESC",
        # A non-key, non-order column follows the ORDER-BY-chosen row (proves the
        # right row is kept, not an arbitrary one).
        "SELECT DISTINCT ON (dept) dept, bonus FROM emp ORDER BY dept, sal DESC",
        # An expression key.
        "SELECT DISTINCT ON (sal % 100) id, sal FROM emp ORDER BY sal % 100, id",
        # Interaction ordering: WHERE -> DISTINCT ON -> ORDER BY -> LIMIT.
        "SELECT DISTINCT ON (dept) id, dept, sal FROM emp "
        "WHERE sal IS NOT NULL ORDER BY dept, sal DESC LIMIT 2",
        # SELECT * must project the base columns only (no internal dedup temporaries).
        "SELECT DISTINCT ON (dept) * FROM emp ORDER BY dept, sal DESC",
    ],
)
def test_distinct_on_ordered(duck, emp, q):
    duck.register("emp", emp)
    got = bt.sql(q, emp=emp).collect()
    assert_same_ordered(got, duck.sql(q))


@pytest.mark.differential
def test_distinct_on_no_order_is_one_row_per_key(duck, emp):
    # With no ORDER BY the kept row is arbitrary (both engines agree only on the key
    # set and the group count), so compare just the distinct keys.
    duck.register("emp", emp)
    q = "SELECT DISTINCT ON (dept) dept FROM emp"
    got = bt.sql(q, emp=emp).collect()
    assert_same(got, duck.sql(q))


@pytest.mark.differential
def test_distinct_on_rejects_group_by(emp):
    # DISTINCT ON combined with aggregation is not supported — it must reject cleanly,
    # never silently drop the wrong rows.
    with pytest.raises(NotImplementedError):
        bt.sql("SELECT DISTINCT ON (dept) dept, count(*) FROM emp GROUP BY dept", emp=emp).collect()
