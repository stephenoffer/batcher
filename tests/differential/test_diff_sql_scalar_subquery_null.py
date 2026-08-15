"""A scalar subquery whose single row is NULL is a NULL, not a crash.

`_scalar_subquery` collects the inner SELECT and substitutes its value as a literal.
It handled the *no rows* case (SQL says that is NULL, and `_typed_null` supplies one
of the right type), but not the case where the subquery returns exactly one row whose
value happens to be NULL. That is not an exotic shape: it is what every aggregate
returns over an empty or all-null column, so the canonical threshold query

    SELECT ... WHERE x > (SELECT AVG(x) FROM t)

crashed on empty input with ``TypeError: unsupported literal type: NoneType`` raised
from deep inside ``to_ir``, where DuckDB simply returns no rows. The IR has no untyped
null literal -- a bare SQL ``NULL`` lowers to ``NULLIF(1, 1)`` -- so `lit(None)` has no
wire form at all and the failure was a bare `TypeError` rather than a typed
`PlanError`.

The fix routes the NULL value through the same `_typed_null` the empty case already
used, so both spellings of "the subquery produced no value" agree.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

#: Inputs whose aggregates are NULL, alongside an ordinary one as the control.
_TABLES = {
    "all_null": pa.table({"i": pa.array([None, None], pa.int64()), "k": pa.array(["a", "b"])}),
    "empty": pa.table({"i": pa.array([], pa.int64()), "k": pa.array([], pa.string())}),
    "one_null": pa.table({"i": pa.array([None, 2, 3], pa.int64()), "k": pa.array(["a", "b", "c"])}),
    "normal": pa.table({"i": pa.array([1, 2, 3], pa.int64()), "k": pa.array(["a", "b", "c"])}),
}

#: Every shape that puts a possibly-NULL scalar subquery somewhere a literal must go.
_QUERIES = [
    "select k, i from t where i > (select avg(i) from t)",
    "select k, i from t where i = (select min(i) from t)",
    "select k, i from t where i <= (select max(i) from t where i > 100)",
    "select k, (select avg(i) from t) as m from t",
    "select k, (select sum(i) from t where k = 'zz') as s from t",
    "select k, i + (select max(i) from t where i > 100) as p from t",
]


@pytest.mark.parametrize("table", sorted(_TABLES))
@pytest.mark.parametrize("query", _QUERIES)
def test_scalar_subquery_null_matches_duckdb(duck, table, query):
    """The subquery's NULL flows into the enclosing expression as DuckDB's does."""
    tbl = _TABLES[table]
    got = bt.sql(query, t=bt.from_arrow(tbl)).collect()

    duck.register("t", tbl)
    assert_same(got, duck.sql(query))


def test_a_null_scalar_subquery_does_not_raise():
    """The regression proper: this raised `TypeError` from inside `to_ir`."""
    tbl = _TABLES["all_null"]
    got = bt.sql(
        "select k, i from t where i > (select avg(i) from t)", t=bt.from_arrow(tbl)
    ).to_pydict()
    assert got == {"k": [], "i": []}


def test_the_empty_and_null_subquery_cases_agree():
    """Zero rows and one NULL row are the same answer; only one of them used to work."""
    no_rows = bt.sql(
        "select (select i from t where k = 'zz') as v from t",
        t=bt.from_arrow(_TABLES["normal"]),
    ).to_pydict()
    one_null_row = bt.sql(
        "select (select avg(i) from t) as v from t", t=bt.from_arrow(_TABLES["all_null"])
    ).to_pydict()
    assert set(no_rows["v"]) == {None}
    assert set(one_null_row["v"]) == {None}
