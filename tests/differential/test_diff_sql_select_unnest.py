"""``SELECT unnest(xs)`` — an unnest written in the SELECT list rather than the FROM.

DuckDB's shorthand for ``FROM t, UNNEST(t.xs) AS u(x)``, and the spelling most SQL in the
wild uses. It reached the scalar path as an unhandled node and raised ``unsupported SQL
expression: Explode``, so only the FROM-clause form worked.

It expands the relation the projection reads, which fixes where it has to run: after
``WHERE`` (which SQL evaluates first, on the un-expanded rows) and before the projection,
so an expression *around* the unnest — ``unnest(xs) * 2`` — is evaluated per element.
Both of those orderings are covered below, because getting either backwards changes the
answer rather than raising.

Several unnests in one SELECT list *zip* in DuckDB rather than multiplying. `explode`
cannot express that, so it is declined; the last test pins the decline, since answering
with a cross product would be a silent wrong answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            # An empty list contributes no rows, which is the non-outer default.
            "xs": pa.array([[10, 20], [30], []], pa.list_(pa.int64())),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, unnest(xs) AS v FROM t",
        # Un-aliased: the output takes the expression's name, `unnest(xs)`.
        "SELECT unnest(xs) FROM t",
        # Over a literal list, with no table at all.
        "SELECT unnest([1, 2, 3]) AS n",
        # An expression around the unnest is evaluated per element.
        "SELECT id, unnest(xs) * 2 AS v FROM t",
        # WHERE runs before the expansion, on the un-expanded rows.
        "SELECT id, unnest(xs) AS v FROM t WHERE id < 3",
        # The expanded rows feed the rest of the query normally.
        "SELECT sum(v) AS s FROM (SELECT unnest(xs) AS v FROM t)",
        "SELECT id, unnest(xs) AS v FROM t ORDER BY v DESC",
        "SELECT count(*) AS n FROM (SELECT unnest(xs) AS v FROM t)",
    ],
)
def test_select_unnest_matches_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_unaliased_unnest_takes_duckdbs_column_name(tables, duck):
    query = "SELECT unnest(xs) FROM t"
    assert bt.sql(query, **tables).collect().column_names == list(duck.sql(query).df().columns)


def test_an_empty_list_contributes_no_rows(tables):
    """The default is non-outer, so `id = 3` (an empty list) must not appear."""
    got = bt.sql("SELECT id, unnest(xs) AS v FROM t", **tables).collect().to_pydict()
    assert sorted(got["id"]) == [1, 1, 2]


def test_two_unnests_in_one_select_are_declined(tables):
    """SQL zips them; a cross product would be a silent wrong answer."""
    with pytest.raises(NotImplementedError, match="UNNEST calls in one SELECT list"):
        bt.sql("SELECT unnest(xs), unnest(xs) FROM t", **tables).collect()
