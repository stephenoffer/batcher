"""Differential tests for `EXISTS` combined with `OR`, against DuckDB.

An `EXISTS` under `AND` folds into a semi/anti join, which is strictly the better plan. An
`EXISTS` under `OR` cannot: the join would drop the very rows the `OR` is there to keep. The
translator used to refuse the whole shape with `NotImplementedError`, which is what made
TPC-DS q10 and q35 unplannable.

Spark's answer is an **ExistenceJoin** — a left join that emits, per outer row, a boolean
saying whether the subquery matched — after which the original boolean is evaluated over that
column like any other predicate. `subquery/core.py::_exists_marker` is that rewrite, built
from the primitives already in the module: the inner relation is reduced to its *distinct*
correlation keys, so the left join matches each outer row at most once and cannot multiply
rows, and the tag is null exactly where nothing matched.

`EXISTS` is the one subquery form with no three-valued subtlety — it is TRUE or FALSE, never
NULL — which is what makes a boolean marker exact rather than approximate. `IN` is the harder
case, because `x IN (…)` *is* NULL when `x` is NULL or the list holds one, and a plain boolean
marker cannot carry that; it was refused for exactly that reason and is now answered.
`test_in_under_or_matches_duckdb_including_the_null_rows` holds it to DuckDB on the rows that
decide it, `NOT IN` against a null-bearing list included.

The fixture keeps nulls in both the outer key and the inner key, so a rewrite that confused
"no match" with "null key" would show up here rather than in a benchmark.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# `ck` carries a null so a null outer key cannot be mistaken for a match; `ws.wk` carries one
# so a null on the *inner* side cannot silently satisfy an equality either. `cs` overlaps `ws`
# on some keys and not others, so `OR` between the two is not the same as either alone.
CUST = pa.table({"ck": [1, 2, 3, 4, 5, None], "nm": ["a", "b", "c", "d", "e", "f"]})
WS = pa.table({"wk": [1, 1, 3, None], "amt": [10, 20, 30, 40]})
CS = pa.table({"ck2": [2, 5], "amt": [7, 8]})


@pytest.fixture
def shop(duck):
    duck.register("cust", CUST)
    duck.register("ws", WS)
    duck.register("cs", CS)


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        # The TPC-DS q10 / q35 shape: two correlated EXISTS joined by OR.
        "SELECT ck FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.wk = c.ck)"
        " OR exists (SELECT 1 FROM cs WHERE cs.ck2 = c.ck)",
        # EXISTS OR an ordinary predicate — the marker has to coexist with a normal filter.
        "SELECT ck FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.wk = c.ck) OR c.ck = 4",
        # Negation, which inverts the marker rather than the join.
        "SELECT ck FROM cust c WHERE NOT exists (SELECT 1 FROM ws WHERE ws.wk = c.ck) OR c.ck = 1",
        "SELECT ck FROM cust c WHERE NOT exists (SELECT 1 FROM ws WHERE ws.wk = c.ck)"
        " OR NOT exists (SELECT 1 FROM cs WHERE cs.ck2 = c.ck)",
        # An OR nested under an AND: the conjunct still folds to a join, the disjunct does not.
        "SELECT ck FROM cust c WHERE c.nm <> 'zz'"
        " AND (exists (SELECT 1 FROM ws WHERE ws.wk = c.ck) OR c.ck = 5)",
        # A local predicate inside the subquery must be applied to the inner relation before
        # it is reduced to distinct keys, not to the outer one.
        "SELECT ck FROM cust c WHERE exists"
        " (SELECT 1 FROM ws WHERE ws.wk = c.ck AND ws.amt > 15) OR c.ck = 3",
        # Uncorrelated EXISTS under OR: a whole-relation emptiness test, so the marker is a
        # constant. Both polarities, since one keeps every row and the other keeps none.
        "SELECT ck FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.amt > 999) OR c.ck = 2",
        "SELECT ck FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.amt > 0) OR c.ck = 2",
        # The bare AND-shapes must keep taking the semi/anti join path unchanged.
        "SELECT ck FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.wk = c.ck)",
        "SELECT ck FROM cust c WHERE NOT exists (SELECT 1 FROM ws WHERE ws.wk = c.ck)",
        # Three EXISTS in one predicate, so the marker naming has to stay unique.
        "SELECT ck FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.wk = c.ck)"
        " OR exists (SELECT 1 FROM cs WHERE cs.ck2 = c.ck)"
        " OR exists (SELECT 1 FROM ws WHERE ws.wk = c.ck AND ws.amt > 25)",
    ],
)
def test_exists_under_or_matches_duckdb(duck, shop, query):
    assert_same(bt.sql(query, cust=CUST, ws=WS, cs=CS).collect(), duck.sql(query))


@pytest.mark.differential
def test_the_marker_column_does_not_leak_into_the_result(duck, shop):
    """The synthetic existence column is an implementation detail, not an output column."""
    out = bt.sql(
        "SELECT * FROM cust c WHERE exists (SELECT 1 FROM ws WHERE ws.wk = c.ck) OR c.ck = 4",
        cust=CUST,
        ws=WS,
    ).collect()
    assert set(out.column_names) == {"ck", "nm"}, out.column_names


@pytest.mark.differential
@pytest.mark.parametrize(
    "predicate",
    [
        "c.ck IN (SELECT wk FROM ws) OR c.ck = 4",
        "c.ck NOT IN (SELECT wk FROM ws) OR c.ck = 4",
        "c.ck IN (SELECT wk FROM ws)",
    ],
)
def test_in_under_or_matches_duckdb_including_the_null_rows(duck, shop, predicate):
    """`IN` under `OR`, held against DuckDB on the shape that decides it: the nulls.

    This was a *refusal*, and the refusal was the right call at the time: `x IN (…)` is NULL
    when `x` is null or the list holds a null, so it cannot be carried by a boolean existence
    marker the way `EXISTS` can, and answering it with one would have been silently wrong on
    exactly these rows. Both fixtures carry a null (`cust.ck` and `ws.wk`), and `NOT IN` is
    the sharpest case of all — a null anywhere in the list makes it never true, so DuckDB
    keeps only `ck = 4`, the row the `OR` rescues.

    The engine now answers all three and agrees. Pinning the *answer* rather than the
    refusal is what keeps that true: a test asserting `NotImplementedError` passes whether
    the feature is absent or present-and-wrong, and stops testing anything the moment it
    lands.
    """
    query = f"SELECT ck FROM cust c WHERE {predicate}"
    assert_same(bt.sql(query, cust=CUST, ws=WS).collect(), duck.sql(query))
