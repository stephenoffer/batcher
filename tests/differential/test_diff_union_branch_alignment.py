"""A union whose branches disagree on a column's type returns what DuckDB returns.

`kyber.rules.extra.setops.align_union_branch_types` casts each branch column to the union's
declared type inside the branch (a `TRY_CAST`, as the engine's own union coercion is), so a
constant there -- TPC-DS q5's `CAST(0 AS DECIMAL(7,2))` beside a `double` -- is cast once
instead of per row at the union. Held to DuckDB over the type pairs a union meets, NULLs,
a distinct union, an aggregate over the union, and with the rule's positive control: the
rewrite actually fires on these plans.
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_N = 3_000

_T = pa.table(
    {
        "k": pa.array([i % 7 for i in range(_N)], pa.int64()),
        "d": pa.array([None if i % 11 == 0 else i * 0.25 for i in range(_N)], pa.float64()),
        "i": pa.array([None if i % 13 == 0 else i for i in range(_N)], pa.int64()),
        "m": pa.array(
            [None if i % 17 == 0 else decimal.Decimal(i) / 100 for i in range(_N)],
            pa.decimal128(9, 2),
        ),
        "dt": pa.array([10_000 + i % 400 for i in range(_N)], pa.date32()),
        "ts": pa.array(
            [(10_000 + i % 400) * 86_400_000_000 for i in range(_N)], pa.timestamp("us")
        ),
    }
)

_QUERIES = [
    # q5's shape: a double column beside a decimal constant, both ways round, then summed.
    "SELECT k, sum(x) AS sx, sum(y) AS sy FROM ("
    "SELECT k, d AS x, CAST(0 AS DECIMAL(7,2)) AS y FROM t "
    "UNION ALL SELECT k, CAST(0 AS DECIMAL(7,2)), d FROM t) u GROUP BY k",
    # int beside double.
    "SELECT x FROM (SELECT i AS x FROM t UNION ALL SELECT d FROM t) u",
    # Decimals of differing scale.
    "SELECT x FROM (SELECT m AS x FROM t UNION ALL SELECT CAST(i AS DECIMAL(12,4)) FROM t) u",
    # date beside timestamp.
    "SELECT x FROM (SELECT dt AS x FROM t UNION ALL SELECT ts FROM t) u",
    # A NULL literal branch.
    "SELECT x, y FROM (SELECT d AS x, k AS y FROM t UNION ALL SELECT NULL, NULL FROM t) u",
    # A distinct union across mismatched types.
    "SELECT x FROM (SELECT i AS x FROM t UNION SELECT d FROM t) u",
]


def _session():
    s = bt.Session()
    s.register("t", _T)
    return s


@pytest.mark.parametrize("query", _QUERIES)
def test_a_mismatched_union_matches_duckdb(duck, query):
    duck.register("t", _T)
    assert_same_for_query(_session().sql(query).collect(), duck.sql(query), query)


def test_the_alignment_fires_on_a_mismatched_union():
    """The positive control: the rule rewrites the q5 shape, so the queries above test it."""
    from batcher.kyber.rules.extra import setops

    fired: list[bool] = []
    plan = _session().sql(_QUERIES[0])._plan
    unions = [n for n in _walk(plan) if type(n).__name__ == "Union"]
    assert unions, "the query has no union to align"
    for u in unions:
        fired.append(setops.align_union_branch_types(u, None) is not None)
    assert any(fired), "the alignment did not rewrite a mismatched union"


def _walk(node):
    from batcher.plan.visitor import walk

    return list(walk(node))
