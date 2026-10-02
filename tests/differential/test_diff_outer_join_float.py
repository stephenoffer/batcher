"""A left/semi/anti join floated above the inner joins that read only its kept side.

`kyber.rules.joins.outer_float` rewrites `(A LEFT JOIN B) JOIN C ON a = c` into
`(A JOIN C) LEFT JOIN B`. The cases below are the ones that rewrite could get wrong: NULL join
keys on either side, a fact row with several matching returns and one with none, a returns
column read through `coalesce` and `count` (which see the null extension), an inner key that
names the null-supplying side (which must not float), a filter on the null-extended column,
and the semi/anti forms.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_N = 400
_FACT = pa.table(
    {
        "f_item": pa.array([None if i % 37 == 0 else i % 23 for i in range(_N)], pa.int64()),
        "f_ticket": pa.array([i // 3 for i in range(_N)], pa.int64()),
        "f_date": pa.array([i % 50 for i in range(_N)], pa.int64()),
        "f_amt": pa.array([float(i % 11) + 0.5 for i in range(_N)], pa.float64()),
    }
)
_RETURNS = pa.table(
    {
        # Every third fact row has a return, some have two, and a few returns match nothing.
        "r_item": pa.array([None if i % 29 == 0 else (i * 3) % 23 for i in range(160)], pa.int64()),
        "r_ticket": pa.array([(i * 3) // 3 + (i % 2) for i in range(160)], pa.int64()),
        "r_amt": pa.array([float(i % 7) for i in range(160)], pa.float64()),
        "r_reason": pa.array([i % 5 for i in range(160)], pa.int64()),
    }
)
_DATES = pa.table(
    {
        "d_sk": pa.array(list(range(50)), pa.int64()),
        "d_year": pa.array([2000 + (i // 10) for i in range(50)], pa.int64()),
    }
)
_ITEMS = pa.table(
    {
        "i_sk": pa.array(list(range(23)), pa.int64()),
        "i_price": pa.array([float(i) for i in range(23)], pa.float64()),
    }
)
_REASONS = pa.table({"re_sk": pa.array([0, 1, 2, 3], pa.int64()), "re_name": ["a", "b", "c", "d"]})

_LEFT = (
    "FROM fact LEFT JOIN returns ON f_item = r_item AND f_ticket = r_ticket, dates, items "
    "WHERE f_date = d_sk AND f_item = i_sk AND d_year = 2001 AND i_price > 5"
)

_ANTI_THEN_DATE = (
    "SELECT d_year, f_item, sum(f_amt) AS s, count(*) AS n FROM fact "
    "LEFT JOIN returns ON f_item = r_item AND f_ticket = r_ticket "
    "JOIN dates ON f_date = d_sk WHERE r_ticket IS NULL AND d_year = 2001 GROUP BY d_year, f_item"
)

_QUERIES = [
    "SELECT i_sk, sum(f_amt) AS s, sum(coalesce(r_amt, 0)) AS r, count(r_amt) AS nr, "
    f"count(*) AS n {_LEFT} GROUP BY i_sk",
    f"SELECT f_ticket, f_item, r_amt {_LEFT}",
    # A filter on the null-extended side: keeps only the fact rows with no return.
    f"SELECT count(*) AS n, sum(f_amt) AS s {_LEFT} AND r_ticket IS NULL",
    # The inner key names the null-supplying side. A bare key rejects the padded rows, so
    # `outer_to_inner_under_join` makes the left join inner (see the test of that below).
    "SELECT count(*) AS n, count(re_name) AS nm FROM fact "
    "LEFT JOIN returns ON f_item = r_item AND f_ticket = r_ticket "
    "JOIN reasons ON r_reason = re_sk JOIN dates ON f_date = d_sk WHERE d_year = 2002",
    "SELECT count(*) AS n, sum(f_amt) AS s FROM fact, dates, items "
    "WHERE f_date = d_sk AND f_item = i_sk AND d_year = 2003 "
    "AND EXISTS (SELECT 1 FROM returns WHERE r_item = f_item AND r_ticket = f_ticket)",
    "SELECT count(*) AS n, sum(f_amt) AS s FROM fact, dates, items "
    "WHERE f_date = d_sk AND f_item = i_sk AND d_year = 2003 "
    "AND NOT EXISTS (SELECT 1 FROM returns WHERE r_item = f_item AND r_ticket = f_ticket)",
    # TPC-DS q78's channel subquery: the `IS NULL` makes the left join an anti join, which must
    # end up above the selective date join rather than be sunk back below it.
    _ANTI_THEN_DATE,
    # The null-side key again, coalesced: a padded row's NULL reason becomes 0, which `reasons`
    # holds, so the padded rows match and the left join can be neither strengthened nor floated.
    "SELECT count(*) AS n, count(re_name) AS nm FROM fact "
    "LEFT JOIN returns ON f_item = r_item AND f_ticket = r_ticket "
    "JOIN reasons ON coalesce(r_reason, 0) = re_sk JOIN dates ON f_date = d_sk WHERE d_year = 2002",
]

_TABLES = {
    "fact": _FACT,
    "returns": _RETURNS,
    "dates": _DATES,
    "items": _ITEMS,
    "reasons": _REASONS,
}


def _session() -> bt.Session:
    s = bt.Session()
    for name, table in _TABLES.items():
        s.register(name, table)
    return s


@pytest.mark.parametrize("query", _QUERIES)
def test_floated_outer_join_matches_duckdb(duck, query):
    for name, table in _TABLES.items():
        duck.register(name, table)
    got = _session().sql(query).collect()
    assert_same_for_query(got, duck.sql(query), query)


def _joins_under(ir: object, found: list[tuple[str, list[str]]]) -> None:
    """Every hash join in `ir`, pre-order, as `(join_type, child join types)`."""
    if isinstance(ir, dict):
        if ir.get("op") == "hash_join":
            below: list[str] = []
            for side in ("left", "right"):
                stack = [ir[side]]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        if node.get("op") == "hash_join":
                            below.append(node["join_type"])
                        stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
                    elif isinstance(node, list):
                        stack.extend(node)
            found.append((ir["join_type"], below))
        for v in ir.values():
            _joins_under(v, found)
    elif isinstance(ir, list):
        for v in ir:
            _joins_under(v, found)


def test_the_outer_join_really_floats():
    """The positive control: the left join ends up *above* the date and item joins."""
    from batcher.kyber import optimize

    ds = _session().sql(_QUERIES[0])
    joins: list[tuple[str, list[str]]] = []
    _joins_under(json.loads(optimize(ds._plan, sources=ds._sources).to_json()), joins)
    left = [below for kind, below in joins if kind == "left"]
    assert left, "the query has a left join"
    assert left[0].count("inner") == 2, f"expected both inner joins under the left join: {joins}"


def test_a_key_on_the_null_side_does_not_float():
    """The negative control: the `reasons` join reads `r_reason` and must stay above it.

    Keyed on `coalesce(r_reason, 0)`, so the padded rows can match: with a bare `r_reason` key
    the left join is strengthened to inner instead (the next test), and there is no left join
    left for this control to watch.
    """
    from batcher.kyber import optimize

    ds = _session().sql(_QUERIES[-1])
    joins: list[tuple[str, list[str]]] = []
    _joins_under(json.loads(optimize(ds._plan, sources=ds._sources).to_json()), joins)
    assert any(kind == "inner" and "left" in below for kind, below in joins), joins


def test_a_bare_key_on_the_null_side_makes_the_left_join_inner():
    """`reasons ON r_reason = re_sk` never matches a padded row, so no left join survives."""
    from batcher.kyber import optimize

    ds = _session().sql(_QUERIES[3])
    joins: list[tuple[str, list[str]]] = []
    _joins_under(json.loads(optimize(ds._plan, sources=ds._sources).to_json()), joins)
    assert joins and all(kind == "inner" for kind, _ in joins), joins


def test_an_anti_join_stays_above_a_more_selective_inner_join():
    """`push_semijoin_through_join` and the float used to undo each other; the cheaper order wins.

    Once a run has measured the date column, the date join is known to keep a fifth of the fact
    rows and the anti join most of them, so the anti join runs second. Before the two rules
    shared `semijoin_goes_first`, the sink ran last and put the anti join under the date join,
    over every fact row. Read from `explain()`: the plan the session runs, statistics and all.
    """
    ds = _session().sql(_ANTI_THEN_DATE)
    for _ in range(2):
        ds.collect(distributed=False)
    lines = ds.explain().splitlines()
    anti = next(i for i, ln in enumerate(lines) if "hash_join  [anti" in ln)
    inner = next(i for i, ln in enumerate(lines) if "hash_join  [inner" in ln)
    assert anti < inner and _depth(lines[inner]) > _depth(lines[anti]), ds.explain()


def _depth(line: str) -> int:
    """How deep an `explain()` line sits: the width of its tree-drawing prefix."""
    return len(line) - len(line.lstrip("│├└─ ⋯"))
