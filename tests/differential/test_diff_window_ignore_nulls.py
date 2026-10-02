"""Window `IGNORE NULLS` vs DuckDB.

`IGNORE NULLS` makes a value function skip nulls when picking its answer. `first_value`,
`last_value` and `nth_value` carry it to the engine as the window function's `ignore_nulls`
flag, which picks among the non-null values of any frame. They used to lower two shapes onto
the fill primitives (`last_value` over the default frame as a forward fill, `first_value`
over ``CURRENT ROW AND UNBOUNDED FOLLOWING`` as a backward fill); both still hold, and the
peer-group test below is what the fills got wrong.

`lag`/`lead` with IGNORE NULLS are rewritten by the SQL front-end into `nth_value` /
`last_value` with IGNORE NULLS over a frame starting just past the current row
(`_sql/parser/windowing/derived.py`), so they are checked against DuckDB here too.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def gaps(duck):
    # Leading null (nothing to fill from), interior runs of nulls, trailing null.
    table = pa.table(
        {
            "k": ["a", "a", "a", "b", "b", "b", "b"],
            "i": [1, 2, 3, 1, 2, 3, 4],
            "v": [None, 2, None, 5, None, None, 8],
        }
    )
    duck.register("gaps", table)
    return table


@pytest.mark.differential
def test_last_value_ignore_nulls_is_a_forward_fill(duck, gaps):
    """Default frame: carry the most recent non-null forward, NULL before the first."""
    query = "SELECT i, k, last_value(v IGNORE NULLS) OVER (ORDER BY k, i) AS x FROM gaps"
    assert_same(bt.sql(query, gaps=gaps).collect(), duck.sql(query))


@pytest.mark.differential
def test_first_value_ignore_nulls_is_a_backward_fill(duck, gaps):
    """Forward-looking frame: take the next non-null, NULL after the last."""
    query = (
        "SELECT i, k, first_value(v IGNORE NULLS) OVER "
        "(ORDER BY k, i ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS x FROM gaps"
    )
    assert_same(bt.sql(query, gaps=gaps).collect(), duck.sql(query))


@pytest.mark.differential
def test_ignore_nulls_respects_partitions(duck, gaps):
    """A fill must not carry a value across a partition boundary."""
    query = (
        "SELECT i, k, last_value(v IGNORE NULLS) OVER (PARTITION BY k ORDER BY i) AS x FROM gaps"
    )
    assert_same(bt.sql(query, gaps=gaps).collect(), duck.sql(query))


@pytest.mark.differential
def test_ignore_nulls_differs_from_respecting_nulls(duck, gaps):
    """The whole point: IGNORE NULLS must not equal the plain form.

    Without the flag `last_value` returns the current row's value, nulls and all. If the
    mapping were dropped the query would still run and quietly return that instead.
    """
    ignoring = "SELECT i, last_value(v IGNORE NULLS) OVER (ORDER BY i) AS x FROM gaps"
    respecting = "SELECT i, last_value(v) OVER (ORDER BY i) AS x FROM gaps"
    a = bt.sql(ignoring, gaps=gaps).collect().to_pydict()
    b = bt.sql(respecting, gaps=gaps).collect().to_pydict()
    assert a != b, "IGNORE NULLS produced the null-respecting result"


@pytest.mark.differential
def test_all_null_column_ignore_nulls(duck):
    """Nothing to fill from — every row stays NULL rather than erroring."""
    table = pa.table({"i": [1, 2, 3], "v": pa.array([None, None, None], pa.int64())})
    duck.register("allnull", table)
    query = "SELECT i, last_value(v IGNORE NULLS) OVER (ORDER BY i) AS x FROM allnull"
    assert_same(bt.sql(query, allnull=table).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize(
    "expr",
    [
        "last_value(v IGNORE NULLS) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)",
        "nth_value(v, 2 IGNORE NULLS) OVER (PARTITION BY k ORDER BY i)",
        "first_value(v IGNORE NULLS) OVER (PARTITION BY k ORDER BY i)",
        "last_value(v IGNORE NULLS) OVER (PARTITION BY k ORDER BY i ROWS BETWEEN UNBOUNDED "
        "PRECEDING AND UNBOUNDED FOLLOWING)",
    ],
)
def test_framed_ignore_nulls_shapes_match_duckdb(duck, gaps, expr):
    """Every frame takes IGNORE NULLS once the runtime picks among non-null values itself."""
    query = f"SELECT i, k, {expr} AS x FROM gaps"
    assert_same(bt.sql(query, gaps=gaps).collect(), duck.sql(query))


@pytest.mark.differential
def test_ignore_nulls_reads_the_whole_peer_group(duck):
    """A tie on the ORDER BY key is one peer group, whatever order its rows arrive in.

    The fills this used to lower to walked rows physically, so the NULL row tied with the 5
    read 1 when it arrived first. DuckDB's running frame includes the whole peer group.
    """
    table = pa.table({"k": [1, 2, 2, 3], "x": pa.array([1, None, 5, None], pa.int64())})
    duck.register("ties", table)
    query = "SELECT k, x, last_value(x IGNORE NULLS) OVER (ORDER BY k) AS r FROM ties"
    assert_same(bt.sql(query, ties=table).collect(), duck.sql(query))


#: `lag`/`lead` with IGNORE NULLS, in both spellings sqlglot reads (the standard one after
#: the call and DuckDB's inside it), across offsets, a default, both directions of the
#: ORDER BY and a negative offset that swaps lag for lead.
_NAVIGATION = [
    "lag(v) IGNORE NULLS OVER (PARTITION BY k ORDER BY i)",
    "lead(v) IGNORE NULLS OVER (PARTITION BY k ORDER BY i)",
    "lag(v, 2 IGNORE NULLS) OVER (PARTITION BY k ORDER BY i)",
    "lead(v, 2 IGNORE NULLS) OVER (PARTITION BY k ORDER BY i DESC)",
    "lag(v, 2, -1 IGNORE NULLS) OVER (PARTITION BY k ORDER BY i)",
    "lead(v, 1, -1) IGNORE NULLS OVER (ORDER BY k, i)",
    "lag(v, -2 IGNORE NULLS) OVER (PARTITION BY k ORDER BY i NULLS FIRST)",
    "lag(v, 0, -1 IGNORE NULLS) OVER (PARTITION BY k ORDER BY i)",
    "lag(v + 1, 3 IGNORE NULLS) OVER (ORDER BY k, i)",
]


@pytest.mark.differential
@pytest.mark.parametrize("expr", _NAVIGATION)
def test_lag_and_lead_ignore_nulls_match_duckdb(duck, gaps, expr):
    """The k-th non-null neighbour, compared row by row (ordered by the unique ``(k, i)``).

    DuckDB spells the clause inside the call only, so its query is the canonical spelling
    of the same window.
    """
    query = f"SELECT k, i, {expr} AS x FROM gaps ORDER BY k, i"
    canonical = query.replace(") IGNORE NULLS OVER", " IGNORE NULLS) OVER")
    got = bt.sql(query, gaps=gaps).collect()
    assert [tuple(r.values()) for r in got.to_pylist()] == duck.sql(canonical).fetchall()


@pytest.mark.differential
def test_lag_ignore_nulls_differs_from_the_null_respecting_lag(duck, gaps):
    """Positive control: on this fixture skipping nulls changes the answer."""
    query = "SELECT k, i, lag(v{}) OVER (PARTITION BY k ORDER BY i) AS x FROM gaps ORDER BY k, i"
    ignoring = bt.sql(query.format(" IGNORE NULLS"), gaps=gaps).to_pydict()["x"]
    respecting = bt.sql(query.format(""), gaps=gaps).to_pydict()["x"]
    assert ignoring != respecting
