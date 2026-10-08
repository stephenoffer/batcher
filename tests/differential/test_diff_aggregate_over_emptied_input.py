"""An aggregate whose input turns out empty only at run time, vs DuckDB.

When no row reaches a streaming aggregate, the empty-input answer -- one row of `COUNT` 0 and
`SUM` NULL for a keyless aggregate, no rows for a grouped one -- is taken from the oracle. It is
computed over an empty relation of the input's *schema* (`stream::breaker`), rather than by
re-running the subtree beneath the aggregate, so what is checked here is that the schema it is
given yields the same types and values as the real input would: integer, float, string and date
columns, `COUNT(*)` against `COUNT(col)`, `MIN`/`MAX` of each type, `AVG`, and a grouped
aggregate that must return nothing at all.

The inputs empty *after* real work over enough rows to shard (an `EXCEPT` that removes every
row, a join whose keys never meet, a filter nothing passes), which is the case the shortcut
exists for. A control query over the same shapes that does *not* empty proves the fixtures are
not empty to begin with.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_ROWS = 120_000


@pytest.fixture(scope="module")
def tables() -> dict[str, pa.Table]:
    base = dt.date(2020, 1, 1)
    a = pa.table(
        {
            "k": pa.array([i % 5_000 for i in range(_ROWS)], pa.int64()),
            "x": pa.array([None if i % 13 == 0 else i * 0.5 for i in range(_ROWS)], pa.float64()),
            "s": pa.array([f"s{i % 97}" for i in range(_ROWS)]),
            "d": pa.array([base + dt.timedelta(days=i % 400) for i in range(_ROWS)], pa.date32()),
        }
    )
    # Every key of `b` is a key of `a`, so `a.k EXCEPT b.k` keeps the 1,000 keys `b` lacks and
    # `b.k EXCEPT a.k` keeps nothing.
    b = pa.table({"k": pa.array([i % 4_000 for i in range(_ROWS)], pa.int64())})
    far = pa.table({"k": pa.array([10_000_000 + i for i in range(_ROWS)], pa.int64())})
    return {"a": a, "b": b, "far": far}


_AGGS = (
    "count(*) AS n, count(x) AS nx, sum(x) AS sx, avg(x) AS ax, min(x) AS mnx, max(s) AS mxs, "
    "min(d) AS mnd, sum(a.k) AS sk"
)

_EMPTIED = {
    "except": "SELECT count(*) AS n, sum(k) AS sk, min(k) AS mk FROM "
    "(SELECT k FROM b EXCEPT SELECT k FROM a) u",
    "join": f"SELECT {_AGGS} FROM a JOIN far ON a.k = far.k",
    "filter": f"SELECT {_AGGS} FROM a WHERE x < -1",
    "grouped": "SELECT s, count(*) AS n, sum(x) AS sx FROM a JOIN far ON a.k = far.k GROUP BY s",
    "left_then_filter": f"SELECT {_AGGS} FROM a LEFT JOIN far ON a.k = far.k WHERE far.k > 0",
}

_CONTROL = {
    "except": "SELECT count(*) AS n, sum(k) AS sk, min(k) AS mk FROM "
    "(SELECT k FROM a EXCEPT SELECT k FROM b) u",
    "filter": f"SELECT {_AGGS} FROM a WHERE x > 10",
}


def _session(tables: dict[str, pa.Table]) -> bt.Session:
    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
    return s


def _register(duck, tables: dict[str, pa.Table]) -> None:
    for name, t in tables.items():
        duck.register(name, t)


@pytest.mark.parametrize("name", sorted(_EMPTIED))
def test_an_aggregate_over_an_emptied_input_matches_duckdb(duck, tables, name) -> None:
    _register(duck, tables)
    sql = _EMPTIED[name]
    got = _session(tables).sql(sql).collect()
    want = duck.sql(sql)
    assert_same(got, want)
    if name != "grouped":
        assert got.num_rows == 1
        assert got.column("n" if "n" in got.column_names else got.column_names[0])[0].as_py() == 0


@pytest.mark.parametrize("name", sorted(_EMPTIED))
def test_the_empty_answer_keeps_the_column_types(tables, name) -> None:
    """The schema the empty relation carries must be the input's, so the output types match
    what the same aggregate produces over a non-empty input of that shape."""
    ds = _session(tables).sql(_EMPTIED[name])
    assert ds.collect().schema.types == ds.schema.types


@pytest.mark.parametrize("name", sorted(_CONTROL))
def test_the_controls_are_not_empty(duck, tables, name) -> None:
    _register(duck, tables)
    sql = _CONTROL[name]
    got = _session(tables).sql(sql).collect()
    assert got.column("n")[0].as_py() > 0
    assert_same(got, duck.sql(sql))


@pytest.mark.parametrize("mode", ["iter_batches", "spill"])
def test_streamed_and_spilled_paths_agree(duck, tables, mode) -> None:
    _register(duck, tables)
    sql = _EMPTIED["join"]
    ds = _session(tables).sql(sql)
    if mode == "spill":
        got = ds.collect(spill=True, num_partitions=4)
    else:
        got = pa.Table.from_batches(list(ds.iter_batches()))
    assert_same(got, duck.sql(sql))
