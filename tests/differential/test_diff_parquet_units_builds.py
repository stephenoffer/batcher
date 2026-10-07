"""A row-group execution's other Parquet scans, read by the engine, return what DuckDB returns.

When the engine reads the driving scan row group by row group (`core.execute_local_parquet`),
the plan's other plain Parquet scans -- the join build sides -- are read by the engine too
(`bc_py::chunked::resident`) instead of by the control plane and handed over through pyarrow.
These tests write both sides as multi-file Parquet with small row groups, hold the results to
DuckDB, and check that the native reads were really requested: build sides with NULL keys, a
predicate pruning some of a build side's row groups, one no row group can satisfy, a dimension
filtered on a string, several build sides at once, LEFT and anti joins, and a source bound
twice, which keeps the control plane's shared read.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FACT_FILES = 3
_FACT_ROWS = 4_000
_ROW_GROUP = 400


def _fact(part: int) -> pa.Table:
    rows = range(part * _FACT_ROWS, (part + 1) * _FACT_ROWS)
    return pa.table(
        {
            "f_ok": pa.array([None if i % 173 == 0 else i % 2_100 for i in rows], pa.int64()),
            "f_ck": pa.array([i % 410 for i in rows], pa.int64()),
            "f_amt": pa.array([float(i % 997) + 0.25 for i in rows]),
            "f_tag": pa.array(["xyzw"[i % 4] for i in rows]),
        }
    )


def _orders(part: int) -> pa.Table:
    # Each file a contiguous key range, so a pushed range predicate prunes whole row groups.
    rows = range(part * 1_000, (part + 1) * 1_000)
    return pa.table(
        {
            "o_ok": pa.array([None if i % 97 == 5 else i for i in rows], pa.int64()),
            "o_day": pa.array([i // 10 for i in rows], pa.int64()),
            "o_prio": pa.array(
                [None if i % 31 == 0 else ["HIGH", "MED", "LOW"][i % 3] for i in rows]
            ),
        }
    )


def _customer(part: int) -> pa.Table:
    rows = range(part * 250, (part + 1) * 250)
    return pa.table(
        {
            "c_ck": pa.array(list(rows), pa.int64()),
            "c_seg": pa.array(["AUTO", "BUILD", "FURN", "MACH", "HOUSE"][i % 5] for i in rows),
        }
    )


_QUERIES = [
    # A build side read whole, NULL keys on both sides.
    "SELECT o_prio, count(*) AS n, sum(f_amt) AS s FROM fact JOIN orders ON f_ok = o_ok "
    "GROUP BY o_prio",
    # A range on the build side's clustered column: its footers prune most row groups.
    "SELECT count(*) AS n, sum(f_amt) AS s FROM fact JOIN orders ON f_ok = o_ok "
    "WHERE o_day BETWEEN 20 AND 45",
    # Two build sides, one filtered on a string column.
    "SELECT c_seg, o_prio, count(*) AS n FROM fact JOIN orders ON f_ok = o_ok "
    "JOIN customer ON f_ck = c_ck WHERE c_seg IN ('AUTO', 'MACH') GROUP BY c_seg, o_prio",
    # An anti join keeps the probe rows the build side does not hold.
    "SELECT f_tag, count(*) AS n FROM fact WHERE NOT EXISTS "
    "(SELECT 1 FROM orders WHERE o_ok = f_ok AND o_prio = 'HIGH') GROUP BY f_tag",
]


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    root = tmp_path_factory.mktemp("units_builds")
    parts = {"fact": (_fact, _FACT_FILES), "orders": (_orders, 3), "customer": (_customer, 2)}
    for name, (make, files) in parts.items():
        (root / name).mkdir()
        for part in range(files):
            pq.write_table(make(part), root / name / f"{part}.parquet", row_group_size=_ROW_GROUP)
    whole = {
        name: pa.concat_tables(make(p) for p in range(files))
        for name, (make, files) in parts.items()
    }
    return root, whole


@pytest.fixture
def native_reads(monkeypatch):
    """Force the row-group read on these small inputs; record which scans the engine read."""
    import batcher.core as core
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    seen: list[dict] = []
    real = core.execute_local_parquet

    def spy(*args, **kwargs):
        seen.append(dict(args[6]) if len(args) > 6 and args[6] else {})
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "execute_local_parquet", spy)
    return seen


def _session(root) -> bt.Session:
    s = bt.Session()
    for name in ("fact", "orders", "customer"):
        s.register(name, bt.read.parquet(str(root / name / "*.parquet")))
    return s


@pytest.mark.parametrize("query", _QUERIES)
def test_native_build_reads_match_duckdb(duck, tables, native_reads, query):
    root, whole = tables
    for name, table in whole.items():
        duck.register(name, table)
    got = _session(root).sql(query).collect()
    assert native_reads, "the query did not take the engine's row-group read"
    assert any(native_reads), "no build side was read by the engine"
    assert_same_for_query(got, duck.sql(query), query)


#: Shapes the row-group route may decline, or Kyber may answer from the footers before anything
#: runs, so only the answer is asserted: a build side no row group can satisfy (an empty
#: relation), and a LEFT join grouped on the build side's column.
_ANSWER_ONLY = [
    "SELECT count(*) AS n, sum(f_amt) AS s FROM fact JOIN orders ON f_ok = o_ok "
    "WHERE o_day > 100000",
    "SELECT o_prio, count(*) AS n, count(o_ok) AS matched FROM fact "
    "LEFT JOIN orders ON f_ok = o_ok GROUP BY o_prio",
]


@pytest.mark.parametrize("query", _ANSWER_ONLY)
def test_shapes_the_route_may_decline_still_answer(duck, tables, native_reads, query):
    root, whole = tables
    for name, table in whole.items():
        duck.register(name, table)
    assert_same_for_query(_session(root).sql(query).collect(), duck.sql(query), query)


def test_a_source_bound_twice_keeps_the_shared_read(duck, tables, native_reads):
    """Two bindings of one source among the build sides are read once, by the control plane."""
    root, whole = tables
    query = (
        "SELECT count(*) AS n FROM fact JOIN orders a ON f_ok = a.o_ok "
        "JOIN orders b ON f_ck = b.o_ok WHERE a.o_prio = 'LOW'"
    )
    for name, table in whole.items():
        duck.register(name, table)
    got = _session(root).sql(query).collect()
    assert native_reads, "the query did not take the engine's row-group read"
    assert not any(len(r) > 1 for r in native_reads), "a shared source was read per binding"
    assert_same_for_query(got, duck.sql(query), query)
