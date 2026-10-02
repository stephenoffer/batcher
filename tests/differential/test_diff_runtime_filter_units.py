"""A runtime join filter sized from the streamed probe side returns what DuckDB returns.

On the engine's row-group read (`bc_interp::stream::chunked::execute_units`) the driving relation
is a zero-row carrier until its workers read it, so the runtime filter's placement is told its
real size from the Parquet footers (`UnitSource::rows`). A probe side at least 16x its build side
then gets the wide bitmap digest (`KeyFilter::build_for_probe`), which is what the build side
here needs: 70,000 keys spaced 100 apart, too many for the hash set and too sparse for the
probe-limit bitmap. Every query is held to DuckDB with the filter forced on and switched off, on
inner and semi joins, with NULL probe keys and keys outside the build side's range, and with a
nullable predicate directly above the filtered scan.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 4
_ROWS_PER_FILE = 300_000
_ROW_GROUP = 50_000
_BUILD_KEYS = 70_000

_QUERIES = [
    "SELECT o_prio, count(*) AS n, sum(l_qty) AS q "
    "FROM lineitem JOIN orders ON l_ok = o_ok WHERE l_qty < 40 GROUP BY o_prio",
    "SELECT count(*) AS n, sum(l_qty) AS q FROM lineitem JOIN orders ON l_ok = o_ok",
    # A nullable disjunction directly above the scan the runtime join filter is placed on.
    "SELECT l_flag, count(*) AS n, sum(l_qty) AS q FROM lineitem JOIN orders ON l_ok = o_ok "
    "WHERE l_qty < 10 OR l_flag = 'F' GROUP BY l_flag",
    "SELECT l_flag, count(*) AS n FROM lineitem "
    "WHERE EXISTS (SELECT 1 FROM orders WHERE o_ok = l_ok AND o_prio = 'HIGH') GROUP BY l_flag",
]


def _lineitem(part: int) -> pa.Table:
    rows = range(part * _ROWS_PER_FILE, (part + 1) * _ROWS_PER_FILE)
    return pa.table(
        {
            # Half the keys fall past the build side's largest key, and a few are NULL.
            "l_ok": pa.array(
                [None if i % 997 == 0 else (i * 37) % (_BUILD_KEYS * 200) for i in rows],
                pa.int64(),
            ),
            # NULL on some rows, so a predicate directly above the filtered scan meets NULLs.
            "l_qty": pa.array([None if i % 101 == 0 else float(i % 50) for i in rows]),
            "l_flag": pa.array(["AFRN"[i % 4] for i in rows]),
        }
    )


_ORDERS = pa.table(
    {
        "o_ok": pa.array([i * 100 for i in range(_BUILD_KEYS)], pa.int64()),
        "o_prio": pa.array(["HIGH", "MED", "LOW"][i % 3] for i in range(_BUILD_KEYS)),
    }
)


@pytest.fixture(scope="module")
def lineitem_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineitem_rf_units")
    for part in range(_FILES):
        pq.write_table(_lineitem(part), root / f"part-{part}.parquet", row_group_size=_ROW_GROUP)
    return root


@pytest.fixture
def engine_read(monkeypatch):
    """Force the engine's row-group read on these inputs and count its calls."""
    import batcher.core as core
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    calls: list[int] = []
    real = core.execute_local_parquet

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "execute_local_parquet", spy)
    return calls


@pytest.mark.parametrize("switch", ["force", "0"])
@pytest.mark.parametrize("query", _QUERIES)
def test_runtime_filter_on_the_row_group_read_matches_duckdb(
    duck, lineitem_dir, engine_read, monkeypatch, switch, query
):
    monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", switch)
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    duck.register("orders", _ORDERS)
    session = bt.Session()
    session.register("lineitem", bt.read.parquet(str(lineitem_dir / "*.parquet")))
    session.register("orders", _ORDERS)
    got = session.sql(query).collect()
    assert engine_read, "the query did not take the engine's row-group read"
    assert_same_for_query(got, duck.sql(query), query)
