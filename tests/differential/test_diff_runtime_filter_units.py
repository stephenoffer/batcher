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


@pytest.fixture(scope="module")
def lineitem_int32_dir(tmp_path_factory):
    """The same rows with the join key stored as Int32, which the engine widens to Int64."""
    root = tmp_path_factory.mktemp("lineitem_rf_units_int32")
    for part in range(_FILES):
        t = _lineitem(part)
        t = t.set_column(0, "l_ok", t["l_ok"].cast(pa.int32()))
        pq.write_table(t, root / f"part-{part}.parquet", row_group_size=_ROW_GROUP)
    return root


@pytest.mark.parametrize("query", _QUERIES)
def test_a_narrow_key_is_tested_during_decode_as_the_engine_sees_it(
    duck, lineitem_int32_dir, engine_read, monkeypatch, query
):
    """The decode tests the key before the engine widens it, and must test the widened value.

    A runtime filter handed to the reader (`UnitSource::read_keyed`) runs on the batch as it
    comes off the Parquet page, where this key is still Int32; the digest takes Int64. The mask
    normalizes the batch first, exactly as the engine does on the way to its own filter, so a key
    that matches must match there too.
    """
    monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", "force")
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    duck.register("orders", _ORDERS)
    session = bt.Session()
    session.register("lineitem", bt.read.parquet(str(lineitem_int32_dir / "*.parquet")))
    session.register("orders", _ORDERS)
    got = session.sql(query).collect()
    assert engine_read, "the query did not take the engine's row-group read"
    assert_same_for_query(got, duck.sql(query), query)


_WIDE_ROWS = 200_000


@pytest.fixture(scope="module")
def wide_dir(tmp_path_factory):
    """A probe side worth deferring: a key, then a wide string the join only reads on survivors.

    Late materialization installs only when the columns it would defer are at least as large as
    the ones its stages read (`LateFilter::worth_deferring`), so the main fixture -- an `Int64`
    key ahead of a 50-value `Float64` -- correctly declines. A payload ~60 bytes a row is the
    shape the keyed read exists for.
    """
    root = tmp_path_factory.mktemp("lineitem_rf_units_wide")
    for part in range(2):
        rows = range(part * _WIDE_ROWS, (part + 1) * _WIDE_ROWS)
        pq.write_table(
            pa.table(
                {
                    "l_ok": pa.array(
                        [(i * 37) % ((_BUILD_KEYS - 1) * 100) for i in rows], pa.int64()
                    ),
                    "l_note": pa.array([f"note-{i:08d}-" + "x" * (40 + i % 17) for i in rows]),
                }
            ),
            root / f"part-{part}.parquet",
            row_group_size=_ROW_GROUP,
        )
    return root


_WIDE_QUERY = (
    "SELECT o_prio, count(*) AS n, sum(length(l_note)) AS w "
    "FROM lineitem JOIN orders ON l_ok = o_ok GROUP BY o_prio"
)


def test_the_read_drops_refuted_rows_before_the_join_sees_them(duck, wide_dir, monkeypatch):
    """The positive control: the keys reach the reader, which removes rows while decoding.

    Every differential above would pass if the reader ignored the keys, because the engine
    applies the same filter again after the scan. What only the reader can do is make the scan
    *itself* emit fewer rows than the files hold. Every probe key here lies inside the build
    side's range, so Kyber's plan-time key range prunes nothing and the drop is the digest's.
    """
    import batcher.core as core
    from batcher.api.orchestration import chunked

    monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", "force")
    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    seen: list[list[dict]] = []
    real = core.execute_local_parquet

    def spy(*args, **kwargs):
        out, ops, usage = real(*args, **kwargs)
        seen.append(ops)
        return out, ops, usage

    monkeypatch.setattr(core, "execute_local_parquet", spy)
    session = bt.Session()
    session.register("lineitem", bt.read.parquet(str(wide_dir / "*.parquet")))
    session.register("orders", _ORDERS)
    got = session.sql(_WIDE_QUERY).collect()
    duck.register("lineitem", pq.read_table(wide_dir))
    duck.register("orders", _ORDERS)
    assert_same_for_query(got, duck.sql(_WIDE_QUERY), _WIDE_QUERY)
    assert seen, "the query did not take the engine's row-group read"
    scans = [op for ops in seen for op in ops if op.get("kind") == "scan"]
    probe = max(scans, key=lambda op: op["rows_in"])
    assert probe["rows_out"] < 2 * _WIDE_ROWS, (probe, scans)
