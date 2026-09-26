"""The largest source streamed into the engine in chunks returns what the resident path returns.

`api.orchestration.chunked` hands the driving Parquet source to the engine a group of files at a
time (`bc_interp::stream::chunked`). It engages on large inputs only, so these tests lower the
threshold and the chunk size until every file is its own chunk, and hold the result to DuckDB
over a TPC-H-shaped schema: an inner join to a filtered dimension, a left join, a semi and an
anti join on the probe spine, a global and a grouped aggregate, a sort and limit above, NULL
keys, a file whose rows are all filtered away, and a predicate pushed into the chunked read.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 5
_ROWS_PER_FILE = 3_000

_QUERIES = [
    "SELECT o_prio, sum(l_price * (1 - l_disc)) AS rev, count(*) AS n, avg(l_qty) AS q "
    "FROM lineitem JOIN orders ON l_ok = o_ok WHERE o_prio <> 'LOW' AND l_ship < 40 "
    "GROUP BY o_prio ORDER BY rev DESC LIMIT 10",
    "SELECT count(*) AS n, sum(l_qty) AS q, min(l_price) AS lo, max(l_price) AS hi "
    "FROM lineitem WHERE l_ship BETWEEN 10 AND 20",
    "SELECT o_prio, count(o_ok) AS matched, count(*) AS n "
    "FROM lineitem LEFT JOIN orders ON l_ok = o_ok GROUP BY o_prio",
    "SELECT l_flag, count(*) AS n FROM lineitem "
    "WHERE EXISTS (SELECT 1 FROM orders WHERE o_ok = l_ok AND o_prio = 'HIGH') "
    "GROUP BY l_flag ORDER BY l_flag",
    "SELECT l_flag, count(*) AS n FROM lineitem "
    "WHERE NOT EXISTS (SELECT 1 FROM orders WHERE o_ok = l_ok) GROUP BY l_flag",
    # TPC-H q15's shape: a grouped sum compared for *equality* with the max of the same sums,
    # which the control plane evaluates first and folds in as a literal. Inexact factors on
    # purpose — if the two evaluations summed in different orders the max would match no row.
    "WITH rev AS (SELECT l_ok AS k, sum(l_price * 0.07 + l_qty * 0.013) AS r FROM lineitem "
    "WHERE l_ship < 50 GROUP BY l_ok) SELECT k, r FROM rev WHERE r = (SELECT max(r) FROM rev)",
    # Every row of the last file fails the predicate, so one chunk contributes nothing.
    "SELECT count(*) AS n, sum(l_price) AS s FROM lineitem WHERE l_ship < 3",
]


def _lineitem(part: int) -> pa.Table:
    base = part * _ROWS_PER_FILE
    rows = range(base, base + _ROWS_PER_FILE)
    return pa.table(
        {
            "l_ok": pa.array([None if i % 211 == 0 else i % 1_700 for i in rows], pa.int64()),
            "l_qty": pa.array([float(i % 50) for i in rows]),
            "l_price": pa.array([float(i % 997) + 0.5 for i in rows]),
            # Quarters and halves: every product and sum is exact in binary, so the ordered
            # comparison below needs no float tolerance however the chunks reassociate the sums.
            "l_disc": pa.array([(i % 4) / 4 for i in rows]),
            # The last file ships only on days >= 60, so `l_ship < 3` prunes it entirely.
            "l_ship": pa.array([60 + i % 30 if part == _FILES - 1 else i % 90 for i in rows]),
            "l_flag": pa.array(["AFRN"[i % 4] for i in rows]),
        }
    )


_ORDERS = pa.table(
    {
        "o_ok": pa.array(list(range(0, 1_500)), pa.int64()),
        "o_prio": pa.array(["HIGH", "MED", "LOW"][i % 3] for i in range(1_500)),
    }
)


@pytest.fixture(scope="module")
def lineitem_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineitem")
    for part in range(_FILES):
        pq.write_table(_lineitem(part), root / f"part-{part}.parquet")
    return root


@pytest.fixture
def streamed(monkeypatch):
    """Force the chunked path on these small inputs, one file per chunk, and count its calls.

    The engine's own row-group read (`parquet.units`) takes precedence for a plain Parquet
    source and has its own suite, `test_diff_parquet_units`; it is switched off here so this
    file keeps testing the control-plane chunks it is about.
    """
    import batcher.core as core
    from batcher.api.orchestration import chunked
    from batcher.dist.spill import scratch

    monkeypatch.setattr(chunked, "_unit_read", lambda *_args: None)
    monkeypatch.setattr(chunked, "chunk_worthy", lambda _bytes: True)
    monkeypatch.setattr(scratch, "spill_chunk_bytes", lambda: 1)
    calls: list[int] = []
    real = core.execute_local_chunked

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "execute_local_chunked", spy)
    return calls


def _session(lineitem_dir) -> bt.Session:
    s = bt.Session()
    s.register("lineitem", bt.read.parquet(str(lineitem_dir / "*.parquet")))
    s.register("orders", _ORDERS)
    return s


@pytest.mark.parametrize("query", _QUERIES)
def test_chunked_scan_matches_duckdb(duck, lineitem_dir, streamed, query):
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    duck.register("orders", _ORDERS)
    got = _session(lineitem_dir).sql(query).collect()
    assert streamed, "the query did not take the chunked path, so nothing here was tested"
    assert_same_for_query(got, duck.sql(query), query)


def test_a_self_join_is_still_correct(duck, lineitem_dir, streamed):
    """A query naming the driving source twice answers correctly whichever path it takes.

    Whether it streams depends on the plan the optimizer hands the engine: common-subplan reuse
    can materialize one side, which leaves the other scan the only one and the plan chunkable.
    The engine's own refusal of a doubly-scanned source is `bc_interp::stream::chunked`'s test.
    """
    query = (
        "SELECT count(*) AS n FROM lineitem a JOIN lineitem b ON a.l_ok = b.l_ok "
        "WHERE a.l_ship < 5 AND b.l_ship < 5"
    )
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    got = _session(lineitem_dir).sql(query).collect()
    assert_same_for_query(got, duck.sql(query), query)
