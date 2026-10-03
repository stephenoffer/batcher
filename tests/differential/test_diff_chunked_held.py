"""An aggregate over a held input, run first on the chunked path, returns what DuckDB returns.

`api.orchestration.chunked_sideways.run_staged_held` stages the join-free subtree that
aggregates a scan the chunks do not drive, when that scan is too large to hold whole, and runs
the rest of the plan over its result. It changes which executor computes the aggregate, so it is
held to DuckDB on the shape it exists for -- TPC-H q18's `IN (... GROUP BY ... HAVING ...)` over
the same fact table the query streams -- and on a correlated average, with NULL keys and a
relation spread over several files. A spy proves the staged path produced the result, and the
TPC-H q15 shape, where the same aggregate is computed twice and compared for equality, proves it
declines: run on two executors, the two sums could differ in the last bit and lose the row.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 4
_ROWS_PER_FILE = 50_000
_ORDERS = 30_000

_STAGED = [
    # TPC-H q18's shape: the HAVING aggregate reads the fact table the final join streams.
    "SELECT o_k, o_c, sum(l_qty) AS q FROM orders, lineitem "
    "WHERE o_k = l_k AND o_k IN (SELECT l_k FROM lineitem GROUP BY l_k HAVING sum(l_qty) > 120) "
    "GROUP BY o_k, o_c ORDER BY q DESC, o_k LIMIT 50",
    # A correlated average over the fact table, beside a second scan of it.
    "SELECT count(*) AS n, sum(l1.l_price) AS p FROM lineitem l1, orders "
    "WHERE o_k = l1.l_k AND o_c < 40 "
    "AND l1.l_qty > (SELECT avg(l2.l_qty) FROM lineitem l2 WHERE l2.l_k = l1.l_k)",
]

# TPC-H q15's shape: one view's sums, compared for equality with their own max.
_DECLINED = (
    "WITH rev AS (SELECT l_s AS s, sum(l_price) AS total FROM lineitem GROUP BY l_s) "
    "SELECT s, total FROM rev WHERE total = (SELECT max(total) FROM rev) ORDER BY s"
)


def _lineitem(part: int) -> pa.Table:
    rows = range(part * _ROWS_PER_FILE, (part + 1) * _ROWS_PER_FILE)
    return pa.table(
        {
            "l_k": pa.array(
                [None if i % 401 == 0 else (i * 7) % _ORDERS for i in rows], pa.int64()
            ),
            "l_s": pa.array([i % 11 for i in rows], pa.int64()),
            "l_qty": pa.array([float(i % 50) for i in rows]),
            "l_price": pa.array([float(i % 997) + 0.25 for i in rows]),
        }
    )


_ORDERS_T = pa.table(
    {
        "o_k": pa.array(list(range(_ORDERS)), pa.int64()),
        "o_c": pa.array([i % 100 for i in range(_ORDERS)], pa.int64()),
    }
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("held")
    (root / "lineitem").mkdir()
    for part in range(_FILES):
        path = root / "lineitem" / f"p{part}.parquet"
        pq.write_table(_lineitem(part), path, row_group_size=20_000)
    pq.write_table(_ORDERS_T, root / "orders.parquet")
    return root


@pytest.fixture
def held(monkeypatch):
    """Force the chunked path, make every non-driving scan 'too large to hold', record outcomes."""
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    monkeypatch.setattr(chunked, "chunk_worthy", lambda _bytes: True)
    monkeypatch.setattr(chunked, "_held_limit", lambda: 1)
    outcomes: list[int | None] = []
    real = chunked.run_staged_held

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        outcomes.append(None if out is None else sum(b.num_rows for b in out))
        return out

    monkeypatch.setattr(chunked, "run_staged_held", spy)
    return outcomes


def _session(data_dir) -> bt.Session:
    s = bt.Session()
    s.register("lineitem", bt.read.parquet(str(data_dir / "lineitem" / "*.parquet")))
    s.register("orders", bt.read.parquet(str(data_dir / "orders.parquet")))
    return s


def _duck(duck):
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    duck.register("orders", _ORDERS_T)
    return duck


@pytest.mark.parametrize("query", _STAGED)
def test_the_held_aggregate_runs_first_and_matches_duckdb(duck, data_dir, held, query):
    got = _session(data_dir).sql(query).collect()
    assert any(o is not None for o in held), f"the held stage did not produce this result: {held}"
    assert_same_for_query(got, _duck(duck).sql(query), query)


def test_an_aggregate_computed_twice_is_not_staged(duck, data_dir, held):
    got = _session(data_dir).sql(_DECLINED).collect()
    assert held, "the chunked path was not reached, so the decline was not tested"
    assert all(o is None for o in held), f"the q15 shape was staged: {held}"
    assert got.num_rows == 1
    assert_same_for_query(got, _duck(duck).sql(_DECLINED), _DECLINED)
