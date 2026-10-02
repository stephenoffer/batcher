"""A decorrelated aggregate streamed in two sideways stages returns what DuckDB returns.

`api.orchestration.chunked_sideways` runs a plan whose outer spine drives the chunks, and whose
decorrelated aggregate reads the same Parquet relation again, in two stages: the outer side
first, then the rest with the aggregate's scan restricted to the outer side's keys and driving
the chunks itself. It changes which rows each operator sees, so it is held to DuckDB on the
shapes it exists for — TPC-H q17's correlated average and q21's `EXISTS`/`NOT EXISTS` pair —
with TPC-H q4's semi join over the largest relation and its anti twin, NULL keys, keys the
outer side lacks, and a restricted relation spread over several files
of several row groups each. A spy proves the staged path produced the result; without it a
path that quietly declined would pass every comparison below.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 4
_ROWS_PER_FILE = 75_000
_PARTS = 20_000

_QUERIES = [
    # A correlated average over the fact table, correlated on the *outer fact* column. (TPC-H
    # q17 correlates on the dimension's key instead, which `agg_semijoin` restricts from the
    # cheap dimension side at plan time, so Kyber rightly gives it no sideways verdict.)
    "SELECT sum(l1.l_price) / 7.0 AS avg_yearly FROM lineitem l1, part "
    "WHERE p_k = l1.l_k AND p_brand = 'B7' "
    "AND l1.l_qty < (SELECT 0.2 * avg(l2.l_qty) FROM lineitem l2 WHERE l2.l_k = l1.l_k)",
    # TPC-H q21's shape: an EXISTS and a NOT EXISTS over the same fact table, both correlated on
    # the key and anti-correlated on a second column.
    "SELECT p_brand, count(*) AS n FROM lineitem l1, part "
    "WHERE p_k = l1.l_k AND p_brand IN ('B3', 'B7') AND l1.l_late = 1 "
    "AND EXISTS (SELECT 1 FROM lineitem l2 WHERE l2.l_k = l1.l_k AND l2.l_s <> l1.l_s) "
    "AND NOT EXISTS (SELECT 1 FROM lineitem l3 WHERE l3.l_k = l1.l_k "
    "AND l3.l_s <> l1.l_s AND l3.l_late = 1) "
    "GROUP BY p_brand ORDER BY p_brand",
    # TPC-H q4's shape: a semi join whose build side is the largest relation, and the anti join
    # beside it. Neither is chunkable as written; the staged second stage replaces the right side
    # with its restricted key set, an aggregate the chunks can drive.
    "SELECT p_brand, count(*) AS n FROM part WHERE p_brand IN ('B3', 'B7') "
    "AND EXISTS (SELECT 1 FROM lineitem WHERE l_k = p_k) "
    "GROUP BY p_brand ORDER BY p_brand",
    "SELECT p_brand, count(*) AS n FROM part WHERE p_brand IN ('B3', 'B7') "
    "AND NOT EXISTS (SELECT 1 FROM lineitem WHERE l_k = p_k AND l_qty < 48) "
    "GROUP BY p_brand ORDER BY p_brand",
]


def _lineitem(part: int) -> pa.Table:
    rows = range(part * _ROWS_PER_FILE, (part + 1) * _ROWS_PER_FILE)
    return pa.table(
        {
            "l_k": pa.array(
                [None if i % 499 == 0 else (i * 13) % (_PARTS * 2) for i in rows], pa.int64()
            ),
            "l_s": pa.array([i % 5 for i in rows], pa.int64()),
            "l_qty": pa.array([float(i % 50) for i in rows]),
            "l_price": pa.array([float(i % 997) + 0.5 for i in rows]),
            "l_late": pa.array([int(i % 3 == 0) for i in rows], pa.int64()),
        }
    )


_PART = pa.table(
    {
        "p_k": pa.array(list(range(_PARTS)), pa.int64()),
        "p_brand": pa.array([f"B{i % 50}" for i in range(_PARTS)]),
    }
)


@pytest.fixture(scope="module")
def lineitem_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineitem_sideways")
    for part in range(_FILES):
        pq.write_table(_lineitem(part), root / f"part-{part}.parquet", row_group_size=20_000)
    return root


@pytest.fixture
def staged(monkeypatch):
    """Force the chunked path on these inputs and record every staged result it returned."""
    from batcher.api.orchestration import chunked
    from batcher.kyber.optimizer import facade

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    monkeypatch.setattr(chunked, "chunk_worthy", lambda _bytes: True)
    # The verdict's row floor is a production size; the fixture's relations are small enough
    # that Kyber's own derived key-range filter takes the semi join's right side under it.
    monkeypatch.setattr(facade, "SIDEWAYS_MIN_ROWS", 1)
    results: list[int] = []
    real = chunked.run_staged_sideways

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        if out is not None:
            results.append(sum(b.num_rows for b in out))
        return out

    monkeypatch.setattr(chunked, "run_staged_sideways", spy)
    return results


@pytest.mark.parametrize("query", _QUERIES)
def test_the_staged_sideways_path_matches_duckdb(duck, lineitem_dir, staged, query):
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    duck.register("part", _PART)
    session = bt.Session()
    session.register("lineitem", bt.read.parquet(str(lineitem_dir / "*.parquet")))
    # Parquet too: the staged path runs each stage on the chunked executor, which streams only
    # a source that can read itself in chunks, so an in-memory outer side declines the staging.
    pq.write_table(_PART, lineitem_dir.parent / "part.parquet")
    session.register("part", bt.read.parquet(str(lineitem_dir.parent / "part.parquet")))
    got = session.sql(query).collect()
    assert staged, "the staged sideways path did not produce this result"
    assert_same_for_query(got, duck.sql(query), query)
