"""A join's build side restricted by another build's key set returns what DuckDB returns.

Two joins on one probe spine that test the same probe column (TPC-H q9: `lineitem` against the
"green" `part` rows on `l_partkey`, then against `partsupp` on `(l_partkey, l_suppkey)`) let the
larger build drop the rows the smaller one refutes before it is hashed
(`bc_interp::stream::runtime_filter::restrict_builds`). Each query is held to DuckDB with the
restriction forced on and with every runtime filter switched off, over a streamed Parquet probe
and an in-memory one, with NULL keys on both sides, an emptied build, and a `LEFT` join that must
not be restricted.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 3
_ROWS_PER_FILE = 100_000
_PARTS = 2_000
_SUPPS = 10

_QUERIES = [
    # q9's shape: the `partsupp` build is restricted to the green parts.
    "SELECT l_flag, count(*) AS n, sum(l_qty * ps_cost) AS v "
    "FROM lineitem JOIN part ON l_pk = p_pk JOIN partsupp ON l_pk = ps_pk AND l_sk = ps_sk "
    "WHERE p_name LIKE '%green%' GROUP BY l_flag",
    # The same with the joins in the other order on the spine.
    "SELECT count(*) AS n, sum(ps_cost) AS v "
    "FROM lineitem JOIN partsupp ON l_pk = ps_pk AND l_sk = ps_sk JOIN part ON l_pk = p_pk "
    "WHERE p_name LIKE '%green%'",
    # A semi join provides.
    "SELECT count(*) AS n, sum(ps_cost) AS v "
    "FROM lineitem JOIN partsupp ON l_pk = ps_pk AND l_sk = ps_sk "
    "WHERE l_pk IN (SELECT p_pk FROM part WHERE p_name LIKE '%green%')",
    # No part matches: the restricted build is emptied, and the answer is empty.
    "SELECT count(*) AS n, sum(ps_cost) AS v "
    "FROM lineitem JOIN part ON l_pk = p_pk JOIN partsupp ON l_pk = ps_pk AND l_sk = ps_sk "
    "WHERE p_name = 'no such part'",
    # A LEFT join keeps unmatched probe rows, so its build must not be restricted.
    "SELECT count(*) AS n, count(ps_cost) AS m, sum(ps_cost) AS v "
    "FROM lineitem JOIN part ON l_pk = p_pk LEFT JOIN partsupp ON l_pk = ps_pk AND l_sk = ps_sk "
    "WHERE p_name LIKE '%green%'",
]


def _lineitem(part: int) -> pa.Table:
    rows = range(part * _ROWS_PER_FILE, (part + 1) * _ROWS_PER_FILE)
    return pa.table(
        {
            # A few NULL keys, and some part keys past every part row.
            "l_pk": pa.array(
                [None if i % 991 == 0 else (i * 7) % (_PARTS + 500) for i in rows], pa.int64()
            ),
            "l_sk": pa.array([(i * 13) % _SUPPS for i in rows], pa.int64()),
            "l_qty": pa.array([float(i % 50) for i in rows]),
            "l_flag": pa.array(["AFRN"[i % 4] for i in rows]),
        }
    )


_PART = pa.table(
    {
        "p_pk": pa.array(range(_PARTS), pa.int64()),
        "p_name": pa.array(["forest green" if i % 3 else "navy" for i in range(_PARTS)]),
    }
)

# Every (part, supplier) pair a lineitem row can carry, plus a few NULL part keys.
_PARTSUPP = pa.table(
    {
        "ps_pk": pa.array(
            [None if i % 409 == 0 else i // _SUPPS for i in range(_PARTS * _SUPPS)], pa.int64()
        ),
        "ps_sk": pa.array([i % _SUPPS for i in range(_PARTS * _SUPPS)], pa.int64()),
        "ps_cost": pa.array([float(i % 97) for i in range(_PARTS * _SUPPS)]),
    }
)


@pytest.fixture(scope="module")
def lineitem_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineitem_build_restriction")
    for part in range(_FILES):
        pq.write_table(_lineitem(part), root / f"part-{part}.parquet", row_group_size=25_000)
    return root


@pytest.mark.parametrize("streamed", [True, False], ids=["parquet", "memory"])
@pytest.mark.parametrize("switch", ["force", "0"])
@pytest.mark.parametrize("query", _QUERIES)
def test_a_restricted_build_matches_duckdb(
    duck, lineitem_dir, monkeypatch, streamed, switch, query
):
    monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", switch)
    lineitem = pa.concat_tables(_lineitem(p) for p in range(_FILES))
    duck.register("lineitem", lineitem)
    duck.register("part", _PART)
    duck.register("partsupp", _PARTSUPP)
    session = bt.Session()
    session.register(
        "lineitem",
        bt.read.parquet(str(lineitem_dir / "*.parquet")) if streamed else bt.from_arrow(lineitem),
    )
    session.register("part", _PART)
    session.register("partsupp", _PARTSUPP)
    assert_same_for_query(session.sql(query).collect(), duck.sql(query), query)
