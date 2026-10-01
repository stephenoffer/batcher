"""A ``FULL JOIN`` whose build side is far larger than its probe side, vs DuckDB.

Where nothing above the join can observe its row order — a count, a grouped aggregate — the
streaming executor builds on whichever side is materially smaller
(`bc_interp::join_par::order_free_swap_pays`). A full join emits every unmatched row of *both*
sides, so a swap that lost track of which side a column came from would still return the right
number of rows and the wrong counts per side: the per-side ``count``/``sum`` columns here are
what catch that. Keys carry nulls on both sides (a null key matches nothing, so its row is
unmatched on its own side), duplicates on the build side, and unmatched keys on each side.

The probe side is 30,000 rows and the build side 200,000, so the swap's ratio and floor are both
cleared and the build side alone is past the 65,536 rows below which nothing shards. A root
``ORDER BY ... LIMIT`` over the same join is compared in order, where no swap may happen.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


@pytest.fixture(scope="module")
def tables() -> dict[str, pa.Table]:
    small = pa.table(
        {
            "k": pa.array([None if i % 50 == 7 else i * 3 for i in range(30_000)], pa.int64()),
            "v": pa.array(range(30_000), pa.int64()),
        }
    )
    big = pa.table(
        {
            "k": pa.array(
                [None if i % 70 == 3 else i % 60_000 for i in range(200_000)], pa.int64()
            ),
            "d": pa.array(range(200_000), pa.int64()),
        }
    )
    return {"small": small, "big": big}


_QUERIES = {
    "global": "SELECT count(*) AS n, count(s.k) AS nl, count(b.k) AS nr, sum(v) AS sv, "
    "sum(d) AS sd FROM small s FULL OUTER JOIN big b ON s.k = b.k",
    "grouped": "SELECT b.k % 7 AS g, count(*) AS n, count(s.k) AS nl, sum(v) AS sv "
    "FROM small s FULL OUTER JOIN big b ON s.k = b.k GROUP BY b.k % 7",
    "reversed": "SELECT count(*) AS n, count(s.k) AS nl, count(b.k) AS nr "
    "FROM big b FULL OUTER JOIN small s ON s.k = b.k",
    "right": "SELECT count(*) AS n, count(s.k) AS nl, sum(d) AS sd "
    "FROM small s RIGHT JOIN big b ON s.k = b.k",
    "left": "SELECT count(*) AS n, count(b.k) AS nr, sum(v) AS sv "
    "FROM small s LEFT JOIN big b ON s.k = b.k",
}

_ORDERED = (
    "SELECT s.k AS lk, v, b.k AS rk, d FROM small s FULL OUTER JOIN big b ON s.k = b.k "
    "ORDER BY d NULLS FIRST, v NULLS FIRST LIMIT 40"
)


def _session(tables: dict[str, pa.Table]) -> bt.Session:
    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
    return s


@pytest.mark.parametrize("name", sorted(_QUERIES))
def test_an_order_free_lopsided_outer_join_matches_duckdb(duck, tables, name) -> None:
    for t, v in tables.items():
        duck.register(t, v)
    sql = _QUERIES[name]
    assert_same(_session(tables).sql(sql).collect(), duck.sql(sql))


def test_an_ordered_limit_over_the_lopsided_full_join_keeps_the_oracle_order(duck, tables) -> None:
    for t, v in tables.items():
        duck.register(t, v)
    assert_same_ordered(_session(tables).sql(_ORDERED).collect(), duck.sql(_ORDERED))
