"""Differential tests for the sort paths that skip a permutation or most of the candidates.

Two shapes, both checked against DuckDB **in order**:

* `SELECT k FROM t ORDER BY k` — every output column is a key, so the parallel sample-sort
  sorts each range's values themselves (`radix_sort::sorted_values`, and
  `packed_multi_sorted_values` for several keys) instead of a permutation.
  Rows equal on the key are then indistinguishable, which is what makes an ordered comparison
  of the whole result well defined without a tie-break column.
* `ORDER BY lead, … LIMIT k` over many morsels — the morsels share a rank bound
  (`ops::RankBound`), so a morsel selected late hands the merge only rows that can still win.
  The leading key ties heavily *at* the cut-off, where a skip that was not strict would drop a
  row the answer needs; a unique trailing key makes DuckDB's answer the one total order.

The fixtures are over the parallel thresholds (> 131,072 rows), so the sample-sort and the
morsel-wise top-N are the paths that run.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

N = 200_000

_DIRECTIONS = ["ASC NULLS LAST", "ASC NULLS FIRST", "DESC NULLS LAST", "DESC NULLS FIRST"]


def _keys() -> pa.Table:
    return pa.table(
        {
            "f": pa.array(
                [None if i % 101 == 0 else ((i * 7919) % 9_000) / 4.0 - 100.0 for i in range(N)]
            ),
            "i": pa.array(
                [None if i % 89 == 0 else (i * 104_729) % 50_000 - 20_000 for i in range(N)],
                pa.int64(),
            ),
            "n": pa.array([(i * 31) % 1_000 for i in range(N)], pa.int32()),
        }
    )


@pytest.mark.parametrize("columns", [["f"], ["i"], ["n"], ["n", "i"], ["i", "n"]])
@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_a_key_only_sort_matches_duckdb_in_order(duck, columns, direction):
    """One key or two; with two, the second runs the opposite way so the fields differ."""
    t = _keys()
    duck.register("t", t)
    session = bt.Session()
    session.register("t", t)
    flipped = "DESC" if direction.startswith("ASC") else "ASC"
    order = ", ".join(
        f"{c} {direction if j == 0 else flipped + ' NULLS FIRST'}" for j, c in enumerate(columns)
    )
    sql = f"SELECT {', '.join(columns)} FROM t ORDER BY {order}"
    assert_same_ordered(session.sql(sql).collect(), duck.sql(sql))


def _topn() -> pa.Table:
    return pa.table(
        {
            "s": pa.array(
                [
                    None
                    if i % 97 == 0
                    else ("shared-prefix-x" if i % 13 == 0 else f"v{(i * 7) % 900:03d}")
                    for i in range(N)
                ]
            ),
            "i": pa.array(
                [None if i % 61 == 0 else (i * 11) % 2_000 - 1_000 for i in range(N)], pa.int64()
            ),
            "f": pa.array([((i * 3) % 4_000) / 8.0 for i in range(N)]),
            "id": pa.array(range(N), pa.int64()),
        }
    )


@pytest.mark.parametrize("lead", ["s", "i", "f"])
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("k", [1, 25, 300])
def test_a_bounded_top_n_matches_duckdb_in_order(duck, lead, direction, k):
    t = _topn()
    duck.register("t", t)
    session = bt.Session()
    session.register("t", t)
    for sql in (
        # One key, and only the key out, so ties are identical rows: the single-key selection.
        f"SELECT {lead} FROM t ORDER BY {lead} {direction} LIMIT {k}",
        f"SELECT {lead}, id FROM t ORDER BY {lead} {direction}, id LIMIT {k}",
        f"SELECT * FROM t ORDER BY {lead} {direction}, id DESC LIMIT {k}",
    ):
        assert_same_ordered(session.sql(sql).collect(), duck.sql(sql))
