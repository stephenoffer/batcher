"""A selective build side over the streamed relation, run as its own stage, returns DuckDB's rows.

On the chunked path the driving relation is streamed through a join's probe side, so a join that
builds on it is swapped and its *other* input is hashed. When a filter makes the driving side
the small one (TPC-H q12: `orders JOIN lineitem` with `lineitem` filtered to a sliver), that
build side runs first as its own streamed stage and the rest of the plan reads its result
(`api.orchestration.chunked.run_staged_build`). Each query is held to DuckDB, and the test
asserts the stage was taken where the filter is selective and not where it is not.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FILES = 4
_ROWS_PER_FILE = 150_000
_ORDERS = 120_000

# q12's shape: the filter on the streamed `lineitem` keeps ~1% of it.
_SELECTIVE = [
    "SELECT l_mode, sum(CASE WHEN o_prio = 'HIGH' THEN 1 ELSE 0 END) AS hi, count(*) AS n "
    "FROM orders JOIN lineitem ON o_ok = l_ok "
    "WHERE l_mode IN ('MAIL', 'SHIP') AND l_day BETWEEN 100 AND 104 GROUP BY l_mode",
    # NULL join keys on the staged side.
    "SELECT o_prio, count(*) AS n, sum(l_qty) AS q FROM orders JOIN lineitem ON o_ok = l_ok "
    "WHERE l_day = 7 GROUP BY o_prio",
]

# No row survives the filter, which statistics may settle before any stage is considered: held
# to DuckDB only.
_EMPTY = (
    "SELECT count(*) AS n, sum(l_qty) AS q FROM orders JOIN lineitem ON o_ok = l_ok "
    "WHERE l_day > 10000"
)

# Keeps half of `lineitem`: not staged, and still correct.
_NOT_SELECTIVE = (
    "SELECT o_prio, count(*) AS n, sum(l_qty) AS q FROM orders JOIN lineitem ON o_ok = l_ok "
    "WHERE l_day < 182 GROUP BY o_prio"
)


def _lineitem(part: int) -> pa.Table:
    rows = range(part * _ROWS_PER_FILE, (part + 1) * _ROWS_PER_FILE)
    return pa.table(
        {
            "l_ok": pa.array(
                [None if i % 503 == 0 else (i * 31) % (_ORDERS + 1_000) for i in rows], pa.int64()
            ),
            "l_mode": pa.array(["MAIL", "SHIP", "AIR", "RAIL", "TRUCK"][i % 5] for i in rows),
            "l_day": pa.array([(i * 7) % 365 for i in rows], pa.int64()),
            "l_qty": pa.array([float(i % 50) for i in rows]),
        }
    )


_ORDERS_TABLE = pa.table(
    {
        "o_ok": pa.array(range(_ORDERS), pa.int64()),
        "o_prio": pa.array(["HIGH", "MED", "LOW"][i % 3] for i in range(_ORDERS)),
    }
)


@pytest.fixture(scope="module")
def lineitem_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineitem_staged_build")
    for part in range(_FILES):
        pq.write_table(_lineitem(part), root / f"part-{part}.parquet", row_group_size=30_000)
    return root


@pytest.fixture
def staged(monkeypatch):
    """Force the chunked path on these inputs and record each staging verdict."""
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "chunk_worthy", lambda *_a, **_k: True)
    monkeypatch.setattr(chunked, "units_worthy", lambda *_a, **_k: True)
    verdicts: list[bool] = []
    real = chunked._selective_build

    def spy(*args, **kwargs):
        found = real(*args, **kwargs)
        verdicts.append(found is not None)
        return found

    monkeypatch.setattr(chunked, "_selective_build", spy)
    return verdicts


def _run(duck, lineitem_dir, query):
    duck.register("lineitem", pa.concat_tables(_lineitem(p) for p in range(_FILES)))
    duck.register("orders", _ORDERS_TABLE)
    session = bt.Session()
    session.register("lineitem", bt.read.parquet(str(lineitem_dir / "*.parquet")))
    session.register("orders", _ORDERS_TABLE)
    assert_same_for_query(session.sql(query).collect(), duck.sql(query), query)


@pytest.mark.parametrize("query", _SELECTIVE)
def test_a_selective_build_side_is_staged_and_matches_duckdb(duck, lineitem_dir, staged, query):
    _run(duck, lineitem_dir, query)
    assert any(staged), "the selective build side was not staged"


def test_a_non_selective_build_side_is_not_staged(duck, lineitem_dir, staged):
    _run(duck, lineitem_dir, _NOT_SELECTIVE)
    assert staged, "the chunked path was not taken, so the verdict was never asked"
    assert not any(staged)


def test_an_empty_build_side_matches_duckdb(duck, lineitem_dir, staged):
    _run(duck, lineitem_dir, _EMPTY)
