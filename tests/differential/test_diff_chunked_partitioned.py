"""A plan whose build side outgrows the envelope, run in key-partitioned passes, matches DuckDB.

`api.orchestration.chunked_sideways.run_partitioned_build` handles the streamed path's one
refusal it can recover from: build sides that do not fit (the streaming executor's do not
spill). It restricts the largest build to `hash(key) mod P == p`, runs the plan's top
aggregate once per `p`, and combines the passes' sums, counts, minima and maxima. That changes
which rows each pass sees, so it is held to DuckDB on the TPC-H q9 shape it exists for -- a
fact table streamed past two dimension builds into a grouped sum -- with NULL keys, a
`count`/`min`/`max` beside the sum, a global aggregate, and a semi join. A spy proves the
passes produced the result. A left join over the oversized build must decline: a probe row
unmatched in one pass would be emitted, null-extended, in every pass.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FACT_FILES = 4
_FACT_ROWS = 60_000
_BIG = 40_000

_PARTITIONED = [
    "SELECT d_g, sum(f_v * b_w) AS s, count(*) AS n, min(f_v) AS lo, max(b_w) AS hi "
    "FROM fact, big, dim WHERE f_k = b_k AND f_d = d_k GROUP BY d_g ORDER BY d_g",
    "SELECT sum(f_v) AS s, count(f_v) AS n FROM fact, big WHERE f_k = b_k AND b_w >= 1",
    "SELECT d_g, count(*) AS n FROM fact, dim WHERE f_d = d_k "
    "AND f_k IN (SELECT b_k FROM big WHERE b_w >= 1) GROUP BY d_g ORDER BY d_g",
]
_DECLINED = (
    "SELECT count(*) AS n, count(b_w) AS m, sum(f_v) AS s FROM fact LEFT JOIN big ON f_k = b_k"
)


def _fact(part: int) -> pa.Table:
    rows = range(part * _FACT_ROWS, (part + 1) * _FACT_ROWS)
    return pa.table(
        {
            "f_k": pa.array(
                [None if i % 211 == 0 else (i * 7) % (_BIG * 2) for i in rows], pa.int64()
            ),
            "f_d": pa.array([i % 50 for i in rows], pa.int64()),
            "f_v": pa.array([float(i % 13) for i in rows]),
        }
    )


_BIG_T = pa.table(
    {
        "b_k": pa.array(list(range(_BIG)), pa.int64()),
        "b_w": pa.array([float(i % 7) for i in range(_BIG)]),
    }
)
_DIM_T = pa.table(
    {"d_k": pa.array(list(range(50)), pa.int64()), "d_g": [f"g{i % 5}" for i in range(50)]}
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("partitioned")
    (root / "fact").mkdir()
    for part in range(_FACT_FILES):
        path = root / "fact" / f"p{part}.parquet"
        pq.write_table(_fact(part), path, row_group_size=20_000)
    pq.write_table(_BIG_T, root / "big.parquet")
    pq.write_table(_DIM_T, root / "dim.parquet")
    return root


@pytest.fixture
def passes(monkeypatch):
    """Force the chunked path under a budget the big build overflows; record the outcomes."""
    from batcher.api.orchestration import chunked

    monkeypatch.setattr(chunked, "units_worthy", lambda _bytes: True)
    monkeypatch.setattr(chunked, "chunk_worthy", lambda _bytes: True)
    monkeypatch.setattr(chunked, "_held_budget", lambda: 400_000)
    monkeypatch.setattr(chunked, "_held_limit", lambda: 1 << 40)
    outcomes: list[int | None] = []
    real = chunked.run_partitioned_build

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        outcomes.append(None if out is None else sum(b.num_rows for b in out))
        return out

    monkeypatch.setattr(chunked, "run_partitioned_build", spy)
    return outcomes


def _session(data_dir) -> bt.Session:
    s = bt.Session()
    s.register("fact", bt.read.parquet(str(data_dir / "fact" / "*.parquet")))
    s.register("big", bt.read.parquet(str(data_dir / "big.parquet")))
    s.register("dim", bt.read.parquet(str(data_dir / "dim.parquet")))
    return s


def _duck(duck):
    duck.register("fact", pa.concat_tables(_fact(p) for p in range(_FACT_FILES)))
    duck.register("big", _BIG_T)
    duck.register("dim", _DIM_T)
    return duck


@pytest.mark.parametrize("query", _PARTITIONED)
def test_partitioned_passes_match_duckdb(duck, data_dir, passes, query):
    got = _session(data_dir).sql(query).collect()
    assert any(o is not None for o in passes), f"no partitioned pass produced this: {passes}"
    assert_same_for_query(got, _duck(duck).sql(query), query)


def test_a_left_join_over_the_oversized_build_is_not_partitioned(duck, data_dir, passes):
    got = _session(data_dir).sql(_DECLINED).collect()
    assert all(o is None for o in passes), f"the left join was partitioned: {passes}"
    assert_same_for_query(got, _duck(duck).sql(_DECLINED), _DECLINED)
