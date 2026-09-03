"""Planned row-group splits are cached per (file, predicate, target size) -- soundly.

Building the splits for a Parquet source is pure Python over the file's physical layout:
decode `.statistics` for every row group and every predicate column, prune, pack the
survivors into runs. The footer cache removes the I/O under that and none of the work, so
it was redone on every query over the same immutable file. On TPC-H sf1000 (1,000 files,
49,000 row groups) that was 666 ms of driver time per query with all 65 nodes idle, because
no task can be submitted until the last split exists.

Caching it is only safe if the key is complete. The two ways to get this wrong are the two
things these tests pin: serving one predicate's surviving row groups to a *different*
predicate, and serving a rewritten file's splits from the version before it was rewritten.
Both would silently lose rows -- a pruned row group is never read, never filtered, and never
counted -- so each is checked against DuckDB rather than against Batcher's own other answer.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.io

_ROWS = 40_000
_ROW_GROUP = 500  # 80 row groups, so pruning has something to prune


@pytest.fixture
def graded(tmp_path) -> str:
    """One file whose `a` is strictly ascending, so every row group has distinct bounds."""
    path = str(tmp_path / "graded.parquet")
    pq.write_table(
        pa.table({"a": pa.array(range(_ROWS), pa.int64())}), path, row_group_size=_ROW_GROUP
    )
    return path


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(0, 10), (5_000, 5_001), (20_000, 39_999), (0, 39_999), (150, 160), (0, 10)],
)
def test_each_predicate_gets_its_own_splits(graded, lo, hi):
    """The same source, read repeatedly under different ranges, still matches DuckDB.

    The repeated `(0, 10)` is deliberate: it is the case that *must* hit the cache, run after
    four other predicates have written entries for the same file.
    """
    got = (
        bt.read.parquet(graded)
        .filter((col("a") >= lo) & (col("a") <= hi))
        .agg(n=bt.count(), s=col("a").sum())
        .collect()
        .to_pydict()
    )
    want = duckdb.sql(
        f"SELECT count(*) AS n, sum(a) AS s FROM '{graded}' WHERE a >= {lo} AND a <= {hi}"
    ).fetchall()[0]
    assert (got["n"][0], got["s"][0]) == want


def test_the_cache_actually_hits(graded, monkeypatch):
    """The positive control. Without it every assertion above passes on a cache that never
    returns anything, and this file would be pinning a memo that does not exist.

    Proved by making the uncached path *impossible*: after the first planning pass, reading
    a footer raises. A second call that still answers can only have been served from the
    cache, and one that raises tells you the memo never took.
    """
    from batcher.io.splits import parquet as mod

    first = mod.parquet_row_group_splits(graded, None, None, None)
    assert first, "no splits were planned, so this file exercises nothing"

    def refuse(*_a, **_k):
        raise AssertionError("the footer was re-read, so the split cache did not hit")

    monkeypatch.setattr(mod, "_parquet_footer", refuse)
    second = mod.parquet_row_group_splits(graded, None, None, None)
    assert second == first
    assert second is not first, "the cache must hand back a fresh list, not its own"


def test_a_rewritten_file_is_not_served_its_old_splits(tmp_path):
    """The identity is `(path, size, mtime)`, not the path -- a deterministic sink overwrites
    its own output, and a path-keyed entry would prune the new file against the old bounds."""
    path = str(tmp_path / "rewritten.parquet")
    pq.write_table(
        pa.table({"a": pa.array(range(_ROWS), pa.int64())}), path, row_group_size=_ROW_GROUP
    )
    first = bt.read.parquet(path).filter(col("a") < 100).agg(n=bt.count()).collect().to_pydict()
    assert first["n"][0] == 100

    # Same path, disjoint values: every row group the old bounds admitted is now wrong.
    pq.write_table(
        pa.table({"a": pa.array(range(_ROWS, 2 * _ROWS), pa.int64())}),
        path,
        row_group_size=_ROW_GROUP,
    )
    got = bt.read.parquet(path).filter(col("a") < 100).agg(n=bt.count()).collect().to_pydict()
    want = duckdb.sql(f"SELECT count(*) FROM '{path}' WHERE a < 100").fetchall()[0][0]
    assert got["n"][0] == want == 0
