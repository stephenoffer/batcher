"""Rewriting a path must invalidate the statistics this session cached for it.

`collect_source_stats` memoizes a source's footer/manifest statistics per session,
because re-reading every row-group footer on each query costs seconds on a large table.
The memo is keyed by path, and a path is not immutable: every copy-on-write pattern —
`write.merge`, `ds.scd.*`, `ds.scd.apply_changes` — reads a table and rewrites it, often
many times in one session.

Statistics are not merely a cost hint. A column's min/max is a **zone map**: the optimizer
prunes predicates and whole join sides with it, and answers some terminal operations from
it without executing. So serving a stale entry does not produce a slower plan, it produces
a *wrong answer* — silently. These tests pin the invalidation that prevents it.

The bug they cover was invisible to `write.merge` and `ds.scd.type1`, whose results happen
to be identical whether or not the stale zone map prunes the anti-join. It surfaced only
where the anti-join must actually remove a row: a CDC delete.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.api.orchestration import _SOURCE_STATS_CACHE, invalidate_source_stats

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clean_cache():
    """The cache is session-global; isolate each test from its neighbours."""
    _SOURCE_STATS_CACHE.clear()
    yield
    _SOURCE_STATS_CACHE.clear()


def test_a_write_evicts_the_cached_statistics_for_that_path(tmp_path):
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1]}).write(path, format="parquet")
    bt.read.parquet(path).count()  # populates the memo
    assert f"parquet:{path}" in _SOURCE_STATS_CACHE
    bt.from_pydict({"id": [2]}).write(path, format="parquet")
    assert f"parquet:{path}" not in _SOURCE_STATS_CACHE


def test_invalidate_uses_the_same_key_the_reader_caches_under(tmp_path):
    """A key mismatch would make the eviction a silent no-op."""
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1]}).write(path, format="parquet")
    bt.read.parquet(path).count()
    assert _SOURCE_STATS_CACHE
    invalidate_source_stats(path, "parquet")
    assert not _SOURCE_STATS_CACHE


def test_a_filter_after_an_overwrite_sees_the_new_zone_map(tmp_path):
    """A stale zone map would say `r` is always "EU" and prune this predicate to false."""
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"r": ["EU", "EU"]}).write(path, format="parquet")
    assert bt.read.parquet(path).filter(bt.col("r") == "US").count() == 0
    bt.from_pydict({"r": ["EU", "US"]}).write(path, format="parquet")
    assert bt.read.parquet(path).filter(bt.col("r") == "US").count() == 1


def test_a_count_after_an_overwrite_is_not_answered_from_the_old_metadata(tmp_path):
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1, 2, 3]}).write(path, format="parquet")
    assert bt.read.parquet(path).count() == 3
    bt.from_pydict({"id": [1]}).write(path, format="parquet")
    assert bt.read.parquet(path).count() == 1


def test_an_anti_join_against_a_rewritten_table_still_removes_matching_rows(tmp_path):
    """The shape a CDC delete lowers to: the row to remove is outside the stale zone map."""
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1]}).write(path, format="parquet")
    bt.read.parquet(path).count()  # cache a zone map claiming `id` is exactly [1, 1]
    bt.from_pydict({"id": [1, 2]}).write(path, format="parquet")

    probe = bt.from_pydict({"id": [2]})
    survivors = bt.read.parquet(path).join(probe, on="id", how="anti").to_pydict()
    assert survivors == {"id": [1]}


def test_a_repeated_copy_on_write_merge_loop_converges(tmp_path):
    """Many read-modify-write cycles over one path, the shape every ETL job has."""
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1], "v": ["a"]}).write(path, format="parquet")
    for i in range(2, 6):
        bt.from_pydict({"id": [i], "v": [chr(96 + i)]}).write.merge(path, on="id")
    got = pq.read_table(path).to_pydict()
    assert sorted(got["id"]) == [1, 2, 3, 4, 5]


def test_an_in_memory_source_is_never_cached(tmp_path):
    """Its identity is shape-based, so two different tables would collide."""
    t = pa.table({"id": pa.array([1], pa.int64())})
    bt.from_arrow(t).count()
    assert not any(k.startswith("mem:") for k in _SOURCE_STATS_CACHE)
