"""A dataset too large to count exactly still reports a size to the planner.

Every scan path refuses an O(files) footer sweep past `_MAX_FOOTER_PLAN_FILES`, because a
million metadata round trips on the driver *is* the query. Refusing the sweep is right;
reporting nothing in its place was not, and that is the shape these tests pin. A table big
enough to decline the sweep is exactly the one whose join order, build side, spill budget,
and worker fan-out are worth getting right — and above the ceiling all four were sized on a
default.

The estimate must be close enough to plan on and must never be mistaken for an exact count.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.source import ParquetSource
from batcher.io.stats.row_estimate import estimate_rows_from_footer_sample


@pytest.fixture
def many_files(tmp_path):
    """120 Parquet files of deliberately unequal size, and their exact total row count."""
    total = 0
    for i in range(120):
        rows = 100 + i
        total += rows
        pq.write_table(pa.table({"v": list(range(rows))}), tmp_path / f"f{i:04d}.parquet")
    return str(tmp_path), total


def test_estimate_is_close_without_reading_every_footer(many_files):
    path, total = many_files
    source = ParquetSource(path)
    read = []

    def counted(file_path: str) -> int | None:
        read.append(file_path)
        return source._file_row_count(file_path)

    estimate = estimate_rows_from_footer_sample(
        source._fs,
        source._files(),
        counted,
        sample_files=16,
    )
    assert len(read) == 16, "the cost must be the sample size, not the file count"
    assert abs(estimate - total) / total < 0.05


def test_sample_spans_the_listing_not_its_head(many_files):
    """Sampling the front of a listing would estimate a partitioned table from one day."""
    path, _ = many_files
    source = ParquetSource(path)
    read: list[str] = []
    estimate_rows_from_footer_sample(
        source._fs,
        files := source._files(),
        lambda p: (read.append(p), source._file_row_count(p))[1],
        sample_files=8,
    )
    assert read[0] == files[0]
    assert read[-1] >= files[len(files) * 3 // 4], "the sample must reach the tail"


def test_small_dataset_prefers_the_exact_count(many_files):
    """At or below the sample size the exact answer is affordable, so none is estimated."""
    path, _ = many_files
    source = ParquetSource(path)
    assert (
        estimate_rows_from_footer_sample(
            source._fs,
            source._files()[:10],
            source._file_row_count,
            sample_files=64,
        )
        is None
    )


def test_statistics_above_the_ceiling_estimate_and_stay_inexact(many_files, monkeypatch):
    """Past the sweep ceiling `statistics()` reports an advisory count, never an exact one."""
    path, total = many_files
    monkeypatch.setattr(ParquetSource, "_too_many_files_to_sweep", lambda self: True)
    stats = ParquetSource(path).statistics()
    assert stats.exact_rows is False
    assert abs(stats.row_count - total) / total < 0.05
    assert stats.byte_size > 0


def test_unreadable_footers_yield_no_estimate_rather_than_a_wrong_one(many_files):
    path, _ = many_files
    source = ParquetSource(path)
    assert (
        estimate_rows_from_footer_sample(
            source._fs,
            source._files(),
            lambda _path: None,
            sample_files=8,
        )
        is None
    )


# ---- the Hive-partitioned reader, which has its own discovery ---------------------------


@pytest.fixture
def hive_tree(tmp_path):
    """A partitioned tree with 300 unequal files, and its exact total row count."""
    import batcher.io.formats.structured.parquet.dataset as mod

    total = 0
    for day in range(30):
        directory = tmp_path / f"day={day:03d}"
        directory.mkdir()
        for f in range(10):
            rows = 3 + (day + f) % 7
            total += rows
            pq.write_table(pa.table({"v": list(range(rows))}), directory / f"p{f}.parquet")
    return str(tmp_path), total, mod


def test_the_partitioned_reader_discovers_the_tree_once(hive_tree):
    """Discovery is a recursive listing of the whole tree; six methods must share one.

    Rebuilding it per call turned the most expensive thing this class does into a per-method
    cost, and it is the one thing the class exists to keep off the driver.
    """
    path, _, mod = hive_tree
    source = mod.ParquetDatasetSource(path)
    built = []
    discover = mod.ParquetDatasetSource._discover
    mod.ParquetDatasetSource._discover = lambda self: (built.append(1), discover(self))[1]
    try:
        source.schema()
        source.row_count()
        source.statistics()
        source.splits()
    finally:
        mod.ParquetDatasetSource._discover = discover
    assert len(built) == 1, f"the tree was walked {len(built)} times"


def test_a_pickled_source_does_not_carry_the_listing(hive_tree):
    """The discovered dataset *is* the file listing, so it must not ride to a worker."""
    import pickle

    path, _, mod = hive_tree
    source = mod.ParquetDatasetSource(path)
    source.schema()  # populates the cache
    restored = pickle.loads(pickle.dumps(source))
    assert restored._built is None
    assert restored.schema() == source.schema()


def test_the_partitioned_reader_estimates_above_its_ceiling(hive_tree, monkeypatch):
    path, total, mod = hive_tree
    monkeypatch.setattr(mod, "_MAX_FOOTER_PLAN_FILES", 50)  # the tree has 300 files
    source = mod.ParquetDatasetSource(path)
    assert source.row_count() is None, "the exact count is an O(files) footer sweep"
    stats = source.statistics()
    assert stats.exact_rows is False
    assert abs(stats.row_count - total) / total < 0.25
    assert stats.partition_keys == ("day",)
