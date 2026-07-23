"""Metadata caches must not survive the file they describe being rewritten.

Every metadata cache in the read path — Parquet footers, per-file row counts, the session
`SourceStatistics` memo — was keyed on the **path**, justified by "Parquet is write-once".
That holds for an immutable lake and not for what pipelines do: `FileSink` writes
deterministic names (`part-00000.parquet`), so a re-run overwrites its own output, and
just as often the path is rewritten by the job upstream, a Spark run, or a compaction.

The consequences are not "slightly stale" — they are wrong answers with no error:

* a cached **row count** answers `count()` with the previous file's total while
  `collect()` returns the new rows, so the two contradict each other;
* a cached **zone map** prunes against bounds the data no longer has, and the scan
  returns *no rows at all* for a predicate that every row satisfies;
* a cached **footer** hands `RowGroupSplit` row-group offsets from the old file, which
  then index into the middle of the new bytes.

The fix keys each cache on the file's identity (`path, size, mtime`) instead. These tests
are the regression: each one fails against a path-only key.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.unit


@pytest.fixture
def rewritten(tmp_path):
    """A path written once, read (warming every cache), then overwritten in place."""

    def _write(values: list[int]) -> str:
        path = str(tmp_path / "part-00000.parquet")
        pq.write_table(pa.table({"x": values}), path)
        return path

    return _write


def test_count_agrees_with_collect_after_an_overwrite(rewritten) -> None:
    """The sharpest form: a metadata shortcut contradicting the data it summarizes."""
    path = rewritten(list(range(100)))
    assert bt.read.parquet(path).count() == 100  # warm the caches

    path = rewritten(list(range(5)))
    ds = bt.read.parquet(path)

    assert ds.count() == 5
    assert ds.collect().num_rows == 5


def test_a_stale_zone_map_does_not_prune_the_new_data_away(rewritten) -> None:
    """The worst form: an empty result for a predicate every row satisfies.

    Values 0..99 give a max of 99, so `x > 500` prunes the file — correctly. Rewriting
    with 1000..1009 makes every row match, but a zone map cached under the path still says
    max 99, and the scan skips the file entirely. No error, no rows.
    """
    path = rewritten(list(range(100)))
    assert bt.read.parquet(path).filter(col("x") > 500).count() == 0  # correct, and warms

    path = rewritten(list(range(1000, 1010)))

    assert bt.read.parquet(path).filter(col("x") > 500).count() == 10


def test_the_schema_reflects_the_new_file(rewritten) -> None:
    path = rewritten([1, 2, 3])
    assert bt.read.parquet(path).count() == 3

    other = str(pytest.importorskip("pathlib").Path(path))
    pq.write_table(pa.table({"x": [1.5, 2.5]}), other)

    assert bt.read.parquet(other).schema.field("x").type == pa.float64()


def test_the_footer_cache_is_keyed_on_the_file_version(tmp_path) -> None:
    """`RowGroupSplit` passes this metadata straight to `ParquetFile`, so a stale footer
    is not merely a wrong count — it is old offsets read against new bytes."""
    from batcher.io.splits.parquet import _parquet_footer

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(100))}), path)
    assert _parquet_footer(path).num_rows == 100

    pq.write_table(pa.table({"x": list(range(7))}), path)

    assert _parquet_footer(path).num_rows == 7


def test_the_row_count_cache_is_keyed_on_the_file_version(tmp_path) -> None:
    from batcher.io.formats.structured.parquet.source import ParquetSource

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(100))}), path)
    assert ParquetSource(path).row_count() == 100

    pq.write_table(pa.table({"x": list(range(9))}), path)

    assert ParquetSource(path).row_count() == 9


def test_a_source_reports_a_version_that_changes_with_its_content(tmp_path) -> None:
    from batcher.io.formats.structured.parquet.source import ParquetSource

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": [1]}), path)
    before = ParquetSource(path).stats_version()

    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    after = ParquetSource(path).stats_version()

    assert before is not None
    assert before != after


def test_the_version_is_stable_when_nothing_changes(tmp_path) -> None:
    """It must not change spuriously, or the memo never hits and the cost it exists to
    avoid comes back."""
    from batcher.io.formats.structured.parquet.source import ParquetSource

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)

    assert ParquetSource(path).stats_version() == ParquetSource(path).stats_version()


def test_repeated_reads_of_an_unchanged_file_still_hit_the_memo(tmp_path) -> None:
    """The memo must keep working — a version check that never hits would trade a
    correctness bug for a performance one."""
    from batcher.api import source_stats
    from batcher.io.formats.structured.parquet.source import ParquetSource

    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(50))}), path)
    source_stats._SOURCE_STATS_CACHE.clear()

    source = ParquetSource(path)
    source_stats.collect_source_stats([source], None)
    size_after_first = len(source_stats._SOURCE_STATS_CACHE)
    source_stats.collect_source_stats([ParquetSource(path)], None)

    assert size_after_first == 1
    assert len(source_stats._SOURCE_STATS_CACHE) == 1, "a second read added a second entry"


def test_the_native_reader_does_not_use_a_stale_footer(tmp_path) -> None:
    """The Rust footer cache had the same defect, with a nastier symptom.

    `bc-io`'s `meta_cache` stored `(size, metadata)` keyed by URI and served hits without
    checking the size. Reading an overwritten file then handed the decoder row-group
    offsets from the *previous* footer, which raises
    ``Parquet error: Column cannot have more than one dictionary`` — a corrupt-file error
    on a perfectly valid file, which is worse than a wrong number because it sends anyone
    debugging it to look at the data.
    """
    from batcher._internal.native import engine

    native = engine()
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(100))}), path, row_group_size=25)
    assert sum(b.num_rows for b in native.read_parquet(path, [0, 1, 2, 3], None, 8192)) == 100

    # Overwrite with different content *and* a different row-group layout, so stale
    # offsets cannot accidentally still be valid.
    pq.write_table(pa.table({"x": list(range(1000, 1007))}), path, row_group_size=3)

    batches = native.read_parquet(path, [0, 1, 2], None, 8192)
    values = [v for b in batches for v in b.column("x").to_pylist()]

    assert values == list(range(1000, 1007))
