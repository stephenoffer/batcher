"""Metadata for the partitioned-Parquet and warehouse connectors, and its Kyber wiring.

Two connectors that reached the estimator nearly blind now surface full metadata:

* **`ParquetDatasetSource`** — the PB-scale Hive-partitioned reader mined only an exact row
  count while the flat `ParquetSource` mined the footers for per-column bounds, byte size,
  row-group count, and partition keys. It now runs the same extractor, and the stats reach
  the Kyber storage shortcuts (partition pruning keys on `partition_keys`).
* **`BigQuerySource`** — a `table=` read now reads `row_count`/`size_bytes` from the free
  ``__TABLES__`` metadata, mapped to an exact `SourceStatistics`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.formats.sql.bigquery import BigQuerySource, _tables_statistics
from batcher.io.formats.structured.parquet.dataset import ParquetDatasetSource

pytestmark = pytest.mark.unit


# --- partitioned Parquet --------------------------------------------------------


@pytest.fixture
def partitioned(tmp_path):
    """A two-partition Parquet tree with a numeric data column."""
    path = str(tmp_path / "ds")
    bt.from_pydict({"region": ["us", "us", "eu", "eu"], "amt": [10, 20, 30, 40]}).write.parquet(
        path, partition_by=["region"]
    )
    return path


def test_partitioned_statistics_full_footer(partitioned) -> None:
    """The partitioned reader surfaces row count, byte size, row groups, and column bounds."""
    stats = ParquetDatasetSource(partitioned).statistics()
    assert stats is not None
    assert stats.row_count == 4
    assert stats.exact_rows is True
    assert stats.byte_size and stats.byte_size > 0
    assert stats.row_group_count and stats.row_group_count >= 1
    amt = stats.columns.get("amt")
    assert amt is not None and amt.min == 10 and amt.max == 40
    assert amt.null_count_is_exact  # footer null counts are exact


def test_partitioned_statistics_records_partition_keys(partitioned) -> None:
    """The Hive partition column is recorded — it is what partition pruning keys on."""
    stats = ParquetDatasetSource(partitioned).statistics()
    assert stats is not None
    assert "region" in stats.partition_keys


def test_partitioned_statistics_reach_kyber_shortcuts(partitioned) -> None:
    """The collected stats flow into the Kyber storage shortcuts end to end."""
    from batcher.api.source_stats import collect_source_stats
    from batcher.kyber.shortcuts import storage

    stats = collect_source_stats([ParquetDatasetSource(partitioned)], None)
    assert storage.row_count(stats) == 4
    assert storage.total_bytes(stats) > 0
    assert storage.is_partitioned(stats)
    assert storage.partition_keys(stats) == ("region",)
    assert storage.row_group_count(stats) is not None


def test_partitioned_large_dataset_skips_footer_sweep(partitioned, monkeypatch) -> None:
    """Above the footer-plan ceiling nothing sweeps a footer per file — including the count.

    This used to assert that ``row_count()`` "is still cheaply available" above the ceiling.
    It is not: `pyarrow.dataset.count_rows` opens a footer per data file, which is the same
    O(files) driver sweep the ceiling exists to refuse — a million object-store round trips
    before a task launches. It now declines like every other O(files) path, and
    `statistics()` answers with what it can state without the sweep.
    """
    import batcher.io.formats.structured.parquet.dataset as mod

    monkeypatch.setattr(mod, "_MAX_FOOTER_PLAN_FILES", 0)
    assert ParquetDatasetSource(partitioned).row_count() is None

    stats = ParquetDatasetSource(partitioned).statistics()
    assert stats is not None, "declining the sweep must not mean declining every fact"
    assert stats.exact_rows is False, "nothing above the ceiling may pass as an exact count"
    assert stats.byte_size and stats.byte_size > 0  # the listing already reported the sizes
    assert "region" in stats.partition_keys  # the directory names cost nothing to read
    assert stats.columns == {}, "a sampled bound is not provable, so none is published"


# --- BigQuery __TABLES__ --------------------------------------------------------


def test_bigquery_tables_statistics_mapping() -> None:
    """A ``__TABLES__`` row maps to an exact row count and byte size."""
    stats = _tables_statistics([{"row_count": 1_000_000, "size_bytes": 5_242_880}])
    assert stats is not None
    assert stats.row_count == 1_000_000
    assert stats.byte_size == 5_242_880
    assert stats.exact_rows is True


def test_bigquery_tables_statistics_empty() -> None:
    """A missing table (no ``__TABLES__`` row) yields no statistics."""
    assert _tables_statistics([]) is None


def test_bigquery_table_read_queries_tables(monkeypatch) -> None:
    """A `table=` read reads its stats from ``__TABLES__`` via the metadata seam."""
    captured = {}

    def fake_query(self, sql):
        captured["sql"] = sql
        return [{"row_count": 500, "size_bytes": 2048}]

    monkeypatch.setattr(BigQuerySource, "_run_metadata_query", fake_query)
    stats = BigQuerySource(project="p", table="proj.ds.orders").statistics()
    assert stats is not None
    assert stats.row_count == 500 and stats.byte_size == 2048
    assert "__TABLES__" in captured["sql"]
    assert "orders" in captured["sql"]


def test_bigquery_query_read_has_no_catalog(monkeypatch) -> None:
    """A `query=` read is an arbitrary expression with no ``__TABLES__`` entry."""
    monkeypatch.setattr(
        BigQuerySource, "_run_metadata_query", lambda self, sql: pytest.fail("should not query")
    )
    assert BigQuerySource(project="p", query="SELECT 1").statistics() is None


def test_bigquery_unqualified_table_declines(monkeypatch) -> None:
    """A table name that is not ``project.dataset.table`` cannot locate ``__TABLES__``."""
    monkeypatch.setattr(
        BigQuerySource, "_run_metadata_query", lambda self, sql: pytest.fail("should not query")
    )
    assert BigQuerySource(project="p", table="bare").statistics() is None


def test_bigquery_field_reads_row_or_dict() -> None:
    """The row accessor tolerates both a mapping and an attribute-style BigQuery Row."""
    from batcher.io.formats.sql.bigquery import _bq_field

    class Row:
        row_count = 7

    assert _bq_field({"row_count": 5}, "row_count") == 5
    assert _bq_field(Row(), "row_count") == 7
    assert _bq_field({}, "row_count") is None
