"""Writing a directory per distinct value of a high-cardinality column is called out.

Hive partitioning buys directory-level skipping at the price of a directory per distinct
value, and that trade inverts once the values are many and small: the write becomes a PUT
per directory and every later query pays a listing over all of them. Partitioning by an id
or a second-resolution timestamp is how people arrive there, and the symptom -- a write
that takes hours and a table that is slow forever after -- never names its cause.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

from batcher._internal.errors import PerformanceWarning
from batcher.io import ParquetSink

pytestmark = pytest.mark.integration


def _write(tmp_path, distinct: int, threshold: int, monkeypatch):
    # The threshold lives with the Hive helpers, which is where the warning is raised —
    # patching the sink module instead would silently no-op and the test would pass
    # against nothing.
    import batcher.io.base._hive as hive

    monkeypatch.setattr(hive, "_HIGH_CARDINALITY_PARTITIONS", threshold)
    table = pa.table({"g": [f"k{i}" for i in range(distinct)], "v": list(range(distinct))})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ParquetSink().write_partitioned(table, str(tmp_path / "out"), partition_by=["g"])
    return [w for w in caught if issubclass(w.category, PerformanceWarning)]


def test_a_reasonable_partition_count_says_nothing(tmp_path, monkeypatch):
    assert _write(tmp_path, distinct=5, threshold=10, monkeypatch=monkeypatch) == []


def test_the_threshold_itself_is_not_over_it(tmp_path, monkeypatch):
    assert _write(tmp_path, distinct=10, threshold=10, monkeypatch=monkeypatch) == []


def test_an_unreasonable_partition_count_is_reported_once(tmp_path, monkeypatch):
    warned = _write(tmp_path, distinct=12, threshold=10, monkeypatch=monkeypatch)
    assert len(warned) == 1


def test_the_message_names_the_count_the_columns_and_a_way_out(tmp_path, monkeypatch):
    (warned,) = _write(tmp_path, distinct=12, threshold=10, monkeypatch=monkeypatch)
    message = str(warned.message)
    assert "12" in message
    assert "'g'" in message
    assert "sort_by" in message


def test_an_unpartitioned_write_is_never_warned_about(tmp_path, monkeypatch):
    import batcher.io.base._hive as hive

    monkeypatch.setattr(hive, "_HIGH_CARDINALITY_PARTITIONS", 1)
    table = pa.table({"v": list(range(50))})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ParquetSink().write_partitioned(table, str(tmp_path / "flat"), max_rows_per_file=1)
    assert [w for w in caught if issubclass(w.category, PerformanceWarning)] == []


def test_the_rows_are_written_correctly_regardless(tmp_path, monkeypatch):
    import batcher as bt

    _write(tmp_path, distinct=12, threshold=10, monkeypatch=monkeypatch)
    assert bt.read.parquet(str(tmp_path / "out")).count() == 12
