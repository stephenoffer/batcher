"""A merge-on-read Hudi table must not report its base-file footers as its exact row count.

`HudiSource.row_count` summed the base files' Parquet footers and `statistics()` marked the
sum exact, which is what `count()` answers from without executing. A merge-on-read slice's
log files hold inserts the base file lacks and deletes it still carries, so the footer sum
is a stale count: deleted rows counted, log-file inserts missed (BT-257).

A real merge-on-read table needs a Spark or Flink writer to produce, so the file slices are
stubbed at the one call the source makes for them; the footer read is stubbed too, so the
copy-on-write control reaches a number without a file on disk.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.lakehouse import hudi as hudi_mod

pytestmark = pytest.mark.unit


class _Slice:
    def __init__(self, logs: list[str]) -> None:
        self._logs = logs

    def base_file_relative_path(self) -> str:
        return "p/base.parquet"

    def log_files_relative_paths(self) -> list[str]:
        return self._logs


def _source(monkeypatch, slices: list[_Slice]) -> hudi_mod.HudiSource:
    # `HudiSource` uses `__slots__`, so the stubs go on the class and module.
    monkeypatch.setattr(hudi_mod.HudiSource, "_file_slices", lambda self, predicate=None: slices)
    monkeypatch.setattr(hudi_mod, "_footer_rows", lambda uri, paths: [100] * len(paths))
    monkeypatch.setattr(hudi_mod.HudiSource, "_partition_keys", lambda self: ())
    return hudi_mod.HudiSource("file:///tmp/t")


def test_a_copy_on_write_table_counts_from_its_footers(monkeypatch) -> None:
    # Positive control: the decline below is about the log files, not the stubs.
    source = _source(monkeypatch, [_Slice([]), _Slice([])])
    assert source.row_count() == 200
    stats = source.statistics()
    assert stats is not None and stats.exact_rows and stats.row_count == 200


def test_a_slice_with_log_files_declines_the_count(monkeypatch) -> None:
    source = _source(monkeypatch, [_Slice([]), _Slice([".base.log.1_0-1-1"])])
    assert source.row_count() is None
    assert source.statistics() is None  # nothing may claim an exact count of 200
