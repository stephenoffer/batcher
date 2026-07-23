"""A finished write must be distinguishable from a half-written one.

`FileSink.commit` was a no-op, so a distributed write that died at 90% left a directory of
perfectly valid Parquet files that read back cleanly and silently short. Nothing downstream
could tell the difference. `commit` now publishes a `_SUCCESS` marker after every shard's
manifest has merged.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt
from batcher.io import ParquetSink, WriteManifest, WrittenFile

pytestmark = pytest.mark.unit


def test_a_completed_directory_write_publishes_a_success_marker(tmp_path):
    """A partitioned write is a directory of data files, which is what a marker describes."""
    out = str(tmp_path / "out")
    bt.from_pydict({"x": [1, 2, 3], "p": ["a", "a", "b"]}).write.parquet(out, partition_by=["p"])
    assert os.path.exists(os.path.join(out, "_SUCCESS"))


def test_a_single_file_write_gets_no_marker_inside_it(tmp_path):
    """`write.parquet("out.parquet")` produces one file AT the path; `path/_SUCCESS` would be
    a nonsense location inside that file."""
    out = str(tmp_path / "one.parquet")
    bt.from_pydict({"x": [1, 2, 3]}).write.parquet(out)
    assert os.path.isfile(out)
    assert not os.path.exists(os.path.join(out, "_SUCCESS"))


def test_the_marker_is_absent_when_nothing_was_written(tmp_path):
    """An empty manifest must not mark an absent directory complete."""
    out = str(tmp_path / "never")
    ParquetSink().commit(WriteManifest(), out)
    assert not os.path.exists(os.path.join(out, "_SUCCESS"))


def test_the_marker_does_not_join_the_data_on_read_back(tmp_path):
    """Readers skip `_`-prefixed files; a marker that got read as Parquet would raise."""
    out = str(tmp_path / "out")
    bt.from_pydict({"x": [1, 2, 3], "p": ["a", "a", "b"]}).write.parquet(out, partition_by=["p"])
    assert sorted(bt.read(out, format="parquet").collect().to_pydict()["x"]) == [1, 2, 3]


def test_a_marker_failure_does_not_fail_an_otherwise_successful_write(tmp_path, monkeypatch):
    """The data is already durable at this point; a marker is best-effort."""
    sink = ParquetSink()
    monkeypatch.setattr(
        type(sink), "_resolve", lambda _self, _p: (_ for _ in ()).throw(OSError("read-only fs"))
    )
    out = str(tmp_path / "out")
    sink.commit(WriteManifest((WrittenFile(out + "/part-0.parquet", 3, 10),)), out)
