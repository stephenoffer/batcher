"""`require_success=True` refuses a directory the producing write never finished.

Every data file is published atomically, so no file is ever half-written -- but a run that
died at 90% leaves a directory of *valid* files that reads back cleanly and silently short.
That is the one failure per-file atomicity cannot cover, which is why `FileSink.commit`
publishes a `_SUCCESS` marker only after every shard's manifest is merged. This is the half
that reads it.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt
from batcher._internal.errors import IOError as BatcherIOError

pytestmark = pytest.mark.integration

FORMATS = ["parquet", "csv", "json", "arrow", "orc", "avro"]


def _written(tmp_path, fmt: str) -> str:
    out = str(tmp_path / fmt)
    getattr(bt.from_pydict({"v": [1, 2, 3]}).write, fmt)(out, max_rows_per_file=1)
    return out


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_completed_write_reads_with_the_check_on(tmp_path, fmt):
    out = _written(tmp_path, fmt)
    assert bt.read(out, format=fmt, require_success=True).count() == 3


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_unmarked_directory_is_refused(tmp_path, fmt):
    out = _written(tmp_path, fmt)
    os.remove(os.path.join(out, "_SUCCESS"))
    with pytest.raises(BatcherIOError, match="_SUCCESS"):
        bt.read(out, format=fmt, require_success=True).count()


def test_the_check_is_off_by_default(tmp_path):
    # A directory Batcher did not write has no marker and is not thereby incomplete.
    out = _written(tmp_path, "parquet")
    os.remove(os.path.join(out, "_SUCCESS"))
    assert bt.read.parquet(out).count() == 3


def test_a_single_file_needs_no_marker(tmp_path):
    # There is nothing partial about one atomically-written file.
    path = str(tmp_path / "one.parquet")
    bt.from_pydict({"v": [1]}).write.parquet(path)
    assert bt.read.parquet(path, require_success=True).count() == 1


def test_the_message_names_the_path_and_the_way_out(tmp_path):
    out = _written(tmp_path, "parquet")
    os.remove(os.path.join(out, "_SUCCESS"))
    with pytest.raises(BatcherIOError) as excinfo:
        bt.read.parquet(out, require_success=True).count()
    message = str(excinfo.value)
    assert out in message
    assert "require_success" in message


def test_a_partitioned_write_is_marked_too(tmp_path):
    out = str(tmp_path / "part")
    bt.from_pydict({"g": ["a", "b"], "v": [1, 2]}).write.parquet(out, partition_by=["g"])
    assert bt.read.parquet(out, require_success=True).count() == 2


def test_a_partitioned_read_keeps_its_partition_columns_with_the_check_on(tmp_path):
    """The check is a property of the *path*, so it must not change which reader runs.

    Wired into the reader instead, it forced the flat Parquet reader and quietly dropped
    the partition columns -- trading one silent loss for another.
    """
    out = str(tmp_path / "part")
    bt.from_pydict({"g": ["a", "b"], "v": [1, 2]}).write.parquet(out, partition_by=["g"])
    back = bt.read.parquet(out, require_success=True).sort("v")
    assert sorted(back.columns) == ["g", "v"]
    assert back.to_pydict() == {"v": [1, 2], "g": ["a", "b"]}


def test_an_absent_path_is_not_reported_as_unmarked(tmp_path):
    # It is not a directory, so the marker check says nothing; the read raises its own
    # "does not exist", which is the error the caller can act on.
    with pytest.raises(BatcherIOError) as excinfo:
        bt.read.parquet(str(tmp_path / "never-there"), require_success=True).count()
    assert "_SUCCESS" not in str(excinfo.value)
