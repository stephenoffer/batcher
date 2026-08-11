"""Options survive the branches of `ds.write` that re-enter it on a derived dataset.

Three branches restart the call on a different `Dataset`: `sort_by` sorts first,
`replace_where` unions the kept rows with the new ones, and an expression `partition_by`
derives its key column. Each one hand-writes the argument list it forwards, and an option
left off that list is dropped in silence — `distributed` in particular, on the
copy-on-write `replace_where` path, which rewrites the whole table and is therefore the
write that least wants to run on the driver alone.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.manifest import WriteManifest

pytestmark = pytest.mark.unit


@pytest.fixture
def capture(monkeypatch):
    """Replace `_write` with a recorder, so a write plans but never executes."""
    seen: list[dict] = []

    def _fake(plan, sources, columns, path, fmt, **kwargs):
        seen.append({"path": path, "fmt": fmt, **kwargs})
        return WriteManifest()

    monkeypatch.setattr("batcher.api.terminal._write", _fake)
    return seen


def test_sort_by_forwards_auto_compact(capture):
    bt.from_pydict({"x": [2, 1]}).write.parquet("out", sort_by=["x"], auto_compact=True)
    assert capture[-1]["auto_compact"] is True


def test_sort_by_forwards_the_rest_of_the_layout(capture):
    # `resume` is deliberately absent: a sorted plan refuses it (row-to-file assignment
    # is not stable across runs), and that refusal is checked in the resume tests.
    bt.from_pydict({"x": [2, 1]}).write.parquet(
        "out", sort_by=["x"], max_rows_per_file=7, num_workers=3, distributed=True
    )
    call = capture[-1]
    assert call["max_rows_per_file"] == 7
    assert call["num_workers"] == 3
    assert call["distributed"] is True


def test_an_expression_partition_key_forwards_the_whole_call(capture):
    bt.from_pydict({"x": [1, 2]}).write.parquet(
        "out",
        partition_by=[(bt.col("x") % 2).alias("bucket")],
        max_rows_per_file=5,
        num_workers=2,
        auto_compact=True,
        distributed=True,
    )
    call = capture[-1]
    assert call["partition_by"] == ["bucket"]
    assert call["max_rows_per_file"] == 5
    assert call["num_workers"] == 2
    assert call["auto_compact"] is True
    assert call["distributed"] is True


def test_replace_where_forwards_distributed(capture, tmp_path):
    # The copy-on-write path rewrites the entire table, so running it on the driver is
    # the one thing it must not silently do.
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"dt": ["a", "b"], "v": [1, 2]}), path)
    bt.from_pydict({"dt": ["a"], "v": [9]}).write.parquet(
        path, replace_where=bt.col("dt") == "a", distributed=True, num_workers=4
    )
    call = capture[-1]
    assert call["distributed"] is True
    assert call["num_workers"] == 4


def test_replace_where_forwards_auto_compact(capture, tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"dt": ["a", "b"], "v": [1, 2]}), path)
    bt.from_pydict({"dt": ["a"], "v": [9]}).write.parquet(
        path, replace_where=bt.col("dt") == "a", auto_compact=True
    )
    assert capture[-1]["auto_compact"] is True
