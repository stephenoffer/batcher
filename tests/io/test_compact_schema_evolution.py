"""Compaction changes a table's file sizes, never its content.

`compact` reads a directory, rewrites it into fewer files, and **deletes the originals**.
So any column the read fails to see is not merely absent from that one result — it is gone
from the table. `read` types a directory from its *first* file, which meant a table that
gained a column midway was compacted down to the first file's columns and the rest were
destroyed, with every row still present and no error anywhere.
"""

from __future__ import annotations

import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.io


@pytest.fixture
def evolved_flat(tmp_path):
    """Three files: `a`, then `a,b`, then `a,b,c` — a table that grew twice."""
    out = str(tmp_path / "t")
    os.makedirs(out)
    pq.write_table(pa.table({"a": pa.array([1, 2], pa.int32())}), f"{out}/f1.parquet")
    pq.write_table(pa.table({"a": pa.array([3], pa.int64()), "b": ["x"]}), f"{out}/f2.parquet")
    pq.write_table(
        pa.table({"a": pa.array([4], pa.int64()), "b": ["y"], "c": [1.5]}), f"{out}/f3.parquet"
    )
    return out


def test_compaction_keeps_every_column_the_files_hold(evolved_flat):
    bt.compact(evolved_flat)
    assert len(glob.glob(f"{evolved_flat}/*.parquet")) == 1
    back = bt.read.parquet(evolved_flat, schema_mode="union").to_pydict()
    assert sorted(back["a"]) == [1, 2, 3, 4]
    assert sorted(v for v in back["b"] if v is not None) == ["x", "y"]
    assert [v for v in back["c"] if v is not None] == [1.5]


@pytest.fixture
def evolved_hive(tmp_path):
    """A Hive tree whose two partitions disagree: one has `b`, the other does not."""
    out = str(tmp_path / "t")
    for i, cols in enumerate([{"a": [1]}, {"a": [2], "b": ["x"]}]):
        os.makedirs(f"{out}/g=k{i}")
        pq.write_table(pa.table(cols), f"{out}/g=k{i}/p.parquet")
    return out


def test_union_on_a_hive_tree_keeps_the_partition_column_too(evolved_hive):
    """Union used to force the flat reader, trading the evolved column for the key.

    `schema_mode="union"` was not on the partition-aware reader's option list, so asking
    for it silently fell back to the flat reader — which reads every file but recovers no
    partition column at all. You could have `b` or you could have `g`, never both.
    """
    assert sorted(bt.read.parquet(evolved_hive, schema_mode="union").columns) == ["a", "b", "g"]


def test_compacting_a_hive_tree_keeps_the_layout_and_every_column(evolved_hive):
    bt.compact(evolved_hive)
    back = bt.read.parquet(evolved_hive, schema_mode="union")
    assert sorted(back.columns) == ["a", "b", "g"]
    assert sorted(back.to_pydict()["a"]) == [1, 2]
    dirs = {os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{evolved_hive}/*/*.parquet")}
    assert dirs == {"g=k0", "g=k1"}


def test_a_directory_whose_files_agree_is_unaffected(tmp_path):
    """The default path must not change: same rows, same columns, fewer files."""
    out = str(tmp_path / "t")
    os.makedirs(out)
    for i in range(6):
        pq.write_table(pa.table({"a": [i], "b": [str(i)]}), f"{out}/f{i}.parquet")
    bt.compact(out, num_files=1)
    back = bt.read.parquet(out).to_pydict()
    assert len(glob.glob(f"{out}/*.parquet")) == 1
    assert sorted(back["a"]) == list(range(6))
    assert sorted(back["b"]) == [str(i) for i in range(6)]
