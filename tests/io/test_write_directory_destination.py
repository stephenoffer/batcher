"""A destination that is already a directory is rewritten as one.

Writing a single file *at* a path that is a directory of part files means renaming a temp
file over the directory: `IsADirectoryError` on local disk, and stale files left behind on
an object store. It is reached by `replace_where` on an unpartitioned output and by any
plain rewrite of a sharded one, so the shape of the write is decided from the destination
rather than left to the caller to remember.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def _parts(root) -> list[str]:
    return sorted(p.name for p in root.glob("*.parquet"))


def test_replace_where_over_a_flat_directory_rewrites_the_directory(tmp_path):
    out = str(tmp_path / "flat")
    bt.from_pydict({"k": [1, 2, 3], "v": [1, 2, 3]}).write.parquet(out, max_rows_per_file=1)
    bt.from_pydict({"k": [2], "v": [99]}).write.parquet(out, replace_where=bt.col("k") == 2)
    assert os.path.isdir(out)
    assert bt.read.parquet(out).sort("k").to_pydict() == {"k": [1, 2, 3], "v": [1, 99, 3]}


def test_a_plain_overwrite_of_a_sharded_directory_stays_a_directory(tmp_path):
    out = str(tmp_path / "sharded")
    bt.from_pydict({"v": [1, 2, 3]}).write.parquet(out, max_rows_per_file=1)
    assert len(_parts(tmp_path / "sharded")) == 3
    bt.from_pydict({"v": [7, 8]}).write.parquet(out)
    assert os.path.isdir(out)
    assert _parts(tmp_path / "sharded") == ["part-00000.parquet"]
    assert sorted(bt.read.parquet(out).to_pydict()["v"]) == [7, 8]


def test_a_new_single_file_write_is_still_one_file(tmp_path):
    path = str(tmp_path / "one.parquet")
    bt.from_pydict({"v": [1]}).write.parquet(path)
    assert os.path.isfile(path)
    assert bt.read.parquet(path).to_pydict() == {"v": [1]}


def test_replace_where_over_a_partitioned_directory_keeps_its_layout(tmp_path):
    # The backfill replaces rows, never the table's organization -- and the partition
    # columns must survive, which needs the read to recover them in the first place.
    out = str(tmp_path / "part")
    bt.from_pydict({"dt": ["a", "b", "c"], "v": [1, 2, 3]}).write.parquet(out, partition_by=["dt"])
    bt.from_pydict({"dt": ["b"], "v": [99]}).write.parquet(out, replace_where=bt.col("dt") == "b")
    assert sorted(p.name for p in (tmp_path / "part").glob("dt=*")) == ["dt=a", "dt=b", "dt=c"]
    got = bt.read.parquet(out).sort("dt").to_pydict()
    assert got["dt"] == ["a", "b", "c"]
    assert got["v"] == [1, 99, 3]


def test_replace_where_matching_nothing_leaves_the_table_plus_the_new_rows(tmp_path):
    out = str(tmp_path / "part")
    bt.from_pydict({"dt": ["a"], "v": [1]}).write.parquet(out, partition_by=["dt"])
    bt.from_pydict({"dt": ["z"], "v": [9]}).write.parquet(out, replace_where=bt.col("dt") == "z")
    got = bt.read.parquet(out).sort("dt").to_pydict()
    assert got["dt"] == ["a", "z"]
    assert got["v"] == [1, 9]


def test_single_file_onto_an_existing_directory_says_so(tmp_path):
    """The promise cannot be kept, so it is refused rather than quietly broken.

    Left alone this became a directory rewrite (silently breaking `single_file`) or an
    `IsADirectoryError` from inside a rename three layers down.
    """
    from batcher._internal.errors import PlanError

    out = str(tmp_path / "occupied")
    bt.from_pydict({"v": [1, 2]}).write.parquet(out, max_rows_per_file=1)
    with pytest.raises(PlanError, match="already a directory"):
        bt.from_pydict({"v": [3]}).write.parquet(out, single_file=True)


def test_single_file_to_a_fresh_path_is_unaffected(tmp_path):
    out = str(tmp_path / "fresh.parquet")
    bt.from_pydict({"v": [1, 2]}).write.parquet(out, single_file=True)
    assert os.path.isfile(out)
    assert bt.read.parquet(out).to_pydict() == {"v": [1, 2]}
