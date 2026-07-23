"""Wave-8 write/sink regressions: overwrite must REPLACE, not leave stale files.

A file sink names its shards ``part-NNNNN`` (and Hive ``col=v/…``) from the *current*
write's shape — its shard/chunk count and partition values — not the previous write's.
So overwriting a differently shaped prior output left the extra files behind, and the
next read unioned the stale rows back in. That is silent data corruption, not a missing
feature: ``mode="overwrite"`` claims to replace the output.

Each test fails on the pre-fix behavior (stale files survive the overwrite).
"""

from __future__ import annotations

import pyarrow.dataset as pds
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def test_overwrite_flat_directory_drops_extra_shards(tmp_path):
    out = str(tmp_path / "tbl")
    # 10 rows, 2 per file -> five part files.
    bt.from_pydict({"x": list(range(10))}).write(out, "parquet", max_rows_per_file=2)
    # Overwrite with 4 rows -> two part files; the other three must not survive.
    bt.from_pydict({"x": [100, 101, 102, 103]}).write(
        out, "parquet", mode="overwrite", max_rows_per_file=2
    )
    got = sorted(bt.read(out, format="parquet").collect().to_pydict()["x"])
    assert got == [100, 101, 102, 103]


def test_overwrite_partitioned_drops_removed_partition(tmp_path):
    out = str(tmp_path / "tbl")
    bt.from_pydict({"c": ["a", "b"], "x": [1, 2]}).write(out, "parquet", partition_by=["c"])
    # Overwrite with only partition ``a``: the ``c=b`` rows must be gone.
    bt.from_pydict({"c": ["a"], "x": [99]}).write(
        out, "parquet", mode="overwrite", partition_by=["c"]
    )
    got = pds.dataset(out, format="parquet", partitioning="hive").to_table().to_pydict()
    assert sorted(got["x"]) == [99]


def test_overwrite_single_file_still_replaces(tmp_path):
    out = str(tmp_path / "t.parquet")
    bt.from_pydict({"x": [1, 2, 3]}).write(out, "parquet")
    bt.from_pydict({"x": [9]}).write(out, "parquet", mode="overwrite")
    assert bt.read(out, format="parquet").collect().to_pydict()["x"] == [9]


def test_overwrite_csv_directory_drops_extra_shards(tmp_path):
    out = str(tmp_path / "tbl")
    bt.from_pydict({"x": list(range(10))}).write(out, "csv", max_rows_per_file=2)
    bt.from_pydict({"x": [1, 2]}).write(out, "csv", mode="overwrite", max_rows_per_file=2)
    assert sorted(bt.read(out, format="csv").collect().to_pydict()["x"]) == [1, 2]


def test_resume_overwrite_keeps_existing_files(tmp_path):
    # `resume=True` intentionally keeps already-present (complete) files; the stale-file
    # prune must NOT run for it, or a re-run would delete work it meant to skip.
    out = str(tmp_path / "tbl")
    bt.from_pydict({"x": list(range(10))}).write(out, "parquet", max_rows_per_file=2)
    before = sorted(p.name for p in (tmp_path / "tbl").iterdir())
    bt.from_pydict({"x": [1, 2, 3, 4]}).write(
        out, "parquet", mode="overwrite", resume=True, max_rows_per_file=2
    )
    after = sorted(p.name for p in (tmp_path / "tbl").iterdir())
    assert before == after
