"""Resumable writes — ``resume=True`` skips already-committed output (G1/G2).

Bypassing the Ray object store costs Batcher Ray's free lineage fault-tolerance, so
resumability is provided explicitly: combined with atomic writes (a file exists only
if fully committed), ``resume=True`` skips shards whose file is already present, so a
job re-run after a crash or spot preemption finishes only the unwritten shards. Ray
Data needs external bookkeeping for this.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt


def _ds() -> bt.Dataset:
    return bt.from_arrow(pa.table({"g": [1, 1, 2, 2], "v": [10, 20, 30, 40]}))


def test_resume_skips_committed_and_rewrites_lost_partition(tmp_path):
    path = str(tmp_path / "out")
    _ds().write.parquet(path, partition_by=["g"])
    f1, f2 = f"{path}/g=1/part-00000.parquet", f"{path}/g=2/part-00000.parquet"
    assert os.path.exists(f1) and os.path.exists(f2)
    kept_mtime = os.path.getmtime(f1)

    # Simulate a crash that lost the g=2 shard, then resume.
    os.remove(f2)
    _ds().write.parquet(path, partition_by=["g"], resume=True)

    assert os.path.getmtime(f1) == kept_mtime  # g=1 was committed → skipped, untouched
    assert os.path.exists(f2)  # g=2 was missing → rewritten
    assert pq.read_table(f1).column("v").to_pylist() == [10, 20]
    assert pq.read_table(f2).column("v").to_pylist() == [30, 40]


def test_resume_on_complete_output_rewrites_nothing(tmp_path):
    path = str(tmp_path / "out")
    _ds().write.parquet(path, partition_by=["g"])
    f1, f2 = f"{path}/g=1/part-00000.parquet", f"{path}/g=2/part-00000.parquet"
    m1, m2 = os.path.getmtime(f1), os.path.getmtime(f2)

    _ds().write.parquet(path, partition_by=["g"], resume=True)
    assert os.path.getmtime(f1) == m1
    assert os.path.getmtime(f2) == m2


def test_resume_single_file_skips_existing(tmp_path):
    path = str(tmp_path / "out.parquet")
    bt.from_arrow(pa.table({"v": [1, 2, 3]})).write.parquet(path)
    mtime = os.path.getmtime(path)
    bt.from_arrow(pa.table({"v": [1, 2, 3]})).write.parquet(path, resume=True)
    assert os.path.getmtime(path) == mtime  # untouched


def test_without_resume_rewrites(tmp_path):
    # Default (resume=False) overwrites — the new write replaces the file.
    path = str(tmp_path / "out.parquet")
    bt.from_arrow(pa.table({"v": [1, 2, 3]})).write.parquet(path)
    first = pq.read_table(path).num_rows
    bt.from_arrow(pa.table({"v": [9, 9]})).write.parquet(path)
    assert first == 3
    assert pq.read_table(path).column("v").to_pylist() == [9, 9]  # overwritten


def test_resume_rejects_nondeterministic_plan(tmp_path):
    # Resume identifies done work by file position, so it is only exactly-once on a
    # deterministic plan. A group_by/join/sort can place different rows in the same
    # part file between runs — resuming it risks dropping/duplicating data, so the
    # write must refuse up front rather than silently corrupt the output.
    import pytest

    from batcher import col
    from batcher._internal.errors import PlanError

    agg = _ds().group_by("g").agg(s=col("v").sum())
    with pytest.raises(PlanError, match="resume=True"):
        agg.write.parquet(str(tmp_path / "agg"), resume=True)


def test_resume_allows_deterministic_etl_plan(tmp_path):
    # The streamable ETL path (read → filter/select/map_batches → write) keeps a stable
    # row→file assignment, so resume stays allowed.
    from batcher import col

    path = str(tmp_path / "etl")
    manifest = _ds().filter(col("v") > 15).select("g", "v").write.parquet(path, resume=True)
    assert manifest.num_files >= 1
