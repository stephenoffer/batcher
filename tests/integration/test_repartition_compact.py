"""`Dataset.repartition` (output layout) and `bt.compact` (in-place small-file fix)."""

from __future__ import annotations

import glob

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _n_files(path: str) -> int:
    return len(glob.glob(f"{path}/*.parquet"))


def test_repartition_num_files(tmp_path):
    out = f"{tmp_path}/u"
    bt.from_arrow(pa.table({"id": list(range(100))})).repartition(num_files=4).write.parquet(out)
    assert _n_files(out) == 4
    assert sorted(bt.read.parquet(out).collect().to_pydict()["id"]) == list(range(100))


def test_repartition_by_column_hive_partitions(tmp_path):
    out = f"{tmp_path}/h"
    bt.from_arrow(pa.table({"dt": ["a", "a", "b"], "v": [1, 2, 3]})).repartition(
        by="dt"
    ).write.parquet(out)
    # One Hive subdir per distinct partition value.
    assert sorted(p.name for p in (tmp_path / "h").glob("dt=*")) == ["dt=a", "dt=b"]


def test_repartition_target_size_coalesces(tmp_path):
    out = f"{tmp_path}/s"
    # 50k small rows; a tiny target → multiple files, but far fewer than per-row.
    bt.from_arrow(pa.table({"id": list(range(50_000))})).repartition(
        target_size_mb=0.05
    ).write.parquet(out)
    assert _n_files(out) >= 2


def test_repartition_rejects_conflicting_options():
    ds = bt.from_arrow(pa.table({"a": [1]}))
    with pytest.raises(PlanError):
        ds.repartition(num_files=2, target_size_mb=10)
    with pytest.raises(PlanError):
        ds.repartition(num_files=0)
    with pytest.raises(PlanError):
        ds.repartition()


def test_compact_reduces_files_and_removes_stale(tmp_path):
    out = f"{tmp_path}/t"
    bt.from_arrow(pa.table({"id": list(range(20))})).write.parquet(out, max_rows_per_file=1)
    assert _n_files(out) == 20
    manifest = bt.compact(out, num_files=2, format="parquet")
    assert manifest.num_files == 2
    assert _n_files(out) == 2  # stale 18 part-files were removed
    assert sorted(bt.read.parquet(out).collect().to_pydict()["id"]) == list(range(20))


def test_compact_target_size_default(tmp_path):
    out = f"{tmp_path}/t2"
    bt.from_arrow(pa.table({"id": list(range(30))})).write.parquet(out, max_rows_per_file=1)
    bt.compact(out, format="parquet")  # default target_size_mb coalesces to 1 file
    assert _n_files(out) == 1
    assert sorted(bt.read.parquet(out).collect().to_pydict()["id"]) == list(range(30))
