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


def test_compact_into_a_partitioned_layout_leaves_no_stale_flat_parts(tmp_path):
    """Compaction removes what it replaced even when the *layout* changed.

    The replaced files sat flat at the root while the rewritten ones live in ``g=v/``
    directories, so every old file shares a basename with some new one. Comparing by
    basename kept them all, and the next read unioned their rows back in.
    """
    out = f"{tmp_path}/p"
    bt.from_arrow(pa.table({"g": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})).write.parquet(
        out, max_rows_per_file=1
    )
    assert _n_files(out) == 4

    bt.compact(out, by="g", format="parquet")
    assert _n_files(out) == 0, "the flat part files were replaced, so none may survive"
    assert sorted(p.name for p in (tmp_path / "p").glob("g=*")) == ["g=a", "g=b"]
    back = bt.read(out, format="parquet").sort("v").to_pydict()
    assert back["v"] == [1, 2, 3, 4], "a surviving stale file would duplicate rows"
    assert back["g"] == ["a", "b", "a", "b"]


def test_compact_carries_an_existing_hive_layout_forward(tmp_path):
    """Compaction changes file sizes, not how the table is organized.

    Reading a partitioned tree and writing it back flat leaves every row intact and
    destroys the layout the table is queried by, so the next partition-pruned scan reads
    all of it.
    """
    out = f"{tmp_path}/h"
    bt.from_arrow(pa.table({"dt": ["x", "x", "y", "y"], "v": [1, 2, 3, 4]})).write.parquet(
        out, partition_by=["dt"], max_rows_per_file=1
    )
    root = tmp_path / "h"
    assert len(list(root.glob("dt=*/*.parquet"))) == 4

    bt.compact(out, num_files=1, format="parquet")
    assert sorted(p.name for p in root.glob("dt=*")) == ["dt=x", "dt=y"]
    assert len(list(root.glob("dt=*/*.parquet"))) == 2, "one file per partition after compaction"
    assert _n_files(out) == 0, "nothing was flattened to the root"

    back = bt.read(out, format="parquet").sort("v").to_pydict()
    assert back["v"] == [1, 2, 3, 4]
    assert back["dt"] == ["x", "x", "y", "y"]


def test_compact_can_still_be_told_to_repartition_differently(tmp_path):
    out = f"{tmp_path}/r"
    bt.from_arrow(pa.table({"dt": ["x", "y"], "g": ["a", "a"], "v": [1, 2]})).write.parquet(
        out, partition_by=["dt"]
    )
    bt.compact(out, by="g", format="parquet")
    root = tmp_path / "r"
    assert sorted(p.name for p in root.glob("*=*")) == ["g=a"]
    assert bt.read(out, format="parquet").count() == 2


def test_compact_refuses_to_flatten_a_partitioned_directory_it_cannot_read_back(tmp_path):
    """Only the Parquet reader recovers a Hive partition column from the directory name.

    For every other format a rewrite would silently drop the partitioning *and* the column
    the table is organized by, so it is refused with the reason rather than performed.
    """
    out = f"{tmp_path}/c"
    bt.from_arrow(pa.table({"dt": ["a", "a", "b"], "v": [1, 2, 3]})).write.csv(
        out, partition_by=["dt"], max_rows_per_file=1
    )
    with pytest.raises(PlanError, match="does not recover partition columns"):
        bt.compact(out, num_files=1, format="csv")
    # Nothing was touched.
    assert sorted(p.name for p in (tmp_path / "c").glob("dt=*")) == ["dt=a", "dt=b"]


def test_compacting_a_flat_directory_of_any_format_still_works(tmp_path):
    out = f"{tmp_path}/f"
    bt.from_arrow(pa.table({"v": [1, 2, 3]})).write.csv(out, max_rows_per_file=1)
    bt.compact(out, num_files=1, format="csv")
    assert sorted(bt.read.csv(out).to_pydict()["v"]) == [1, 2, 3]


def test_compact_can_cluster_the_rewritten_rows(tmp_path):
    """`sort_by` reaches the rewrite, so a compaction can tighten min/max bounds too.

    The rows are being rewritten anyway; clustering them is what makes the *next* query
    skip files, which is half the reason to compact at all.
    """
    out = f"{tmp_path}/s"
    bt.from_arrow(pa.table({"x": [3, 1, 2, 4]})).write.parquet(out, max_rows_per_file=1)
    bt.compact(out, num_files=1, format="parquet", sort_by=["x"])
    assert _n_files(out) == 1
    assert bt.read.parquet(out).to_pydict() == {"x": [1, 2, 3, 4]}
