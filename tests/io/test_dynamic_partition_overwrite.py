"""``mode="overwrite_partitions"`` — replace only the partitions the new data covers.

Reloading one day into a table holding years of them is the single most common
partitioned write, and a plain ``"overwrite"`` does the wrong thing at full speed: it
replaces the output, so the other years are deleted. Spark spells the fix as a session
conf (``partitionOverwriteMode="dynamic"``) and Hive as ``INSERT OVERWRITE``; both name
this mode here.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _seed(path: str) -> None:
    bt.from_pydict({"dt": ["a", "b", "c"], "v": [1, 2, 3]}).write.parquet(path, partition_by=["dt"])


def test_only_the_covered_partition_is_replaced(tmp_path):
    out = str(tmp_path / "t")
    _seed(out)
    bt.from_pydict({"dt": ["b"], "v": [99]}).write.parquet(
        out, partition_by=["dt"], mode="overwrite_partitions"
    )
    got = bt.read.parquet(out).sort("dt").to_pydict()
    assert got["dt"] == ["a", "b", "c"]
    assert got["v"] == [1, 99, 3]


def test_a_plain_overwrite_still_replaces_everything(tmp_path):
    out = str(tmp_path / "t")
    _seed(out)
    bt.from_pydict({"dt": ["b"], "v": [77]}).write.parquet(
        out, partition_by=["dt"], mode="overwrite"
    )
    assert bt.read.parquet(out).to_pydict() == {"v": [77], "dt": ["b"]}


def test_a_new_partition_is_added_without_disturbing_the_others(tmp_path):
    out = str(tmp_path / "t")
    _seed(out)
    bt.from_pydict({"dt": ["d"], "v": [4]}).write.parquet(
        out, partition_by=["dt"], mode="overwrite_partitions"
    )
    got = bt.read.parquet(out).sort("dt").to_pydict()
    assert got["dt"] == ["a", "b", "c", "d"]
    assert got["v"] == [1, 2, 3, 4]


def test_a_replaced_partition_does_not_keep_its_old_rows(tmp_path):
    # The partition's own stale part files must go, or the reload duplicates rows.
    out = str(tmp_path / "t")
    bt.from_pydict({"dt": ["b", "b", "b"], "v": [1, 2, 3]}).write.parquet(
        out, partition_by=["dt"], max_rows_per_file=1
    )
    assert len(list((tmp_path / "t" / "dt=b").glob("*.parquet"))) == 3
    bt.from_pydict({"dt": ["b"], "v": [9]}).write.parquet(
        out, partition_by=["dt"], mode="overwrite_partitions"
    )
    assert bt.read.parquet(out).to_pydict() == {"v": [9], "dt": ["b"]}


@pytest.mark.parametrize("spelling", ["dynamic", "INSERT_OVERWRITE", "overwritePartitions"])
def test_the_other_engines_spellings_resolve_to_the_same_mode(tmp_path, spelling):
    out = str(tmp_path / spelling)
    _seed(out)
    bt.from_pydict({"dt": ["b"], "v": [99]}).write.parquet(out, partition_by=["dt"], mode=spelling)
    assert sorted(bt.read.parquet(out).to_pydict()["v"]) == [1, 3, 99]


def test_it_needs_partition_columns_to_know_what_a_partition_is(tmp_path):
    out = str(tmp_path / "t")
    with pytest.raises(PlanError, match="partition_by"):
        bt.from_pydict({"v": [1]}).write.parquet(out, mode="overwrite_partitions")


def test_a_transactional_table_is_pointed_at_replace_where(tmp_path):
    out = str(tmp_path / "t")
    with pytest.raises(PlanError, match="replace_where"):
        bt.from_pydict({"dt": ["a"], "v": [1]}).write.delta(
            out, partition_by=["dt"], mode="overwrite_partitions"
        )
