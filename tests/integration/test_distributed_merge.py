"""A merge on a cluster must produce exactly what the same merge produces on one core.

This is invariant 7 (`single-node == distributed via mergeable algebra`) applied to
``MERGE``. It holds for a structural reason, and the tests are here to keep that reason
true: a merge is **not** a new operator. `api.merge.compose` builds it out of joins, a
chained ``CASE``, and a ``union`` — so the thing that runs on a cluster is an ordinary
relational plan, executed by the ordinary distributed executor. There is no second merge
implementation with its own semantics to drift.

What *is* merge-specific, and what these tests actually guard:

* the **pruning** decision (which files to rewrite) is made once, on the driver, from file
  statistics — so every worker must agree on it, and a worker must never read a file the
  driver pruned away;
* the **swap** (write new files, delete the ones they replace) is a driver-side commit over
  the manifest the workers return, so no data crosses the driver and no worker deletes
  anything;
* the new files carry a per-merge **token**, and that token has to survive the trip to the
  worker — a sink rebuilt on the worker without it would name its shard ``part-00000.parquet``
  and overwrite a file that pruning had deliberately preserved. That is silent data loss, and
  it is the failure this file exists to catch.
"""

from __future__ import annotations

import glob

import pyarrow as pa
import pytest

import batcher as bt
from batcher import lit, source_col, target_col

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    from conftest import init_test_ray, shutdown_test_ray

    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _target(path: str, rows: int = 2_000, rows_per_file: int = 100) -> pa.Table:
    table = pa.table(
        {
            "id": pa.array(range(rows), pa.int64()),
            "v": pa.array([i * 10 for i in range(rows)], pa.int64()),
            "tag": pa.array([f"t{i % 7}" for i in range(rows)]),
        }
    )
    bt.from_arrow(table).write.parquet(path, max_rows_per_file=rows_per_file)
    return table


def _rows(path: str) -> dict:
    out = bt.read.parquet(path).collect().to_pydict()
    return {i: (v, t) for i, v, t in zip(out["id"], out["v"], out["tag"], strict=True)}


def _merge(path: str, changes: pa.Table, *, distributed, workers=None):
    builder = bt.from_arrow(changes).write.merge_into(
        path,
        on="id",
        format="parquet",
        distributed=distributed,
        **({"num_workers": workers} if workers else {}),
    )
    return builder.when_matched().update_all().when_not_matched().insert_all().execute()


def test_distributed_merge_equals_single_node(tmp_path):
    """The same merge, one core vs four workers — identical tables."""
    changes = pa.table(
        {
            "id": pa.array([5, 500, 1_500, 9_999], pa.int64()),
            "v": pa.array([-1, -2, -3, -4], pa.int64()),
            "tag": pa.array(["new", "new", "new", "new"]),
        }
    )

    solo = str(tmp_path / "solo")
    _target(solo)
    _merge(solo, changes, distributed=False)

    dist = str(tmp_path / "dist")
    _target(dist)
    _merge(dist, changes, distributed=True, workers=4)

    assert _rows(solo) == _rows(dist)
    # And it really is the merged content, not two identically-broken tables.
    assert _rows(solo)[5] == (-1, "new")
    assert _rows(solo)[9_999] == (-4, "new")  # the new key was inserted
    assert _rows(solo)[6] == (60, "t6")  # an untouched row is untouched


def test_distributed_merge_does_not_clobber_the_files_it_pruned(tmp_path):
    """The token test: a worker must not name its shard over a file pruning preserved.

    Without a per-write token the worker's sink writes ``part-00000.parquet`` — which is a
    *live data file* of the target that this merge never read. The row count is what catches
    it: clobbering one 100-row file loses 100 rows.
    """
    path = str(tmp_path / "t")
    _target(path, rows=2_000, rows_per_file=100)
    before = _rows(path)

    changes = pa.table(
        {
            "id": pa.array([1_950], pa.int64()),  # one key, so exactly one file is touched
            "v": pa.array([-99], pa.int64()),
            "tag": pa.array(["z"]),
        }
    )
    _merge(path, changes, distributed=True, workers=4)

    after = _rows(path)
    assert len(after) == len(before) == 2_000, "rows went missing — a live file was overwritten"
    assert after[1_950] == (-99, "z")
    for key in (0, 99, 100, 999, 1_000, 1_999):
        if key != 1_950:
            assert after[key] == before[key], f"row {key} changed but nothing should have"


def test_distributed_merge_with_every_clause_kind(tmp_path):
    """All three populations, guarded, on the distributed path — vs the single-node result."""
    changes = pa.table(
        {
            "id": pa.array([10, 20, 30, 5_000], pa.int64()),
            "v": pa.array([1, 999_999, 3, 4], pa.int64()),
            "tag": pa.array(["a", "b", "c", "d"]),
        }
    )

    def run(path: str, *, distributed: bool):
        _target(path, rows=200, rows_per_file=25)
        (
            bt.from_arrow(changes)
            .write.merge_into(path, on="id", format="parquet", distributed=distributed)
            .when_matched(source_col("v") > lit(1_000))
            .delete()
            .when_matched(source_col("v") > target_col("v"))
            .update({"v": source_col("v")})
            .when_matched()
            .update({"tag": source_col("tag")})
            .when_not_matched()
            .insert_all()
            .when_not_matched_by_source(target_col("id") < lit(3))
            .update({"tag": lit("expired")})
            .execute()
        )
        return _rows(path)

    solo = run(str(tmp_path / "solo"), distributed=False)
    dist = run(str(tmp_path / "dist"), distributed=True)
    assert solo == dist

    assert 20 not in solo  # the guarded DELETE fired
    assert solo[5_000] == (4, "d")  # the new key was inserted
    assert solo[0][1] == "expired"  # not-matched-by-source, guarded, fired
    assert solo[100][1] == "t2"  # not-matched-by-source, guard false → untouched


def test_repeated_distributed_merges_stay_consistent(tmp_path):
    """Five merges in a row: no duplicate keys, no lost rows, layout intact."""
    path = str(tmp_path / "t")
    _target(path, rows=1_000, rows_per_file=100)

    for i in range(5):
        changes = pa.table(
            {
                "id": pa.array([7, 2_000 + i], pa.int64()),
                "v": pa.array([1_000 + i, i], pa.int64()),
                "tag": pa.array([f"r{i}", f"n{i}"]),
            }
        )
        _merge(path, changes, distributed=True, workers=4)

    rows = _rows(path)
    assert len(rows) == 1_005, "a key was duplicated or a row was lost"
    assert rows[7] == (1_004, "r4"), "the last update did not win"
    assert rows[999] == (9_990, "t5"), "an untouched row was disturbed"
    assert len(glob.glob(f"{path}/*.parquet")) > 1, "the table collapsed to one file"
