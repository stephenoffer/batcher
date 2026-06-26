"""Clustered writes — `write(sort_by=...)` (W7).

Sorting rows before writing tightens each file / row-group's min/max so downstream
zonemap + bloom skipping prunes far more data — the engine-side slice of liquid
clustering (the user picks the keys; there is no managed table service).
"""

from __future__ import annotations

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def test_sort_by_clusters_into_non_overlapping_files(tmp_path):
    # A shuffled column written with sort_by must land in files whose value ranges are
    # disjoint and ascending — exactly what lets a range predicate skip whole files.
    vals = list(range(1000))
    random.Random(0).shuffle(vals)
    path = str(tmp_path / "clustered")
    bt.from_arrow(pa.table({"v": vals})).write.parquet(path, sort_by=["v"], max_rows_per_file=200)
    files = sorted(tmp_path.rglob("*.parquet"))
    ranges = []
    total = 0
    for f in files:
        col = pq.read_table(str(f)).column("v").to_pylist()
        ranges.append((col[0], col[-1]))
        total += len(col)
    assert total == 1000
    assert len(files) == 5
    # Each file's max is below the next file's min: perfectly clustered.
    assert all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1))


def test_sort_by_round_trips_correctly(tmp_path):
    # Clustering must not change the data — only its layout.
    vals = list(range(500))
    random.Random(1).shuffle(vals)
    path = str(tmp_path / "rt")
    bt.from_arrow(pa.table({"v": vals})).write.parquet(path, sort_by=["v"])
    back = sorted(bt.read(path, format="parquet").to_pydict()["v"])
    assert back == list(range(500))


def test_sort_by_rejected_on_unbounded_stream(tmp_path):
    # A global sort has no meaning over an unbounded stream.
    stream = bt.read.rate(rows_per_second=10)
    with pytest.raises(PlanError, match="cannot sort an unbounded stream"):
        stream.write.parquet(str(tmp_path / "s"), sort_by=["value"])
