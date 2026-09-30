"""The distributed single-node fallback reads only the sources its plan scans.

A late adaptive stage runs with every source of the query still bound. TPC-H q15's last stage
is a `Project` over a one-row intermediate, and reading the unscanned `lineitem` beside it --
whole, with no projection -- OOM-killed the SF10 driver at 22 GB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.executors.ray_runtime import lifecycle
from batcher.io.source import InMemorySource

pytestmark = pytest.mark.unit


def test_an_unscanned_source_is_not_read(monkeypatch):
    real = lifecycle.read_source
    read = []

    def spy(src, *args, **kwargs):
        read.append(src)
        return real(src, *args, **kwargs)

    monkeypatch.setattr(lifecycle, "read_source", spy)
    scanned = InMemorySource([pa.record_batch({"a": [1, 2, 3]})])
    bystander = InMemorySource([pa.record_batch({"b": [9]})])
    ds = bt.from_arrow(pa.table({"a": [1, 2, 3]})).select((bt.col("a") * 2).alias("a2"))
    plan = ds._plan
    table = lifecycle._single_node(plan, [scanned, bystander])
    assert table.column("a2").to_pylist() == [2, 4, 6]
    assert read == [scanned]
