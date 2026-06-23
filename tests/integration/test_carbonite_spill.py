"""Carbonite auto-spill: a query that won't fit the envelope goes out-of-core.

Phase 1 wired credits; Phase 2 lets Carbonite *decide* to spill. With a tiny
configured memory cap the resource manager estimates the aggregation's breaker
won't fit and routes `collect()` through the partition-and-spill executor — with
no `spill=True` from the user — and the result must still equal the in-memory
result. This is the "a node doesn't OOM under pressure" guarantee.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import Config, MemoryConfig, config_context

pytest.importorskip("batcher._native", reason="native engine not built")


def _data():
    rng = np.random.default_rng(3)
    n = 100_000
    return pa.table(
        {"k": rng.integers(0, 500, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    }


def test_auto_spill_grouped_aggregate_matches_in_memory():
    t = _data()

    def agg(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count(), a=col("v").mean())

    in_memory = agg(bt.from_arrow(t)).collect()

    # A 1-byte cap makes every sized breaker exceed the budget, so Carbonite forces
    # the out-of-core path without the caller asking for it.
    tiny = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
    with config_context(tiny):
        auto_spilled = agg(bt.from_arrow(t)).collect()

    assert _norm(in_memory) == _norm(auto_spilled)


def test_normal_query_does_not_auto_spill():
    # With the default (generous) envelope a small query stays in memory and is
    # unaffected — the decision only triggers under real pressure.
    t = _data()
    result = bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect()
    assert result.num_rows == 500  # all groups present, ran fine in memory


def test_auto_spill_recurses_on_skew(tmp_path):
    # N13: a tiny per-bucket budget forces grace recursion (each over-large bucket is
    # re-partitioned by a secondary hash of the key and reduced sub-bucket by
    # sub-bucket). The result must still equal the in-memory aggregate.
    t = _data()

    def agg(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count(), a=col("v").mean())

    in_memory = agg(bt.from_arrow(t)).collect()

    recursive = Config().replace(
        memory=MemoryConfig(
            max_memory_bytes=1,  # force auto-spill
            spill_dir=str(tmp_path / "skew"),
            spill_bucket_max_bytes=1,  # force recursion on every non-empty bucket
        )
    )
    with config_context(recursive):
        spilled = agg(bt.from_arrow(t)).collect()

    assert _norm(in_memory) == _norm(spilled)


def test_auto_spill_overflows_to_object_storage(tmp_path):
    # C14: when local disk is "full" (budget 0) the auto-spill path overflows to
    # object storage (an in-memory fsspec FS here) and still returns the correct
    # result — the PB-scale "don't die when local disk fills" guarantee.
    pytest.importorskip("fsspec", reason="fsspec (cloud extra) not installed")
    t = _data()

    def agg(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count())

    in_memory = agg(bt.from_arrow(t)).collect()

    spilled_cfg = Config().replace(
        memory=MemoryConfig(
            max_memory_bytes=1,  # force auto-spill
            spill_dir=str(tmp_path / "local"),
            spill_remote_uri="memory://batcher-auto-spill",
            spill_local_budget_bytes=0,  # local already full → every bucket overflows
        )
    )
    with config_context(spilled_cfg):
        overflowed = agg(bt.from_arrow(t)).collect()

    assert _norm(in_memory) == _norm(overflowed)
