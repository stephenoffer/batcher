"""The metadata loop (Core collects → Kyber reads → Core executes) closes on every
execution path — single-node native, UDF/map_batches, and distributed — and the
distributed path produces results identical to single-node."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core, count, kyber


def _identity(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch


def test_udf_path_collects_metadata():
    # Unique column names so the process-wide hub hasn't already learned them.
    hub = core.default_hub()
    t = pa.table({"mlk": [i % 5 for i in range(500)], "mlv": list(range(500))})
    bt.from_arrow(t).map_batches(_identity).collect()  # UDF executor path
    learned = kyber.load_learned_stats(hub)
    ndv = learned.get("__column_ndv__", {})
    assert abs(ndv.get("mlk", 0) - 5) < 1  # ~5 distinct
    assert ndv.get("mlv", 0) > 400  # ~500 distinct
    assert "mlk" in learned.get("__column_quantiles__", {})


def test_native_path_still_collects_metadata():
    hub = core.default_hub()
    t = pa.table({"nlk": [i % 3 for i in range(300)], "nlv": list(range(300))})
    bt.from_arrow(t).filter(col("nlv") > 10).collect()  # local native path
    ndv = kyber.load_learned_stats(hub).get("__column_ndv__", {})
    assert abs(ndv.get("nlk", 0) - 3) < 1


def test_adaptive_path_collects_metadata():
    # Adaptive stages now run through the shared Kyber→Carbonite→Core orchestrator,
    # so each stage learns column stats from its scanned input (previously skipped).
    hub = core.default_hub()
    fact = pa.table({"adk": [i % 7 for i in range(700)], "adv": list(range(700))})
    dim = pa.table({"adk": list(range(7)), "adn": [f"x{i}" for i in range(7)]})
    q = (
        bt.from_arrow(fact)
        .group_by("adk")
        .agg(s=col("adv").sum())
        .join(bt.from_arrow(dim), on="adk")
        .select("adk", "adn", "s")
    )
    q.collect(adaptive=True)
    ndv = kyber.load_learned_stats(hub).get("__column_ndv__", {})
    assert abs(ndv.get("adk", 0) - 7) < 1  # learned from the fact scan


def test_native_path_records_cpu_utilization():
    # The CPU half of the adaptive loop: a real run measures per-operator CPU time
    # (Rust `cpu_ns`) and transcribes it to a [0, 1] utilization on the hub, which
    # Kyber later folds into each task's `num_cpus`. A CPU-heavy aggregate over
    # enough rows registers measurable CPU time regardless of timer granularity.
    hub = core.default_hub()
    n = 1_000_000
    t = pa.table({"cuk": [i % 17 for i in range(n)], "cuv": list(range(n))})
    bt.from_arrow(t).group_by("cuk").agg(s=col("cuv").sum()).collect()
    rows = [r for rs in hub.op_stats_by_kind().values() for r in rs]
    assert rows, "operator feedback was recorded"
    assert any(r.get("cpu_utilization", 0.0) > 0.0 for r in rows), (
        "at least one operator registered measurable CPU utilization"
    )


def _op_stats_count(hub) -> int:
    return sum(len(v) for v in hub.op_stats_by_kind().values())


def test_streaming_path_collects_metadata():
    # The streaming relational path previously executed with feedback=None and
    # learned nothing; it now feeds each micro-batch's op_stats into the hub.
    hub = core.default_hub()
    t = pa.table({"smk": [i % 4 for i in range(400)], "smv": list(range(400))})
    before = _op_stats_count(hub)
    list(bt.from_arrow(t).filter(col("smv") > 10).iter_batches())
    assert _op_stats_count(hub) > before  # streaming now records operator feedback


def test_distributed_equals_single_node_and_collects():
    pytest.importorskip("ray")
    hub = core.default_hub()
    t = pa.table({"dk": [i % 6 for i in range(600)], "dv": list(range(600))})
    ds = bt.from_arrow(t).filter(col("dv") > 100).group_by("dk").agg(n=count())

    single = ds.collect().sort_by("dk").to_pydict()
    dist = ds.collect(distributed=True, num_workers=2).sort_by("dk").to_pydict()
    assert single == dist  # single-node == distributed (mergeable algebra)

    # The distributed path also fed the metadata loop.
    ndv = kyber.load_learned_stats(hub).get("__column_ndv__", {})
    assert abs(ndv.get("dk", 0) - 6) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
