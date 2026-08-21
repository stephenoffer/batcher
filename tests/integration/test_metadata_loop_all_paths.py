"""The metadata loop (Core collects → Kyber reads → Core executes) closes on every
execution path — single-node native, UDF/map_batches, and distributed — and the
distributed path produces results identical to single-node.

The **out-of-core** routes are covered next door, in
`test_spill_closes_learning_loops.py`, and deliberately so: they close a *subset* of these
loops (everything except `learn_column_stats`, which measures the scanned batches an
out-of-core run never holds), so asserting the full set here would be wrong. That split is
also how the gap survived — this file's "every execution path" once meant the three named
above, and the three spill routes returned early, recording almost nothing.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core, count, kyber
from batcher.plan.source_stats import source_stats_key


def _identity(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch


def _learned(hub, ds, stat_key: str = kyber.NDV_KEY) -> dict:
    """The column statistics learned **for `ds`'s base source**.

    Statistics are filed per `(source, column)`: a bare column name identifies nothing,
    since two tables both have an `id`. Reading them back therefore means naming the source
    they were measured from.
    """
    learned = kyber.load_learned_stats(hub)
    return kyber.columns_for(learned, stat_key, source_stats_key(ds._sources[0]))


def test_udf_path_collects_metadata():
    # Unique column names so the process-wide hub hasn't already learned them.
    hub = core.default_hub()
    t = pa.table({"mlk": [i % 5 for i in range(500)], "mlv": list(range(500))})
    ds = bt.from_arrow(t)
    ds.map_batches(_identity).collect()  # UDF executor path
    ndv = _learned(hub, ds)
    assert abs(ndv.get("mlk", 0) - 5) < 1  # ~5 distinct
    assert ndv.get("mlv", 0) > 400  # ~500 distinct
    assert "mlk" in _learned(hub, ds, kyber.QUANTILES_KEY)


def test_native_path_still_collects_metadata():
    """The native path learns the distinct count of a column the estimator will consult.

    A *group key* is the canonical such column: `_estimate_aggregate` reads its ndv to size
    the output. This asserted the ndv of a column the query never mentioned, which the
    deliberate `learnable_columns` bounding in `learn_column_stats` stopped sketching — that
    sketch is O(rows x columns), and running it over every column of every scan cost 22.9s
    on top of a 0.73s read to learn things nothing would ever ask for.
    """
    hub = core.default_hub()
    t = pa.table({"nlk": [i % 3 for i in range(300)], "nlv": list(range(300))})
    ds = bt.from_arrow(t)
    ds.group_by("nlk").agg(s=col("nlv").sum()).collect()  # local native path
    assert abs(_learned(hub, ds).get("nlk", 0) - 3) < 1


def test_native_path_does_not_sketch_a_column_nothing_consults():
    """The other half of the bounding, and the reason the test above had to change: a
    column no estimator will read is deliberately left unmeasured, so the first query that
    *can* use it is the one that pays for it."""
    hub = core.default_hub()
    t = pa.table({"unread": [i % 3 for i in range(300)], "nlv2": list(range(300))})
    ds = bt.from_arrow(t)
    ds.filter(col("nlv2") > 10).collect()
    assert "unread" not in _learned(hub, ds)


def test_adaptive_path_collects_metadata():
    # Adaptive stages now run through the shared Kyber→Carbonite→Core orchestrator,
    # so each stage learns column stats from its scanned input (previously skipped).
    hub = core.default_hub()
    fact = pa.table({"adk": [i % 7 for i in range(700)], "adv": list(range(700))})
    dim = pa.table({"adk": list(range(7)), "adn": [f"x{i}" for i in range(7)]})
    fact_ds = bt.from_arrow(fact)
    q = (
        fact_ds.group_by("adk")
        .agg(s=col("adv").sum())
        .join(bt.from_arrow(dim), on="adk")
        .select("adk", "adn", "s")
    )
    q.collect(adaptive=True)
    assert abs(_learned(hub, fact_ds).get("adk", 0) - 7) < 1  # learned from the fact scan


def test_native_path_reports_a_measured_cpu_utilization_not_a_constant():
    """The CPU half of the adaptive loop reports a *measured* figure, never a constant.

    The morsel meter carries two numbers now: `elapsed_ns`, transform time summed over every
    morsel and every worker, and `wall_span_ns`, the interval the operator actually occupied.
    `plan.feedback.cpu_utilization` divides the first by ``span x threads``, so the result is
    a real per-core busy fraction rather than the `0` this tier used to report.

    **What the guard has to catch, and what it must not.** The regression it exists for was
    the meter handing `elapsed_ns` back as `cpu_ns` while the consumer also divided by
    `elapsed_ns` — a number over itself, so *every operator of every query* scored exactly
    ``1 / threads`` (a hardcoded 6.25% on a 16-core box), and `explain(analyze=True)` called
    a query running at 8-10x parallelism "CPU idle". The fingerprint of that bug is that it
    is **universal**.

    A single operator legitimately scoring ``1 / threads`` is not that bug. A materializing
    aggregate's combine is sequential, so one worker busy across its whole span out of the
    pool's `threads` is exactly ``1 / threads`` — the truthful answer, and measured. Asserting
    that *no* operator may show it therefore fails on every query containing an aggregate
    while still not distinguishing the constant from a measurement; the two conditions below
    do distinguish it. Measured here: aggregate 1/25, filter 0.154, scan 0.00006.
    """
    hub = core.default_hub()
    before = _op_stats_count(hub)
    n = 1_000_000
    t = pa.table({"cuk": [i % 17 for i in range(n)], "cuv": list(range(n))})
    # A filter as well as the aggregate, so the query spans operators whose utilizations
    # must *differ* — which is the property the fabricated constant cannot have.
    bt.from_arrow(t).filter(col("cuv") > 10).group_by("cuk").agg(s=col("cuv").sum()).collect()
    rows = [r for rs in hub.op_stats_by_kind().values() for r in rs]
    assert len(rows) > before, "this query's operator feedback was recorded"
    utils = [r.get("cpu_utilization") for r in rows]
    assert all(u is not None for u in utils), "the field must be carried, not dropped"
    threads = [r.get("threads") or 1 for r in rows]
    fingerprint = [
        u > 0.0 and abs(u - 1.0 / th) < 1e-9 for u, th in zip(utils, threads, strict=True)
    ]
    assert not all(fingerprint), (
        "every operator scored exactly 1/threads: utilization is dividing a number by itself"
    )
    assert any(u > 0.0 for u in utils), "the tier reports no utilization at all"


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
    base = bt.from_arrow(t)
    ds = base.filter(col("dv") > 100).group_by("dk").agg(n=count())

    single = ds.collect().sort_by("dk").to_pydict()
    dist = ds.collect(distributed=True, num_workers=2).sort_by("dk").to_pydict()
    assert single == dist  # single-node == distributed (mergeable algebra)

    # The distributed path also fed the metadata loop.
    assert abs(_learned(hub, base).get("dk", 0) - 6) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
