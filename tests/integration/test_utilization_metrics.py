"""What a real query reports about the machine it ran on, end to end.

The unit tests cover each fold in isolation against synthetic events. This file runs actual
queries and asserts the numbers arrive — which is the half that kept breaking, because every
piece can be individually correct while nothing publishes into it. Two of the gaps closed
here were exactly that shape: the whole per-operator record was measured and three fields of
it were published, and Carbonite measured every envelope it governs and put none of it on
the bus.
"""

from __future__ import annotations

import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher import col
from batcher.observe import metrics_snapshot, prometheus_text, reset_metrics, start_metrics


@pytest.fixture
def metrics():
    """Zeroed counters attached to the bus, detached afterwards.

    The detach matters: a subscriber left attached tells the engine that per-query profiles
    are being consumed, which changes what every later test measures.
    """
    import batcher.observe.metrics as m

    start_metrics()
    reset_metrics()
    yield
    if m._detach is not None:
        m._detach()
        m._detach = None
    reset_metrics()


def _run_a_query() -> None:
    """A scan → filter → aggregate over enough rows to consume measurable CPU."""
    rows = 200_000
    ds = bt.from_pydict({"a": list(range(rows)), "b": [float(i % 7) for i in range(rows)]})
    ds.filter(col("a") > 1).group_by(k=col("a") % 1000).agg(s=col("b").sum()).collect()


def test_a_collect_reports_the_rows_and_bytes_it_scanned(metrics):
    _run_a_query()
    snap = metrics_snapshot()
    # The headline regression: `collect()` publishes no progress events, so the scanned
    # counters read zero for every non-streaming query however much it read.
    assert snap["rows"]["scanned_total"] == 200_000
    assert snap["bytes"]["scanned_total"] > 0


def test_a_collect_reports_what_it_cost_the_machine(metrics):
    _run_a_query()
    snap = metrics_snapshot()
    # Measured at the FFI boundary, so this holds on the streaming tier too — which is
    # where this query runs, and which cannot attribute OS counters to an operator.
    assert snap["cpu"]["time_ms_total"] > 0.0
    assert snap["cpu"]["execution_ms_total"] > 0.0
    assert snap["cpu"]["cores_busy"] > 0.0
    assert snap["memory"]["minor_faults_total"] > 0


def test_the_per_operator_counters_carry_more_than_time_and_rows(metrics):
    _run_a_query()
    operators = metrics_snapshot()["operators"]
    assert {"scan", "filter", "aggregate"} <= set(operators)
    agg = operators["aggregate"]
    assert agg["count"] == 1
    assert agg["rows_in_total"] > 0
    assert agg["elapsed_ms_total"] > 0.0
    # Every summed field is present whether or not this tier could measure it, so a
    # consumer never has to branch on which executor ran.
    assert {"cpu_ms_total", "spill_bytes_total", "io_read_bytes_total"} <= set(agg)


def test_carbonite_resource_gauges_reach_the_snapshot(metrics):
    _run_a_query()
    resources = metrics_snapshot()["resources"]
    # Measured all along, and readable only by holding the manager that owned it.
    assert "memory" in resources
    assert resources["memory"]["envelope_bytes"] > 0
    assert "admission" in resources


def test_the_backend_split_says_which_tier_ran_the_row_work(metrics):
    _run_a_query()
    backends = metrics_snapshot()["backends"]
    assert backends, "no operator reported an execution backend"
    assert set(backends) <= {"interp", "jit", "interp+jit"}


def test_stats_surfaces_the_machine_cost_to_the_user(metrics):
    rows = 200_000
    ds = bt.from_pydict({"a": list(range(rows))})
    st = ds.filter(col("a") > 1).group_by(k=col("a") % 1000).agg(n=col("a").sum()).stats()
    assert st.usage.measured
    assert st.usage.cpu_ms > 0.0
    assert st.usage.cores_busy > 0.0
    assert "machine:" in st.summary()
    assert st.to_dict()["usage"]["cpu_ms"] > 0.0


def test_the_exposition_stays_parseable_after_a_real_query(metrics):
    _run_a_query()
    for line in prometheus_text().splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, value = line.partition(" ")
        assert name.startswith("batcher_"), line
        float(value)


def test_the_active_query_gauge_returns_to_zero(metrics):
    _run_a_query()
    snap = metrics_snapshot()
    assert snap["queries"]["total"] >= 1
    assert snap["queries"]["active"] == 0


def test_an_out_of_core_run_reports_the_volume_it_spilled(metrics):
    """The one execution mode that spills by definition used to report no spill at all.

    It streams the plan through thousands of *unmetered* engine dispatches, so the
    operator-level spill flags never fire for it and the engine's own whole-execution
    reading never happens. Both are now measured around the phase instead of inside it.
    """
    from batcher.config import Config, MemoryConfig, active_config, config_context

    rows = 2_000_000
    squeezed = active_config().replace(
        memory=MemoryConfig(max_memory_bytes=32 * 1024 * 1024),
    )
    assert isinstance(squeezed, Config)
    with config_context(squeezed):
        ds = bt.from_pydict({"a": list(range(rows)), "b": [float(i % 7) for i in range(rows)]})
        out = ds.group_by(k=col("a") % 50_000).agg(s=col("b").sum()).collect()
    assert len(out) == 50_000
    snap = metrics_snapshot()
    if snap["spills"]["out_of_core_phases_total"] == 0:
        pytest.skip("this host had headroom to keep the query in memory")
    assert snap["spills"]["bytes_total"] > 0
    assert snap["resources"]["spill"]["buckets_written"] > 0
    # The store drops each bucket as it reads it back, so the held figures are zero by the
    # end and only the high-water mark records what was ever on disk at once.
    assert snap["resources"]["spill"]["peak_local_bytes"] > 0
    assert snap["cpu"]["time_ms_total"] > 0.0


def test_a_map_batches_pipeline_reports_like_a_relational_one(metrics):
    """The one shape the ML surface exists for used to report nothing at all.

    A UDF pipeline has no engine IR, so the event-log writer bailed before assembling a
    profile — no document, no spans, no stage events, and therefore no operators, no rows
    scanned, no machine cost. Only `query_start` and `query_end` reached the bus, while the
    orchestrator had measured every stage into the `StageRecorder` the whole time.
    """
    rows = 50_000
    ds = bt.from_pydict({"a": list(range(rows))})
    out = ds.map_batches(lambda batch: batch).collect()
    assert len(out) == rows

    snap = metrics_snapshot()
    # Reported under the engine's vocabulary, not the logical tree's class names, so a
    # `scan` is one series whichever tree the profile came from.
    assert {"scan", "map_batches"} <= set(snap["operators"])
    assert snap["operators"]["map_batches"]["rows_out_total"] == rows
    assert snap["rows"]["scanned_total"] == rows
