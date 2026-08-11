"""Unit tests for the distributed / inference observability consumers.

These drive the consumer side with synthetic bus events only — no GPU, no cluster, no
network. They pin the rendered line, the aggregation across stages and devices, the
field-guided diagnostics, and the property that matters for a 12-hour job: the store stays
bounded no matter how many events arrive.
"""

from __future__ import annotations

import time

import pytest

from batcher._internal import events
from batcher.observe import metrics
from batcher.observe.inference import InferenceProgress
from batcher.observe.inference.progress import _GPU_WINDOW

pytestmark = pytest.mark.unit


def _event(kind: str, *, query_id: str = "q1", name: str = "", ts: float = 0.0, **fields):
    """A synthetic bus event with a caller-controlled monotonic `ts`."""
    return events.Event(
        kind=kind, ts=ts, wall=time.time(), query_id=query_id, name=name, fields=fields
    )


# --- partition progress -----------------------------------------------------


def test_partitions_aggregate_and_render_n_of_m():
    store = InferenceProgress()
    for i in range(3):
        store.handle(_event(events.PARTITION, name="infer", ts=float(i), total=8, rows=1000))
    snap = store.snapshot()
    assert snap is not None
    assert snap["partitions"]["done"] == 3
    assert snap["partitions"]["total"] == 8
    line = store.render()
    assert "3 of 8 partitions" in line
    assert "infer" in line


def test_partition_total_unknown_renders_bare_count():
    store = InferenceProgress()
    store.handle(_event(events.PARTITION, name="scan", ts=1.0, rows=500))
    snap = store.snapshot()
    assert snap["partitions"]["total"] is None
    assert snap["partitions"]["fraction"] is None
    assert "1 partitions" in store.render()
    assert "of" not in store.render()


def test_partition_totals_span_multiple_stages():
    store = InferenceProgress()
    store.handle(_event(events.PARTITION, name="scan", ts=1.0, total=4, rows=10))
    store.handle(_event(events.PARTITION, name="infer", ts=2.0, total=6, rows=10))
    snap = store.snapshot()
    assert snap["partitions"]["done"] == 2
    assert snap["partitions"]["total"] == 10
    assert set(snap["partitions"]["stages"]) == {"scan", "infer"}


def test_one_unbudgeted_stage_makes_aggregate_total_unknown():
    store = InferenceProgress()
    store.handle(_event(events.PARTITION, name="scan", ts=1.0, total=4, rows=10))
    store.handle(_event(events.PARTITION, name="infer", ts=2.0, rows=10))  # no total
    assert store.snapshot()["partitions"]["total"] is None


# --- rows/sec ---------------------------------------------------------------


def test_rows_per_sec_is_positive_and_surfaced():
    store = InferenceProgress()
    for i in range(1, 6):
        store.handle(_event(events.INFER, name="infer", ts=float(i), rows=1000, latency_ms=50.0))
    snap = store.snapshot()
    assert snap["rows_per_sec"] > 0
    assert "rows/s" in store.render()


# --- GPU --------------------------------------------------------------------


def test_gpu_utilization_and_vram_render():
    store = InferenceProgress()
    store.handle(
        _event(
            events.GPU,
            ts=1.0,
            device="cuda:0",
            util_pct=78.0,
            mem_used_bytes=74,
            mem_total_bytes=100,
        )
    )
    line = store.render()
    assert "GPU 78%" in line
    assert "VRAM 74%" in line
    snap = store.snapshot()
    assert snap["gpu"]["cuda:0"]["mem_fraction"] == pytest.approx(0.74)


def test_severe_gpu_underuse_is_critical():
    store = InferenceProgress()
    store.handle(_event(events.GPU, ts=1.0, device="cuda:0", util_pct=12.0))
    diags = store.diagnostics()
    codes = {d["code"]: d["severity"] for d in diags}
    assert codes["gpu_underused"] == "critical"


def test_gpu_oscillation_flags_data_starvation():
    store = InferenceProgress()
    # Alternating idle/saturated within the window — the starvation signal an average hides.
    for i in range(_GPU_WINDOW):
        util = 100.0 if i % 2 == 0 else 0.0
        store.handle(_event(events.GPU, ts=float(i), device="cuda:0", util_pct=util))
    diags = store.diagnostics()
    assert any(d["code"] == "gpu_starved" for d in diags)
    # A pure oscillation must be reported as starvation, not as plain under-use.
    assert not any(d["code"] == "gpu_underused" for d in diags)
    assert store.snapshot()["gpu"]["cuda:0"]["starved"] is True


def test_high_vram_warns_about_oom():
    store = InferenceProgress()
    store.handle(
        _event(
            events.GPU,
            ts=1.0,
            device="cuda:0",
            util_pct=80.0,
            mem_used_bytes=95,
            mem_total_bytes=100,
        )
    )
    assert any(d["code"] == "gpu_memory_high" for d in store.diagnostics())


def test_healthy_gpu_has_no_findings():
    store = InferenceProgress()
    store.handle(
        _event(
            events.GPU,
            ts=1.0,
            device="cuda:0",
            util_pct=80.0,
            mem_used_bytes=75,
            mem_total_bytes=100,
        )
    )
    assert store.diagnostics() == []


# --- blocked time / pipeline bottleneck -------------------------------------


def test_rising_blocked_time_flags_pipeline_bottleneck():
    store = InferenceProgress()
    for i in range(12):
        # Blocked time climbing across the run: workers increasingly waiting on input.
        store.handle(
            _event(
                events.INFER,
                name="infer",
                ts=float(i),
                rows=100,
                latency_ms=20.0,
                blocked_ms=float(i) * 10,
            )
        )
    assert any(d["code"] == "pipeline_bottleneck" for d in store.diagnostics())


def test_steady_blocked_time_is_not_flagged():
    store = InferenceProgress()
    for i in range(12):
        store.handle(
            _event(
                events.INFER, name="infer", ts=float(i), rows=100, latency_ms=20.0, blocked_ms=5.0
            )
        )
    assert not any(d["code"] == "pipeline_bottleneck" for d in store.diagnostics())


# --- skipped rows -----------------------------------------------------------


def test_skipped_files_aggregate_and_warn():
    """`SKIPPED` counts whole unreadable inputs, and must say so.

    The status line and the diagnostic both used to call them rows, and the diagnostic
    named `on_read_error`, which is not an option this engine has. A reader chasing a short
    result was told to look at the wrong flag for the wrong unit.
    """
    store = InferenceProgress()
    store.handle(_event(events.SKIPPED, ts=1.0, count=5, reason="read_error"))
    store.handle(_event(events.SKIPPED, ts=2.0, count=3, reason="read_error"))
    snap = store.snapshot()
    assert snap["skipped"]["total"] == 8
    assert snap["skipped"]["by_reason"]["read_error"] == 8
    assert "8 files skipped" in store.render()
    finding = next(d for d in store.diagnostics() if d["code"] == "skipped_files")
    assert "on_error='skip'" in finding["message"]


def test_malformed_rows_are_counted_apart_from_skipped_files():
    """Rows dropped inside a readable file are a different unit and a different cause.

    Folding them into one total would answer neither "how many files went missing" nor
    "how many rows", and the two want different responses: an unreadable file is usually
    infrastructure, a malformed record is usually the producer upstream.
    """
    store = InferenceProgress()
    store.handle(_event(events.SKIPPED, ts=1.0, count=2, reason="read_error"))
    store.handle(_event(events.MALFORMED, ts=2.0, count=3, source="csv"))
    snap = store.snapshot()
    assert snap["skipped"]["total"] == 2
    assert snap["skipped"]["malformed_rows_total"] == 3
    assert "2 files skipped" in store.render()
    assert "3 bad rows" in store.render()
    codes = {d["code"] for d in store.diagnostics()}
    assert {"skipped_files", "malformed_rows"} <= codes


# --- actor pool -------------------------------------------------------------


def test_pool_size_tracked_and_rendered():
    store = InferenceProgress()
    store.handle(_event(events.POOL, name="infer", ts=1.0, size=8, pending=20))
    snap = store.snapshot()
    assert snap["pool"] == {"size": 8, "pending": 20}
    assert "8 actors" in store.render()


# --- selection / empties ----------------------------------------------------


def test_render_and_snapshot_empty_when_no_job():
    store = InferenceProgress()
    assert store.render() == ""
    assert store.snapshot() is None
    assert store.diagnostics() == []
    assert store.snapshot("missing") is None


def test_default_picks_most_recently_active_job():
    store = InferenceProgress()
    store.handle(_event(events.PARTITION, query_id="old", name="a", ts=1.0, total=2, rows=1))
    store.handle(_event(events.PARTITION, query_id="new", name="b", ts=5.0, total=2, rows=1))
    assert store.snapshot()["query_id"] == "new"
    assert store.snapshot("old")["query_id"] == "old"


# --- boundedness (the 12-hour-job property) ---------------------------------


def test_store_stays_bounded_under_a_long_run():
    """A single job emitting a flood of events must not grow the per-job state."""
    store = InferenceProgress()
    for i in range(50_000):
        ts = float(i)
        store.handle(_event(events.PARTITION, name="infer", ts=ts, total=100_000, rows=10))
        store.handle(
            _event(events.INFER, name="infer", ts=ts, rows=10, latency_ms=5.0, blocked_ms=1.0)
        )
        store.handle(
            _event(
                events.GPU,
                ts=ts,
                device="cuda:0",
                util_pct=80.0,
                mem_used_bytes=1,
                mem_total_bytes=2,
            )
        )
        store.handle(_event(events.POOL, name="infer", ts=ts, size=4, pending=1))
    job = store._jobs["q1"]
    # One stage, one device, one GPU ring capped at the window, one job total.
    assert len(store._jobs) == 1
    assert len(job.stages) == 1
    assert len(job.gpus) == 1
    assert len(job.gpus["cuda:0"].recent) <= _GPU_WINDOW
    assert len(job.blocked_trend) <= 16
    # The counters still reflect every event.
    assert job.stages["infer"].done == 50_000
    assert job.infer_batches == 50_000


def test_job_count_is_capped():
    store = InferenceProgress(max_jobs=4)
    for i in range(20):
        store.handle(
            _event(events.PARTITION, query_id=f"q{i}", name="s", ts=float(i), total=1, rows=1)
        )
    assert len(store._jobs) == 4


def test_skip_reason_cardinality_is_capped():
    store = InferenceProgress()
    for i in range(500):
        store.handle(_event(events.SKIPPED, ts=float(i), count=1, reason=f"reason-{i}"))
    job = store._jobs["q1"]
    assert len(job.skipped) <= 65  # 64 distinct + the "other" bucket
    assert job.skipped_total == 500


def test_unrelated_event_kinds_are_ignored():
    store = InferenceProgress()
    store.handle(_event(events.QUERY_START, ts=1.0))
    store.handle(_event(events.STAGE_END, ts=2.0))
    assert store._jobs == {}


# --- attach / detach --------------------------------------------------------


def test_attach_receives_bus_events_and_detach_stops():
    store = InferenceProgress()
    detach = store.attach()
    try:
        events.publish(events.PARTITION, query_id="q1", name="infer", total=2, rows=1)
        assert store.snapshot("q1")["partitions"]["done"] == 1
    finally:
        detach()
    events.publish(events.PARTITION, query_id="q1", name="infer", total=2, rows=1)
    # After detach the store no longer receives events, so the count is unchanged.
    assert store.snapshot("q1")["partitions"]["done"] == 1


# --- the cumulative metrics surface -----------------------------------------


@pytest.fixture
def collecting():
    """Attach the process collector for one test, and detach it afterwards.

    The detach is the part that was missing. `start_metrics` attaches a bus sink and
    leaves it attached for the life of the process, and *any* attached sink tells the
    engine that per-query profiles are being consumed — so a later test asserting that a
    disabled event log builds no collector failed, in a file that never mentions metrics.
    """
    metrics.start_metrics()
    metrics.reset_metrics()
    yield
    metrics.stop_metrics()
    metrics.reset_metrics()


def test_metrics_snapshot_folds_inference_events(collecting):
    events.publish(events.PARTITION, query_id="q", name="infer", total=4, rows=100)
    events.publish(
        events.INFER, query_id="q", name="infer", rows=100, latency_ms=20.0, blocked_ms=3.0
    )
    events.publish(events.SKIPPED, query_id="q", count=7, reason="read_error")
    events.publish(
        events.GPU,
        query_id="q",
        device="cuda:0",
        util_pct=80.0,
        mem_used_bytes=5,
        mem_total_bytes=8,
    )
    snap = metrics.metrics_snapshot()
    assert snap["partitions"]["done_total"] == 1
    assert snap["inference"]["batches_total"] == 1
    assert snap["inference"]["rows_total"] == 100
    assert snap["inference"]["latency_ms_mean"] == pytest.approx(20.0)
    assert snap["skipped"]["total"] == 7
    assert snap["gpu"]["util_pct_max"] == 80.0
    assert snap["gpu"]["devices"]["cuda:0"]["util_pct"] == 80.0


def test_prometheus_text_exposes_inference_series(collecting):
    events.publish(events.SKIPPED, query_id="q", count=3, reason="read_error")
    events.publish(events.GPU, query_id="q", device="cuda:0", util_pct=42.0)
    text = metrics.prometheus_text()
    assert "batcher_skipped_total 3" in text
    assert 'batcher_gpu_utilization_percent{device="cuda:0"} 42.0' in text
