"""What a run cost the machine, from the engine's measurement to the Prometheus line.

The counters in `observe.metrics` used to answer "what did this process do" and could not
answer "what did it consume". These cover the path that fills that gap end to end: the
engine's whole-execution reading (`QueryUsage`), the per-operator work fold
(`WorkCounters`), the resource gauges Carbonite publishes (`ResourceGauges`), and the two
export shapes they reach — `metrics_snapshot()` and `prometheus_text()`.

The properties worth pinning are the ones that are wrong in a way no shape check catches:
a level summed as if it were a counter, an unmeasured operator folded in as a zero, a
cardinality that grows without bound, and a scrape line a scraper cannot parse.
"""

from __future__ import annotations

import pytest

from batcher._internal import events
from batcher._internal.errors import FormatError
from batcher.observe import metrics as metrics_mod
from batcher.observe.counters import ResourceGauges, WorkCounters
from batcher.plan.profile import QueryUsage, UsageStopwatch

pytestmark = pytest.mark.unit


@pytest.fixture
def bus():
    """Isolate each test from any sink another test (or the conductor) left attached."""
    saved = events._subscribers
    events._subscribers = ()
    yield events
    events._subscribers = saved


@pytest.fixture
def collector(bus):
    """A freshly-zeroed process collector, restored — subscribers *and* handle — afterwards.

    Both halves of the attachment have to be put back together, and getting that wrong is
    silent. `metrics_snapshot` attaches the collector lazily and records the unsubscribe
    handle in module state; the `bus` fixture puts `_subscribers` back without the handle
    knowing. Detach without restoring the handle and the saved tuple comes back still
    holding the collector while the module believes nothing is attached — so the next
    `start_metrics()` subscribes it a *second* time and a later test counts every event
    twice, in a file that never mentions metrics.
    """
    saved_detach = metrics_mod._detach
    metrics_mod.reset_metrics()
    yield metrics_mod._collector
    metrics_mod.stop_metrics()
    metrics_mod._detach = saved_detach
    metrics_mod.reset_metrics()


def _usage(**over) -> dict[str, float]:
    """A `QueryUsage.to_dict()`-shaped payload with every field set."""
    base = {
        "wall_ms": 10.0,
        "cpu_ms": 40.0,
        "peak_rss_bytes": 1024,
        "minor_faults": 7,
        "major_faults": 1,
        "vol_ctx_switches": 3,
        "invol_ctx_switches": 5,
        "io_read_bytes": 100,
        "io_write_bytes": 200,
    }
    base.update(over)
    return base


# --- QueryUsage -------------------------------------------------------------


def test_usage_reads_the_engine_block_in_nanoseconds():
    usage = QueryUsage.from_metrics({"wall_ns": 2_000_000, "cpu_ns": 8_000_000, "major_faults": 4})
    assert usage.wall_ms == 2.0
    assert usage.cpu_ms == 8.0
    assert usage.major_faults == 4
    # Four core-milliseconds of CPU per wall millisecond is four cores busy.
    assert usage.cores_busy == 4.0


def test_usage_from_a_missing_block_is_all_zero_rather_than_an_error():
    # An engine build that predates the measurement reports no `query` object at all.
    assert QueryUsage.from_metrics(None) == QueryUsage()
    assert QueryUsage.from_metrics({}).measured is False


def test_merging_sums_the_counters_but_takes_the_max_resident_peak():
    a = QueryUsage(wall_ms=1.0, cpu_ms=2.0, peak_rss_bytes=100, io_read_bytes=5)
    b = QueryUsage(wall_ms=3.0, cpu_ms=4.0, peak_rss_bytes=80, io_read_bytes=7)
    merged = a.merged(b)
    assert merged.wall_ms == 4.0
    assert merged.cpu_ms == 6.0
    assert merged.io_read_bytes == 12
    # A resident-set peak is a level per process. Summing two workers' peaks would report
    # memory no single machine ever held.
    assert merged.peak_rss_bytes == 100


def test_cores_busy_is_zero_when_the_platform_measured_nothing():
    assert QueryUsage().cores_busy == 0.0
    assert QueryUsage(cpu_ms=5.0).cores_busy == 0.0


def test_usage_stopwatch_reports_nonnegative_deltas_shaped_like_the_engine_block():
    watch = UsageStopwatch()
    sum(range(200_000))  # burn measurable CPU
    reading = watch.finish()
    assert reading["wall_ns"] > 0
    assert set(reading) >= {
        "wall_ns",
        "cpu_ns",
        "peak_rss_bytes",
        "minor_faults",
        "major_faults",
        "vol_ctx_switches",
        "invol_ctx_switches",
        "io_read_bytes",
        "io_write_bytes",
    }
    assert all(value >= 0 for value in reading.values())
    # The same reader consumes the engine's block and this one; that is the whole point of
    # matching the shape.
    assert QueryUsage.from_metrics(reading).wall_ms > 0


# --- WorkCounters -----------------------------------------------------------


def test_an_unmeasured_stage_is_not_folded_into_the_per_kind_counters():
    work = WorkCounters()
    work.record_stage("scan", {"measured": False, "rows_out": 0, "elapsed_ms": 0.0})
    # An out-of-core run reports its whole plan and measures none of it. Counting those
    # operators would add a zero-row, zero-time entry per operator on every such query and
    # pull every per-kind average toward zero.
    assert work.operators() == {}


def test_scanned_rows_and_bytes_come_from_the_scan_operators():
    work = WorkCounters()
    work.record_stage("scan", {"rows_out": 1_000, "result_bytes": 8_000})
    work.record_stage("filter", {"rows_in": 1_000, "rows_out": 10, "result_bytes": 80})
    # Not the filter's output: "how much did this read" is what the scans produced.
    assert work.scanned() == (1_000, 8_000)


def test_scanned_is_zero_before_any_scan_is_measured():
    assert WorkCounters().scanned() == (0, 0)


def test_per_kind_cardinality_is_capped_so_a_bad_event_cannot_grow_it_forever():
    work = WorkCounters()
    for i in range(200):
        work.record_stage(f"kind{i}", {"rows_out": 1})
    kinds = work.operators()
    assert len(kinds) <= 65  # the cap, plus the "other" bucket everything past it folds into
    assert "other" in kinds


def test_process_totals_come_from_the_whole_execution_reading_not_the_operator_sums():
    work = WorkCounters()
    # Deliberately contradictory: the streaming tier reports operator hardware counters as
    # zero, and the process figures must still be the ones the engine measured per query.
    work.record_stage("scan", {"rows_out": 5, "cpu_ms": 0.0, "io_read_bytes": 0})
    work.record_query(_usage())
    totals = work.totals()
    assert totals["cpu_ms"] == 40.0
    assert totals["io_read_bytes"] == 100
    assert totals["cores_busy"] == 4.0
    assert totals["peak_rss_bytes_max"] == 1024


def test_the_resident_peak_across_queries_is_the_largest_not_the_sum():
    work = WorkCounters()
    work.record_query(_usage(peak_rss_bytes=1024))
    work.record_query(_usage(peak_rss_bytes=512))
    assert work.totals()["peak_rss_bytes_max"] == 1024


def test_a_malformed_usage_payload_is_ignored_rather_than_raising():
    work = WorkCounters()
    work.record_query("not a dict")
    work.record_query({"cpu_ms": None, "wall_ms": "nope"})
    # A bus sink must never raise: by contract observability cannot fail a query.
    assert work.totals()["cpu_ms"] == 0.0


def test_an_out_of_core_phase_contributes_its_written_volume_to_the_spill_counter():
    work = WorkCounters()
    work.record_stage("aggregate", {"spilled": True, "spill_bytes": 100})
    work.record_spill_store({"bytes_written": 900})
    totals = work.totals()
    # Both are logical bytes routed off memory, so they belong in one counter — and the
    # out-of-core path contributes nothing to the operator-level flags.
    assert totals["spill_bytes"] == 1000
    assert totals["out_of_core_phases"] == 1
    assert work.spills_total == 1


def test_reset_zeroes_every_counter():
    work = WorkCounters()
    work.record_stage("scan", {"rows_out": 5})
    work.record_query(_usage())
    work.record_spill_store({"bytes_written": 10})
    work.reset()
    assert work.operators() == {}
    assert work.totals()["cpu_ms"] == 0.0
    assert work.totals()["spill_bytes"] == 0
    assert work.scanned() == (0, 0)


def test_render_exports_the_process_totals_even_before_anything_is_measured():
    lines = WorkCounters().render()
    # A scrape config must not have to be conditional on whether a query has run yet.
    assert any(line.startswith("batcher_cpu_ms_total ") for line in lines)
    assert any(line.startswith("batcher_cores_busy ") for line in lines)


def test_render_labels_per_operator_series_by_kind():
    work = WorkCounters()
    work.record_stage("hash_join", {"rows_in": 10, "rows_out": 4, "elapsed_ms": 1.5})
    lines = work.render()
    assert 'batcher_operator_rows_out_total{kind="hash_join"} 4.0' in lines


# --- ResourceGauges ---------------------------------------------------------


def test_a_reading_replaces_its_group_rather_than_accumulating():
    gauges = ResourceGauges()
    gauges.record("memory", {"used_bytes": 10})
    gauges.record("memory", {"used_bytes": 4})
    # These describe a level. A consumer differencing successive readings of a summed
    # gauge would get noise, not a rate.
    assert gauges.snapshot() == {"memory": {"used_bytes": 4}}


def test_a_non_dict_reading_is_ignored():
    gauges = ResourceGauges()
    gauges.record("memory", ["not", "a", "dict"])
    assert gauges.snapshot() == {}


def test_nested_readings_flatten_into_path_named_series():
    gauges = ResourceGauges()
    gauges.record("memory", {"pool": {"used_bytes": 7}, "envelope_bytes": 9})
    lines = gauges.render()
    assert "batcher_memory_pool_used_bytes 7" in lines
    assert "batcher_memory_envelope_bytes 9" in lines


def test_a_string_leaf_becomes_a_state_set_series():
    gauges = ResourceGauges()
    gauges.record("memory", {"pressure_level": "HIGH"})
    # Prometheus stores floats, so an enumerated level is conventionally a labelled 1.
    assert 'batcher_memory_pressure_level{state="HIGH"} 1' in gauges.render()


def test_a_boolean_leaf_becomes_zero_or_one():
    gauges = ResourceGauges()
    gauges.record("spill", {"compressed": True, "overflowed": False})
    lines = gauges.render()
    assert "batcher_spill_compressed 1" in lines
    assert "batcher_spill_overflowed 0" in lines


def test_group_cardinality_is_capped():
    gauges = ResourceGauges()
    for i in range(100):
        gauges.record(f"group{i}", {"x": 1})
    assert len(gauges.snapshot()) <= 32


def test_reset_forgets_every_reading():
    gauges = ResourceGauges()
    gauges.record("memory", {"used_bytes": 1})
    gauges.reset()
    assert gauges.snapshot() == {}
    assert gauges.render() == []


# --- the process collector --------------------------------------------------


def test_active_queries_rises_on_start_and_falls_on_end(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.QUERY_START, query_id="q1")
    assert collector.snapshot()["queries"]["active"] == 1
    bus.publish(bus.QUERY_END, query_id="q1", ok=True, rows=1, total_ms=1.0)
    assert collector.snapshot()["queries"]["active"] == 0


def test_the_active_gauge_never_goes_negative(collector, bus):
    bus.subscribe(collector.handle)
    # Collection can start mid-query, so an end whose start predates the collector is
    # ordinary. Left unclamped it would pin the gauge negative for the whole process.
    bus.publish(bus.QUERY_END, query_id="q1", ok=True, rows=0, total_ms=1.0)
    assert collector.snapshot()["queries"]["active"] == 0


def test_failures_are_counted_by_exception_type_not_by_message(collector, bus):
    bus.subscribe(collector.handle)
    for row in ("PlanError: no column 'a' in [x, y]", "PlanError: no column 'b' in [p, q]"):
        bus.publish(bus.QUERY_END, query_id="q", ok=False, rows=0, total_ms=1.0, error=row)
    # Keying on the message would leak predicate literals into a metrics label and grow
    # the map without bound.
    assert collector.snapshot()["queries"]["failed_by_error"] == {"PlanError": 2}


def test_a_query_end_folds_its_usage_into_the_process_sections(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.QUERY_END, query_id="q", ok=True, rows=1, total_ms=1.0, usage=_usage())
    snap = collector.snapshot()
    assert snap["cpu"]["time_ms_total"] == 40.0
    assert snap["cpu"]["cores_busy"] == 4.0
    assert snap["memory"]["major_faults_total"] == 1
    assert snap["io"]["write_bytes_total"] == 200


def test_a_spill_resource_reading_is_both_a_gauge_and_a_counter(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.RESOURCE, name="spill", stats={"bytes_written": 512, "local_bytes": 0})
    snap = collector.snapshot()
    assert snap["resources"]["spill"]["bytes_written"] == 512
    # A store is created and torn down per out-of-core phase, so its lifetime volume is
    # that phase's whole contribution; gauges alone would only show the last phase.
    assert snap["spills"]["bytes_total"] == 512
    assert snap["spills"]["out_of_core_phases_total"] == 1


def test_stage_events_populate_the_operator_and_scan_counters(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(
        bus.STAGE_END,
        query_id="q",
        name="scan",
        op_id=0,
        measured=True,
        rows_out=100,
        result_bytes=800,
        elapsed_ms=2.0,
    )
    snap = collector.snapshot()
    # The headline regression this fixes: a `collect()` publishes no progress events, so
    # rows scanned read zero no matter how much the query read.
    assert snap["rows"]["scanned_total"] == 100
    assert snap["bytes"]["scanned_total"] == 800
    assert snap["operators"]["scan"]["count"] == 1


def test_progress_events_count_as_streamed_not_as_scanned(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.PROGRESS, query_id="q", name="stream", rows=50, bytes=400)
    snap = collector.snapshot()
    # `iter_batches` progress measures rows *delivered*, which is not what a scan read.
    assert snap["rows"]["streamed_total"] == 50
    assert snap["rows"]["scanned_total"] == 0


def test_the_snapshot_carries_every_documented_section(collector):
    snap = collector.snapshot()
    assert set(snap) == {
        "backends",
        "bytes",
        "cpu",
        "data_quality",
        "gpu",
        "inference",
        "io",
        "logs",
        "memory",
        "node",
        "operators",
        "partitions",
        "queries",
        "recovery",
        "resources",
        "rows",
        "skipped",
        "spills",
        "streaming",
        "uptime_seconds",
        "writes",
    }


# --- the exposition ---------------------------------------------------------


def test_every_exposition_line_is_a_comment_or_a_named_sample(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.QUERY_END, query_id="q", ok=True, rows=1, total_ms=1.0, usage=_usage())
    bus.publish(bus.RESOURCE, name="memory", stats={"pool": {"used_bytes": 1}})
    bus.publish(bus.STAGE_END, query_id="q", name="scan", op_id=0, measured=True, rows_out=1)
    text = metrics_mod.prometheus_text()
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        assert name.startswith("batcher_"), line
        # A sample is `name{labels} value`; the value must parse as a float either way.
        float(value)


def test_the_exposition_carries_the_utilization_and_resource_series(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.QUERY_END, query_id="q", ok=True, rows=1, total_ms=1.0, usage=_usage())
    bus.publish(bus.RESOURCE, name="admission", stats={"active": 2})
    text = metrics_mod.prometheus_text()
    for series in (
        "batcher_cpu_ms_total",
        "batcher_cores_busy",
        "batcher_io_write_bytes_total",
        "batcher_major_page_faults_total",
        "batcher_involuntary_context_switches_total",
        "batcher_peak_rss_bytes",
        "batcher_queries_active",
        "batcher_spill_bytes_total",
        "batcher_admission_active",
    ):
        assert f"\n{series}" in f"\n{text}", series


# --- publishers that had no publisher ---------------------------------------


def test_a_skipped_file_reaches_the_bus(bus):
    """Silent data loss is the one condition that must reach a metrics backend.

    `corrupt_files()` answers it for whoever already suspects it and holds the source; the
    warning log answers it for a human reading a terminal. Neither reaches a scrape loop,
    and a job that quietly read 98% of its corpus produces a plausible answer and no error.
    """
    from batcher.io.base._tolerance import ErrorPolicy

    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    ErrorPolicy("skip").tolerate(
        "/data/part-0042.parquet", ValueError("truncated"), format_name="parquet"
    )

    skipped = [e for e in seen if e.kind == events.SKIPPED]
    assert len(skipped) == 1
    assert skipped[0].fields["count"] == 1
    assert skipped[0].fields["reason"] == "ValueError"
    # The path is deliberately absent: a metrics label built from one is unbounded
    # cardinality, and a path can itself be sensitive.
    assert "part-0042" not in str(skipped[0].fields)


def test_the_same_unreadable_file_is_announced_once(bus):
    from batcher.io.base._tolerance import ErrorPolicy

    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    policy = ErrorPolicy("skip")
    # Schema inference, the footer row count, split planning and the read itself each meet
    # the same bad file, and a counter must not multiply one loss by four.
    for _ in range(4):
        policy.tolerate("/data/bad.parquet", ValueError("truncated"), format_name="parquet")
    assert sum(1 for e in seen if e.kind == events.SKIPPED) == 1


def test_a_raising_policy_publishes_nothing(bus):
    from batcher.io.base._tolerance import ErrorPolicy

    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    with pytest.raises(FormatError):
        ErrorPolicy("raise").tolerate("/data/bad.parquet", ValueError("x"), format_name="parquet")
    assert not [e for e in seen if e.kind == events.SKIPPED]


def test_the_inference_pool_reports_each_micro_batch(bus):
    """The pool measured all of this for its own controller and then discarded it.

    `observe.InferenceProgress` and the `inference` metrics section were both written
    against these events, and neither had a publisher — so a multi-hour batch-inference
    job, the workload with the longest gap between "started" and "finished" of anything
    the engine runs, reported no progress at all.
    """
    pa = pytest.importorskip("pyarrow")
    from batcher.ml.inference.pool import InferencePool

    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    batch = pa.RecordBatch.from_pydict({"x": list(range(64))})
    pool = InferencePool(lambda: lambda b: b, num_workers=2, target_batch_rows=16)
    out = list(pool.run([batch]))

    assert sum(b.num_rows for b in out) == 64
    infer = [e for e in seen if e.kind == events.INFER]
    assert infer, "the pool published no per-batch event"
    assert sum(e.fields["rows"] for e in infer) == 64
    assert all(e.fields["latency_ms"] >= 0.0 for e in infer)
    assert all(e.fields["blocked_ms"] >= 0.0 for e in infer)
    pool_events = [e for e in seen if e.kind == events.POOL]
    assert pool_events and pool_events[0].fields["size"] == 2


def test_the_inference_pool_publishes_nothing_when_nobody_listens(bus):
    pa = pytest.importorskip("pyarrow")
    from batcher.ml.inference.pool import InferencePool

    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3]})
    pool = InferencePool(lambda: lambda b: b, num_workers=1)
    # The publish path runs per micro-batch, so with no sink attached it must cost a
    # tuple check and nothing else.
    assert sum(b.num_rows for b in pool.run([batch])) == 3


# --- streaming ---------------------------------------------------------------


def _progress(**over) -> dict[str, float]:
    """A `STREAM` event payload with every field set."""
    base = {
        "batch_id": 4,
        "input_rows": 1_000,
        "output_rows": 40,
        "duration_ms": 250.0,
        "behind_by_ms": 30.0,
        "input_rows_per_second": 4_000.0,
        "processed_rows_per_second": 160.0,
        "state_rows": 12,
        "state_bytes": 4_096,
        "duration_addBatch_ms": 200.0,
    }
    base.update(over)
    return base


def test_streaming_rows_accumulate_and_rates_replace():
    from batcher.observe.counters import StreamCounters

    streams = StreamCounters()
    streams.record("ingest", _progress())
    streams.record("ingest", _progress(input_rows=500, input_rows_per_second=2_000.0))
    entry = streams.snapshot()["ingest"]
    assert entry["batches"] == 2
    assert entry["input_rows"] == 1_500  # a counter
    assert entry["input_rows_per_second"] == 2_000.0  # a level
    # The per-phase breakdown sums too: "is the query slow or is the checkpoint slow" is
    # the first thing to rule out when a stream falls behind, and the two differ in fix.
    assert entry["duration_addBatch_ms"] == 400.0


def test_streaming_queries_are_kept_apart_by_name():
    from batcher.observe.counters import StreamCounters

    streams = StreamCounters()
    streams.record("ingest", _progress(input_rows=10))
    streams.record("enrich", _progress(input_rows=7))
    snap = streams.snapshot()
    # A process running two streams needs to know *which* one fell behind.
    assert snap["ingest"]["input_rows"] == 10
    assert snap["enrich"]["input_rows"] == 7


def test_streaming_query_cardinality_is_capped():
    from batcher.observe.counters import StreamCounters

    streams = StreamCounters()
    for i in range(100):
        streams.record(f"q{i}", _progress())
    snap = streams.snapshot()
    assert len(snap) <= 33  # the cap, plus the "other" bucket the rest folds into
    assert snap["other"]["batches"] == 68


def test_the_streaming_exposition_is_absent_until_a_batch_lands():
    from batcher.observe.counters import StreamCounters

    streams = StreamCounters()
    # Absent rather than zeroed, unlike the always-on process counters: a zero here is
    # indistinguishable from a query that has stopped.
    assert streams.render() == []
    streams.record("ingest", _progress())
    lines = streams.render()
    assert 'batcher_streaming_input_rows_total{query="ingest"} 1000.0' in lines
    assert 'batcher_streaming_behind_by_ms{query="ingest"} 30.0' in lines


def test_a_completed_micro_batch_reaches_the_bus(bus):
    """The progress record went only to a listener the user had to write and register.

    That is the right shape for "do something when a batch lands" and the wrong one for
    "chart this forever", which is why the longest-lived workload the engine runs was the
    one a scrape loop could not see.
    """
    from batcher.plan.streaming import StateOperatorProgress, StreamingQueryProgress
    from batcher.plan.streaming.listener import notify_query_progress

    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    notify_query_progress(
        "ingest",
        StreamingQueryProgress(
            3,
            1_000,
            40,
            250.0,
            0.0,
            behind_by_ms=30.0,
            state_operators=(
                StateOperatorProgress(
                    "windowed_aggregate", num_rows_total=12, memory_used_bytes=4096
                ),
            ),
            duration_breakdown_ms=(("addBatch", 200.0),),
        ),
    )
    published = [e for e in seen if e.kind == events.STREAM]
    assert len(published) == 1
    fields = published[0].fields
    assert published[0].name == "ingest"
    assert fields["input_rows"] == 1_000
    assert fields["behind_by_ms"] == 30.0
    assert fields["state_rows"] == 12
    assert fields["state_bytes"] == 4096
    assert fields["duration_addBatch_ms"] == 200.0


def test_the_collector_folds_streaming_progress(collector, bus):
    bus.subscribe(collector.handle)
    bus.publish(bus.STREAM, name="ingest", **_progress())
    snap = collector.snapshot()
    assert snap["streaming"]["ingest"]["input_rows"] == 1_000
    assert snap["streaming"]["ingest"]["behind_by_ms"] == 30.0


# --- the operator vocabulary -------------------------------------------------


def test_a_logical_tree_operator_is_reported_under_its_ir_tag():
    from batcher.api.terminal.event_log import _metric_kind

    # A relational profile names operators by their IR tag; a UDF profile has no IR and
    # names them by class. Folded together unreconciled, `scan` and `Scan` are two series
    # for one operator — and "rows read from sources" misses every ML pipeline, because the
    # scan it looks for is spelled the other way.
    assert _metric_kind("Scan") == "scan"
    assert _metric_kind("HashJoin") == "hash_join"
    assert _metric_kind("scan") == "scan"
    assert _metric_kind("hash_join") == "hash_join"


def test_a_per_row_adapter_stays_apart_from_a_vectorized_one():
    from batcher.api.terminal.event_log import _metric_kind

    # The one distinction the logical tree adds that the IR cannot: both are the same node
    # and 10-100x apart in cost, so normalizing must not merge them.
    assert _metric_kind("MapRows") != _metric_kind("MapBatches")
    assert _metric_kind("MapRows") == "map_rows"
    assert _metric_kind("MapBatches") == "map_batches"


def test_a_streaming_query_cannot_grow_its_own_field_set_forever():
    from batcher.observe.counters import StreamCounters

    streams = StreamCounters()
    # The engine's phase set is fixed at four; a caller deriving one from data would add a
    # key per micro-batch, and this is the workload whose uptime is measured in weeks.
    for i in range(200):
        streams.record("ingest", _progress(**{f"duration_phase{i}_ms": 1.0}))
    assert len(streams.snapshot()["ingest"]) <= 32
    # The bounded fields still accumulate correctly — the cap drops new keys, not counting.
    assert streams.snapshot()["ingest"]["batches"] == 200


# --- the write side ----------------------------------------------------------


def test_committed_writes_accumulate_overall_and_per_format():
    from batcher.observe.counters import WriteCounters

    writes = WriteCounters()
    writes.record("parquet", {"files": 4, "rows": 1_000, "bytes": 2_048})
    writes.record("parquet", {"files": 1, "rows": 10, "bytes": 64})
    writes.record("delta", {"files": 2, "rows": 5, "bytes": 32})
    snap = writes.snapshot()
    assert snap["commits_total"] == 3
    assert snap["rows_total"] == 1_015
    assert snap["files_total"] == 7
    # Per format, because "the parquet sink stopped producing" and "the delta sink did" are
    # different incidents with different owners.
    assert snap["by_format"]["parquet"]["rows"] == 1_010
    assert snap["by_format"]["delta"]["rows"] == 5


def test_write_totals_are_exported_before_anything_is_written():
    from batcher.observe.counters import WriteCounters

    lines = WriteCounters().render()
    # A job that writes nothing is a fact, so the roll-ups are always present and a scrape
    # config never has to be conditional; only the per-format series wait for a write.
    assert "batcher_rows_written_total 0" in lines
    assert not [line for line in lines if "by_format" in line]


def test_write_format_cardinality_is_capped():
    from batcher.observe.counters import WriteCounters

    writes = WriteCounters()
    for i in range(100):
        writes.record(f"fmt{i}", {"files": 1, "rows": 1, "bytes": 1})
    snap = writes.snapshot()
    assert len(snap["by_format"]) <= 33  # the cap plus the "other" bucket
    # The roll-up stays exact whatever the breakdown had to fold away.
    assert snap["rows_total"] == 100


def test_a_committed_write_reaches_the_bus(bus, tmp_path):
    """The read side was always countable and the write side never was.

    Every sink already returns a `WrittenFile` per file carrying its rows and its size on
    storage, and `WriteManifest` already rolls them up — the numbers existed and stopped at
    whoever held the manifest, so an ETL job could not count the thing it exists to produce.
    """
    import batcher as bt

    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    bt.from_pydict({"a": [1, 2, 3]}).write.parquet(str(tmp_path / "out"))

    written = [e for e in seen if e.kind == events.WRITE]
    assert len(written) == 1, "one event per commit, not one per file"
    assert written[0].name == "parquet"
    assert written[0].fields["rows"] == 3
    assert written[0].fields["files"] >= 1
    assert written[0].fields["bytes"] > 0
