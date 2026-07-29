"""Stats, metrics, and dashboard ergonomics — the "what did my job do?" surface.

`RunStats` is built from a `QueryProfile` here rather than from a real run, so these stay
unit tests with no engine dependency. The metrics collector is fed synthetic bus events
for the same reason.
"""

from __future__ import annotations

import json

import pytest

from batcher._internal import events
from batcher.api.stats import OpStat, RunStats
from batcher.observe import metrics_snapshot, prometheus_text, reset_metrics, start_metrics

pytestmark = pytest.mark.unit


def _stats() -> RunStats:
    """A two-operator run: a scan that spilled, then a cheap filter."""
    return RunStats(
        ops=(
            OpStat(
                op_id=0,
                kind="scan",
                rows_in=1_000,
                rows_out=1_000,
                elapsed_ms=80.0,
                result_bytes=2048,
                spilled=True,
                backend="native",
                est_rows=900.0,
            ),
            OpStat(
                op_id=1,
                kind="filter",
                rows_in=1_000,
                rows_out=10,
                elapsed_ms=20.0,
                result_bytes=64,
                spilled=False,
                backend="jit",
            ),
        ),
        total_ms=100.0,
        rows=10,
    )


# --- OpStat -------------------------------------------------------------------


def test_op_stat_to_dict_is_json_encodable():
    """The dict is what reaches a JSON profile artifact, so it must survive the round trip."""
    d = _stats().ops[0].to_dict()
    assert json.loads(json.dumps(d)) == d


def test_op_stat_to_dict_includes_the_derived_fields():
    d = _stats().ops[1].to_dict()
    assert d["selectivity"] == pytest.approx(0.01)
    assert "est_error" in d


def test_selectivity_of_an_empty_operator_is_one_not_a_zero_division():
    empty = OpStat(0, "scan", 0, 0, 0.0, 0, False, "native")
    assert empty.selectivity == 1.0


# --- RunStats aggregates ------------------------------------------------------


def test_rows_in_counts_only_the_scans():
    """Summing rows_in over every operator would double-count a pipelined run."""
    assert _stats().rows_in == 1_000


def test_rows_out_matches_the_run_row_count():
    assert _stats().rows_out == 10


def test_spill_count_and_spilled_agree():
    stats = _stats()
    assert stats.spill_count == 1
    assert stats.spilled is True


def test_peak_memory_bytes_is_the_largest_operator_output():
    assert _stats().peak_memory_bytes == 2048


def test_bottleneck_picks_the_slowest_operator():
    assert _stats().bottleneck.kind == "scan"


def test_empty_run_has_no_bottleneck_and_does_not_raise():
    empty = RunStats(ops=(), total_ms=0.0, rows=0)
    assert empty.bottleneck is None
    assert empty.peak_memory_bytes == 0
    assert empty.spill_count == 0
    assert "no operators executed" in empty.bottleneck_summary()
    assert empty.to_dict()["bottleneck"] is None


# --- RunStats rendering and export --------------------------------------------


def test_repr_shows_the_table_not_the_dataclass_fields():
    text = repr(_stats())
    assert "rows_out" in text
    assert not text.startswith("RunStats(ops=")


def test_str_and_repr_agree():
    stats = _stats()
    assert repr(stats) == str(stats)


def test_summary_reports_time_rows_and_spill():
    text = _stats().summary()
    assert text.startswith("wall time:")
    assert "1,000 read -> 10 out" in text
    assert "1 operator(s) spilled" in text


def test_to_dict_is_json_encodable_and_has_the_totals():
    d = _stats().to_dict()
    json.dumps(d)
    assert d["total_ms"] == 100.0
    assert d["rows_out"] == 10
    assert d["spill_count"] == 1
    assert d["bottleneck"] == "scan"
    assert len(d["ops"]) == 2


def test_repr_html_renders_a_table_flagging_the_spill():
    html = _stats()._repr_html_()
    assert "<table" in html and "</table>" in html
    assert "scan" in html
    assert "yes" in html  # the spill column


def test_repr_html_of_an_empty_run_is_still_valid():
    assert "<table" in RunStats(ops=(), total_ms=0.0, rows=0)._repr_html_()


def test_to_pandas_has_one_row_per_operator():
    pd = pytest.importorskip("pandas")
    assert pd
    df = _stats().to_pandas()
    assert len(df) == 2
    assert list(df["kind"]) == ["scan", "filter"]


# --- metrics ------------------------------------------------------------------


@pytest.fixture
def clean_metrics():
    """A collector attached to the bus with every counter zeroed, detached afterwards.

    The detach matters: a bus subscriber left attached tells the engine that per-query
    profiles are being consumed, which silently changes behavior for every later test.
    """
    import batcher.observe.metrics as m

    start_metrics()
    reset_metrics()
    yield
    if m._detach is not None:
        m._detach()
        m._detach = None
    reset_metrics()


def test_metrics_snapshot_shape(clean_metrics):
    snap = metrics_snapshot()
    assert sorted(snap) == [
        "bytes",
        "gpu",
        "inference",
        "logs",
        "operators",
        "partitions",
        "queries",
        # Fault-tolerance actions by kind (recompute, worker_lost, backup_won, ...).
        # Recovery used to be entirely unobservable, so a query that transparently
        # survived losing two workers looked identical to one that was merely slow.
        "recovery",
        "rows",
        "skipped",
        "spills",
        "uptime_seconds",
    ]


def test_metrics_snapshot_is_json_encodable(clean_metrics):
    """The snapshot is served over HTTP as JSON, so every value must be encodable."""
    snap = metrics_snapshot()
    assert json.loads(json.dumps(snap)) == snap


def test_metrics_count_a_finished_query(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=True, total_ms=12.5, rows=100)
    snap = metrics_snapshot()
    assert snap["queries"]["total"] == 1
    assert snap["queries"]["failed"] == 0
    assert snap["queries"]["duration_ms_total"] == 12.5
    assert snap["rows"]["out_total"] == 100


def test_metrics_count_a_failed_query(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=False, total_ms=1.0, rows=0)
    snap = metrics_snapshot()
    assert snap["queries"]["failed"] == 1
    assert snap["queries"]["succeeded"] == 0


def test_metrics_count_operator_time_and_spills(clean_metrics):
    events.publish(
        events.STAGE_END,
        query_id="q1",
        name="scan",
        op_id=0,
        rows_out=50,
        elapsed_ms=9.0,
        spilled=True,
    )
    snap = metrics_snapshot()
    assert snap["operators"]["scan"]["count"] == 1
    assert snap["operators"]["scan"]["elapsed_ms_total"] == 9.0
    assert snap["spills"]["total"] == 1


def test_metrics_count_scanned_rows_and_bytes(clean_metrics):
    events.publish(events.PROGRESS, query_id="q1", name="p", rows=7, bytes=4096)
    snap = metrics_snapshot()
    assert snap["rows"]["scanned_total"] == 7
    assert snap["bytes"]["scanned_total"] == 4096


def test_duration_histogram_buckets_are_cumulative(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=True, total_ms=7.0, rows=0)
    # Keys are strings: the snapshot is JSON, and JSON object keys are strings.
    buckets = metrics_snapshot()["queries"]["duration_ms_buckets"]
    assert buckets["1"] == 0
    assert buckets["10"] == 1
    assert buckets["1000"] == 1


def test_reset_metrics_zeroes_the_counters(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=True, total_ms=1.0, rows=1)
    reset_metrics()
    assert metrics_snapshot()["queries"]["total"] == 0


def test_prometheus_text_exposes_the_counters(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=True, total_ms=12.5, rows=100)
    text = prometheus_text()
    assert "batcher_queries_total 1" in text
    assert "batcher_rows_out_total 100" in text
    assert text.endswith("\n")


def test_prometheus_text_emits_a_well_formed_histogram(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=True, total_ms=12.5, rows=0)
    text = prometheus_text()
    assert "# TYPE batcher_query_duration_ms histogram" in text
    assert 'batcher_query_duration_ms_bucket{le="+Inf"} 1' in text
    assert "batcher_query_duration_ms_sum 12.5" in text
    assert "batcher_query_duration_ms_count 1" in text


def test_prometheus_every_sample_line_has_a_numeric_value(clean_metrics):
    events.publish(events.QUERY_END, query_id="q1", name="q", ok=True, total_ms=3.0, rows=1)
    samples = [ln for ln in prometheus_text().splitlines() if ln and not ln.startswith("#")]
    assert samples  # an exposition with no samples would pass the loop vacuously
    for line in samples:
        float(line.rsplit(" ", 1)[1])


def test_a_snapshot_starts_collection_so_no_explicit_setup_is_needed(clean_metrics):
    assert isinstance(metrics_snapshot()["queries"]["total"], int)


def test_observability_sinks_do_not_attach_a_metrics_bus_sink():
    """Attaching a bus sink makes the engine assemble a profile on every query.

    A process that never exports metrics must not pay that, so `ensure_sinks` must leave
    the bus untouched when the user has not asked for progress, the UI, or metrics.
    """
    import batcher.observe.metrics as m
    from batcher._internal import events
    from batcher.observe.control import ensure_sinks

    detach, m._detach = m._detach, None
    try:
        ensure_sinks()
        assert m._detach is None, "ensure_sinks must not start metrics collection"
    finally:
        m._detach = detach
        assert events.listening() or not events.listening()


# --- findings surface in stats(), not only in the dashboard ---------------------------
# The engine already derived what was wrong with a run, and showed it only to someone who
# had started the web UI. A user reading `stats()` in a terminal — the surface every
# performance guide sends them to — was told nothing.


def _profile(ops, total_ms=700.0):
    from batcher.plan.profile import OpProfile, QueryProfile

    return QueryProfile(
        ops=tuple(OpProfile(measured=True, **{"depth": 0, **op}) for op in ops),
        total_ms=total_ms,
        rows=1000,
        measured=True,
    )


def test_a_starved_gpu_is_reported_by_stats_itself():
    profile = _profile(
        [
            {
                "op_id": 0,
                "kind": "MapBatches",
                "rows_in": 1000,
                "rows_out": 1000,
                "elapsed_ms": 100.0,
                "backend": "gpu",
            },
            {
                "op_id": 1,
                "kind": "MapBatches",
                "rows_in": 1000,
                "rows_out": 1000,
                "elapsed_ms": 500.0,
            },
        ]
    )
    stats = RunStats.from_profile(profile)
    assert any(f["rule"] == "gpu-starved" for f in stats.findings)
    rendered = str(stats)
    assert "findings" in rendered
    # The finding is useless without what to do about it, so the action prints too.
    assert "more CPU per GPU" in rendered


def test_a_healthy_run_prints_no_findings_section():
    """An empty section under every clean run is how a reader learns to skip the one that
    mattered."""
    profile = _profile(
        [
            {
                "op_id": 0,
                "kind": "scan",
                "rows_in": 1000,
                "rows_out": 1000,
                "elapsed_ms": 40.0,
                "cpu_util": 0.95,
            },
            {
                "op_id": 1,
                "kind": "aggregate",
                "rows_in": 1000,
                "rows_out": 10,
                "elapsed_ms": 45.0,
                "cpu_util": 0.95,
            },
        ],
        total_ms=90.0,
    )
    stats = RunStats.from_profile(profile)
    assert "findings" not in str(stats)


def test_info_findings_are_carried_but_not_printed():
    """`findings` is the full list for a program to read; the printed table shows only what
    needs acting on."""
    profile = _profile(
        [
            {
                "op_id": 0,
                "kind": "MapBatches",
                "rows_in": 2000,
                "rows_out": 20000,
                "elapsed_ms": 300.0,
            },
        ]
    )
    stats = RunStats.from_profile(profile)
    assert any(f["rule"] == "row-exploding-stage" for f in stats.findings)
    assert "findings" not in str(stats)


def test_deriving_findings_never_breaks_a_measurement(monkeypatch):
    """Findings are commentary on a measurement that already succeeded."""
    import batcher.observe.insights as insights

    def boom(_profile):
        raise RuntimeError("rule exploded")

    monkeypatch.setattr(insights, "derive_insights", boom)
    stats = RunStats.from_profile(
        _profile([{"op_id": 0, "kind": "scan", "rows_in": 1, "rows_out": 1, "elapsed_ms": 1.0}])
    )
    assert stats.findings == ()
    assert stats.total_ms == 700.0


def test_overlapping_stages_do_not_report_a_share_above_100_percent():
    """A pipelined chain runs its stages concurrently, so their times legitimately sum past
    the wall clock. Dividing by wall time produced shares like "325%", which reads as a bug
    rather than as concurrency — so the denominator switches and says which one it used."""
    stats = RunStats(
        ops=(
            OpStat(0, "MapBatches", 100, 100, 30.0, 0, False, "gpu"),
            OpStat(1, "MapBatches", 100, 100, 70.0, 0, False, ""),
        ),
        total_ms=40.0,  # less than 30 + 70: the stages overlapped
        rows=100,
    )
    line = stats.bottleneck_summary()
    assert "70%" in line, line
    assert "operator time (stages overlap)" in line


def test_a_sequential_run_still_reports_against_wall_time():
    """The wall-clock reading is the right one when nothing overlapped, and must not be
    silently replaced by the operator-time one."""
    stats = RunStats(
        ops=(
            OpStat(0, "scan", 100, 100, 20.0, 0, False, ""),
            OpStat(1, "filter", 100, 50, 30.0, 0, False, ""),
        ),
        total_ms=100.0,
        rows=50,
    )
    line = stats.bottleneck_summary()
    assert "30% of wall time" in line, line
