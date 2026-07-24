"""Streaming ergonomics: interval parsing, Trigger/OutputMode/Watermark, query handle.

These exercise only the pure control-plane value types and the `StreamingQuery`
handle against a fake engine. Nothing here starts a real stream or blocks on an
unbounded source.
"""

from __future__ import annotations

import datetime

import pytest

from batcher._internal.errors import PlanError
from batcher.api.streaming import StreamingQuery
from batcher.plan.streaming import parse_interval_seconds
from batcher.plan.streaming.spec import (
    OutputMode,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
    Watermark,
)

pytestmark = pytest.mark.unit


# --- parse_interval_seconds -------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10, 10.0),
        (2.5, 2.5),
        ("10 seconds", 10.0),
        ("1 minute", 60.0),
        ("100ms", 0.1),
        ("2 hours", 7200.0),
        ("500 milliseconds", 0.5),
        (datetime.timedelta(seconds=5), 5.0),
        (datetime.timedelta(minutes=2), 120.0),
    ],
)
def test_parse_interval_accepts_numbers_strings_and_timedelta(value, expected):
    assert parse_interval_seconds(value) == expected


def test_parse_interval_unparseable_is_actionable():
    with pytest.raises(PlanError, match="cannot parse interval"):
        parse_interval_seconds("banana")


def test_parse_interval_unknown_unit_names_the_unit():
    with pytest.raises(PlanError, match="unknown interval unit 'furlongs'"):
        parse_interval_seconds("5 furlongs")


def test_parse_interval_negative_rejected():
    with pytest.raises(PlanError, match="non-negative"):
        parse_interval_seconds(-1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_parse_interval_rejects_non_finite(value):
    # A NaN cadence busy-loops the micro-batch thread and an infinite lateness overflows
    # the microsecond literal it lowers to — both must fail loudly at the parse gate.
    with pytest.raises(PlanError, match="finite"):
        parse_interval_seconds(value)


def test_trigger_rejects_non_finite_interval():
    with pytest.raises(PlanError, match="finite"):
        Trigger.processing_time(float("inf"))


def test_watermark_rejects_non_finite_delay():
    with pytest.raises(PlanError, match="finite"):
        Watermark.of("ts", float("nan"))


def test_parse_interval_bad_type_rejected():
    with pytest.raises(PlanError, match="must be a number"):
        parse_interval_seconds(None)  # type: ignore[arg-type]


def test_parse_interval_bool_rejected():
    with pytest.raises(PlanError, match="not a bool"):
        parse_interval_seconds(True)


# --- Trigger ----------------------------------------------------------------
def test_trigger_snake_case_factories():
    assert Trigger.processing_time("5 seconds").interval_seconds == 5.0
    assert Trigger.once().kind == "once"
    assert Trigger.available_now().kind == "available_now"
    assert Trigger.continuous("1 second").interval_seconds == 1.0


def test_trigger_accepts_timedelta():
    assert Trigger.processing_time(datetime.timedelta(seconds=3)).interval_seconds == 3.0


def test_trigger_spark_capitalized_aliases_match_snake_case():
    assert Trigger.ProcessingTime("5 seconds") == Trigger.processing_time("5 seconds")
    assert Trigger.Once() == Trigger.once()
    assert Trigger.AvailableNow() == Trigger.available_now()
    assert Trigger.Continuous("1 second") == Trigger.continuous("1 second")


def test_trigger_bad_interval_is_actionable():
    with pytest.raises(PlanError, match="cannot parse interval"):
        Trigger.processing_time("banana")


def test_trigger_unknown_kind_rejected_with_suggestion():
    with pytest.raises(PlanError, match="Did you mean 'processing_time'"):
        Trigger("procesing_time", 5.0)  # type: ignore[arg-type]


# --- OutputMode -------------------------------------------------------------
def test_output_mode_constants():
    assert OutputMode.APPEND == "append"
    assert OutputMode.COMPLETE == "complete"
    assert OutputMode.UPDATE == "update"


def test_output_mode_validate_accepts_canonical():
    assert OutputMode.validate("append") == "append"
    assert OutputMode.validate("complete") == "complete"


def test_output_mode_validate_suggests_closest():
    with pytest.raises(PlanError, match="Did you mean 'complete'"):
        OutputMode.validate("compelte")


def test_output_mode_validate_rejects_spark_capitalization_with_suggestion():
    # Downstream sink guards match the exact lowercase string, so "Append" is a
    # suggestion, not a silent normalization.
    with pytest.raises(PlanError, match="Did you mean 'append'"):
        OutputMode.validate("Append")


def test_output_mode_validate_rejects_non_string():
    with pytest.raises(PlanError, match="must be a string"):
        OutputMode.validate(3)  # type: ignore[arg-type]


# --- Watermark --------------------------------------------------------------
def test_watermark_of_accepts_human_delay():
    wm = Watermark.of("event_time", "10 minutes")
    assert wm.time_col == "event_time"
    assert wm.lateness_micros == 600_000_000
    assert wm.lateness_seconds == 600.0


def test_watermark_of_accepts_timedelta():
    assert Watermark.of("ts", datetime.timedelta(seconds=5)).lateness_micros == 5_000_000


def test_watermark_of_rejects_empty_time_col():
    with pytest.raises(PlanError, match="non-empty event-time column"):
        Watermark.of("", "1 minute")


# --- progress / status human summaries --------------------------------------
def test_progress_throughput_and_summary():
    p = StreamingQueryProgress(
        batch_id=1, num_input_rows=20, num_output_rows=20, duration_ms=4.0, timestamp=1.0
    )
    assert p.input_rows_per_second == pytest.approx(5000.0)
    assert "batch 1" in str(p)
    assert "rows/s" in str(p)


def test_progress_output_throughput_is_distinct_from_input():
    # A filtering micro-batch: 100 in, 10 out over 10ms -> 10000 in/s, 1000 out/s.
    p = StreamingQueryProgress(
        batch_id=0, num_input_rows=100, num_output_rows=10, duration_ms=10.0, timestamp=0.0
    )
    assert p.input_rows_per_second == pytest.approx(10000.0)
    assert p.output_rows_per_second == pytest.approx(1000.0)
    # Zero duration is reported as zero throughput, never a division error.
    zero = StreamingQueryProgress(0, 5, 5, 0.0, 0.0)
    assert zero.output_rows_per_second == 0.0


def test_status_summary_readable():
    s = StreamingQueryStatus(True, True, False, "running", 3)
    assert "running" in str(s)
    assert "3 batches" in str(s)


# --- StreamingQuery handle (against a fake engine) --------------------------
class _FakeEngine:
    def __init__(self, active=True):
        self.is_active = active
        self.exception = None

    def status(self):
        return StreamingQueryStatus(self.is_active, True, False, "running", 2)

    def recent_progress(self):
        return [
            StreamingQueryProgress(0, 10, 10, 5.0, 0.0),
            StreamingQueryProgress(1, 20, 20, 4.0, 1.0),
        ]

    def stop(self):
        self.is_active = False

    def await_termination(self, timeout=None):
        return not self.is_active


def _query(active=True):
    return StreamingQuery("q-test", _FakeEngine(active))


def test_query_repr_shows_name_state_and_progress():
    r = repr(_query())
    assert "q-test" in r
    assert "active" in r
    assert "batch 1" in r


def test_query_repr_handles_no_progress():
    class _Empty(_FakeEngine):
        def recent_progress(self):
            return []

    r = repr(StreamingQuery("q0", _Empty()))
    assert "no batches yet" in r


def test_query_core_accessors():
    q = _query()
    assert q.name == "q-test"
    assert q.id == "q-test"
    assert q.is_active is True
    assert q.last_progress.batch_id == 1
    assert len(q.recent_progress()) == 2
    assert q.status.batches_processed == 2


def test_query_spark_aliases_delegate():
    q = _query(active=False)
    assert q.isActive is False
    assert q.lastProgress.batch_id == 1
    assert len(q.recentProgress()) == 2
    assert q.awaitTermination(0.0) is True
    assert q.processAllAvailable() is True


# --- active-query registry guard --------------------------------------------
def test_register_rejects_a_duplicate_active_name():
    from batcher.api.streaming._query import _deregister, _register

    name = "dup-active-test"
    first = _query(active=True)
    _register(name, first)
    try:
        with pytest.raises(PlanError, match="already active"):
            _register(name, _query(active=True))
    finally:
        _deregister(name)


def test_register_allows_reusing_a_stopped_name():
    from batcher.api.streaming._query import _deregister, _register

    name = "reuse-stopped-test"
    _register(name, _query(active=False))  # a prior query that has since stopped
    try:
        _register(name, _query(active=True))  # reuse is fine — no active clash
    finally:
        _deregister(name)
