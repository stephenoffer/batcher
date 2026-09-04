"""The terminal progress reporter: what it measures, what it reports, and what it restores.

Each test here pins a property the reporter did not have. Three of them are the reasons
this file exists:

* Throughput was an average since the query started, drawn as a sparkline and documented as
  showing "a stall, a ramp, a stutter". It could show none of them — a cumulative average
  moves by a few percent per second thirty seconds in, so the line stayed flat through
  exactly the event a person watches for.
* The bus carries skipped inputs, malformed rows, spill volume, written files and recovery
  actions. None of it reached the terminal, so a run that quietly read 98% of its corpus,
  or that transparently survived losing two workers, printed the same line as a clean one.
* The cursor was hidden by nothing and restored by nothing, and a reporter detached
  mid-query left a frozen bar on the terminal.
"""

from __future__ import annotations

import io

import pytest

from batcher._internal import events
from batcher.observe.console import ConsoleReporter, RunState, should_render

pytestmark = pytest.mark.unit


class _Tty(io.StringIO):
    """A stream that claims to be a terminal, so live rendering is exercised."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def _event(kind: str, qid: str = "q", name: str = "", ts: float = 1.0, **fields):
    return events.Event(kind, ts, ts, qid, name, fields)


# --- windowed throughput -----------------------------------------------------


def test_rate_falls_to_zero_during_a_stall():
    """The defect: a cumulative average cannot fall, so a stall was invisible."""
    run = RunState("q", est=None, t0=0.0)
    for second in range(1, 11):  # ten seconds at 1000 rows/s
        run.observe(1000)
        run.tick(float(second))
    assert run.rate == pytest.approx(1000, rel=0.2)

    for second in range(11, 21):  # ten seconds of nothing arriving
        run.tick(float(second))
    # 2% of the pre-stall reading. The exact figure depends on the display smoothing; what
    # matters is that it collapses at all, which the cumulative average could not do.
    assert run.rate < 20, "a stalled query must read as stalled, not as its historical mean"


def test_rate_tracks_a_step_change_within_the_window():
    run = RunState("q", est=None, t0=0.0)
    t = 0.0
    for _ in range(20):  # 200 rows per 0.1s = 2000/s
        t += 0.1
        run.observe(200)
        run.tick(t)
    fast = run.rate
    for _ in range(40):  # halve the arrival rate
        t += 0.1
        run.observe(100)
        run.tick(t)
    assert run.rate < fast * 0.75


def test_the_window_holds_its_last_reading_rather_than_dividing_by_nothing():
    """Two repaints inside a millisecond must not produce an enormous invented rate."""
    run = RunState("q", est=None, t0=0.0)
    run.observe(1000)
    run.tick(0.0005)
    run.observe(1000)
    run.tick(0.001)
    assert run.rate == 0.0


# --- progress the estimate cannot give ---------------------------------------


def test_partition_counts_drive_the_bar_because_they_are_exact():
    """A distributed stage knows it has 64 buckets; the row estimate is a guess."""
    run = RunState("q", est=1_000_000.0, t0=0.0)
    run.note_partition(64, rows=10)
    run.note_partition(64, rows=10)
    assert run.fraction == pytest.approx(2 / 64)
    assert run.rows == 20


def test_a_partition_total_learned_once_is_never_erased_by_a_later_unknown():
    run = RunState("q", est=None, t0=0.0)
    run.note_partition(8, rows=1)
    run.note_partition(None, rows=1)
    assert run.partitions_total == 8


def test_an_unbudgeted_run_reports_no_fraction_rather_than_a_fabricated_one():
    run = RunState("q", est=None, t0=0.0)
    run.observe(5000)
    assert run.fraction is None
    assert run.eta_s is None


def test_eta_is_omitted_once_the_estimate_has_been_beaten():
    run = RunState("q", est=100.0, t0=0.0)
    run.observe(400)
    assert run.eta_s is None


# --- what the summary line now says ------------------------------------------


def _run_query(handler_events, *, live=False):
    stream = _Tty() if live else io.StringIO()
    reporter = ConsoleReporter(stream=stream, live=live)
    reporter.handle(_event(events.QUERY_START, label="ingest"))
    for event in handler_events:
        reporter.handle(event)
    reporter.handle(_event(events.QUERY_END, ts=2.0, ok=True, rows=8192, total_ms=50.0))
    return stream.getvalue()


def test_skipped_inputs_reach_the_person_running_the_job():
    """A job that read 98% of its corpus produces a plausible answer and no error."""
    out = _run_query(
        [
            _event(events.SKIPPED, count=2, reason="FileNotFoundError", source="parquet"),
            _event(events.SKIPPED, count=1, reason="OSError", source="parquet"),
        ]
    )
    assert "3 inputs skipped" in out


def test_malformed_rows_are_counted_separately_from_unreadable_inputs():
    out = _run_query([_event(events.MALFORMED, count=417, reason="bad_line", source="csv")])
    assert "417 bad rows dropped" in out
    assert "skipped" not in out


def test_spill_volume_is_reported_not_merely_the_fact_of_spilling():
    out = _run_query([_event(events.STAGE_END, spill_bytes=3 * 1024**3, op_id=1)])
    assert "spilled 3.0 GiB" in out


def test_what_the_job_wrote_is_reported_at_all():
    """The read side was always countable and the write side never was."""
    out = _run_query([_event(events.WRITE, name="parquet", files=12, rows=8192, bytes=1024**2)])
    assert "wrote 12 files" in out
    assert "1.0 MiB" in out


def test_a_transparently_recovered_run_does_not_look_like_a_clean_one():
    out = _run_query(
        [
            _event(events.RECOVERY, event="worker_lost", shuffle="join", worker="node-7"),
            _event(events.RECOVERY, event="recompute", shuffle="join", epoch=3),
            _event(events.RECOVERY, event="recompute", shuffle="join", epoch=4),
        ]
    )
    assert "worker lost" in out and "node-7" in out  # named as it happens
    assert "1x worker lost" in out and "2x recompute" in out  # and counted at the end


def test_a_failed_data_quality_contract_is_unmissable():
    out = _run_query(
        [
            _event(
                events.DQ,
                name="orders_have_a_customer",
                ok=False,
                severity="error",
                violations=1_400,
                rows=1_000_000,
            )
        ]
    )
    assert "orders_have_a_customer" in out
    assert "1.4K" in out and "1.0M" in out


def test_a_passing_data_quality_check_says_nothing():
    out = _run_query([_event(events.DQ, name="ok_check", ok=True, violations=0, rows=10)])
    assert "ok_check" not in out


def test_a_clean_run_carries_no_caveat_clause():
    out = _run_query([_event(events.PROGRESS, rows=8192)])
    assert "8.2K rows" in out
    assert "skipped" not in out and "spilled" not in out


def test_events_for_a_query_the_reporter_never_saw_start_are_dropped():
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, live=False)
    reporter.handle(_event(events.PROGRESS, qid="ghost", rows=5))
    reporter.handle(_event(events.SKIPPED, qid="ghost", count=1))
    assert stream.getvalue() == ""


# --- terminal hygiene --------------------------------------------------------


def test_the_cursor_is_hidden_while_a_bar_animates_and_restored_when_it_stops():
    stream = _Tty()
    reporter = ConsoleReporter(stream=stream, live=True)
    detach = reporter.attach()
    reporter.handle(_event(events.QUERY_START, label="scan"))
    reporter.handle(_event(events.PROGRESS, rows=1000))
    assert "\x1b[?25l" in stream.getvalue(), "a blinking block at the end of the bar"
    reporter.handle(_event(events.QUERY_END, ts=2.0, ok=True, rows=1000, total_ms=10.0))
    assert stream.getvalue().endswith("\x1b[?25h")
    detach()


def test_detaching_mid_query_clears_the_bar_and_restores_the_cursor():
    """A frozen progress line and a hidden cursor outlive the process that left them."""
    stream = _Tty()
    reporter = ConsoleReporter(stream=stream, live=True)
    detach = reporter.attach()
    reporter.handle(_event(events.QUERY_START, label="scan"))
    reporter.handle(_event(events.PROGRESS, rows=1000))
    stream.truncate(0)
    stream.seek(0)
    detach()
    written = stream.getvalue()
    assert "\x1b[2K\r" in written and written.endswith("\x1b[?25h")


def test_a_non_live_reporter_emits_no_escape_codes_at_all():
    """A captured or redirected stream gets a record, not styling."""
    out = _run_query([_event(events.PROGRESS, rows=10)], live=False)
    assert "\x1b" not in out


def test_several_queries_in_flight_say_so_rather_than_hiding_each_other():
    stream = _Tty()
    reporter = ConsoleReporter(stream=stream, live=True)
    for i in range(3):
        reporter.handle(_event(events.QUERY_START, qid=f"q{i}", label=f"query{i}"))
        # Repaints are rate-limited to ~20 fps against the real clock, so a test that
        # delivers three queries in a microsecond must let the limiter through.
        reporter._last_draw = 0.0
        reporter.handle(_event(events.PROGRESS, qid=f"q{i}", rows=10, ts=1.0 + i))
    assert "+2 more" in stream.getvalue()
    assert "query2" in stream.getvalue(), "the most recently started run is the one drawn"


def test_a_double_width_label_does_not_shear_the_columns():
    """A dataset named in Japanese costs two columns per character, not one."""
    from batcher._internal.humanize import display_width
    from batcher.observe.console.paint import LABEL_W, compose
    from batcher.observe.theme import Glyphs, Palette

    run = RunState("注文明細表データ", est=100.0, t0=0.0)
    run.observe(50)
    run.tick(1.0)
    line = compose(run, 1.0, palette=Palette(0), glyphs=Glyphs(unicode=True), frame=1, bar_width=12)
    # Eight double-width characters are sixteen columns, and `str.ljust` would have added
    # ten spaces to reach eighteen instead of two — shifting every column to the right of it.
    assert display_width("注文明細表データ") == 16
    assert "注文明細表データ  " in line
    # A label too wide for the field is clipped to the field, not allowed to overflow it.
    wide = RunState("注" * 40, est=None, t0=0.0)
    clipped = compose(
        wide, 1.0, palette=Palette(0), glyphs=Glyphs(unicode=True), frame=1, bar_width=12
    )
    assert display_width(clipped.split("  ")[1]) <= LABEL_W


# --- when not to render ------------------------------------------------------


def test_continuous_integration_gets_no_animation(monkeypatch):
    """A CI runner is a terminal nobody is watching; thousands of frames bury the log."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert should_render("auto", _Tty()) is False
    assert should_render("on", _Tty()) is True, "an explicit request still wins"
