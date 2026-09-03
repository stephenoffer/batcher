"""The one line every script leaves behind has to describe the run it just did.

The console summary is the most-read thing Batcher prints, and its throughput field was
computed from the rows *returned*. Every query that filters or aggregates returns fewer rows
than it processed, so the figure understated the engine by exactly the query's selectivity:
a `group_by` reducing 400,000 rows to five in 104 ms printed

    OK  aggregate   5 rows  |  104ms  |  48 rows/s

which reads as a system that is broken. The real figure was 3.8M rows/s. And the rows
returned were already the first field on the same line, so the rate was repeating one number
the reader had and misrepresenting the one they did not.

The rows read are on the bus already -- every `stage_end` carries `rows_in` -- so this is
about which number the line is built from, not about measuring anything new.
"""

from __future__ import annotations

import io

import pytest

from batcher._internal import events
from batcher.observe import ConsoleReporter

pytestmark = pytest.mark.unit


def _run(events_to_send):
    """Feed a reporter a hand-built event sequence and return what it printed."""
    buffer = io.StringIO()
    reporter = ConsoleReporter(stream=buffer, live=False)
    detach = reporter.attach()
    try:
        for event in events_to_send:
            reporter.handle(event)
    finally:
        detach()
    return buffer.getvalue()


def _event(kind, *, name="", **fields):
    return events.Event(kind=kind, ts=0.0, wall=0.0, query_id="q1", name=name, fields=fields)


#: A query that read 400,000 rows and returned five, in 100 ms.
SELECTIVE = [
    _event(events.QUERY_START, name="aggregate", label="aggregate"),
    _event(events.STAGE_END, name="scan", rows_in=400_000, rows_out=400_000),
    _event(events.STAGE_END, name="aggregate", rows_in=400_000, rows_out=5),
    _event(events.QUERY_END, ok=True, rows=5, total_ms=100.0),
]


def test_throughput_is_measured_on_the_rows_read():
    """400,000 rows in 100 ms is 4M rows/s, however few rows came back."""
    line = _run(SELECTIVE)
    assert "4.0M rows/s" in line, line


def test_the_rows_read_are_named_when_they_differ_from_the_rows_returned():
    """Otherwise the rate has a denominator the reader cannot see."""
    line = _run(SELECTIVE)
    assert "400.0K read" in line, line
    assert "5 rows" in line, line


def test_the_output_based_rate_is_gone():
    """The control. 5 rows / 100 ms is 50 rows/s, which is what used to be printed."""
    line = _run(SELECTIVE)
    assert "50 rows/s" not in line, line


def test_a_query_that_returns_what_it_read_says_it_once():
    """No `read` field when it would repeat the row count, and the rate is unchanged."""
    line = _run(
        [
            _event(events.QUERY_START, name="scan", label="scan"),
            _event(events.STAGE_END, name="scan", rows_in=1_000, rows_out=1_000),
            _event(events.QUERY_END, ok=True, rows=1_000, total_ms=100.0),
        ]
    )
    assert "read" not in line, line
    assert "10.0K rows/s" in line, line


def test_rows_read_counts_the_scans_only():
    """Summing every operator's `rows_in` would multiply throughput by the plan's depth.

    Six operators each passing the same 1,000 rows would report 60.0K rows/s for a query
    that moved 1,000 rows, which is the failure mode of the obvious fix.
    """
    line = _run(
        [
            _event(events.QUERY_START, name="filter", label="filter"),
            _event(events.STAGE_END, name="scan", rows_in=1_000, rows_out=1_000),
            *[
                _event(events.STAGE_END, name="project", rows_in=1_000, rows_out=1_000)
                for _ in range(5)
            ],
            _event(events.QUERY_END, ok=True, rows=10, total_ms=100.0),
        ]
    )
    assert "1.0K read" in line, line
    assert "10.0K rows/s" in line, line


def test_a_multi_source_query_adds_its_scans():
    """A join reads both sides, and both are rows the engine moved."""
    line = _run(
        [
            _event(events.QUERY_START, name="hash_join", label="hash_join"),
            _event(events.STAGE_END, name="scan", rows_in=600, rows_out=600),
            _event(events.STAGE_END, name="scan", rows_in=400, rows_out=400),
            _event(events.QUERY_END, ok=True, rows=12, total_ms=100.0),
        ]
    )
    assert "1.0K read" in line, line


def test_the_row_noun_is_inflected():
    """`1 rows` on the line a script leaves behind is the kind of thing people notice."""
    line = _run(
        [
            _event(events.QUERY_START, name="aggregate", label="aggregate"),
            _event(events.QUERY_END, ok=True, rows=1, total_ms=1.0),
        ]
    )
    assert "1 row " in line, line
    assert "1 rows" not in line, line


def test_a_commit_is_reported_even_though_it_outlives_its_query():
    """The write is what an ETL job exists to produce, and nothing printed it.

    A commit lands *after* the query that produced the rows has ended -- the sink writes its
    files, commits, and only then is there a manifest -- so the `WRITE` event names a query
    the reporter has already summarized and popped. Folded into a run that no longer exists,
    it was dropped, which is why the `wrote 24 files (3.1 GiB)` the documentation advertised
    never appeared for a `ds.write.parquet(...)`.
    """
    line = _run(
        [
            _event(events.QUERY_START, name="scan", label="scan"),
            _event(events.QUERY_END, ok=True, rows=2_000, total_ms=48.0),
            _event(events.WRITE, name="parquet", files=24, rows=2_000, bytes=3_400_000),
        ]
    )
    assert "wrote parquet" in line, line
    assert "24 files" in line, line
    assert "3.2 MiB" in line, line


def test_a_commit_inside_a_live_query_still_folds_into_its_summary():
    """The ordering is not guaranteed, so the in-flight case must keep working too.

    The control for the test above: it would pass if every write were unconditionally given
    its own line, which would print the volume twice for a sink that commits mid-query.
    """
    line = _run(
        [
            _event(events.QUERY_START, name="scan", label="scan"),
            _event(events.WRITE, name="parquet", files=3, rows=100, bytes=4096),
            _event(events.QUERY_END, ok=True, rows=100, total_ms=10.0),
        ]
    )
    assert "wrote 3 files" in line, line
    assert line.count("wrote") == 1, line


def test_a_commit_of_one_file_is_not_reported_as_one_files():
    """`plural` exists for this; the line a script leaves behind is read by people."""
    line = _run(
        [
            _event(events.QUERY_START, name="scan", label="scan"),
            _event(events.QUERY_END, ok=True, rows=1, total_ms=1.0),
            _event(events.WRITE, name="csv", files=1, rows=1, bytes=12),
        ]
    )
    assert "1 file " in line, line
    assert "1 files" not in line, line
