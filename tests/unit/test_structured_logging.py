"""What a Batcher log record has to carry to be usable by anything but a human eye.

Every property here is one a log shipper needs and the records did not have. A line that
cannot be joined to a query, timestamped in a format nothing parses, with a value that
breaks the logfmt it claims to be, is a line that reaches an index and answers no question.
"""

from __future__ import annotations

import json
import logging
import re

import pytest

from batcher._internal import events
from batcher._internal import logging as blog
from batcher.config import ObservabilityConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def records(tmp_path):
    """Configure JSON logging to a file and yield a reader for the records written."""
    path = tmp_path / "engine.json"
    blog._applied = None
    blog.configure(
        ObservabilityConfig(log_level="DEBUG", console=False, log_file=str(path), log_format="json")
    )

    def read():
        for handler in blog.get_logger().handlers:
            handler.flush()
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    yield read
    blog._applied = None


@pytest.fixture
def human(tmp_path):
    """The same, for the human (logfmt) formatter."""
    path = tmp_path / "engine.log"
    blog._applied = None
    blog.configure(
        ObservabilityConfig(
            log_level="DEBUG", console=False, log_file=str(path), log_format="human"
        )
    )

    def read():
        for handler in blog.get_logger().handlers:
            handler.flush()
        return path.read_text().splitlines()

    yield read
    blog._applied = None


# --- correlation -------------------------------------------------------------


def test_a_record_written_inside_a_query_names_that_query(records):
    """The field that makes a log line joinable to a plan, a profile, and an event log.

    All three are already keyed by `query_id`; the records were not, so a shipper could
    show a line and never say which of the forty queries in the job it belonged to.
    """
    with events.query_scope("q-abc123"):
        blog.get_logger("kyber").warning("chose a broadcast join")
    assert records()[0]["query_id"] == "q-abc123"


def test_correlation_is_read_at_format_time_so_a_plain_call_gets_it_too(records):
    """A subsystem deep in the engine has no id to pass; it must not have to."""
    with events.query_scope("q-1"):
        blog.log_kv(blog.get_logger("carbonite"), logging.INFO, "admitted", bytes=1024)
    record = records()[0]
    assert record["query_id"] == "q-1"
    assert record["fields"] == {"bytes": 1024}


def test_a_record_outside_any_query_carries_no_query_id(records):
    blog.get_logger("io").warning("no query here")
    assert "query_id" not in records()[0]


def test_the_human_format_carries_the_same_correlation(human):
    with events.query_scope("q-xyz"):
        blog.get_logger("core").warning("something")
    assert "query_id=q-xyz" in human()[0]


# --- timestamps --------------------------------------------------------------


def test_the_json_timestamp_is_rfc3339_utc(records):
    """``2026-08-25 12:34:56,789`` is rejected by every shipper and falls back to ingest
    time, which silently reorders a stream whose whole value is its order."""
    blog.get_logger("core").warning("tick")
    stamp = records()[0]["time"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", stamp), stamp


# --- logfmt validity ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        ("No such file or directory", 'detail="No such file or directory"'),
        ("simple", "detail=simple"),
        ("", 'detail=""'),
        (True, "detail=true"),
        (False, "detail=false"),
        (None, "detail=null"),
        (42, "detail=42"),
        ('has "quotes"', 'detail="has \\"quotes\\""'),
    ],
)
def test_a_value_is_quoted_exactly_when_logfmt_requires_it(human, value, rendered):
    """``detail=No such file or directory`` parses as one key and three keyless tokens.

    The fields most likely to contain a space are exception messages and paths, which are
    the ones worth reading. The console reporter quoted correctly and these handlers did
    not, so the same record was valid logfmt through one sink and invalid through another.
    """
    blog.log_kv(blog.get_logger("io"), logging.INFO, "read failed", detail=value)
    assert rendered in human()[0]


# --- exceptions --------------------------------------------------------------


def test_an_exception_is_carried_as_fields_and_not_only_as_a_traceback(records):
    """Alerting on a class of failure needs a field; a string to regex-match is not one."""
    try:
        raise FileNotFoundError("missing.parquet")
    except FileNotFoundError:
        blog.get_logger("io").exception("read failed")
    record = records()[0]
    assert record["exc_type"] == "FileNotFoundError"
    assert record["exc_message"] == "missing.parquet"
    assert "Traceback" in record["exc"]


def test_the_bus_carries_the_exception_so_the_dashboard_can_show_it():
    """The traceback used to stop at the two handlers that format it, neither of which the
    web UI reads — so its log pane showed "best-effort step failed" and nothing else."""
    seen: list[events.Event] = []
    detach = events.subscribe(seen.append)
    blog._applied = None
    blog.configure(ObservabilityConfig(log_level="DEBUG", console=False))
    try:
        try:
            raise ValueError("bad footer")
        except ValueError as exc:
            blog.note_suppressed("kyber", "read footer", exc)
    finally:
        detach()
        blog._applied = None
    logs = [e for e in seen if e.kind == events.LOG]
    assert logs, "the record must reach the bus"
    assert logs[0].fields["fields"]["exc_type"] == "ValueError"
    assert logs[0].fields["fields"]["step"] == "read footer"


# --- distributed readability -------------------------------------------------


def test_records_name_their_process_and_thread(records):
    """The same subsystem logs from the driver and from every worker into one index."""
    import os

    blog.get_logger("dist").warning("shuffling")
    record = records()[0]
    assert record["pid"] == os.getpid()
    assert record["thread"] == "MainThread"
