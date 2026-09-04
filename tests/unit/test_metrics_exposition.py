"""The Prometheus exposition has to be parseable, not merely produced.

A scraper that cannot parse one line drops the **whole** exposition, so an unescaped label
value does not cost one series — it costs every series in the process. `observe.counters`'s
own docstring says exactly this, and four label sites in three modules were interpolating
raw values with no escaping at all, while a second copy of the escaper sat in
`observe.metrics` beside them.
"""

from __future__ import annotations

import re

import pytest

from batcher._internal import events
from batcher.observe import metrics as bm
from batcher.observe.counters import escape_label

pytestmark = pytest.mark.unit

#: A label value that exercises every character the text format reserves.
HOSTILE = 'NVIDIA "A100" \\ SXM\nrev2'


@pytest.fixture
def collecting():
    """Metrics collection on and reset, so each test sees only what it published.

    `stop_metrics` on the way out, not only `reset_metrics`. Resetting zeroes the counters
    and leaves the collector *attached* to the event bus, and `events.listening()` is global
    state the engine reads to decide whether optional work is worth doing at all. So a test
    that started collection and never stopped it silently switches that work on for every
    test running after it in the same process.

    The way this surfaced is the reason it is worth a paragraph.
    `test_device_link_and_gds.py::test_nothing_is_probed_when_nobody_is_listening` asserts
    the *absence* of a GDS probe when nothing is subscribed. Once this fixture has run, that
    test cannot fail -- something is always listening -- so it passed in file order and
    failed only under a reversed run, where this file comes first. An assertion that quietly
    stops being able to fail is worse than one that is missing, because the suite still
    reports it as covered.

    `stop_metrics` rather than the raw unsubscribe handle: it also clears the module-level
    handle, so the next test's `start_metrics` can attach again. Detaching without clearing
    it silences collection for the rest of the process, which `stop_metrics`'s own docstring
    records as the mistake made the first time.
    """
    bm.start_metrics()
    bm.reset_metrics()
    yield
    bm.reset_metrics()
    bm.stop_metrics()


def _label_values(text: str, metric: str, key: str) -> list[str]:
    """Every value of label `key` on series `metric`, as the exposition renders it."""
    pattern = re.compile(
        rf'^{re.escape(metric)}\{{{re.escape(key)}="((?:[^"\\]|\\.)*)"\}} ', re.MULTILINE
    )
    return pattern.findall(text)


def test_there_is_exactly_one_label_escaper():
    """A second copy is how one of them silently stops escaping a character."""
    import inspect

    source = inspect.getsource(bm)
    assert "def _escape_label" not in source
    assert "def escape_label" not in source
    assert "escape_label" in source, "it must still be used, just not redefined"


def test_the_escaper_handles_every_reserved_character():
    escaped = escape_label(HOSTILE)
    assert '\\"' in escaped
    assert "\\\\" in escaped
    assert "\n" not in escaped


def test_a_hostile_device_name_cannot_break_the_exposition(collecting):
    """A GPU's name is vendor text read off a driver this process does not control."""
    events.publish(
        events.GPU,
        query_id="q",
        device=HOSTILE,
        actor="a0",
        util_pct=91.0,
        mem_used_bytes=1024,
        mem_total_bytes=2048,
    )
    text = bm.prometheus_text()
    values = _label_values(text, "batcher_gpu_utilization_percent", "device")
    assert values, text
    assert '\\"A100\\"' in values[0]
    _assert_every_labelled_line_parses(text)


def test_a_hostile_constraint_name_cannot_break_the_exposition(collecting):
    """A regex data-quality constraint embeds its pattern, quotes and all."""
    events.publish(
        events.DQ,
        query_id="q",
        name=r'matches "^\d+$"',
        check="row",
        severity="error",
        violations=3,
        rows=10,
        ok=False,
    )
    _assert_every_labelled_line_parses(bm.prometheus_text())


def test_a_hostile_operator_kind_cannot_break_the_exposition(collecting):
    """Operator kinds reach the per-kind series, and a named UDF stage is user text."""
    events.publish(events.QUERY_START, query_id="q", label="q")
    events.publish(
        events.STAGE_END,
        query_id="q",
        name='MapBatches["score"]',
        op_id=0,
        rows_out=1,
        elapsed_ms=1.0,
    )
    text = bm.prometheus_text()
    kinds = _label_values(text, "batcher_operator_elapsed_seconds_total", "kind")
    assert any("MapBatches" in k for k in kinds), text
    assert any('\\"score\\"' in k for k in kinds), kinds
    _assert_every_labelled_line_parses(text)


def test_a_hostile_recovery_event_name_cannot_break_the_exposition(collecting):
    events.publish(events.RECOVERY, query_id="q", event='worker_lost "n1"', shuffle="join")
    _assert_every_labelled_line_parses(bm.prometheus_text())


def _assert_every_labelled_line_parses(text: str) -> None:
    """Hold the exposition to the text format's own grammar for a labelled sample.

    Deliberately a grammar check rather than a substring check: the point is that a scraper
    would accept the document, and a scraper does not look for the value we happened to
    write.
    """
    sample = re.compile(
        r"^[a-zA-Z_:][a-zA-Z0-9_:]*"  # metric name
        r'(?:\{[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\[\\"n])*"'  # first label
        r'(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\[\\"n])*")*\})?'  # more labels
        r" [^ ]+$"  # value
    )
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        assert sample.match(line), f"a scraper would reject: {line!r}"


def test_the_whole_default_exposition_parses(collecting):
    """Nothing in the standing set of series is malformed either."""
    _assert_every_labelled_line_parses(bm.prometheus_text())
