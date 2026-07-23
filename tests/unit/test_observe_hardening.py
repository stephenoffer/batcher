"""Adversarial tests for observability: the ways it could break a query, and must not.

The rest of the observability tests check that the feature works. These check that it
cannot do harm — because a progress bar that crashes a job, a store that grows without
bound, or a sink that deadlocks the engine is far worse than no observability at all.

Every test here corresponds to a way this code could plausibly fail in a real process:
recursion through the logging bridge, concurrent queries on many threads, a consumer that
abandons a stream, a stream that raises, a terminal that vanishes, unbounded growth in a
long-lived process, and sinks left attached after shutdown.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import urllib.request

import pytest

import batcher as bt
from batcher._internal import events
from batcher._internal import logging as blog
from batcher.config import ObservabilityConfig, active_config, config_context
from batcher.observe.console import ConsoleReporter
from batcher.observe.server import UIServer
from batcher.observe.store import ActivityStore

pytestmark = pytest.mark.unit


class Stream(io.StringIO):
    """A StringIO with a settable `encoding`, which the real class does not allow."""

    def __init__(self, encoding: str = "utf-8") -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding


@pytest.fixture
def bus():
    """Isolate the bus for one test and restore it afterwards.

    Restoration is unconditional: these tests deliberately attach sinks and let them stay
    for the duration, so leak detection belongs in the one test that exercises a real
    attach/detach lifecycle (`test_stopping_the_ui_detaches_its_sink`), not here.
    """
    saved = events._subscribers
    events._subscribers = ()
    try:
        yield events
    finally:
        events._subscribers = saved


# --- the bus must not be able to hurt the engine ----------------------------


def test_a_failing_sink_cannot_recurse_through_the_logging_bridge(bus, tmp_path):
    """Regression: `publish` → sink raises → DEBUG log → LOG event → same sink → ...

    Log records are mirrored onto the bus, so reporting a sink failure *by logging it* fed
    straight back into the failing sink. At DEBUG verbosity that recursed until the stack
    blew, turning a cosmetic sink bug into a killed query.
    """
    blog._applied = None
    blog.configure(
        ObservabilityConfig(log_level="DEBUG", console=False, log_file=str(tmp_path / "l"))
    )
    delivered: list[str] = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("sink is broken")))
    bus.subscribe(lambda e: delivered.append(e.kind))
    for _ in range(200):
        bus.publish(bus.QUERY_START, query_id="q", label="t")
    assert len(delivered) == 200  # every publish still delivered to the healthy sink


def test_reporter_survives_a_failing_sink_under_debug_logging(bus, tmp_path):
    """The same cycle, through the real reporter, which takes a lock while rendering."""
    blog._applied = None
    blog.configure(
        ObservabilityConfig(log_level="DEBUG", console=False, log_file=str(tmp_path / "l"))
    )
    reporter = ConsoleReporter(stream=Stream(), live=False)
    bus.subscribe(reporter.handle)
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))

    done = threading.Event()

    def run() -> None:
        for _ in range(50):
            bus.publish(bus.QUERY_START, query_id="q", label="t")
            bus.publish(bus.QUERY_END, query_id="q", ok=True, rows=1, total_ms=1.0)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(15), "publishing deadlocked or recursed"


def test_publish_is_thread_safe_under_concurrent_subscribe_and_unsubscribe(bus):
    """Sinks attach and detach while other threads publish; nothing may raise or corrupt."""
    seen: list[object] = []
    stop = threading.Event()
    errors: list[BaseException] = []

    def publisher() -> None:
        try:
            while not stop.is_set():
                bus.publish(bus.PROGRESS, query_id="q", rows=1)
        except BaseException as exc:  # pragma: no cover - the failure we are asserting against
            errors.append(exc)

    def churner() -> None:
        try:
            for _ in range(300):
                off = bus.subscribe(seen.append)
                off()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=publisher, daemon=True) for _ in range(3)]
    threads += [threading.Thread(target=churner, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads[3:]:
        t.join(20)
    stop.set()
    for t in threads[:3]:
        t.join(20)
    assert not errors


# --- streaming: the consumer controls the generator -------------------------


def test_abandoning_a_stream_early_closes_the_query_as_ok(bus):
    """`for b in ds.iter_batches(): break` is ordinary use, not a failure."""
    store = ActivityStore()
    bus.subscribe(store.handle)
    ds = bt.from_pydict({"x": list(range(50_000))})
    for _ in ds.iter_batches():
        break
    # The generator closes on GC; force it deterministically.
    import gc

    gc.collect()
    statuses = [q["status"] for q in store.queries()]
    assert statuses and all(s != "running" for s in statuses), statuses
    assert "error" not in statuses


def test_an_exception_mid_stream_marks_the_query_failed_and_re_raises(bus):
    store = ActivityStore()
    bus.subscribe(store.handle)

    def explode(batch):
        raise RuntimeError("udf exploded")

    ds = bt.from_pydict({"x": list(range(10_000))}).map_batches(explode)
    with pytest.raises(RuntimeError, match="udf exploded"):
        list(ds.iter_batches())
    assert store.queries()[0]["status"] == "error"


def test_streaming_rows_are_counted_exactly_once(bus):
    """`batch_size` makes `_iter_batches` recurse; progress must not double-count."""
    store = ActivityStore()
    bus.subscribe(store.handle)
    ds = bt.from_pydict({"x": list(range(30_000))})
    total = sum(b.num_rows for b in ds.iter_batches(batch_size=1000))
    assert total == 30_000
    assert store.queries()[0]["rows_seen"] == 30_000


def test_nested_streams_are_tracked_independently(bus):
    store = ActivityStore()
    bus.subscribe(store.handle)
    outer = bt.from_pydict({"x": list(range(4_000))})
    inner = bt.from_pydict({"y": list(range(2_000))})
    for _ in outer.iter_batches():
        assert sum(b.num_rows for b in inner.iter_batches()) == 2_000
    assert len({q["query_id"] for q in store.queries()}) >= 2
    assert all(q["status"] == "ok" for q in store.queries())


# --- bounded memory in a long-lived process ---------------------------------


def test_store_bounds_queries_logs_and_the_id_index(bus):
    store = ActivityStore(max_queries=5, max_logs=10)
    for i in range(500):
        store.handle(events.Event(events.QUERY_START, 1.0, 1.0, f"q{i}", "", {"label": "x"}))
        store.handle(events.Event(events.LOG, 1.0, 1.0, "", "k", {"level": "INFO", "message": "m"}))
    assert len(store.queries()) == 5
    assert len(store._by_id) == 5
    assert len(store.logs(since=0)["lines"]) == 10


def test_reporter_does_not_accumulate_finished_runs(bus):
    reporter = ConsoleReporter(stream=Stream(), live=False)
    for i in range(1000):
        reporter.handle(events.Event(events.QUERY_START, 1.0, 1.0, f"q{i}", "", {"label": "x"}))
        reporter.handle(events.Event(events.QUERY_END, 2.0, 2.0, f"q{i}", "", {"ok": True}))
    assert reporter._runs == {}


# --- the terminal can vanish or misbehave -----------------------------------


def test_a_closed_stream_never_raises_into_the_query(bus):
    stream = Stream()
    reporter = ConsoleReporter(stream=stream, live=True)
    stream.close()
    for kind in (events.QUERY_START, events.PROGRESS, events.QUERY_END):
        reporter.handle(events.Event(kind, 1.0, 1.0, "q", "", {"ok": True, "rows": 1}))


def test_rendering_survives_a_degenerate_terminal_size(bus, monkeypatch):
    import shutil as shutil_mod

    monkeypatch.setattr(shutil_mod, "get_terminal_size", lambda _d=None: lambda: None)
    monkeypatch.setattr(
        shutil_mod, "get_terminal_size", lambda *_a, **_k: __import__("os").terminal_size((1, 1))
    )
    reporter = ConsoleReporter(stream=Stream(), live=True)
    assert reporter._bar(0.5)
    assert reporter._bar(None)


def test_ascii_stream_never_emits_a_character_it_cannot_encode(bus):
    """A `LANG=C` terminal must get ASCII, not a UnicodeEncodeError or mojibake."""
    stream = Stream("ascii")
    reporter = ConsoleReporter(stream=stream, live=True)
    reporter.handle(events.Event(events.QUERY_START, 1.0, 1.0, "q", "", {"label": "scan"}))
    reporter.handle(
        events.Event(events.QUERY_END, 2.0, 2.0, "q", "", {"ok": True, "rows": 5, "total_ms": 3.0})
    )
    stream.getvalue().encode("ascii")  # raises if a non-ASCII glyph leaked through


def test_a_very_long_label_cannot_break_the_column_layout(bus):
    reporter = ConsoleReporter(stream=Stream(), live=False)
    reporter.handle(events.Event(events.QUERY_START, 1.0, 1.0, "q", "", {"label": "x" * 500}))
    stream_out = io.StringIO()
    reporter._stream = stream_out
    reporter.handle(events.Event(events.QUERY_END, 2.0, 2.0, "q", "", {"ok": True, "rows": 1}))
    line = stream_out.getvalue().strip()
    assert len(line) < 200 and "…" in line


# --- concurrency: many queries at once --------------------------------------


def test_concurrent_queries_are_all_recorded(bus):
    store = ActivityStore()
    bus.subscribe(store.handle)
    reporter = ConsoleReporter(stream=Stream(), live=False)
    bus.subscribe(reporter.handle)
    errors: list[BaseException] = []

    def run(i: int) -> None:
        try:
            ds = bt.from_pydict({"g": ["a", "b"] * 50, "x": list(range(100))})
            ds.group_by("g").agg(s=bt.col("x").sum()).collect()
            list(ds.iter_batches())
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert all(q["status"] == "ok" for q in store.queries())


# --- the dashboard ----------------------------------------------------------


def test_ui_serves_correctly_while_queries_run_concurrently(bus):
    store = ActivityStore()
    server = UIServer(store, port=0)
    url = server.start()
    errors: list[BaseException] = []
    stop = threading.Event()

    def poll() -> None:
        try:
            while not stop.is_set():
                for path in ("/api/summary", "/api/queries", "/api/logs?since=0"):
                    with urllib.request.urlopen(f"{url}{path}", timeout=10) as response:
                        json.loads(response.read())
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    pollers = [threading.Thread(target=poll, daemon=True) for _ in range(4)]
    for t in pollers:
        t.start()
    try:
        for _ in range(20):
            bt.from_pydict({"g": ["a", "b"] * 20, "x": list(range(40))}).group_by("g").agg(
                s=bt.col("x").sum()
            ).collect()
    finally:
        stop.set()
        for t in pollers:
            t.join(20)
        server.stop()
    assert not errors, errors


def test_ui_rejects_a_malformed_cursor_without_failing(bus):
    store = ActivityStore()
    server = UIServer(store, port=0)
    url = server.start()
    try:
        for query in ("since=abc", "since=-5", "since=99999999", "since="):
            with urllib.request.urlopen(f"{url}/api/logs?{query}", timeout=10) as response:
                assert "lines" in json.loads(response.read())
    finally:
        server.stop()


def test_stopping_the_ui_detaches_its_sink(bus):
    before = len(events._subscribers)
    bt.start_ui(port=0)
    assert len(events._subscribers) == before + 1
    bt.stop_ui()
    assert len(events._subscribers) == before


# --- verbosity --------------------------------------------------------------


@pytest.mark.parametrize(
    ("verbosity", "log_level", "progress", "native"),
    [
        ("silent", "CRITICAL", "off", "off"),
        ("quiet", "ERROR", "off", "error"),
        ("normal", "WARNING", "auto", "warn"),
        ("verbose", "INFO", "auto", "info"),
        ("debug", "DEBUG", "auto", "debug"),
        ("trace", "DEBUG", "on", "trace"),
    ],
)
def test_the_verbosity_ladder_resolves_as_documented(verbosity, log_level, progress, native):
    cfg = ObservabilityConfig(verbosity=verbosity)
    assert cfg.resolved_log_level == log_level
    assert cfg.resolved_progress == progress
    assert cfg.resolved_native_log_level == native


def test_integer_and_name_verbosity_are_interchangeable():
    """A CLI counting `-v` flags passes the number straight through."""
    for rank, name in enumerate(("silent", "quiet", "normal", "verbose", "debug", "trace")):
        assert ObservabilityConfig(verbosity=rank).resolved_log_level == (
            ObservabilityConfig(verbosity=name).resolved_log_level
        )
        assert ObservabilityConfig(verbosity=str(rank)).resolved_progress == (
            ObservabilityConfig(verbosity=name).resolved_progress
        )


def test_an_explicit_component_overrides_the_preset():
    cfg = ObservabilityConfig(verbosity="trace", log_level="ERROR", progress="off")
    assert cfg.resolved_log_level == "ERROR"
    assert cfg.resolved_progress == "off"


def test_default_verbosity_reproduces_the_historical_behavior():
    """Adding the dial must have changed nothing for anyone who does not touch it."""
    cfg = ObservabilityConfig()
    assert (cfg.resolved_log_level, cfg.resolved_progress) == ("WARNING", "auto")


@pytest.mark.parametrize("bad", ["loud", "-1", "6", 9, -1, True])
def test_an_invalid_verbosity_is_rejected_by_validation(bad):
    from batcher._internal.errors import ConfigError

    with (
        pytest.raises(ConfigError, match="verbosity"),
        config_context(active_config().replace(observability=ObservabilityConfig(verbosity=bad))),
    ):
        pass


def test_verbosity_drives_the_logger_threshold_end_to_end(tmp_path):
    log_file = tmp_path / "v.log"
    blog._applied = None
    blog.configure(ObservabilityConfig(verbosity="verbose", console=False, log_file=str(log_file)))
    assert blog.get_logger().level == logging.INFO
    blog.log_kv(blog.get_logger("kyber"), logging.INFO, "visible at verbose", n=1)
    blog.log_kv(blog.get_logger("kyber"), logging.DEBUG, "hidden at verbose", n=2)
    for handler in blog.get_logger().handlers:
        handler.flush()
    text = log_file.read_text()
    assert "visible at verbose" in text and "hidden at verbose" not in text


def test_verbosity_reaches_the_native_tracing_bridge(tmp_path):
    blog._applied = None
    blog.configure(
        ObservabilityConfig(verbosity="trace", console=False, log_file=str(tmp_path / "t"))
    )
    assert blog.native_tracing_settings() == ("trace", False)


def test_verbosity_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("BATCHER_OBSERVABILITY_VERBOSITY", "debug")
    from batcher.config.config import Config

    assert Config.from_env().observability.resolved_log_level == "DEBUG"


# --- cost: observability off must stay free ---------------------------------


def test_nothing_is_published_when_no_sink_is_attached(bus):
    """The zero-overhead promise: with no sink, the reporting path must not even build ids."""
    from batcher.api.terminal.event_log import start_query_report

    assert bus.listening() is False
    assert start_query_report("aggregate") == ""


def test_streaming_passes_batches_through_untouched_with_no_sink(bus):
    ds = bt.from_pydict({"x": list(range(20_000))})
    assert bus.listening() is False
    assert sum(b.num_rows for b in ds.iter_batches()) == 20_000


def test_a_quiet_query_emits_nothing_to_the_terminal(bus, capsys):
    with config_context(
        active_config().replace(observability=ObservabilityConfig(verbosity="silent"))
    ):
        bt.from_pydict({"x": [1, 2, 3]}).collect()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_progress_events_do_not_slow_a_stream_measurably(bus):
    """A sanity bound, not a benchmark: the per-batch publish must not dominate."""
    ds = bt.from_pydict({"x": list(range(200_000))})
    t0 = time.perf_counter()
    list(ds.iter_batches())
    baseline = time.perf_counter() - t0
    bus.subscribe(ActivityStore().handle)
    t0 = time.perf_counter()
    list(ds.iter_batches())
    observed = time.perf_counter() - t0
    assert observed < baseline * 20 + 0.5, f"baseline={baseline:.4f}s observed={observed:.4f}s"


def test_verbose_actually_shows_more_than_normal(bus, tmp_path):
    """Each rung must earn its place: `verbose` promised decisions and showed none."""

    def run_at(verbosity: str) -> str:
        log_file = tmp_path / f"{verbosity}.log"
        blog._applied = None
        blog.configure(
            ObservabilityConfig(verbosity=verbosity, console=False, log_file=str(log_file))
        )
        left = bt.from_pydict({"k": [1, 2, 3] * 20, "x": list(range(60))})
        right = bt.from_pydict({"k": [1, 2, 3], "y": [10, 20, 30]})
        left.join(right, on="k").collect()
        for handler in blog.get_logger().handlers:
            handler.flush()
        return log_file.read_text()

    normal, verbose, debug = (run_at(v) for v in ("normal", "verbose", "debug"))
    assert normal.strip() == ""
    assert "plan admitted" in verbose and "join build side" in verbose
    assert "run phase" not in verbose  # per-phase timing is a DEBUG concern
    assert "run phase" in debug  # ...and debug adds it


def test_load_per_core_uses_the_cores_this_process_may_use(monkeypatch):
    """The regression: `load_per_core` divided by `os.cpu_count()`, the *host's* cores. A
    container limited to 4 of 64 divided a saturating load of 4.0 by 64 and reported 0.06 —
    "idle" at exactly the moment it was pegged and throttled, which is the contention this
    metric exists to surface."""
    import os as _os

    from batcher.observe.insights import resources

    monkeypatch.setattr(_os, "getloadavg", lambda: (4.0, 4.0, 4.0))
    monkeypatch.setattr(_os, "cpu_count", lambda: 64)
    monkeypatch.setattr("batcher._internal.hardware.available_cpu_count", lambda: 4, raising=True)
    assert resources.cpu_contention()["load_per_core"] == 1.0
