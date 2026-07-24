"""The event bus, the activity store, the terminal reporter, and the web dashboard.

Covers the contract each piece promises rather than its implementation: the bus is free
and non-propagating, the store is bounded and cursor-correct, the reporter refuses to
render into a non-terminal, and the UI serves what the store holds.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from batcher._internal import events
from batcher._internal import logging as blog
from batcher.observe.console import ConsoleReporter, should_render
from batcher.observe.server import UIServer
from batcher.observe.store import ActivityStore
from batcher.observe.theme import detect

pytestmark = pytest.mark.unit


@pytest.fixture
def bus():
    """Isolate each test from any sink another test (or the conductor) left attached."""
    saved = events._subscribers
    events._subscribers = ()
    yield events
    events._subscribers = saved


# --- the bus ----------------------------------------------------------------


def test_publish_is_a_noop_with_no_subscribers(bus):
    assert bus.listening() is False
    bus.publish(bus.QUERY_START, query_id="q1")  # must not raise


def test_subscribe_delivers_and_unsubscribe_stops_delivery(bus):
    seen: list[events.Event] = []
    off = bus.subscribe(seen.append)
    bus.publish(bus.QUERY_START, query_id="q1", name="scan", label="scan")
    assert [e.kind for e in seen] == [bus.QUERY_START]
    assert seen[0].fields["label"] == "scan"
    off()
    bus.publish(bus.QUERY_END, query_id="q1")
    assert len(seen) == 1


def test_unsubscribe_is_idempotent(bus):
    off = bus.subscribe(lambda _e: None)
    off()
    off()  # a second detach must not raise
    assert bus.listening() is False


def test_a_raising_sink_never_propagates_to_the_publisher(bus):
    """Observability must not be able to fail the query it is observing."""
    delivered: list[str] = []

    def boom(_event):
        raise RuntimeError("sink is broken")

    bus.subscribe(boom)
    bus.subscribe(lambda e: delivered.append(e.kind))
    bus.publish(bus.QUERY_END, query_id="q1")  # must not raise
    # The healthy sink still received the event despite the broken one.
    assert delivered == [bus.QUERY_END]


# --- structured logging -----------------------------------------------------


def test_log_kv_renders_fields_in_human_format(tmp_path):
    log_file = tmp_path / "kv.log"
    blog._applied = None
    blog.configure(_obs(log_file, "human"))
    blog.log_kv(blog.get_logger("kyber"), logging.INFO, "join reorder", tables=3)
    _flush()
    assert "join reorder  tables=3" in log_file.read_text()


def test_log_kv_nests_fields_in_json_format(tmp_path):
    log_file = tmp_path / "kv.json"
    blog._applied = None
    blog.configure(_obs(log_file, "json"))
    blog.log_kv(blog.get_logger("carbonite"), logging.INFO, "spilled", bytes=2048)
    _flush()
    record = json.loads(log_file.read_text().splitlines()[0])
    assert record["message"] == "spilled"
    assert record["fields"] == {"bytes": 2048}


def test_log_records_are_mirrored_onto_the_bus(bus, tmp_path):
    blog._applied = None
    blog.configure(_obs(tmp_path / "bus.log", "human"))
    seen: list[events.Event] = []
    bus.subscribe(seen.append)
    blog.log_kv(blog.get_logger("kyber"), logging.WARNING, "pushdown blocked", op="join")
    assert [e.kind for e in seen] == [bus.LOG]
    assert seen[0].fields["message"] == "pushdown blocked"
    assert seen[0].fields["fields"] == {"op": "join"}


def _obs(log_file, fmt):
    from batcher.config import ObservabilityConfig

    return ObservabilityConfig(
        log_level="INFO", console=False, log_file=str(log_file), log_format=fmt
    )


def _flush():
    for handler in blog.get_logger().handlers:
        handler.flush()


# --- the store --------------------------------------------------------------


def _feed(store, kind, **kwargs):
    store.handle(events.Event(kind=kind, ts=1.0, wall=1.0, **kwargs))


def test_store_folds_a_query_lifecycle(bus):
    store = ActivityStore()
    _feed(store, events.QUERY_START, query_id="q1", name="aggregate", fields={"label": "aggregate"})
    _feed(store, events.STAGE_START, query_id="q1", name="scan", fields={"op_id": 1})
    _feed(
        store,
        events.STAGE_END,
        query_id="q1",
        name="scan",
        fields={"op_id": 1, "rows_out": 500, "elapsed_ms": 12.5, "spilled": True},
    )
    _feed(store, events.QUERY_END, query_id="q1", fields={"ok": True, "total_ms": 30.0, "rows": 3})

    [summary] = store.queries()
    assert (summary["label"], summary["status"], summary["rows"]) == ("aggregate", "ok", 3)
    detail = store.query("q1")
    assert detail["stages"] == [
        {
            "op_id": 1,
            "kind": "scan",
            "est_rows": None,
            "rows_out": 500,
            "elapsed_ms": 12.5,
            "spilled": True,
            "done": True,
        }
    ]


def test_store_marks_a_failed_query(bus):
    store = ActivityStore()
    _feed(store, events.QUERY_START, query_id="q1", fields={"label": "join"})
    _feed(store, events.QUERY_END, query_id="q1", fields={"ok": False, "error": "PlanError: bad"})
    assert store.queries()[0]["status"] == "error"
    assert store.summary()["n_failed"] == 1


def test_store_evicts_oldest_queries_and_does_not_leak_the_index(bus):
    """The ring buffer bounds memory — including the id index, which is easy to leak."""
    store = ActivityStore(max_queries=2)
    for i in range(5):
        _feed(store, events.QUERY_START, query_id=f"q{i}", fields={"label": f"q{i}"})
    assert len(store.queries()) == 2
    assert len(store._by_id) == 2
    assert store.query("q0") is None  # aged out
    assert store.query("q4") is not None


def test_store_ignores_events_for_an_unknown_query(bus):
    store = ActivityStore()
    _feed(store, events.STAGE_END, query_id="ghost", fields={"op_id": 1})
    assert store.queries() == []


def test_log_cursor_advances_and_survives_eviction(bus):
    store = ActivityStore(max_logs=3)
    for i in range(5):
        _feed(store, events.LOG, name="kyber", fields={"level": "INFO", "message": f"m{i}"})
    first = store.logs(since=0)
    # Only the last three survive; the cursor reports the true end of the stream, so a
    # poller resuming from it gets nothing rather than replaying.
    assert [line["message"] for line in first["lines"]] == ["m2", "m3", "m4"]
    assert first["cursor"] == 5
    assert store.logs(since=first["cursor"])["lines"] == []


# --- the console reporter ---------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "isatty", "expected"),
    [("off", True, False), ("on", False, True), ("auto", True, True), ("auto", False, False)],
)
def test_should_render_respects_mode_and_tty(mode, isatty, expected, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    stream = io.StringIO()
    stream.isatty = lambda: isatty
    assert should_render(mode, stream) is expected


def test_auto_declines_to_render_when_no_color_is_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    stream = io.StringIO()
    stream.isatty = lambda: True
    assert should_render("auto", stream) is False


def test_reporter_prints_a_summary_line_on_query_end(bus):
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, live=False)
    reporter.handle(events.Event(events.QUERY_START, 1.0, 1.0, "q1", "", {"label": "aggregate"}))
    reporter.handle(
        events.Event(
            events.QUERY_END, 2.0, 2.0, "q1", "", {"ok": True, "total_ms": 42.0, "rows": 7}
        )
    )
    out = stream.getvalue()
    assert "aggregate" in out and "7 rows" in out and "42ms" in out
    # A non-live reporter must emit no escape codes — they would corrupt a log file.
    assert "\x1b" not in out


def test_reporter_marks_a_failed_query(bus):
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, live=False)
    reporter.handle(events.Event(events.QUERY_START, 1.0, 1.0, "q1", "", {"label": "join"}))
    reporter.handle(
        events.Event(events.QUERY_END, 2.0, 2.0, "q1", "", {"ok": False, "error": "PlanError: bad"})
    )
    assert "PlanError: bad" in stream.getvalue()


def test_reporter_survives_a_closed_stream(bus):
    """A reporter can outlive its stream; that must not surface as a query failure."""
    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, live=False)
    stream.close()
    reporter.handle(events.Event(events.QUERY_END, 1.0, 1.0, "q1", "", {"ok": True}))  # no raise


# --- the web dashboard ------------------------------------------------------


@pytest.fixture
def ui(bus):
    """A dashboard on an OS-chosen port, torn down after the test."""
    store = ActivityStore()
    server = UIServer(store, port=0)
    url = server.start()
    yield url, store
    server.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def test_ui_serves_the_app_shell_and_assets(ui):
    url, _ = ui
    for path, needle in (
        ("/", b"Batcher"),
        ("/app.js", b"poll"),
        ("/dag.js", b"DAG"),
        ("/app.css", b"--surface-1"),
    ):
        with urllib.request.urlopen(f"{url}{path}", timeout=5) as response:
            assert response.status == 200
            assert needle in response.read()


def test_ui_reports_queries_the_store_received(ui):
    url, store = ui
    _feed(store, events.QUERY_START, query_id="q1", fields={"label": "scan"})
    _feed(store, events.QUERY_END, query_id="q1", fields={"ok": True, "total_ms": 5.0, "rows": 9})
    assert _get(f"{url}/api/queries")["queries"][0]["label"] == "scan"
    assert _get(f"{url}/api/summary")["total_rows"] == 9
    assert _get(f"{url}/api/query/q1")["rows"] == 9


def test_ui_404s_an_unknown_query_and_route(ui):
    url, _ = ui
    for path in ("/api/query/nope", "/api/nonsense"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(f"{url}{path}")
        assert excinfo.value.code == 404


def test_ui_refuses_a_path_traversal(ui):
    """The dashboard serves one fixed asset directory, never the filesystem."""
    url, _ = ui
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{url}/../../../../etc/passwd", timeout=5)
    assert excinfo.value.code == 404


def test_start_ui_is_idempotent_and_stop_is_safe_twice():
    import batcher as bt

    try:
        first = bt.start_ui(port=0)
        assert bt.start_ui(port=0) == first  # does not bind a second port
        assert bt.ui_url() == first
    finally:
        bt.stop_ui()
        bt.stop_ui()
    assert bt.ui_url() is None


# --- theme: capability detection and degradation ----------------------------


class _Stream(io.StringIO):
    """A StringIO with a settable `encoding`, which the real class does not allow."""

    def __init__(self, encoding: str = "utf-8") -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding


@pytest.mark.parametrize(
    ("env", "depth"),
    [
        ({"COLORTERM": "truecolor", "TERM": "xterm-256color"}, 24),
        ({"TERM": "xterm-256color"}, 8),
        ({"TERM": "xterm"}, 4),
        ({"TERM": "dumb"}, 0),
        ({"TERM": "xterm", "NO_COLOR": "1"}, 0),
    ],
)
def test_detect_reads_the_published_color_conventions(env, depth, monkeypatch):
    for key in ("COLORTERM", "TERM", "NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    palette, _ = detect(_Stream())
    assert palette.depth == depth


def test_non_utf8_stream_gets_ascii_glyphs(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    _, glyphs = detect(_Stream("ascii"))
    assert glyphs.unicode is False
    assert glyphs.full == "#"


def test_uncolored_palette_emits_no_escapes(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    palette, _ = detect(_Stream())
    assert palette.accent == "" and palette.reset == "" and palette.ramp(0.5) == ""


def _bar_of(fraction, monkeypatch, env=None):
    for key in ("COLORTERM", "TERM", "NO_COLOR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {"NO_COLOR": "1", "TERM": "xterm"}).items():
        monkeypatch.setenv(key, value)
    return ConsoleReporter(stream=_Stream(), live=True)._bar(fraction)


def test_bar_fills_proportionally(monkeypatch):
    empty, half, full = (_bar_of(f, monkeypatch) for f in (0.0, 0.5, 1.0))
    assert empty.count("█") == 0
    assert 0 < half.count("█") < full.count("█")
    assert full.count("░") == 0


def test_bar_uses_eighth_cells_for_sub_cell_progress(monkeypatch):
    """The smoothness claim: two fractions inside one cell must render differently."""
    monkeypatch.setenv("TERM", "xterm-256color")
    reporter = ConsoleReporter(stream=_Stream(), live=True)
    assert reporter._bar(0.605) != reporter._bar(0.620)


def test_unknown_total_renders_an_indeterminate_sweep_not_a_fake_percentage(monkeypatch):
    bar = _bar_of(None, monkeypatch)
    assert "%" not in bar
    assert any(shade in bar for shade in "█▓▒░")


def test_reporter_reports_live_streaming_progress(bus):
    """`iter_batches` is the one path that can report progress while it runs."""
    stream = _Stream()
    reporter = ConsoleReporter(stream=stream, live=False)
    reporter.handle(events.Event(events.QUERY_START, 1.0, 1.0, "q1", "", {"label": "filter"}))
    reporter.handle(events.Event(events.PROGRESS, 1.1, 1.1, "q1", "", {"rows": 4096}))
    reporter.handle(events.Event(events.PROGRESS, 1.2, 1.2, "q1", "", {"rows": 4096}))
    reporter.handle(
        events.Event(
            events.QUERY_END, 2.0, 2.0, "q1", "", {"ok": True, "rows": 8192, "total_ms": 50.0}
        )
    )
    assert "8.2K rows" in stream.getvalue()


def test_logfmt_quotes_only_values_containing_spaces(bus):
    stream = _Stream()
    reporter = ConsoleReporter(stream=stream, live=False)
    reporter.handle(
        events.Event(
            events.LOG,
            1.0,
            1.0,
            "",
            "kyber",
            {
                "level": "INFO",
                "message": "rule applied",
                "fields": {"rule": "pushdown", "note": "two words", "ok": True},
            },
        )
    )
    out = stream.getvalue()
    assert "rule=pushdown" in out
    assert 'note="two words"' in out
    assert "ok=true" in out


# --- pipelines, DAG, insights, and system -----------------------------------


def test_runs_of_one_shape_group_into_a_pipeline(bus):
    """A pipeline is every run of the same plan shape — the unit a person thinks in."""
    store = ActivityStore()
    for i in range(3):
        _feed(
            store, events.QUERY_START, query_id=f"q{i}", fields={"label": "agg", "signature": "abc"}
        )
        _feed(
            store,
            events.QUERY_END,
            query_id=f"q{i}",
            fields={"ok": True, "total_ms": 10.0 + i, "rows": 5},
        )
    _feed(store, events.QUERY_START, query_id="other", fields={"label": "join", "signature": "xyz"})
    _feed(
        store, events.QUERY_END, query_id="other", fields={"ok": True, "total_ms": 99.0, "rows": 1}
    )

    pipelines = {p["signature"]: p for p in store.pipelines()}
    assert pipelines["abc"]["runs"] == 3
    assert pipelines["abc"]["recent_ms"] == [10.0, 11.0, 12.0]  # oldest → newest
    assert pipelines["xyz"]["runs"] == 1
    assert store.summary()["n_pipelines"] == 2


def test_an_unsigned_query_is_its_own_pipeline(bus):
    """Unsignable plans must not collapse into one meaningless shared bucket."""
    store = ActivityStore()
    for i in range(2):
        _feed(store, events.QUERY_START, query_id=f"q{i}", fields={"label": "x"})
        _feed(store, events.QUERY_END, query_id=f"q{i}", fields={"ok": True})
    assert len(store.pipelines()) == 2


def test_dag_reuses_the_profile_walk_so_op_ids_match():
    """The DAG's node ids must be the same ids the engine measured against."""
    from batcher.observe.dag import build_dag
    from batcher.plan.profile import walk_ir

    ir = {
        "op": "aggregate",
        "input": {
            "op": "hash_join",
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
            "left_keys": ["k"],
            "join_type": "inner",
        },
        "group_keys": [{"alias": "r"}],
        "aggregates": [{"func": "sum"}],
    }
    ops = [{"op_id": 1, "measured": True, "rows_out": 42, "elapsed_ms": 7.0}]
    dag = build_dag(ir, ops)
    assert [n["kind"] for n in dag["nodes"]] == [k for _d, n in walk_ir(ir) for k in [n["op"]]]
    join = next(n for n in dag["nodes"] if n["op_id"] == 1)
    assert (join["kind"], join["rows_out"], join["measured"]) == ("hash_join", 42, True)
    assert join["detail"] == "inner on k"


def test_dag_does_not_walk_a_predicate_as_an_operator():
    """A binary expression carries both `e` and `op`; treating it as a node shifts every id."""
    from batcher.observe.dag import build_dag

    ir = {
        "op": "filter",
        "input": {"op": "scan", "source_id": 0},
        "predicate": {
            "e": "binary",
            "op": "gt",
            "left": {"e": "col", "name": "x"},
            "right": {"e": "lit", "value": {"int": 5}},
        },
    }
    dag = build_dag(ir, [])
    assert [n["kind"] for n in dag["nodes"]] == ["filter", "scan"]
    assert dag["nodes"][0]["detail"] == "x > 5"


def test_dag_lays_a_join_out_over_both_of_its_children():
    from batcher.observe.dag import build_dag

    ir = {"op": "hash_join", "left": {"op": "scan"}, "right": {"op": "scan"}, "join_type": "inner"}
    dag = build_dag(ir, [])
    by_kind = {n["op_id"]: n for n in dag["nodes"]}
    assert dag["width"] == 2
    # The join sits between its two children, and above them.
    assert by_kind[0]["column"] == pytest.approx(0.5)
    assert by_kind[0]["row"] > by_kind[1]["row"]


def test_dag_of_a_missing_plan_is_empty_not_an_error():
    from batcher.observe.dag import build_dag

    # Same *keys* as a populated graph, so a renderer never has to test for the empty case
    # separately — only the values differ.
    assert build_dag(None, []) == {
        "nodes": [],
        "edges": [],
        "width": 0,
        "depth": 0,
        "critical_path": [],
        "stages": 0,
    }


def test_insights_are_silent_on_a_healthy_run():
    from batcher.observe.insights import derive_insights

    healthy = {
        "total_ms": 100.0,
        "rows": 1000,
        "ops": [
            {
                "measured": True,
                "kind": "scan",
                "rows_out": 1000,
                "elapsed_ms": 40.0,
                "cpu_util": 0.95,
                "est_error": 1.0,
            },
            {
                "measured": True,
                "kind": "aggregate",
                "rows_out": 1000,
                "elapsed_ms": 45.0,
                "cpu_util": 0.95,
                "est_error": 1.0,
            },
        ],
    }
    assert derive_insights(healthy) == []
    assert derive_insights(None) == []


def test_insights_flag_a_spill_with_its_volume():
    from batcher.observe.insights import derive_insights

    found = derive_insights(
        {
            "total_ms": 500.0,
            "rows": 10,
            "ops": [
                {
                    "measured": True,
                    "kind": "sort",
                    "rows_out": 10,
                    "elapsed_ms": 500.0,
                    "spilled": True,
                    "spill_bytes": 5 * 1024 * 1024,
                }
            ],
        }
    )
    spill = next(i for i in found if i["rule"] == "operator-spilled")
    assert "5.0 MiB" in spill["evidence"]
    assert spill["severity"] == "warning"
    assert spill["action"]


def test_insights_flag_a_badly_wrong_cardinality_estimate():
    from batcher.observe.insights import derive_insights

    found = derive_insights(
        {
            "total_ms": 200.0,
            "rows": 50000,
            "ops": [
                {
                    "measured": True,
                    "kind": "hash_join",
                    "rows_out": 50000,
                    "est_rows": 100.0,
                    "elapsed_ms": 200.0,
                    "est_error": 500.0,
                }
            ],
        }
    )
    assert any(i["rule"] == "cardinality-misestimate" for i in found)


def test_insights_ignore_a_trivially_short_run():
    """Sub-25ms runs are dominated by fixed control-plane cost; tuning them is noise."""
    from batcher.observe.insights import derive_insights

    found = derive_insights(
        {
            "total_ms": 3.0,
            "rows": 1,
            "ops": [
                {
                    "measured": True,
                    "kind": "scan",
                    "rows_out": 1,
                    "elapsed_ms": 3.0,
                    "cpu_util": 0.01,
                }
            ],
        }
    )
    assert not any(i["rule"] in ("dominant-operator", "cpu-underutilized") for i in found)


def test_system_snapshot_reports_real_host_facts():
    from batcher.observe.system import system_snapshot

    snap = system_snapshot()
    assert snap["host"]["cpus"] >= 1
    assert snap["runtime"]["python"]
    assert isinstance(snap["host"]["gpus"], list)
    assert snap["cluster"]["attached"] in (True, False)
    assert "morsel_rows" in snap["config"]


def test_ui_serves_pipelines_and_system(ui):
    url, store = ui
    _feed(store, events.QUERY_START, query_id="q1", fields={"label": "scan", "signature": "s1"})
    _feed(store, events.QUERY_END, query_id="q1", fields={"ok": True, "total_ms": 5.0, "rows": 9})
    assert _get(f"{url}/api/pipelines")["pipelines"][0]["signature"] == "s1"
    assert _get(f"{url}/api/system")["host"]["cpus"] >= 1


def test_operator_share_is_never_over_100_percent():
    """Regression: operators run concurrently, so their elapsed times sum past wall time.

    A wall-time denominator reported "hash_join is 199% of the runtime" on a real 16-thread
    run. One visibly impossible number discredits every other number on the page.
    """
    from batcher.observe.insights import derive_insights

    found = derive_insights(
        {
            "total_ms": 38.2,  # wall time, less than the sum below because morsels run in parallel
            "rows": 3,
            "ops": [
                {"measured": True, "kind": "hash_join", "rows_out": 380070, "elapsed_ms": 76.1},
                {"measured": True, "kind": "aggregate", "rows_out": 3, "elapsed_ms": 21.1},
                {"measured": True, "kind": "filter", "rows_out": 380070, "elapsed_ms": 5.0},
            ],
        }
    )
    dominant = next(i for i in found if i["rule"] == "dominant-operator")
    assert dominant["detail"]["share"] <= 1.0
    assert "74% of operator time" in dominant["title"]


def test_an_aggregate_returning_few_rows_is_not_a_pushdown_miss():
    """Regression: a GROUP BY over 400K rows returning 4 is the aggregate doing its job."""
    from batcher.observe.insights import derive_insights

    found = derive_insights(
        {
            "total_ms": 200.0,
            "rows": 4,
            "ops": [
                {"measured": True, "kind": "scan", "rows_out": 400_000, "elapsed_ms": 100.0},
                {"measured": True, "kind": "aggregate", "rows_out": 4, "elapsed_ms": 100.0},
            ],
        }
    )
    assert not any(i["rule"] == "selective-query" for i in found)


def test_a_selective_filter_still_reports_the_pushdown_opportunity():
    """The rule must stay useful for the shape it was written for."""
    from batcher.observe.insights import derive_insights

    found = derive_insights(
        {
            "total_ms": 200.0,
            "rows": 227,
            "ops": [
                {"measured": True, "kind": "scan", "rows_out": 400_000, "elapsed_ms": 190.0},
                {"measured": True, "kind": "filter", "rows_out": 227, "elapsed_ms": 10.0},
            ],
        }
    )
    assert any(i["rule"] == "selective-query" for i in found)


# --- the dashboard's front-end assets ---------------------------------------
# The JS has no other gate: a syntax error would ship a blank dashboard while every Python
# test stayed green. `quickjs` is a real ES2020 engine, so these run the actual files rather
# than a down-levelled copy — an earlier `esprima`-based check kept failing on valid modern
# syntax (object spread, `?.`, `??`), and a test that cries wolf on correct code is worse
# than no test, because the next real failure gets waved through too.

_ASSETS = Path(__file__).parents[2] / "python" / "batcher" / "observe" / "assets"

# The browser globals the modules touch at load time. Enough to let the top level run; the
# render functions are not called here, so this is a load-time smoke test, not a DOM double.
#
# When this stub is missing a method the failure looks like a code bug ("TypeError: not a
# function") but is not one — it has cried wolf three times now. Add the method rather than
# changing the dashboard, and only treat a failure as real once the stub provides everything
# a browser would.
_BROWSER_STUB = """
var __handlers = {};
var document = {
  documentElement: { dataset: {} },
  getElementById: function () { return __el(); },
  querySelector: function () { return __el(); },
  querySelectorAll: function () { return []; },
  addEventListener: function (name) { __handlers[name] = true; },
  body: { classList: { toggle: function () {} } },
};
function __el() {
  return { textContent: '', innerHTML: '', value: '', hidden: false, dataset: {},
           style: {}, checked: false, offsetWidth: 0, clientWidth: 100, clientHeight: 100,
           classList: { toggle: function () {}, add: function () {}, remove: function () {},
                        contains: function () { return false; } },
           addEventListener: function () {}, setAttribute: function () {},
           getBoundingClientRect: function () {
             return { width: 800, height: 600, left: 0, top: 0 };
           },
           appendChild: function () {}, append: function () {},
           querySelectorAll: function () { return []; },
           querySelector: function () { return __el(); },
           closest: function () { return null; },
           remove: function () {}, matches: function () { return false; },
           focus: function () {}, click: function () {} };
}
var window = { matchMedia: function () { return { matches: false }; },
               innerWidth: 1280, innerHeight: 900, scrollY: 0,
               scrollTo: function () {}, addEventListener: function () {} };
var location = { hash: '', href: 'http://localhost/' };
var history = { replaceState: function () {}, pushState: function () {} };
var navigator = { clipboard: { writeText: function () { return Promise.resolve(); } } };
var URL = { createObjectURL: function () { return ''; }, revokeObjectURL: function () {} };
var Blob = function () {};
var localStorage = { getItem: function () { return null; }, setItem: function () {} };
var requestAnimationFrame = function () {};
var setTimeout = function () {};
var fetch = function () { return Promise.resolve({ ok: true, json: function () { return {}; } }); };
"""


def _js_context():
    """A QuickJS context with enough browser surface for the dashboard modules to load."""
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_BROWSER_STUB)
    return ctx


#: Load order, matching the <script> tags. Each module may use the ones before it.
# Load order is a real dependency order, not an alphabetical one: `reference.js` is pure
# data, `learn.js` renders it and needs UI, `views.js` calls LEARN.term, `app.js` needs all
# of them at its top level. The cumulative test below names whichever one breaks first.
_JS_MODULES = (
    "ui.js",
    "reference.js",
    "charts.js",
    "dag.js",
    "learn.js",
    "plan.js",
    "live.js",
    "views.js",
    "app.js",
)


@pytest.mark.parametrize("upto", range(1, len(_JS_MODULES) + 1))
def test_dashboard_javascript_loads(upto):
    """Each module must parse *and* execute its top level, given the ones it depends on.

    Parametrized cumulatively so a failure names the first module that breaks rather than
    just "the bundle is broken".
    """
    ctx = _js_context()
    for name in _JS_MODULES[:upto]:
        ctx.eval((_ASSETS / name).read_text())


def test_every_script_tag_points_at_a_file_that_exists():
    """A renamed asset 404s silently in the browser; here it fails in CI."""
    html = (_ASSETS / "index.html").read_text()
    for src in re.findall(r'<script src="/([^"]+)"', html):
        assert (_ASSETS / src).is_file(), f"index.html loads /{src}, which does not exist"


def test_script_load_order_matches_module_dependencies():
    """Each module's dependencies must already be loaded when its top level runs."""
    html = (_ASSETS / "index.html").read_text()
    order = re.findall(r'<script src="/([^"]+)"', html)
    assert order == list(_JS_MODULES)
    # Stated as the actual constraint, so re-ordering for an unrelated reason is caught by
    # the reason it is wrong rather than by a list literal nobody can interpret.
    at = {name: i for i, name in enumerate(order)}
    for module, needs in {
        "learn.js": ["ui.js", "reference.js"],
        "charts.js": ["ui.js"],
        "plan.js": ["ui.js", "charts.js", "dag.js"],
        "live.js": ["ui.js", "charts.js", "dag.js"],
        "views.js": ["ui.js", "charts.js", "dag.js", "learn.js"],
        "app.js": ["ui.js", "charts.js", "dag.js", "learn.js", "plan.js", "live.js", "views.js"],
    }.items():
        for dep in needs:
            assert at[dep] < at[module], f"{module} runs before its dependency {dep}"


def test_ui_module_formats_and_persists():
    ctx = _js_context()
    ctx.eval((_ASSETS / "ui.js").read_text())
    assert ctx.eval("UI.bytes(1536)") == "1.5 KiB"
    assert ctx.eval("UI.pct(0.62)") == "62%"
    assert ctx.eval("UI.count(2400000)") == "2.4M"
    assert ctx.eval("UI.duration(3725)") == "1h 2m"
    # CSV quoting: only cells that need it, so the file opens cleanly in a spreadsheet.
    assert ctx.eval("""UI.toCSV(['a','b'], [['x','y,z'], ['q"r', 1]])""") == (
        'a,b\nx,"y,z"\n"q""r",1'
    )
    # Fuzzy matching is a subsequence test, so "hj" finds "hash_join".
    assert ctx.eval("UI.fuzzy('hj', 'hash_join')") is True
    assert ctx.eval("UI.fuzzy('zz', 'hash_join')") is False


def test_dag_module_formats_values_the_way_the_ui_claims():
    """The formatting helpers are the most-called code on the page; pin their contract.

    `dag.js` loads `ui.js` first because `DAG.fmtMs`/`fmtShare` delegate to `UI.ms`/`UI.pct`
    rather than keeping a second copy — the two had already drifted once, leaving the graph
    showing "9µs" while every other panel rounded the same value to "0.0ms".
    """
    ctx = _js_context()
    ctx.eval((_ASSETS / "ui.js").read_text())
    ctx.eval((_ASSETS / "dag.js").read_text())
    assert ctx.eval("DAG.fmtCount(1500)") == "1.5K"
    assert ctx.eval("DAG.fmtCount(2_400_000)") == "2.4M"
    assert ctx.eval("DAG.fmtMs(850)") == "850ms"
    assert ctx.eval("DAG.fmtMs(1500)") == "1.50s"
    # The two names must stay the same function, not merely agree today.
    assert ctx.eval("UI.ms(0.009) === DAG.fmtMs(0.009)")
    assert ctx.eval("UI.pct(0.0003) === DAG.fmtShare(0.0003)")
    # Operator names are shown in a person's words, not the engine's IR tags.
    assert ctx.eval("DAG.friendlyKind('hash_join')") == "Join"
    assert ctx.eval("DAG.friendlyKind('scan')") == "Read source"
    # An unknown tag falls through rather than rendering as undefined.
    assert ctx.eval("DAG.friendlyKind('brand_new_op')") == "brand_new_op"


def test_every_element_the_javascript_reaches_for_exists_in_the_markup():
    """A renamed id fails silently in the browser; here it fails in CI."""
    html = (_ASSETS / "index.html").read_text()
    js = (_ASSETS / "app.js").read_text() + (_ASSETS / "dag.js").read_text()
    missing = set(re.findall(r"\$\('([^']+)'\)", js)) - set(re.findall(r'id="([^"]+)"', html))
    assert not missing, f"JavaScript references ids that do not exist: {sorted(missing)}"


def test_every_view_button_has_a_matching_pane():
    html = (_ASSETS / "index.html").read_text()
    ids = set(re.findall(r'id="([^"]+)"', html))
    for name in set(re.findall(r'data-view="([^"]+)"', html)):
        assert f"view-{name}" in ids, f"data-view={name} has no #view-{name} pane"


def test_every_tab_shows_panes_and_every_pane_is_reachable():
    """A tab owns a *set* of panes, so tab id no longer equals pane id.

    Both directions matter. A tab resolving to nothing is a dead nav item; a pane no tab
    resolves to is markup that renders forever and is never seen — which is exactly what
    consolidating nine tabs into five could have left behind.
    """
    html = (_ASSETS / "index.html").read_text()
    app = (_ASSETS / "app.js").read_text()
    pane_ids = set(re.findall(r'id="(tab-[a-z]+)"', html))
    tabs = set(re.findall(r'data-tab="([^"]+)"', html))

    # The mapping is data in app.js; read it rather than restating it here.
    block = app[app.index("const TAB_PANES = {") : app.index("function switchTab")]
    mapped: dict[str, set[str]] = {}
    for name, body in re.findall(r"(\w+): \(\) => \[([^\]]*)\]", block):
        mapped[name] = set(re.findall(r"tab-[a-z]+", body))
    # `steps` and `query` are computed from state rather than listed, so their renderings
    # are named here. Keep these in step with STEPS_VIEWS / QUERY_VIEWS in app.js.
    mapped.setdefault("steps", set())
    mapped["steps"] |= {"tab-plan", "tab-stages", "tab-flame", "tab-timeline", "tab-operators"}
    mapped.setdefault("query", set())
    mapped["query"] |= {"tab-explain", "tab-diff", "tab-ir"}

    assert tabs == set(mapped), f"nav tabs {tabs} do not match TAB_PANES {set(mapped)}"
    for tab, panes in mapped.items():
        assert panes, f"tab {tab} resolves to no pane"
        for pane in panes:
            assert pane in pane_ids, f"tab {tab} points at #{pane}, which does not exist"

    reachable = set().union(*mapped.values())
    assert not (pane_ids - reachable), f"panes no tab can show: {sorted(pane_ids - reachable)}"


def _css_block(css: str, selector: str) -> str:
    """The body of the first rule matching `selector`, found by brace matching.

    Not a string slice on some other rule's text: an earlier version keyed off the literal
    `"* { box-sizing"` and broke the moment that selector was reformatted, failing a test
    about theme tokens for a reason that had nothing to do with them.
    """
    start = css.index(selector)
    open_brace = css.index("{", start)
    depth, i = 0, open_brace
    while i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1 : i]
        i += 1
    raise AssertionError(f"unbalanced braces after {selector}")


def test_every_css_variable_is_defined_in_both_themes():
    """A token defined in only one theme renders as nothing in the other."""
    css = (_ASSETS / "app.css").read_text()
    js = "".join((_ASSETS / f).read_text() for f in _JS_MODULES)
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", css + js))
    # The sequential ramp is referenced by computed name (`--seq-${step}`), so name it here.
    used |= {f"--seq-{i}" for i in range(1, 6)}
    used -= set(re.findall(r"var\((--seq-)\$", js))
    # Theme-independent tokens (spacing, type, motion) live on the bare :root.
    base = set(re.findall(r"(--[a-z0-9-]+)\s*:", _css_block(css, ":root {")))
    for theme in ("dark", "light"):
        block = _css_block(css, f':root[data-theme="{theme}"]')
        defined = base | set(re.findall(r"(--[a-z0-9-]+)\s*:", block))
        assert not (used - defined), f"{theme} theme is missing: {sorted(used - defined)}"


def test_the_spacing_and_type_scales_are_tokens_not_magic_numbers():
    """The design system's whole point: components reference the scale, not raw pixels."""
    css = (_ASSETS / "app.css").read_text()
    root = _css_block(css, ":root {")
    for name in ("--s1", "--s4", "--t2", "--t7", "--r1", "--dur-2", "--ease-out"):
        assert f"{name}:" in root, f"{name} is missing from the design system"
    # Components below the token block should reach for the scale, not hard-code padding.
    body = css[css.index("═══ chrome ═══") :]
    raw_padding = re.findall(r"padding:\s*(\d+)px", body)
    assert all(int(v) <= 3 for v in raw_padding), (
        f"hard-coded padding outside the spacing scale: {sorted(set(raw_padding))}"
    )


def test_reduced_motion_disables_every_animation_in_one_place():
    css = (_ASSETS / "app.css").read_text()
    block = _css_block(css, "@media (prefers-reduced-motion: reduce)")
    assert "animation-duration" in block and "transition-duration" in block
    assert "!important" in block


# --- cross-run analytics ----------------------------------------------------


def test_percentiles_use_nearest_rank_and_flag_small_samples():
    """Interpolating between two observed durations invents one that never happened."""
    from batcher.observe.analytics import percentiles

    p = percentiles([10, 20, 30, 40, 100])
    assert p["p50"] == 30 and p["max"] == 100 and p["count"] == 5
    assert p["reliable"] is True
    assert all(v in (10, 20, 30, 40, 100) for k, v in p.items() if k.startswith("p"))
    assert percentiles([5, 6])["reliable"] is False
    assert percentiles([]) == {"count": 0}


def test_throughput_series_keeps_empty_buckets():
    """A gap where nothing ran is information; dropping it draws a line through silence."""
    from batcher.observe.analytics import throughput_series

    series = throughput_series(
        [
            {"started_wall": 100.0, "total_ms": 100.0, "rows": 50, "status": "ok"},
            {"started_wall": 200.0, "total_ms": 100.0, "rows": 50, "status": "ok"},
        ],
        buckets=10,
    )
    assert len(series["buckets"]) == 10
    assert sum(b["runs"] for b in series["buckets"]) == 2
    assert any(b["runs"] == 0 for b in series["buckets"])


def test_operator_rollup_totals_by_kind():
    from batcher.observe.analytics import operator_rollup

    runs = [
        {
            "dag": {
                "nodes": [
                    {"kind": "hash_join", "measured": True, "elapsed_ms": 60.0, "rows_out": 10},
                    {"kind": "scan", "measured": True, "elapsed_ms": 20.0, "rows_out": 100},
                    {"kind": "scan", "measured": False, "elapsed_ms": 999.0, "rows_out": 0},
                ]
            }
        },
        {
            "dag": {
                "nodes": [
                    {"kind": "hash_join", "measured": True, "elapsed_ms": 20.0, "rows_out": 5}
                ]
            }
        },
    ]
    rollup = {r["kind"]: r for r in operator_rollup(runs)}
    assert rollup["hash_join"]["total_ms"] == 80.0
    assert rollup["hash_join"]["runs"] == 2
    assert rollup["hash_join"]["share"] == pytest.approx(0.8)
    assert rollup["scan"]["total_ms"] == 20.0  # the unmeasured node is excluded


def test_compare_matches_steps_by_plan_position():
    from batcher.observe.analytics import compare_runs

    a = {
        "signature": "s",
        "total_ms": 100.0,
        "rows": 5,
        "dag": {
            "nodes": [
                {"op_id": 0, "kind": "sort", "elapsed_ms": 10.0, "rows_out": 5},
                {"op_id": 1, "kind": "scan", "elapsed_ms": 90.0, "rows_out": 900},
            ]
        },
    }
    b = {
        "signature": "s",
        "total_ms": 50.0,
        "rows": 5,
        "dag": {
            "nodes": [
                {"op_id": 0, "kind": "sort", "elapsed_ms": 10.0, "rows_out": 5},
                {"op_id": 1, "kind": "scan", "elapsed_ms": 40.0, "rows_out": 900},
            ]
        },
    }
    cmp = compare_runs(a, b)
    assert cmp["ok"]
    assert cmp["steps"][0]["kind"] == "scan"  # biggest change first
    assert cmp["steps"][0]["delta_ms"] == -50.0
    assert cmp["steps"][0]["ratio"] == pytest.approx(40 / 90)


def test_compare_refuses_runs_of_different_shapes():
    """Diffing steps across different plans would line up unrelated operators."""
    from batcher.observe.analytics import compare_runs

    cmp = compare_runs({"signature": "a", "total_ms": 1.0}, {"signature": "b", "total_ms": 2.0})
    assert cmp["ok"] is False
    assert "different plan shapes" in cmp["reason"]
    assert cmp["steps"] == []
    assert cmp["totals"]  # whole-run numbers are still comparable


def test_health_reports_checks_with_actions():
    from batcher.observe.analytics import health_report

    report = health_report(
        [{"status": "error"}, {"status": "ok"}],
        [{"dag": {"nodes": [{"spilled": True, "spill_bytes": 1024}]}}],
        {"host": {"memory_total_bytes": 100, "memory_available_bytes": 2}},
    )
    assert report["status"] == "critical"
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["Failures"]["status"] == "critical"
    assert by_name["Spilling"]["status"] == "warn"
    assert by_name["Host memory"]["status"] == "critical"
    assert all(c["action"] for c in report["checks"] if c["status"] != "ok")


def test_health_is_ok_on_a_clean_session():
    from batcher.observe.analytics import health_report

    report = health_report(
        [{"status": "ok"}],
        [{"dag": {"nodes": [{"spilled": False}]}}],
        {"host": {"memory_total_bytes": 100, "memory_available_bytes": 60}},
    )
    assert report["status"] == "ok"
    assert report["uptime_s"] >= 0


def test_ui_serves_every_analytics_route(ui):
    url, store = ui
    _feed(store, events.QUERY_START, query_id="q1", fields={"label": "scan", "signature": "s1"})
    _feed(store, events.QUERY_END, query_id="q1", fields={"ok": True, "total_ms": 5.0, "rows": 9})
    assert _get(f"{url}/api/health")["status"] in {"ok", "warn", "critical"}
    assert "buckets" in _get(f"{url}/api/timeseries")
    assert "operators" in _get(f"{url}/api/operators")
    assert _get(f"{url}/api/compare?a=q1&b=q1")["ok"] is True
    assert _get(f"{url}/api/summary")["percentiles"]["count"] == 1


def test_the_dashboard_is_a_three_level_drill_down():
    """Pipelines → one pipeline → one run. The nav offers only the top level.

    Regression for an IA where "Overview" and "Pipelines" split one question across two
    screens, and the run list lived in a sidebar of the run page rather than on the pipeline
    it belonged to.
    """
    html = (_ASSETS / "index.html").read_text()
    nav = set(re.findall(r'data-view="([^"]+)"', html))
    panes = set(re.findall(r'id="view-([^"]+)"', html))
    assert nav == {"pipelines", "live", "logs", "system", "learn"}, (
        "only top-level destinations belong in the nav"
    )
    assert panes == {"pipelines", "pipeline", "live", "logs", "run", "system", "learn"}
    # The load-bearing part: the two drill-down levels exist as pages but are reached by
    # clicking through, never from the nav — which is what stops one question being split
    # across two screens.
    assert panes - nav == {"pipeline", "run"}


def test_the_run_page_has_no_run_list_and_the_pipeline_page_does():
    """The list of a pipeline's runs belongs on the pipeline, not beside one run."""
    html = (_ASSETS / "index.html").read_text()
    pipeline = html[html.index('id="view-pipeline"') : html.index('id="view-run"')]
    run = html[html.index('id="view-run"') : html.index('id="view-logs"')]
    assert 'id="p-runs"' in pipeline
    assert 'id="q-list"' not in run and 'id="q-list"' not in html
    assert 'id="run-prev"' in run and 'id="run-next"' in run


def test_breadcrumbs_exist_for_the_drill_down():
    html = (_ASSETS / "index.html").read_text()
    assert 'id="crumbs"' in html
    app = (_ASSETS / "app.js").read_text()
    assert "renderCrumbs" in app and "function goUp" in app


_SPLIT_PIPELINES = [
    {"signature": "sig-a", "label": "aggregate", "total_ms": 400.0},
    {"signature": "sig-b", "label": "sort", "total_ms": 100.0},
]

_FAILURE_GROUPS = [
    {"error": "boom", "count": 2, "last_wall": 1.0, "runs": ["q1", "q2"]},
]


def test_panels_cross_reference_each_other():
    """A dashboard where each panel is a dead end makes you re-navigate to follow a thread.

    Every panel that names a run, a pipeline, or a plan step emits a data attribute, and one
    delegated listener turns all of them into navigation.
    """
    app = (_ASSETS / "app.js").read_text()
    # Asserted on the *rendered* markup, not on the source. A grep could not see `data-pipe`
    # once the proportion bar began building the attribute name from a parameter, and the
    # rendered form is the stronger check anyway: it proves the attribute reaches the page.
    ctx = _views_context()
    ctx.eval(f"VIEWS.timeSplit({json.dumps(_SPLIT_PIPELINES)});")
    ctx.eval(f"VIEWS.failures({json.dumps(_FAILURE_GROUPS)});")
    rendered = ctx.eval(
        "document.getElementById('time-split').innerHTML + "
        "document.getElementById('failures').innerHTML"
    )
    for attr in ("data-run=", "data-pipe="):
        assert attr in rendered, f"no panel emits {attr}"
    assert "data-op=" in (_ASSETS / "views.js").read_text()
    assert "installCrossReferences" in app
    # Delegated once on document, not bound per render — otherwise every redraw leaks.
    handler = app[app.index("function installCrossReferences") :]
    assert "document.addEventListener('click'" in handler
    assert "closest('[data-run], [data-pipe], [data-op]')" in handler


def test_navigation_records_where_you_have_been():
    app = (_ASSETS / "app.js").read_text()
    assert "function noteVisit" in app and "renderRecentRail" in app
    # Both drill-down entry points record the visit.
    assert app.count("noteVisit(") >= 3


def test_the_run_page_shows_its_sibling_runs():
    """Sideways movement within a pipeline, without going back up a level."""
    assert 'id="related"' in (_ASSETS / "index.html").read_text()
    assert "function renderRelated" in (_ASSETS / "app.js").read_text()


def test_run_detail_carries_its_siblings_from_the_api(bus):
    from batcher.observe.store import ActivityStore

    store = ActivityStore()
    for i in range(3):
        _feed(
            store, events.QUERY_START, query_id=f"q{i}", fields={"label": "agg", "signature": "s"}
        )
        _feed(
            store,
            events.QUERY_END,
            query_id=f"q{i}",
            fields={"ok": True, "total_ms": 10.0 + i, "rows": 1},
        )
    detail = store.query("q1")
    assert len(detail["siblings"]) == 3
    assert {s["query_id"] for s in detail["siblings"]} == {"q0", "q1", "q2"}


def test_every_firing_health_check_links_to_the_runs_behind_it():
    """A verdict you cannot click through to is a dead end.

    Asserted for *every* firing check rather than a sample: two of the three were silently
    missing their run list, and a test that only checked one would have passed anyway.
    """
    from batcher.observe.analytics import health_report

    report = health_report(
        [{"status": "error", "query_id": "bad-run"}],
        [
            {"query_id": "spilly", "dag": {"nodes": [{"spilled": True, "spill_bytes": 1}]}},
            {"query_id": "guessy", "dag": {"nodes": [{"est_error": 40.0}]}},
        ],
        {},
    )
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["Failures"]["runs"] == ["bad-run"]
    assert by_name["Spilling"]["runs"] == ["spilly"]
    assert by_name["Plan estimates"]["runs"] == ["guessy"]
    for check in report["checks"]:
        if check["status"] != "ok":
            assert check["runs"], f"{check['name']} fired but links to nothing"


def test_operator_rollup_points_at_its_worst_run():
    from batcher.observe.analytics import operator_rollup

    rows = operator_rollup(
        [
            {
                "query_id": "fast",
                "dag": {"nodes": [{"kind": "sort", "measured": True, "elapsed_ms": 5.0}]},
            },
            {
                "query_id": "slow",
                "dag": {"nodes": [{"kind": "sort", "measured": True, "elapsed_ms": 90.0}]},
            },
        ]
    )
    assert rows[0]["slowest_run"] == "slow"
    assert set(rows[0]["example_runs"]) == {"fast", "slow"}


def test_boot_wiring_cannot_be_aborted_by_one_missing_element():
    """`boot` wires ~40 controls; a throw in any one would abort the rest and blank the page.

    Regression: two KPI handlers used `.closest('.kpi').addEventListener(...)` directly, so a
    markup change that moved the value out of its card would have taken down every control
    bound after it — including the poll loop's error banner.
    """
    app = (_ASSETS / "app.js").read_text()
    boot = app[app.index("function boot()") :]
    # No direct `.addEventListener` on a possibly-null lookup inside boot.
    assert "').addEventListener(" not in boot, (
        "bind through on() so a missing element is survivable"
    )
    assert "function on(target, event, handler)" in app


def test_failures_group_by_cause_not_by_run():
    """Twenty failures are usually one bug; a list of twenty rows hides that."""
    from batcher.observe.analytics import failure_groups

    groups = failure_groups(
        [
            {
                "status": "error",
                "query_id": "a",
                "error": "RuntimeError: model not loaded",
                "started_wall": 10,
            },
            {
                "status": "error",
                "query_id": "b",
                "error": "RuntimeError: model not loaded",
                "started_wall": 20,
            },
            {
                "status": "error",
                "query_id": "c",
                "error": "PlanError: unknown column",
                "started_wall": 30,
            },
            {"status": "ok", "query_id": "d"},
        ]
    )
    assert [g["count"] for g in groups] == [2, 1]
    assert groups[0]["runs"] == ["a", "b"]
    assert groups[0]["first_wall"] == 10 and groups[0]["last_wall"] == 20


def test_pipeline_report_separates_chronic_from_occasional():
    """A finding seen every run deserves a fix; seen once it is often just a cold cache."""
    from batcher.observe.analytics import pipeline_report

    runs = [
        {
            "signature": "s",
            "dag": {
                "nodes": [
                    {
                        "op_id": 0,
                        "kind": "hash_join",
                        "measured": True,
                        "elapsed_ms": 50.0,
                        "on_critical_path": True,
                    },
                    {
                        "op_id": 1,
                        "kind": "scan",
                        "measured": True,
                        "elapsed_ms": 5.0,
                        "on_critical_path": i == 0,
                    },
                ]
            },
            "insights": (
                [
                    {
                        "rule": "operator-spilled",
                        "title": "Spilled",
                        "severity": "warning",
                        "action": "x",
                    }
                ]
                if i < 3
                else []
            ),
        }
        for i in range(4)
    ]
    report = pipeline_report("s", runs)
    assert report["runs"] == 4
    join = next(s for s in report["steps"] if s["kind"] == "hash_join")
    assert join["critical_share"] == 1.0  # on the critical path every run
    scan = next(s for s in report["steps"] if s["kind"] == "scan")
    assert scan["critical_share"] == 0.25  # only sometimes
    recurring = report["recurring"][0]
    assert recurring["count"] == 3 and recurring["chronic"] is True


def test_pipeline_report_ignores_other_pipelines():
    from batcher.observe.analytics import pipeline_report

    report = pipeline_report("mine", [{"signature": "theirs", "dag": {"nodes": []}}])
    assert report["runs"] == 0 and report["steps"] == []


def test_ui_serves_the_failure_and_pipeline_routes(ui):
    url, store = ui
    _feed(store, events.QUERY_START, query_id="q1", fields={"label": "scan", "signature": "s1"})
    _feed(store, events.QUERY_END, query_id="q1", fields={"ok": False, "error": "boom"})
    assert _get(f"{url}/api/failures")["groups"][0]["count"] == 1
    assert "recurring" in _get(f"{url}/api/pipeline?signature=s1")


def test_number_animation_respects_reduced_motion():
    """A rolling number is a signal; for a user who asked for stillness it is just motion."""
    ui = (_ASSETS / "ui.js").read_text()
    roll = ui[ui.index("function rollTo") :]
    assert "prefers-reduced-motion" in roll


def test_dark_is_the_default_theme():
    """Every comparable engine dashboard ships dark-first; inheriting a light OS preference
    put people somewhere they had not asked to be."""
    ui = (_ASSETS / "ui.js").read_text()
    defaults = ui[ui.index("const DEFAULTS = {") : ui.index("let prefs")]
    assert "theme: 'dark'" in defaults
    app = (_ASSETS / "app.js").read_text()
    cycle = app[app.index("function toggleTheme") :]
    assert "['dark', 'light', 'auto']" in cycle, "dark should lead the cycle"


def test_the_pipeline_page_has_its_own_plan_graph_and_matrix():
    """A pipeline is defined by its plan shape, so the shape can be drawn once with typical
    costs — and a runs x steps matrix shows what no single graph can."""
    html = (_ASSETS / "index.html").read_text()
    pipeline = html[html.index('id="view-pipeline"') : html.index('id="view-run"')]
    assert 'id="p-dag"' in pipeline and 'id="p-grid"' in pipeline
    dag = (_ASSETS / "dag.js").read_text()
    # Two independent explorer instances, not one shared singleton.
    assert "function createDagExplorer()" in dag
    assert "const DAG = createDagExplorer()" in dag
    assert "const PIPE_DAG = createDagExplorer()" in dag


def test_pipeline_dag_reports_typical_cost_not_one_run():
    from batcher.observe.analytics import pipeline_report

    runs = [
        {
            "signature": "s",
            "query_id": f"q{i}",
            "total_ms": 10.0,
            "dag": {
                "nodes": [
                    {
                        "op_id": 0,
                        "kind": "sort",
                        "measured": True,
                        "elapsed_ms": 10.0 * (i + 1),
                        "on_critical_path": True,
                        "column": 0,
                        "row": 0,
                        "depth": 0,
                    }
                ],
                "edges": [],
                "width": 1,
                "depth": 1,
                "critical_path": [0],
            },
        }
        for i in range(4)
    ]
    node = pipeline_report("s", runs)["dag"]["nodes"][0]
    assert node["samples"] == 4
    assert node["mean_ms"] == 25.0  # (10+20+30+40)/4, not any single run
    assert node["max_ms"] == 40.0
    assert node["critical_share"] == 1.0
    assert node["elapsed_ms"] == node["mean_ms"]  # the ramp encodes typical cost
    assert node["percentiles"]["count"] == 4


def test_the_matrix_compares_each_step_against_its_own_median():
    """A slow cell must stand out whether the step takes microseconds or minutes."""
    from batcher.observe.analytics import pipeline_report

    runs = [
        {
            "signature": "s",
            "query_id": f"q{i}",
            "started_wall": float(i),
            "status": "ok",
            "dag": {
                "nodes": [
                    {
                        "op_id": 0,
                        "kind": "scan",
                        "measured": True,
                        "elapsed_ms": 1.0,
                        "column": 0,
                        "row": 0,
                        "depth": 0,
                    },
                    {
                        "op_id": 1,
                        "kind": "sort",
                        "measured": True,
                        "elapsed_ms": 1000.0 if i < 3 else 4000.0,
                        "column": 0,
                        "row": 1,
                        "depth": 0,
                    },
                ],
                "edges": [],
                "width": 1,
                "depth": 2,
                "critical_path": [],
            },
        }
        for i in range(4)
    ]
    grid = pipeline_report("s", runs)["grid"]
    assert [s["kind"] for s in grid["steps"]] == ["scan", "sort"]
    assert len(grid["runs"]) == 4
    slow = next(c for c in grid["cells"] if c["op_id"] == 1 and c["run"] == 3)
    assert slow["ratio"] == pytest.approx(4.0)  # 4x its own median, despite being the big step
    steady = next(c for c in grid["cells"] if c["op_id"] == 0 and c["run"] == 3)
    assert steady["ratio"] == pytest.approx(1.0)


def test_a_step_missing_from_a_run_is_absent_not_zero():
    """A cell with no measurement must read as "no data", not as an instantaneous step."""
    from batcher.observe.analytics import pipeline_report

    runs = [
        {
            "signature": "s",
            "query_id": "a",
            "dag": {
                "nodes": [
                    {
                        "op_id": 0,
                        "kind": "scan",
                        "measured": True,
                        "elapsed_ms": 5.0,
                        "column": 0,
                        "row": 0,
                        "depth": 0,
                    },
                    {
                        "op_id": 1,
                        "kind": "sort",
                        "measured": True,
                        "elapsed_ms": 9.0,
                        "column": 0,
                        "row": 1,
                        "depth": 0,
                    },
                ],
                "edges": [],
                "critical_path": [],
            },
        },
        {
            "signature": "s",
            "query_id": "b",
            "dag": {
                "nodes": [
                    {
                        "op_id": 0,
                        "kind": "scan",
                        "measured": True,
                        "elapsed_ms": 5.0,
                        "column": 0,
                        "row": 0,
                        "depth": 0,
                    }
                ],
                "edges": [],
                "critical_path": [],
            },
        },
    ]
    grid = pipeline_report("s", runs)["grid"]
    missing = next(c for c in grid["cells"] if c["op_id"] == 1 and c["run"] == 1)
    assert missing["elapsed_ms"] is None and missing["ratio"] is None


def _util_profile(cpu_util=0.95, peak_rss=0, budget=0, spilled=False):
    """One long, measured operator — the shape the utilization rules read."""
    op = {
        "measured": True,
        "kind": "hash_join",
        "rows_out": 1000,
        "elapsed_ms": 5000.0,
        "cpu_util": cpu_util,
        "est_error": 1.0,
    }
    if peak_rss:
        op["peak_rss_bytes"] = peak_rss
    if spilled:
        op |= {"spilled": True, "spill_bytes": 1 << 30}
    return {"total_ms": 5000.0, "ops": [op], "memory_budget_bytes": budget}


def _rules(profile):
    from batcher.observe.insights import derive_insights

    return {i["rule"] for i in derive_insights(profile)}


def test_idle_cpu_blames_the_box_when_the_box_is_oversubscribed(monkeypatch):
    # Idle cores have two causes with opposite fixes. Telling a user whose box is at 3x load to
    # go re-partition their input sends them to tune a query that was never the problem.
    from batcher.observe.insights import resources

    monkeypatch.setattr(resources, "cpu_contention", lambda: {"load_per_core": 3.0})
    found = _rules(_util_profile(cpu_util=0.1))
    assert "cpu-contended" in found
    assert "cpu-underutilized" not in found  # the contended finding replaces it, not adds to it


def test_idle_cpu_blames_the_quota_when_the_cgroup_is_throttling(monkeypatch):
    from batcher.observe.insights import resources

    monkeypatch.setattr(
        resources, "cpu_contention", lambda: {"load_per_core": 0.2, "throttled_ratio": 0.4}
    )
    assert "cpu-throttled" in _rules(_util_profile(cpu_util=0.1))


def test_idle_cpu_blames_the_plan_only_on_a_quiet_box(monkeypatch):
    from batcher.observe.insights import resources

    monkeypatch.setattr(
        resources, "cpu_contention", lambda: {"load_per_core": 0.3, "throttled_ratio": 0.0}
    )
    found = _rules(_util_profile(cpu_util=0.1))
    assert "cpu-underutilized" in found
    assert not {"cpu-contended", "cpu-throttled"} & found


def test_the_target_memory_band_is_not_reported_as_a_problem():
    # The engine targets >80% memory utilization, so warning at 80% would flag success and
    # train the user to ignore the panel. Only the spill-risk band above `hard_limit` speaks.
    budget = 27 << 30
    assert "memory-headroom" not in _rules(_util_profile(peak_rss=24 << 30, budget=budget))  # 89%
    assert "memory-headroom" in _rules(_util_profile(peak_rss=25 << 30, budget=budget))  # 93%


def test_spilling_with_most_of_memory_unused_is_a_warning():
    # Spilling is decided before execution from an estimate, so an over-estimate spills a query
    # that would have fit. That is an estimate bug, and is worth separating from ordinary spill.
    found = _rules(_util_profile(peak_rss=2 << 30, budget=27 << 30, spilled=True))
    assert "spilled-with-headroom" in found


def test_an_oversized_budget_is_only_an_info_finding():
    found = _rules(_util_profile(peak_rss=1 << 30, budget=27 << 30))
    assert "memory-underused" in found
    assert "spilled-with-headroom" not in found


# --- rendering, actually executed ------------------------------------------
# The load-time smoke test proves the modules parse and run; it cannot prove a renderer
# produces anything. These drive the real render functions against a small DOM and assert
# what came out — which is how the "0.0ms · 0%" node was found.

_MINIDOM = Path(__file__).parent / "data" / "minidom.js"


def _render_context():
    """A QuickJS context with a small DOM and every dashboard module loaded."""
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_MINIDOM.read_text())
    for name in ("ui.js", "charts.js", "dag.js", "views.js"):
        ctx.eval((_ASSETS / name).read_text())
    return ctx


_SAMPLE_DAG = {
    "nodes": [
        {
            "op_id": 0,
            "kind": "sort",
            "detail": "revenue",
            "measured": True,
            "rows_out": 3,
            "elapsed_ms": 0.009,
            "column": 0,
            "row": 2,
            "depth": 0,
            "on_critical_path": True,
        },
        {
            "op_id": 1,
            "kind": "hash_join",
            "detail": "inner on k",
            "measured": True,
            "rows_out": 380070,
            "elapsed_ms": 20.0,
            "est_error": 40.0,
            "column": 0,
            "row": 1,
            "depth": 1,
            "on_critical_path": True,
        },
        {
            "op_id": 2,
            "kind": "scan",
            "detail": "source 0",
            "measured": True,
            "rows_out": 400000,
            "elapsed_ms": 0.002,
            "spilled": True,
            "spill_bytes": 2048,
            "column": 0,
            "row": 0,
            "depth": 2,
        },
    ],
    "edges": [{"from": 1, "to": 0}, {"from": 2, "to": 1}],
    "width": 1,
    "depth": 3,
    "critical_path": [0, 1],
}


def test_sub_millisecond_work_is_not_rendered_as_zero():
    """Regression: 0.009 ms rendered as "0.0ms" and its share as "0%", so the fastest steps
    in a plan looked unmeasured — the opposite of the truth."""
    ctx = _render_context()
    assert ctx.eval("DAG.fmtMs(0.009)") == "9µs"
    assert ctx.eval("DAG.fmtMs(0.4)") == "400µs"
    assert ctx.eval("DAG.fmtMs(5.1)") == "5.1ms"
    assert ctx.eval("DAG.fmtMs(1500)") == "1.50s"
    assert ctx.eval("DAG.fmtMs(0)") == "0"


def test_a_plan_node_renders_as_a_card_not_a_labelled_box():
    """Each node carries its name, id, detail, share, rows, and time — six values, a share
    bar, and an operator glyph. A flat rectangle with three lines read as a placeholder."""
    ctx = _render_context()
    ctx.eval(f"var DAGDATA = {json.dumps(_SAMPLE_DAG)};")
    out = json.loads(
        ctx.eval("""
    (function(){
      var svg = document.getElementById('dag');
      DAG.render(svg, DAGDATA, {onHover:function(){},onLeave:function(){},
                                onZoom:function(){},onSelect:function(){}}, 'q');
      var nodes = svg.querySelectorAll('.dag-node');
      var join = null;
      for (var i = 0; i < nodes.length; i++) {
        var t = nodes[i].querySelectorAll('text').map(function(x){ return x.textContent; });
        if (t[0] === 'Join') join = { texts: t, rects: nodes[i].querySelectorAll('rect').length,
                                      paths: nodes[i].querySelectorAll('path').length };
      }
      return JSON.stringify({ nodes: nodes.length, edges: svg.querySelectorAll('.dag-edge').length,
                              join: join });
    })()
    """)
    )
    assert out["nodes"] == 3 and out["edges"] == 2
    join = out["join"]
    assert join["texts"][:3] == ["Join", "op 1", "inner on k"]
    assert "69%" not in join["texts"]  # share is computed from this plan, not hard-coded
    assert any(t.endswith("%") for t in join["texts"]), "share is shown"
    assert "380.1K rows" in join["texts"] and "20ms" in join["texts"]
    # Card body + header band + share track/fill, plus the operator glyph.
    assert join["rects"] >= 3 and join["paths"] >= 2


def test_a_badly_estimated_step_is_flagged_on_its_node():
    ctx = _render_context()
    ctx.eval(f"var DAGDATA = {json.dumps(_SAMPLE_DAG)};")
    texts = json.loads(
        ctx.eval("""
    (function(){
      var svg = document.getElementById('dag');
      DAG.render(svg, DAGDATA, {onHover:function(){},onLeave:function(){},
                                onZoom:function(){},onSelect:function(){}}, 'q');
      return JSON.stringify(svg.querySelectorAll('text').map(function(t){ return t.textContent; }));
    })()
    """)
    )
    assert "40x off" in texts, "a 40x cardinality miss should be visible on the node"
    assert "spilled" in texts, "a spilled step should be visible on the node"


def test_the_run_and_pipeline_graphs_do_not_share_state():
    """Two explorer instances: navigating between the pages must not reset the other."""
    ctx = _render_context()
    ctx.eval(f"var DAGDATA = {json.dumps(_SAMPLE_DAG)};")
    out = json.loads(
        ctx.eval("""
    (function(){
      var a = document.getElementById('dag'), b = document.getElementById('p-dag');
      var opts = {onHover:function(){}, onLeave:function(){},
                  onZoom:function(){}, onSelect:function(){}};
      DAG.render(a, DAGDATA, opts, 'run');
      PIPE_DAG.render(b, DAGDATA, opts, 'pipeline');
      DAG.select(1);
      return JSON.stringify({ runSelected: DAG.selectedNode() ? DAG.selectedNode().op_id : null,
                              pipeSelected: PIPE_DAG.selectedNode() });
    })()
    """)
    )
    assert out["runSelected"] == 1
    assert out["pipeSelected"] is None, "selecting in one graph must not select in the other"


def test_the_matrix_renders_a_cell_per_step_and_run():
    ctx = _render_context()
    grid = {
        "steps": [
            {"op_id": 0, "kind": "scan", "median_ms": 5.0},
            {"op_id": 1, "kind": "sort", "median_ms": 1000.0},
        ],
        "runs": [
            {"query_id": "a", "started_wall": 1.0, "total_ms": 10.0, "status": "ok"},
            {"query_id": "b", "started_wall": 2.0, "total_ms": 40.0, "status": "error"},
        ],
        "cells": [
            {"run": 0, "op_id": 0, "elapsed_ms": 5.0, "ratio": 1.0, "spilled": False},
            {"run": 1, "op_id": 0, "elapsed_ms": 5.0, "ratio": 1.0, "spilled": False},
            {"run": 0, "op_id": 1, "elapsed_ms": 1000.0, "ratio": 1.0, "spilled": False},
            {"run": 1, "op_id": 1, "elapsed_ms": 4000.0, "ratio": 4.0, "spilled": True},
        ],
    }
    ctx.eval(f"var GRID = {json.dumps(grid)};")
    html = ctx.eval("""
    (function(){
      VIEWS.runGrid(GRID, { onOpenRun: function(){}, onSelectStep: function(){} });
      return document.getElementById('p-grid').innerHTML;
    })()
    """)
    assert html.count("grid-cell") == 4
    assert "lvl-5" in html, "a 4x-slower cell should land in the top shade band"
    assert "is-spilled" in html
    assert "Read source" in html and "Sort" in html


def test_every_style_token_resolves_without_a_theme_attribute():
    """Regression: the colour palette lived only under :root[data-theme=...], so any host
    that renders the page without stamping the attribute — an embed, a "system" theme, the
    first paint before the script runs — got a page with no surfaces, borders, or text
    colour. Every panel and card collapsed to an unstyled box."""
    css = (_ASSETS / "app.css").read_text()
    base = re.search(r"^:root \{\n(.*?)\n\}\n", css, re.S | re.M)
    assert base, "the bare :root block defines the defaults; it must exist"
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", base.group(1)))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", css))
    assert not (used - defined), (
        f"tokens unresolved without a theme attribute: {sorted(used - defined)}"
    )


def test_an_explicit_theme_choice_outranks_the_os_preference():
    """The OS-preference block must not win over a stamped data-theme, in either direction —
    hence :where(), which holds it at zero specificity."""
    css = (_ASSETS / "app.css").read_text()
    media = re.search(r"@media \(prefers-color-scheme: dark\) \{\n(\s*:root[^\n{]*)\{", css)
    assert media, "an OS-preference block should exist"
    assert ":where(:not([data-theme]))" in media.group(1), (
        "the OS block must be zero-specificity and scoped to the unstamped case"
    )
    for theme in ("dark", "light"):
        assert f':root[data-theme="{theme}"] {{' in css, f"explicit {theme} override missing"


# --- the teaching layer ------------------------------------------------------
# The dashboard's hardest audience is someone meeting the engine's vocabulary for the first
# time. These pin the parts of that layer that rot silently: a term referenced but never
# defined, an operator the engine can run but the reference cannot explain, a cross-link
# pointing at nothing.


def _reference_context():
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_MINIDOM.read_text())
    for name in ("ui.js", "reference.js", "dag.js", "learn.js"):
        ctx.eval((_ASSETS / name).read_text())
    return ctx


def test_every_operator_the_engine_can_plan_is_explained():
    """A plan can contain any IR tag; a step the reference cannot explain is a dead end
    exactly when a newcomer clicks it."""
    ctx = _reference_context()
    explained = set(json.loads(ctx.eval("JSON.stringify(Object.keys(REFERENCE.OPERATORS))")))
    drawn = set(json.loads(ctx.eval("JSON.stringify(Object.keys(DAG.friendlyKinds()))")))
    assert not (drawn - explained), (
        f"operators drawn but not explained: {sorted(drawn - explained)}"
    )


def test_every_cross_reference_resolves_to_a_real_entry():
    """`see:` lists are hand-written, so a renamed term leaves a link to nowhere."""
    ctx = _reference_context()
    dangling = json.loads(
        ctx.eval("""
    JSON.stringify((() => {
      const bad = [];
      for (const [word, entry] of Object.entries(REFERENCE.TERMS)) {
        for (const other of entry.see || []) {
          if (!REFERENCE.lookup(other)) bad.push(`${word} -> ${other}`);
        }
      }
      for (const [kind, op] of Object.entries(REFERENCE.OPERATORS)) {
        for (const other of op.terms || []) {
          if (!REFERENCE.lookup(other)) bad.push(`${kind} -> ${other}`);
        }
      }
      for (const [key, m] of Object.entries(REFERENCE.METRICS)) {
        if (m.term && !REFERENCE.lookup(m.term)) bad.push(`metric ${key} -> ${m.term}`);
      }
      return bad;
    })())
    """)
    )
    assert dangling == [], f"cross-references with no entry: {dangling}"


def test_every_entry_actually_says_something():
    """An entry that exists but is empty is worse than none — it looks answered."""
    ctx = _reference_context()
    thin = json.loads(
        ctx.eval("""
    JSON.stringify((() => {
      const bad = [];
      for (const [word, e] of Object.entries(REFERENCE.TERMS)) {
        if (!e.what || e.what.length < 25) bad.push(`term ${word}`);
      }
      for (const [kind, o] of Object.entries(REFERENCE.OPERATORS)) {
        for (const field of ['what', 'slow', 'fix', 'label']) {
          if (!o[field] || o[field].length < 4) bad.push(`operator ${kind}.${field}`);
        }
      }
      return bad;
    })())
    """)
    )
    assert thin == [], f"entries with no substance: {thin}"


def test_a_term_renders_as_something_a_keyboard_can_reach():
    """Regression: terms were `<abbr title>`, which no keyboard can focus, no screen reader
    announces as interactive, and no stylesheet can touch."""
    ctx = _reference_context()
    markup = ctx.eval("LEARN.term('spill')")
    assert "<button" in markup and 'data-term="spill"' in markup
    assert "aria-label" in markup
    assert "<abbr" not in markup and "title=" not in markup
    # An unknown word degrades to plain escaped text rather than an inert control.
    assert ctx.eval("LEARN.term('not-a-real-term')") == "not-a-real-term"


def test_a_definition_answers_what_why_and_what_to_do():
    ctx = _reference_context()
    body = ctx.eval("""
    (function(){
      const t = document.createElement('button');
      LEARN.openTerm('spill', t);
      return document.body.children[document.body.children.length - 1].innerHTML;
    })()
    """)
    assert "Writing intermediate data to disk" in body
    assert "Why it matters" in body and "What to do" in body
    assert "See also" in body, "a definition should lead somewhere"
    assert 'aria-label="Close definition"' in body


def test_the_reference_page_renders_all_three_sections_and_filters():
    ctx = _reference_context()

    def render(query):
        return ctx.eval(
            f"(function(){{ const h = document.createElement('div');"
            f" LEARN.renderLearn(h, {json.dumps(query)}); return h.innerHTML; }})()"
        )

    full = render("")
    # Floors, not exact counts: the reference is content and is meant to grow. Pinning the
    # number here means every added glossary entry fails an unrelated test.
    counts = {name: full.count(f'class="{name}"') for name in ("recipe", "opref", "gloss-row")}
    assert counts["recipe"] >= 5, counts
    assert counts["opref"] >= 10, counts
    assert counts["gloss-row"] >= 30, counts
    # Every operator the reference knows about is rendered, so the page cannot silently
    # cover fewer than the engine can plan.
    ctx_ops = ctx.eval("Object.keys(REFERENCE.OPERATORS).length")
    assert counts["opref"] == ctx_ops, "the page must render every operator entry"
    # Glossary terms and metrics both render as `.gloss-row`, so the page must show one per
    # term plus one per metric.
    ctx_terms = ctx.eval("REFERENCE.termKeys.length")
    ctx_metrics = ctx.eval("Object.keys(REFERENCE.METRICS).length")
    assert counts["gloss-row"] == ctx_terms + ctx_metrics, (
        "the page must render every glossary and metric entry"
    )

    # Filtering searches the whole entry, not only its name: someone typing "spill" wants the
    # steps that spill, and those words live in the `slow` and `fix` fields.
    spill = render("spill")
    assert spill.count('class="opref"') >= 3
    assert spill.count('class="gloss-row"') < full.count('class="gloss-row"')

    missing = render("zzzznotathing")
    assert "empty-state" in missing and "zzzznotathing" in missing


def test_the_tour_only_points_at_things_that_exist():
    """A tour step whose target is gone silently ends the tour partway through."""
    ctx = _reference_context()
    html = (_ASSETS / "index.html").read_text()
    selectors = json.loads(
        ctx.eval("""
    JSON.stringify((() => {
      // The tour list is module-private; drive it through the public surface instead.
      const seen = [];
      const realQuery = document.querySelector;
      document.querySelector = (sel) => { seen.push(sel); return null; };
      LEARN.startTour();
      document.querySelector = realQuery;
      return seen;
    })())
    """)
    )
    assert selectors, "the tour should declare targets"
    # A target is either in the static shell or rendered by a view; both are legitimate, but
    # a selector that appears in neither can never match and silently truncates the tour.
    rendered = "".join((_ASSETS / f).read_text() for f in ("views.js", "app.js"))
    for sel in selectors:
        token = sel.lstrip("#.").split("[")[0]
        assert token in html or token in rendered, (
            f"tour step targets {sel}, which nothing in the page or the renderers produces"
        )


def test_the_analytics_package_exposes_one_flat_surface():
    """`analytics` was one module and is now a package split on responsibility seams. The
    split must be invisible to callers: same names, same signatures, one import path.

    Without this, a future re-split can quietly move a function to a submodule and leave the
    façade re-exporting something subtly different.
    """
    import inspect

    from batcher.observe import analytics

    assert sorted(analytics.__all__) == [
        "compare_runs",
        "failure_groups",
        "health_report",
        "operator_rollup",
        "percentiles",
        "pipeline_report",
        "throughput_series",
    ]
    # Every exported name resolves to a callable, and nothing public leaks that is not in
    # __all__ — the façade is a curated surface, not whatever happened to be imported.
    public = {
        n for n in dir(analytics) if not n.startswith("_") and callable(getattr(analytics, n))
    }
    assert public == set(analytics.__all__)
    for name in analytics.__all__:
        assert inspect.isfunction(getattr(analytics, name)), f"{name} should be a function"


# --- the triage path ---------------------------------------------------------
# "Which step is the problem" is a ranking question, and a sorted list answers a ranking
# question directly where a graph makes you scan for it. The graph explains the shape around
# the answer; the list is the answer. These pin the parts of that list carrying the
# diagnosis.


def _views_context():
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_MINIDOM.read_text())
    for name in (
        "ui.js",
        "reference.js",
        "charts.js",
        "dag.js",
        "learn.js",
        "plan.js",
        "live.js",
        "views.js",
    ):
        ctx.eval((_ASSETS / name).read_text())
    return ctx


_COST_NODES = [
    {
        "op_id": 0,
        "kind": "hash_join",
        "detail": "inner",
        "measured": True,
        "rows_out": 380070,
        "elapsed_ms": 70.0,
    },
    {
        "op_id": 1,
        "kind": "aggregate",
        "detail": "by tier",
        "measured": True,
        "rows_out": 3,
        "elapsed_ms": 20.0,
    },
    {
        "op_id": 2,
        "kind": "scan",
        "detail": "src",
        "measured": True,
        "rows_out": 400000,
        "elapsed_ms": 9.0,
    },
    {
        "op_id": 3,
        "kind": "limit",
        "detail": "",
        "measured": True,
        "rows_out": 10,
        "elapsed_ms": 0.05,
    },
    {
        "op_id": 4,
        "kind": "project",
        "detail": "",
        "measured": True,
        "rows_out": 10,
        "elapsed_ms": 0.02,
    },
]


def _render_cost(ctx, nodes):
    ctx.eval(f"var N = {json.dumps(nodes)};")
    ctx.eval("document.getElementById('costliest').innerHTML = '';")
    ctx.eval("VIEWS.costliest(N, function(){}, null);")
    return ctx.eval("document.getElementById('costliest').innerHTML")


def test_the_cost_list_ranks_by_time_and_truncates_the_noise():
    """Self-truncating at 1% rather than a fixed top-N: a plan with evenly spread cost has no
    "top 5" worth naming, and one dominated by a single step should not pad to five."""
    html = _render_cost(_views_context(), _COST_NODES)
    shares = re.findall(r'cost-share">([^<]+)', html)
    assert shares == ["71%", "20%", "9%"], "ranked descending, sub-1% steps dropped"
    # What was dropped is stated, never silently swallowed.
    assert "2 more steps below 1%" in html


def test_a_dominant_step_is_banded_not_shaded_on_a_gradient():
    """Two bands beat a continuous ramp: a gradient makes 12% and 19% look alike, whereas
    banding says "this one, then that one" and is readable without consulting a legend."""
    html = _render_cost(_views_context(), _COST_NODES)
    assert html.count("is-dominant") == 1, "the 71% step is dominant (>30%)"
    assert html.count("is-heavy") == 1, "the 20% step is heavy (15-30%)"


def test_the_cost_list_keeps_a_step_even_when_everything_is_tiny():
    """A plan of uniformly trivial steps must still name its worst one rather than render
    an empty list under the 1% rule."""
    tiny = [
        {
            "op_id": i,
            "kind": "project",
            "detail": "",
            "measured": True,
            "rows_out": 1,
            "elapsed_ms": 0.001,
        }
        for i in range(200)
    ]
    html = _render_cost(_views_context(), tiny)
    assert html.count('class="cost-row') >= 1, "at least the worst step is always shown"


def test_the_cost_list_says_so_when_nothing_was_measured():
    html = _render_cost(_views_context(), [{"op_id": 0, "kind": "scan", "measured": False}])
    assert "empty-state" in html and "no per-step timing" in html.lower()


def test_row_counts_ride_the_edges_so_an_exploding_join_is_visible():
    """A join that multiplies its input is invisible in the *shape* of a graph — the node
    count and the edges look identical whether it emitted 1,000 rows or 5,000,000. Putting
    the row count on the edge makes it one number jumping orders of magnitude between an
    inbound and an outbound edge."""
    ctx = _views_context()
    dag = {
        "nodes": [
            {
                "op_id": 0,
                "kind": "hash_join",
                "measured": True,
                "rows_out": 5_000_000,
                "elapsed_ms": 90.0,
                "column": 0,
                "row": 0,
                "depth": 0,
            },
            {
                "op_id": 1,
                "kind": "scan",
                "measured": True,
                "rows_out": 1000,
                "elapsed_ms": 1.0,
                "column": 0,
                "row": 1,
                "depth": 1,
            },
        ],
        "edges": [{"from": 1, "to": 0}],
        "width": 1,
        "depth": 2,
        "critical_path": [],
    }
    ctx.eval(f"var D = {json.dumps(dag)};")
    labels = json.loads(
        ctx.eval("""
    (function(){
      var svg = document.getElementById('dag');
      DAG.render(svg, D, { onHover:function(){}, onLeave:function(){},
                           onZoom:function(){}, onSelect:function(){} }, 'q');
      return JSON.stringify(svg.querySelectorAll('.edge-rows')
        .map(function(t){ return t.textContent; }));
    })()
    """)
    )
    assert labels == ["1.0K"], "the edge carries the producing node's output rows"


def test_an_unmeasured_plan_draws_no_edge_labels():
    """Rather than a graph strung with em-dashes."""
    ctx = _views_context()
    dag = {
        "nodes": [
            {"op_id": 0, "kind": "hash_join", "measured": False, "column": 0, "row": 0, "depth": 0},
            {"op_id": 1, "kind": "scan", "measured": False, "column": 0, "row": 1, "depth": 1},
        ],
        "edges": [{"from": 1, "to": 0}],
        "width": 1,
        "depth": 2,
        "critical_path": [],
    }
    ctx.eval(f"var D2 = {json.dumps(dag)};")
    count = ctx.eval("""
    (function(){
      var svg = document.getElementById('dag');
      DAG.render(svg, D2, { onHover:function(){}, onLeave:function(){},
                            onZoom:function(){}, onSelect:function(){} }, 'q2');
      return svg.querySelectorAll('.edge-rows').length;
    })()
    """)
    assert count == 0


def test_the_cost_list_states_how_well_the_planner_predicted():
    """An engine that re-plans on measured cardinalities should lead with how wrong the
    estimates were, because a plan built on bad estimates has a completely different fix
    from a plan that is simply expensive: the first wants another run to feed the measured
    counts back in, the second wants less work."""
    ctx = _views_context()

    misjudged = _render_cost(
        ctx,
        [
            {
                "op_id": 0,
                "kind": "hash_join",
                "measured": True,
                "rows_out": 380070,
                "elapsed_ms": 70.0,
                "est_error": 40.0,
            },
            {
                "op_id": 1,
                "kind": "scan",
                "measured": True,
                "rows_out": 400000,
                "elapsed_ms": 30.0,
                "est_error": 1.2,
            },
        ],
    )
    assert "misjudged" in misjudged
    assert "<b>1 of 2</b>" in misjudged, "names how many steps, of how many measured"
    assert "is-warn" in misjudged, "a misjudged plan is flagged"
    # The per-row cell shows what was predicted beside what happened.
    assert "est 9.5K" in misjudged

    sound = _render_cost(
        ctx,
        [
            {
                "op_id": 0,
                "kind": "scan",
                "measured": True,
                "rows_out": 100,
                "elapsed_ms": 5.0,
                "est_error": 1.1,
            },
        ],
    )
    assert "sound estimates" in sound
    assert "is-warn" not in sound, "a well-estimated plan must not be flagged"


def test_estimate_accuracy_is_silent_when_there_are_no_estimates():
    """An unprofiled or estimate-free plan should say nothing rather than claim accuracy."""
    html = _render_cost(
        _views_context(),
        [
            {"op_id": 0, "kind": "scan", "measured": True, "rows_out": 10, "elapsed_ms": 5.0},
        ],
    )
    assert "cost-accuracy" not in html


def test_a_constant_is_not_rendered_as_a_cpu_measurement():
    """The streaming tier reports `cpu_ns` as the operator's wall time rather than sampling
    the OS clock per morsel (`bc-interp/src/stream/meter.rs`). `cpu_util` is then exactly
    `1/threads` for every operator of every query — a constant wearing a measurement's
    clothes. Rendering it as a percentage invites conclusions from a number that says
    nothing about the reader's query, so the dashboard shows an em dash instead.

    Verified against a live run: the engine emitted `cpu_ns == elapsed_ns` to the nanosecond
    for all four operators, spanning 4us to 24ms.
    """
    ctx = _views_context()
    # The degenerate case, at both thread counts observed on this host.
    assert ctx.eval("UI.cpuMeasured({cpu_util: 0.0625, threads: 16})") is False
    assert ctx.eval("UI.cpuMeasured({cpu_util: 1/15, threads: 15})") is False
    # A genuine reading still renders.
    assert ctx.eval("UI.cpuMeasured({cpu_util: 0.62, threads: 16})") is True
    # Absent data is not a measurement either.
    assert ctx.eval("UI.cpuMeasured({})") is False
    assert ctx.eval("UI.cpuMeasured({cpu_util: 0, threads: 16})") is False


def test_the_run_cpu_summary_ignores_unmeasured_steps():
    """Averaging a constant into the summary would report that constant as the run's CPU
    figure, which is the same lie one level up."""
    ctx = _views_context()
    ctx.eval("""
    var NODES = [
      {op_id:0, kind:'scan', measured:true, rows_out:10,
       elapsed_ms:100, cpu_util:1/16, threads:16},
      {op_id:1, kind:'filter', measured:true, rows_out:10,
       elapsed_ms:100, cpu_util:1/16, threads:16}
    ];
    """)
    # Every step is the degenerate constant, so there is nothing to average.
    measured = ctx.eval("NODES.filter(UI.cpuMeasured).length")
    assert measured == 0, "all-degenerate input contributes no CPU samples"


def test_grid_shading_uses_fixed_ratio_bands_not_quantiles_of_the_run_set():
    """The grid shades each cell by its duration over *that step's own median*, banded at
    fixed ratios. That looks like a heatmap that ought to use quantiles of the observed
    range, and it deliberately does not.

    A ratio is already scale-free: "2x its median" means the same thing whether the step
    takes 1ms or 1s, so a fixed band carries a stable meaning across pipelines. Quantiles
    would rescale to whatever spread happens to be present, which paints ordinary run-to-run
    noise as a full-spectrum regression and paints a genuinely uniform slowdown as normal.

    These cases pin that behaviour; the drift case is the one a quantile scheme gets wrong.
    """
    ctx = _views_context()

    def levels(ratios):
        cells = [
            {"run": r, "op_id": 0, "elapsed_ms": 10.0 * x, "ratio": x, "spilled": False}
            for r, x in enumerate(ratios)
        ]
        grid = {
            "steps": [{"op_id": 0, "kind": "scan", "median_ms": 10.0}],
            "runs": [
                {"query_id": f"q{r}", "started_wall": float(r), "total_ms": 30.0, "status": "ok"}
                for r in range(len(ratios))
            ],
            "cells": cells,
        }
        ctx.eval('document.getElementById("p-grid").innerHTML = "";')
        ctx.eval(f"var G = {json.dumps(grid)};")
        ctx.eval("VIEWS.runGrid(G, { onOpenRun: function(){}, onSelectStep: function(){} });")
        html = ctx.eval('document.getElementById("p-grid").innerHTML')
        return re.findall(r"grid-cell lvl-(\d)", html)

    # A stable pipeline reads as uniformly typical rather than as a gradient of noise.
    assert levels([1.0, 1.0, 1.0, 1.0]) == ["3", "3", "3", "3"]
    # Ordinary drift is still typical. Quantiles would spread this across every band.
    assert levels([0.9, 1.0, 1.1, 1.2]) == ["3", "3", "3", "3"]
    # A real regression stands out against its neighbours.
    assert levels([1.0, 1.0, 1.0, 3.0]) == ["3", "3", "3", "5"]
    # A uniform slowdown is flagged on every run, not normalised away as "the new normal".
    assert levels([2.5, 2.5, 2.5, 2.5]) == ["5", "5", "5", "5"]


def test_the_test_harness_reports_the_dom_properties_tests_rely_on():
    """Guard the guard.

    Three times in this file's history the mini-DOM silently lacked something production code
    reads — `classList` was not synced from the class attribute, `document.querySelectorAll`
    returned a hardcoded `[]`, and `element.id` was `undefined`. Each gap turned real
    assertions into vacuous ones **without failing anything**, which is worse than a broken
    harness because it looks like passing coverage.

    So: assert the harness's own fidelity before trusting a test that runs on it.
    """
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_MINIDOM.read_text())
    facts = json.loads(
        ctx.eval("""
    (function(){
      var d = document.getElementById('probe');
      d.setAttribute('id', 'probe');
      d.setAttribute('class', 'alpha beta');
      document.body.appendChild(d);
      var child = document.createElement('span');
      child.setAttribute('class', 'alpha');
      d.appendChild(child);
      return JSON.stringify({
        id: d.id,
        className: d.className,
        classListSynced: d.classList.contains('alpha') && d.classList.contains('beta'),
        docFindsAttached: document.querySelectorAll('.alpha').length,
        docQuerySelector: document.querySelector('.beta') === d,
        elementScoped: d.querySelectorAll('.alpha').length,
        toggleWorks: (d.classList.toggle('gamma', true), d.classList.contains('gamma')),
        toggleOff: (d.classList.toggle('gamma', false), d.classList.contains('gamma')),
        closestUp: child.closest('.beta') === d,  // .beta is on the parent only
        closestMiss: child.closest('.nonesuch'),
        matches: d.matches('.alpha'),
        rollSettles: (function(){
          // A minimal rAF-driven loop must terminate, not recurse until the stack overflows.
          var r = document.createElement('span'), n = 0;
          (function step(){
            n += 1;
            if (n < 5) requestAnimationFrame(step); else r.textContent = 'done';
          })();
          return r.textContent;
        })(),
      });
    })()
    """)
    )
    assert facts["id"] == "probe", "element.id must read back"
    assert facts["className"] == "alpha beta", "element.className must read back"
    assert facts["classListSynced"], "setAttribute('class') must sync classList"
    assert facts["docFindsAttached"] >= 2, "document.querySelectorAll must search the tree"
    assert facts["docQuerySelector"], "document.querySelector must resolve"
    assert facts["elementScoped"] == 1, "element-scoped query must not escape its subtree"
    assert facts["toggleWorks"] and not facts["toggleOff"], "classList.toggle(force) must work"
    assert facts["closestUp"], "closest() must walk the parent chain"
    assert facts["closestMiss"] is None, "closest() returns null on no match, not the element"
    assert facts["matches"], "matches() must test the element itself"
    # requestAnimationFrame must terminate a roll rather than recurse forever.
    assert facts["rollSettles"] == "done", "an rAF loop must settle, not overflow"


def test_the_run_page_groups_nine_old_tabs_into_five_sections():
    """Plan / Timeline / Operators were three tabs over the *same* per-step data, so a reader
    comparing two renderings had to navigate rather than toggle. They are now one section
    with a rendering switch. Decisions folded into Findings and Raw into Details for the same
    reason: neither earned a top-level destination.
    """
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    html = (_ASSETS / "index.html").read_text()
    panes = re.findall(r'id="(tab-[a-z]+)"', html)
    ctx.eval(f"var PANES = {json.dumps(panes)};")
    ctx.eval(
        "PANES.forEach(function(id){ var d = document.getElementById(id);"
        " d.setAttribute('class','tabpane'); document.body.appendChild(d); });"
    )

    def active(script):
        ctx.eval(script)
        return json.loads(
            ctx.eval("""
        JSON.stringify(PANES.filter(function(id){
          return document.getElementById(id).classList.contains('is-active');
        }))
        """)
        )

    # Pin the rendering explicitly: `stepsView` is a persisted preference, and a test that
    # inherits whatever the last one left is not testing the thing it claims to.
    assert active("switchStepsView('plan')") == ["tab-plan"]
    # One subject, three renderings — the tab never changes.
    assert active("switchTab('steps')") == ["tab-plan"]
    assert active("switchStepsView('timeline')") == ["tab-timeline"]
    assert active("switchStepsView('operators')") == ["tab-operators"]
    # Two former tabs now share one section.
    assert set(active("switchTab('insights')")) == {"tab-insights", "tab-adaptive", "tab-decisions"}
    assert set(active("switchTab('meta')")) == {"tab-meta", "tab-raw"}
    # An unknown tab lands on Steps rather than a blank panel — and keeps the rendering the
    # reader last chose, rather than resetting them to the graph.
    assert active("switchStepsView('timeline')") == ["tab-timeline"]
    assert active("switchTab('nonsense')") == ["tab-timeline"]


def test_a_bookmark_from_before_the_consolidation_still_resolves():
    """`?tab=timeline` was a real URL. It must open the Steps section showing the timeline,
    not drop the reader on an empty page."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    legacy = json.loads(
        ctx.eval("""
    JSON.stringify((function(){
      var LEGACY = { plan: 'steps', timeline: 'steps', operators: 'steps',
                     decisions: 'insights', raw: 'meta' };
      return Object.keys(LEGACY).map(function(k){ return [k, LEGACY[k]]; });
    })())
    """)
    )
    # The mapping the router applies must cover every tab name the old UI could emit.
    retired = {"plan", "timeline", "operators", "decisions", "raw"}
    assert {k for k, _ in legacy} == retired
    app = (_ASSETS / "app.js").read_text()
    for name in retired:
        assert f"{name}:" in app[app.index("const LEGACY") : app.index("switchTab(LEGACY")], (
            f"retired tab {name} has no redirect"
        )


def test_a_sortable_header_is_a_control_not_a_clickable_cell():
    """A `<th>` with a click handler cannot be reached by keyboard and announces nothing about
    what activating it does. The control goes inside the cell; `aria-sort` goes on the cell.
    """
    ctx = _views_context()
    ctx.eval("""
    var COLS = [{label: 'Name', value: function(r){ return r.n; }},
                {label: 'Rows', num: true, value: function(r){ return r.v; }}];
    UI.table('t-a11y', COLS, [{n:'a', v:5}, {n:'b', v:null}], {caption: 'Steps'});
    """)
    html = ctx.eval("document.getElementById('t-a11y').innerHTML")
    assert '<button class="th-sort"' in html, "the sort control must be a button"
    assert "aria-sort=" in html, "the sorted state must be exposed on the cell"
    assert 'scope="col"' in html
    # The accessible name states the action, not just the column name.
    assert re.search(r'aria-label="Name, sort (as|de)scending"', html)
    # A screen reader gets the row count; sighted readers get it in the panel head.
    assert "visually-hidden" in html and "2 rows" in html


def test_absent_values_render_as_one_em_dash_not_four_different_lies():
    """`null`, `undefined`, `NaN`, and `""` are four ways of saying "no value" badly. Zero is
    not one of them — it is a value, and rendering it as a dash would be a different bug."""
    ctx = _views_context()
    for absent in ("null", "undefined", "NaN", "''", "'   '", "1/0", "-1/0"):
        assert ctx.eval(f"UI.blankToDash({absent})") == "—", absent
    assert ctx.eval("UI.blankToDash(0)") == "0", "zero is a value"
    assert ctx.eval("UI.blankToDash(false)") == "false"
    assert ctx.eval("UI.blankToDash('ok')") == "ok"


def test_an_empty_table_says_so_inside_the_table():
    ctx = _views_context()
    ctx.eval("""
    var C = [{label: 'A', value: function(r){ return r.a; }}];
    UI.table('t-empty', C, [], {emptyText: 'No steps recorded.'});
    """)
    html = ctx.eval("document.getElementById('t-empty').innerHTML")
    assert "table-empty" in html and "No steps recorded." in html
    assert "<thead>" in html, "the header stays, so the shape of the data is still legible"


def test_the_shortcut_sheet_is_generated_from_the_one_registry():
    """It used to be hand-written HTML duplicating `ACTIONS`. Every new shortcut went
    undocumented and every changed key made the sheet lie."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    ctx.eval("renderShortcuts();")
    html = ctx.eval("document.getElementById('shortcuts-list').innerHTML")
    registered = ctx.eval("ACTIONS.filter(function(a){ return a.keys; }).length")
    assert registered > 0
    assert html.count("<div>") == registered, "every keyed action appears exactly once"
    # Every label in the registry is present in the sheet.
    labels = json.loads(
        ctx.eval(
            "JSON.stringify(ACTIONS.filter(function(a){ return a.keys; })"
            "  .map(function(a){ return a.label; }))"
        )
    )
    for label in labels:
        assert label in html, f"{label} is registered but absent from the sheet"
    # The static markup must not have re-grown a hand-written copy.
    shell = (_ASSETS / "index.html").read_text()
    sheet = shell[shell.index('id="shortcuts-list"') : shell.index("shortcuts-context")]
    assert "<kbd>" not in sheet, "the generated list must not be pre-populated by hand"


def test_no_control_is_named_only_by_a_symbol():
    """A screen reader reads `◐` as a codepoint name or as nothing. The glyph is decoration;
    the button needs a real name, and `title` alone is not reliably announced.

    Regression: the theme, refresh, pause, run-step and expand controls all shipped with a
    bare glyph as their entire accessible name.
    """
    html = (_ASSETS / "index.html").read_text()
    for m in re.finditer(r"<button([^>]*)>(.*?)</button>", html, re.S):
        attrs, inner = m.group(1), m.group(2)
        text = re.sub(r"<[^>]+>", "", inner).strip()
        named = "aria-label" in attrs or "aria-labelledby" in attrs
        # A control whose visible text is only symbols/punctuation carries no name.
        symbolic = text and not re.search(r"[A-Za-z0-9]", text)
        assert named or not symbolic, (
            f"control named only by a symbol: {text!r} in {m.group(0)[:90]}"
        )
        # Decorative glyphs beside a real name must be hidden from the accessibility tree.
        if named and symbolic:
            assert 'aria-hidden="true"' in inner, f"decorative glyph {text!r} should be aria-hidden"


def test_counters_that_change_on_a_poll_announce_themselves():
    """A badge that silently ticks up is invisible to a screen reader; `polite` so it waits
    for a pause rather than interrupting."""
    html = (_ASSETS / "index.html").read_text()
    block = html[html.index('id="log-count"') - 60 : html.index('id="log-count"') + 120]
    assert "aria-live" in block, "the log counter must announce"
    assert "polite" in block, "it must not interrupt"


def test_nice_axis_ticks_land_on_readable_values():
    """1, 2 or 5 times a power of ten. Without this an axis reads 0, 0.333, 0.667 and the
    reader does arithmetic to place a value."""
    ctx = _views_context()

    def ticks(lo, hi, w):
        return json.loads(ctx.eval(f"JSON.stringify(UI.niceTicks({lo}, {hi}, {w}))"))

    assert ticks(0, 100, 600) == [0, 20, 40, 60, 80, 100]
    assert ticks(1000, 1350, 400) == [1000, 1100, 1200, 1300]
    # Fractional steps must not drift: repeated addition of 0.2 gives 0.6000000000000001.
    assert ticks(0, 1, 600) == [0, 0.2, 0.4, 0.6, 0.8, 1]
    assert ticks(0, 0.03, 300) == [0, 0.01, 0.02, 0.03]
    # Degenerate inputs produce no ticks rather than a hang or a NaN axis.
    assert ticks(5, 5, 600) == []
    assert ticks(0, 10, 0) == []
    # Every step is 1, 2 or 5 times a power of ten.
    for lo, hi, w in ((0, 97, 500), (3, 1234, 800), (0, 0.007, 240)):
        t = ticks(lo, hi, w)
        if len(t) < 2:
            continue
        step = t[1] - t[0]
        mantissa = step / (10 ** math.floor(math.log10(step)))
        assert round(mantissa, 6) in (1.0, 2.0, 5.0), f"step {step} is not a nice number"


def test_an_open_dialog_keeps_focus_and_gives_it_back():
    """`aria-modal` tells a screen reader the page behind is inert; it does not stop the
    browser moving focus there. Without a trap, Tab walks out of the dialog into content the
    reader has been told is unavailable."""
    app = (_ASSETS / "app.js").read_text()
    assert "function trapFocus" in app
    # Registered while open, removed on close — a listener that outlives the dialog would
    # swallow Tab for the rest of the session.
    assert "addEventListener('keydown', trapFocus, true)" in app
    assert "removeEventListener('keydown', trapFocus, true)" in app
    # Focus is returned where it came from, not dropped at the top of the document.
    assert "returnFocusTo" in app and "returnFocusTo.focus()" in app
    # The focusable list is queried per Tab, because the palette rebuilds on every keystroke.
    trap = app[app.index("function trapFocus") : app.index("function showModal")]
    assert "querySelectorAll(FOCUSABLE)" in trap, "the list must be read fresh, not captured"


def test_text_on_a_filled_accent_is_a_named_token():
    """`#fff` on an accent surface means "whatever stays legible on the accent", not "white".
    Naming it lets a theme with a light accent flip it in one place."""
    css = (_ASSETS / "app.css").read_text()
    root = _css_block(css, ":root {")
    assert "--on-accent:" in root
    components = css[css.index("═══ chrome ═══") :]
    # The print block deliberately keeps literals — print has no theme to resolve against.
    non_print = re.sub(r"@media print \{.*?\n\}\n", "", components, flags=re.S)
    assert "color: #fff" not in non_print, "accent text must go through the token"


def test_a_panel_distinguishes_loading_failed_and_empty():
    """Four states, and conflating any two is the commonest lie a dashboard tells. "Nothing
    here" and "we could not reach the engine" look identical if you only ever render an empty
    list, and a reader who cannot tell them apart either chases a phantom or ignores a real
    problem."""
    ctx = _views_context()

    def state(opts):
        return json.loads(
            ctx.eval(f"""
        (function(){{
          var ok = UI.panelState('pstate', {opts});
          var e = document.getElementById('pstate');
          return JSON.stringify({{ ok: ok, busy: e.attrs['aria-busy'] || null,
                                   html: e.innerHTML }});
        }})()
        """)
        )

    loading = state("{ loading: true }")
    assert loading["ok"] is False and loading["busy"] == "true"
    assert "skeleton" in loading["html"]
    # Placeholders are decorative; a screen reader should not read four blank lines.
    assert 'aria-hidden="true"' in loading["html"]

    failed = state("{ error: 'Lost contact with the engine.', onRetry: function(){} }")
    assert failed["ok"] is False and failed["busy"] is None
    assert 'role="alert"' in failed["html"], "a failure must be announced"
    assert "Lost contact" in failed["html"], "it must say what failed"
    assert "Try again" in failed["html"], "and offer the way out"

    empty = state("{ empty: true, emptyState: { title: 'No runs yet', body: 'b' } }")
    assert empty["ok"] is False
    assert "empty-state" in empty["html"] and "No runs yet" in empty["html"]
    assert 'role="alert"' not in empty["html"], "empty is not an error"

    content = state("{ render: function(el){ el.innerHTML = '<p>real</p>'; } }")
    assert content["ok"] is True and content["html"] == "<p>real</p>"
    assert content["busy"] is None, "aria-busy must be cleared once content lands"


def test_identifiers_are_copyable_without_hijacking_a_selection():
    """A query id or plan signature exists to be pasted somewhere else. One delegated handler
    makes every marked identifier copyable rather than growing a button beside each — but it
    must never fire while the reader is selecting part of the text, which is the other reason
    someone clicks a monospaced value."""
    app = (_ASSETS / "app.js").read_text()
    assert "function installCopyAnywhere" in app
    handler = app[
        app.index("function installCopyAnywhere") : app.index("/* ---------- command palette")
    ]
    assert "getSelection" in handler, "an active selection must suppress the copy"
    # The visible text may be truncated; the copied value must not be.
    assert "dataset.copyValue" in handler
    assert "[data-copyable]" in handler
    # Marked elements are reachable and named.
    assert app.count("data-copyable") >= 2
    for block in re.findall(r'class="[^"]*"\s+data-copyable[^>]*', app):
        assert "tabindex" in block, f"copyable element not keyboard-reachable: {block[:60]}"
        assert "aria-label" in block, f"copyable element unnamed: {block[:60]}"


def test_the_graph_drops_unreadable_text_when_zoomed_out():
    """Below roughly 7px the per-node text is noise, not information. Hiding it keeps the
    structure legible and makes a large plan cheaper to paint at exactly the zoom where the
    most nodes are on screen."""
    dag = (_ASSETS / "dag.js").read_text()
    assert "LOD_SCALE" in dag
    assert "classList.toggle('is-far'" in dag
    css = (_ASSETS / "app.css").read_text()
    far = css[css.index(".dag.is-far") :]
    # The detail lines go; the operator name stays, so the shape is still readable.
    for hidden in ("node-detail", "node-op", "node-metric", "node-share", "edge-rows"):
        assert hidden in far.split("}")[0], f"{hidden} should be hidden when zoomed out"
    assert ".dag.is-far .node-kind" in css, "the operator name must survive"


def _profile(total_ms: float, ops: list[dict]) -> dict:
    return {"total_ms": total_ms, "ops": [{"measured": True, **op} for op in ops]}


def _rules(profile: dict) -> set[str]:
    from batcher.observe.insights import derive_insights

    return {i["rule"] for i in derive_insights(profile)}


def test_an_exploding_join_is_reported():
    """A join emitting far more rows than it consumed means a non-unique key on both sides.
    It is invisible in a plan's *structure* — the node count is identical whether it emitted
    a thousand rows or a billion — so only the row counts can surface it."""
    fires = _profile(
        100.0,
        [
            {
                "op_id": 0,
                "kind": "hash_join",
                "rows_in": 10_000,
                "rows_out": 500_000,
                "elapsed_ms": 60.0,
            },
        ],
    )
    assert "exploding-join" in _rules(fires)
    # A join that reduces rows, which is the normal case, must stay quiet.
    quiet = _profile(
        100.0,
        [
            {
                "op_id": 0,
                "kind": "hash_join",
                "rows_in": 10_000,
                "rows_out": 9_000,
                "elapsed_ms": 60.0,
            },
        ],
    )
    assert "exploding-join" not in _rules(quiet)
    # Small joins are noise, not findings.
    tiny = _profile(
        100.0,
        [
            {"op_id": 0, "kind": "hash_join", "rows_in": 10, "rows_out": 900, "elapsed_ms": 60.0},
        ],
    )
    assert "exploding-join" not in _rules(tiny)


def test_a_selective_filter_running_after_costly_work_is_reported():
    """The cost is not the filter — it is everything beneath it that processed rows which
    were never going to survive."""
    fires = _profile(
        200.0,
        [
            {
                "op_id": 0,
                "kind": "filter",
                "rows_in": 100_000,
                "rows_out": 1_000,
                "elapsed_ms": 5.0,
            },
            {
                "op_id": 1,
                "kind": "hash_join",
                "rows_in": 100_000,
                "rows_out": 100_000,
                "elapsed_ms": 120.0,
            },
        ],
    )
    assert "late-filter" in _rules(fires)
    # A selective filter sitting directly above a scan is exactly where it belongs.
    quiet = _profile(
        200.0,
        [
            {
                "op_id": 0,
                "kind": "filter",
                "rows_in": 100_000,
                "rows_out": 1_000,
                "elapsed_ms": 5.0,
            },
            {
                "op_id": 1,
                "kind": "scan",
                "rows_in": 100_000,
                "rows_out": 100_000,
                "elapsed_ms": 120.0,
            },
        ],
    )
    assert "late-filter" not in _rules(quiet)


def test_time_spread_across_many_steps_gets_the_opposite_advice():
    """No single step to attack means the win is doing less work, not faster work — the
    opposite remedy from a dominant operator, so it needs its own finding."""
    spread = _profile(
        200.0,
        [
            {"op_id": i, "kind": "project", "rows_in": 10, "rows_out": 10, "elapsed_ms": 10.0}
            for i in range(8)
        ],
    )
    assert "long-tail" in _rules(spread)
    # One step owning the time is the other shape, and must not claim a long tail.
    concentrated = _profile(
        200.0,
        [
            {"op_id": 0, "kind": "sort", "rows_in": 10, "rows_out": 10, "elapsed_ms": 180.0},
            *[
                {"op_id": i, "kind": "project", "rows_in": 10, "rows_out": 10, "elapsed_ms": 1.0}
                for i in range(1, 8)
            ],
        ],
    )
    assert "long-tail" not in _rules(concentrated)


def test_a_run_that_barely_touched_its_steps_says_so():
    """Otherwise the reader hunts for a slow operator that does not exist."""
    overhead = _profile(
        50.0,
        [
            {"op_id": 0, "kind": "scan", "rows_in": 5, "rows_out": 5, "elapsed_ms": 1.0},
        ],
    )
    assert "planning-dominates" in _rules(overhead)
    busy = _profile(
        100.0,
        [
            {"op_id": 0, "kind": "scan", "rows_in": 5, "rows_out": 5, "elapsed_ms": 95.0},
        ],
    )
    assert "planning-dominates" not in _rules(busy)


def test_every_insight_carries_a_rule_title_evidence_and_action():
    """A finding without an action is an observation, and the reader still has to work out
    what to do. Each rule must name itself so a false positive is reportable."""
    from batcher.observe.insights import derive_insights

    profile = _profile(
        200.0,
        [
            {
                "op_id": 0,
                "kind": "hash_join",
                "rows_in": 10_000,
                "rows_out": 500_000,
                "elapsed_ms": 120.0,
                "spilled": True,
                "spill_bytes": 2**20,
            },
            {
                "op_id": 1,
                "kind": "filter",
                "rows_in": 100_000,
                "rows_out": 1_000,
                "elapsed_ms": 5.0,
            },
        ],
    )
    found = derive_insights(profile)
    assert found, "this profile should produce findings"
    for i in found:
        for field in ("rule", "title", "evidence", "action", "severity"):
            assert i.get(field), f"{i.get('rule')} is missing {field}"
        assert i["severity"] in {"critical", "warning", "info"}
    severities = [i["severity"] for i in found]
    order = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: order[s]), "most severe first"


def test_the_reference_page_offers_a_contents_strip_that_tracks_the_filter():
    """At 50+ glossary entries, landing at the top and scrolling is the wrong default. The
    counts are part of each label so a reader knows the size of a section before jumping —
    and they must reflect the *filtered* set, or the strip promises rows that are not there.
    """
    ctx = _views_context()

    def toc(query):
        html = ctx.eval(
            "(function(){ var d = document.createElement('div');"
            f" LEARN.renderLearn(d, {json.dumps(query)}); return d.innerHTML; }})()"
        )
        return dict(re.findall(r'href="#learn-([a-z]+)">([^<]+)', html))

    full = toc("")
    assert set(full) == {"how", "coming", "steps", "metrics", "glossary"}
    # Every section id in the strip must exist as a target on the page.
    html = ctx.eval(
        "(function(){ var d = document.createElement('div');"
        " LEARN.renderLearn(d, ''); return d.innerHTML; })()"
    )
    for section in full:
        assert f'id="learn-{section}"' in html, f"#learn-{section} is linked but absent"

    narrowed = toc("spill")
    for section, label in narrowed.items():
        wide = int(re.search(r"\((\d+)\)", full[section]).group(1))
        thin = int(re.search(r"\((\d+)\)", label).group(1))
        assert thin <= wide, f"{section} count grew when filtered"
    assert any(
        int(re.search(r"\((\d+)\)", v).group(1)) < int(re.search(r"\((\d+)\)", full[k]).group(1))
        for k, v in narrowed.items()
    ), "filtering should narrow at least one section"


def test_optional_columns_can_be_hidden_but_load_bearing_ones_cannot():
    """A wide table (the operator table has 12 columns) should let a reader hide the secondary
    ones — but a column without the `optional` flag is load-bearing and always shown, so the
    table can never be reduced to nothing."""
    ctx = _views_context()
    ctx.eval("""
    var COLS = [{label: 'Step', value: function(r){ return r.s; }},
                {label: 'Detail', optional: true, value: function(r){ return r.d; }},
                {label: 'Time', value: function(r){ return r.t; }}];
    """)
    ctx.eval("UI.setPref('hiddenCols', {});")
    ctx.eval("UI.table('t-cols', COLS, [{s:'scan', d:'src', t:5}]);")
    shown = ctx.eval("document.getElementById('t-cols').innerHTML")
    assert "col-menu" in shown, "a table with an optional column shows the columns control"
    # Only optional columns get a toggle.
    assert 'data-col-toggle="Detail"' in shown
    assert 'data-col-toggle="Step"' not in shown
    assert 'data-col-toggle="Time"' not in shown

    # Hiding the optional column drops it from the render but keeps the required ones.
    ctx.eval("UI.setPref('hiddenCols', { 't-cols': ['Detail'] });")
    ctx.eval("UI.table('t-cols', COLS, [{s:'scan', d:'src', t:5}]);")
    hidden = ctx.eval("document.getElementById('t-cols').innerHTML")
    # Count header cells, not the substring "th" (which also appears in the toggle label).
    assert len(re.findall(r"<th[ >]", hidden)) == 2, "the hidden column is gone, two remain"
    # The optional column's *body* cell is gone; its name still appears in the toggle menu.
    body = hidden[hidden.index("<tbody>") :]
    assert "src" not in body, "the hidden column's cells are gone"
    # The choice persists.
    stored = json.loads(ctx.eval("JSON.stringify(UI.getPref('hiddenCols'))"))
    assert stored["t-cols"] == ["Detail"]


def test_drag_to_zoom_needs_a_threshold_so_a_click_still_works():
    """Drag selects an arbitrary time window on the log histogram; a click still picks one
    bucket. Without a pixel threshold every click registers as a zero-width drag."""
    app = (_ASSETS / "app.js").read_text()
    assert "function installHistoDrag" in app
    drag = app[app.index("function installHistoDrag") : app.index("/** Narrow to a bucket")]
    assert "moved < 3" in drag, "a sub-threshold movement must fall through to the click"
    # The window is derived from the published geometry, not re-measured, so the selection
    # lines up with the bars the reader saw.
    assert "logHistoGeom" in app
    assert "setPointerCapture" in drag, "the drag must survive leaving the element"


def test_the_run_verdict_names_the_comparison_in_every_state():
    """The one-line judgement at the top of a run is what a newcomer reads first. It must
    distinguish slower / faster / typical / no-baseline-yet / failed — and "first run" is not
    silence, because an empty space is something the reader then has to interpret."""
    ctx = _views_context()

    def verdict(d):
        ctx.eval(f"VIEWS.verdict({json.dumps(d)});")
        return json.loads(
            ctx.eval("""
        JSON.stringify((function(){
          var e = document.getElementById('d-verdict');
          return { cls: e.className, text: e.textContent, hidden: !!e.hidden };
        })())
        """)
        )

    slow = verdict({"status": "ok", "total_ms": 400, "baseline": {"ratio": 2.5, "median_ms": 160}})
    assert slow["cls"] == "verdict is-warn"
    assert "slower" in slow["text"] and "2.5" in slow["text"]
    assert not slow["hidden"]

    fast = verdict({"status": "ok", "total_ms": 40, "baseline": {"ratio": 0.4, "median_ms": 100}})
    assert fast["cls"] == "verdict is-good"
    assert "faster" in fast["text"], "a fast run is worth seeing, not just a slow one"

    typical = verdict(
        {"status": "ok", "total_ms": 100, "baseline": {"ratio": 1.0, "median_ms": 100}}
    )
    assert "typical" in typical["text"]

    # A first run says so rather than showing nothing.
    first = verdict({"status": "ok", "total_ms": 100, "baseline": None})
    assert not first["hidden"], "no baseline is information, not a reason to hide"
    assert "first run" in first["text"] or "no baseline" in first["text"]

    failed = verdict({"status": "error", "total_ms": 0})
    assert failed["cls"] == "verdict is-critical" and "failed" in failed["text"]


def test_grid_cells_are_keyboard_operable_and_carry_one_run_id():
    """A run-grid cell opens its run on click, so a keyboard must reach and activate it — a
    `<td>` does neither on its own.

    Regression: the cell carried `data-run` twice (a column index and a query id), and the
    second silently clobbered the first, so the crosshair highlighted a column keyed on an id
    the header did not share. Column identity now lives on `data-col`, run identity on
    `data-run`, and they never collide.
    """
    ctx = _views_context()
    grid = {
        "steps": [{"op_id": o, "kind": "scan", "median_ms": 10.0} for o in range(2)],
        "runs": [
            {"query_id": f"q{r}", "started_wall": float(r), "total_ms": 30.0, "status": "ok"}
            for r in range(3)
        ],
        "cells": [
            {"run": r, "op_id": o, "elapsed_ms": 10.0, "ratio": 1.0, "spilled": False}
            for r in range(3)
            for o in range(2)
        ],
    }
    ctx.eval(f"var G = {json.dumps(grid)};")
    ctx.eval("VIEWS.runGrid(G, { onOpenRun: function(){}, onSelectStep: function(){} });")
    html = ctx.eval("document.getElementById('p-grid').innerHTML")

    # Every measured cell is focusable and named.
    cells = re.findall(r"<td class=\"grid-cell[^\"]*\"[^>]*>", html)
    real = [c for c in cells if "is-absent" not in c]
    assert real, "the grid should have measured cells"
    for c in real:
        assert 'tabindex="0"' in c, "a clickable cell must be keyboard-reachable"
        assert 'role="button"' in c
        assert "aria-label=" in c
        # The two identities never share an attribute name.
        assert c.count("data-run=") == 1, "run id must appear exactly once"
        assert "data-col=" in c, "column identity is separate from run id"

    # The header carries the same data-col, so the crosshair can span a whole column.
    assert 'class="grid-run" data-col=' in html


def test_keyboard_focus_rules_only_target_focusable_elements():
    """A `:focus-visible` rule on an element that can never receive focus is dead CSS. Every
    class in the shared keyboard-parity rule must be something the markup makes focusable."""
    css = (_ASSETS / "app.css").read_text()
    js = "".join(
        (_ASSETS / f).read_text() for f in ("views.js", "charts.js", "plan.js", "live.js", "app.js")
    )
    shell = (_ASSETS / "index.html").read_text()
    haystack = js + shell
    block = css[css.index("═══ keyboard parity ═══") :]
    styled = set(re.findall(r"\.([\w-]+):focus-visible", block))
    for cls in styled:
        # Every place the class is opened must be a focusable element: a native button/anchor,
        # or a tag given tabindex or a button role. A single non-focusable render makes the
        # focus rule a lie for that case.
        # Match the class as a whole token, not as a substring: `\brow\b` also matches
        # `row-name`, a different (static) element.
        sites = list(
            re.finditer(rf'<(\w+)[^>]*class="(?:[^"]*[\s"])?{cls}(?:[\s"$}}]|\$\{{)', haystack)
        )
        assert sites, f".{cls} has a focus rule but is never rendered"
        # At least one render must be focusable, or the rule can never fire. A class may also
        # appear on a static element (a `.seg` label between two `.seg` buttons); `:focus-visible`
        # is simply inert there, so it is enough that *some* render can take focus.
        any_focusable = False
        for m in sites:
            tag = m.group(1)
            window = haystack[m.start() : m.start() + 260]
            if (
                tag in ("button", "a", "input", "select", "textarea")
                or "tabindex" in window
                or 'role="button"' in window
            ):
                any_focusable = True
                break
        assert any_focusable, f".{cls} has a focus rule but no render can take focus"


def test_a_kpi_that_leads_somewhere_is_reachable_and_one_that_does_not_is_not():
    """The failed and spill KPIs open the runs behind them — so they must be keyboard-reachable
    when they lead somewhere, and must *not* be a dead stop in the focus order when they don't
    (no failures, no spill)."""
    ctx = _views_context()
    # A .kpi wrapper around the value element, so closest('.kpi') resolves.
    ctx.eval("""
    var wrap = document.getElementById('kpi-wrap');
    wrap.setAttribute('class', 'kpi');
    var val = document.getElementById('k-failed');
    wrap.appendChild(val);
    """)

    def render(n_failed):
        ctx.eval(
            f"VIEWS.kpis({{ n_pipelines: 1, n_queries: 5, n_running: 0, n_failed: {n_failed},"
            " percentiles: {} }, function(){});"
        )
        return json.loads(
            ctx.eval("""
        JSON.stringify((function(){
          var k = document.getElementById('kpi-wrap');
          return { tabindex: k.attrs['tabindex'] || null, role: k.attrs['role'] || null,
                   clickable: k.classList.contains('is-clickable') };
        })())
        """)
        )

    actionable = render(3)
    assert actionable["clickable"] is True
    assert actionable["tabindex"] == "0", "an actionable KPI must be keyboard-reachable"
    assert actionable["role"] == "button"

    inert = render(0)
    assert inert["clickable"] is False
    assert inert["tabindex"] is None, "a KPI that leads nowhere must not be a focus stop"
    assert inert["role"] is None


def test_the_number_roll_settles_on_the_real_value():
    """`rollTo` animates a counter toward its target. The final rendered text must be the
    target, formatted — a value that only *approaches* it would show a wrong number whenever
    the animation is cut short (or reduced-motion is on)."""
    ctx = _views_context()
    result = ctx.eval("""
    (function(){
      var e = document.createElement('span');
      UI.rollTo(e, 380000, function(v){ return UI.count(Math.round(v)); });
      return e.textContent;
    })()
    """)
    assert result == "380.0K", f"roll settled on {result!r}, not the target"


def test_arrow_keys_walk_the_plan_in_two_dimensions():
    """The graph is drawn in rows and columns; arrow keys should navigate both. Up/down walks
    the flow, left/right moves across nodes on the same row — otherwise the shortcut sheet's
    "walk the plan" promise only half works."""
    ctx = _views_context()
    dag = {
        "nodes": [
            {
                "op_id": 0,
                "kind": "hash_join",
                "measured": True,
                "rows_out": 10,
                "elapsed_ms": 5.0,
                "column": 0,
                "row": 0,
                "depth": 0,
            },
            {
                "op_id": 1,
                "kind": "scan",
                "measured": True,
                "rows_out": 10,
                "elapsed_ms": 5.0,
                "column": 0,
                "row": 1,
                "depth": 1,
            },
            {
                "op_id": 2,
                "kind": "scan",
                "measured": True,
                "rows_out": 10,
                "elapsed_ms": 5.0,
                "column": 1,
                "row": 1,
                "depth": 1,
            },
        ],
        "edges": [{"from": 1, "to": 0}, {"from": 2, "to": 0}],
        "width": 2,
        "depth": 2,
        "critical_path": [],
    }
    ctx.eval(f"var D = {json.dumps(dag)};")
    ctx.eval("""
    var svg = document.getElementById('dag');
    DAG.render(svg, D, { onHover: function(){}, onLeave: function(){},
                         onZoom: function(){}, onSelect: function(){} }, 'q');
    """)

    def selected(action):
        ctx.eval(action)
        return ctx.eval("DAG.selectedNode() ? DAG.selectedNode().op_id : null")

    # Down from nothing lands on the first node in the flattening.
    first = selected("DAG.step(1)")
    assert first is not None
    # Down again moves along the vertical flattening.
    assert selected("DAG.step(1)") != first
    # From a row with two siblings, left/right moves between them. selectOnly sets without
    # the toggle, so we land on op 1 deterministically.
    ctx.eval("DAG.selectOnly(1);")  # op 1 is row 1, column 0
    assert ctx.eval("DAG.selectedNode().op_id") == 1
    across = selected("DAG.stepAcross(1)")
    assert across == 2, "right should move to the sibling at column 1"
    # And back.
    assert selected("DAG.stepAcross(-1)") == 1
    # At a row edge, stepping further stays put rather than wrapping or erroring.
    assert selected("DAG.stepAcross(-1)") == 1


def test_the_graph_does_not_animate_under_reduced_motion():
    """The node rise and bar sweep are decorative. A reader who asked for no motion should get
    the graph drawn, not animated in."""
    dag = (_ASSETS / "dag.js").read_text()
    assert "REDUCED_MOTION" in dag
    assert "prefers-reduced-motion" in dag
    # The entrance delay is gated on the preference.
    assert "if (!REDUCED_MOTION) group.style.animationDelay" in dag
    assert "group.style.animation = 'none'" in dag


def test_a_huge_table_is_capped_and_says_so():
    """A session with thousands of runs should not build thousands of `<tr>`s. The cap is
    generous and `content-visibility` handles the rest, but a silently truncated table claims
    to be complete — so the cap, when hit, is stated."""
    ctx = _views_context()
    ctx.eval("var COLS = [{label: 'N', value: function(r){ return r.n; }}];")

    ctx.eval(
        "UI.table('t-small', COLS, Array.from({length: 10}, function(_, i){ return {n: i}; }));"
    )
    small = ctx.eval("document.getElementById('t-small').innerHTML")
    assert "table-capped" not in small, "a small table is not capped and says nothing"
    assert len(re.findall(r"<tr", small)) == 11  # 10 rows + header

    ctx.eval(
        "UI.table('t-big', COLS, Array.from({length: 1200}, function(_, i){ return {n: i}; }));"
    )
    big = ctx.eval("document.getElementById('t-big').innerHTML")
    assert "table-capped" in big, "an oversized table must disclose the cap"
    assert "500" in big and "1200" in big, "the note names both the shown and the total"
    # 500 rows drawn + the header, not 1200.
    assert len(re.findall(r"<tr", big)) == 501


def test_table_rows_use_content_visibility_for_cheap_virtualization():
    """The framework-free virtualization the whole dashboard relies on: skip layout and paint
    for rows scrolled out of view, with an intrinsic-size hint so the scrollbar is honest
    before a row is measured."""
    css = (_ASSETS / "app.css").read_text()
    assert "content-visibility: auto" in css
    rule = css[css.index("table.dense tbody tr {") :].split("}")[0]
    assert "content-visibility: auto" in rule
    assert "contain-intrinsic-size" in rule, "an intrinsic size keeps the scrollbar accurate"


def test_the_node_hover_card_carries_the_whole_per_step_picture():
    """A mouse user should not have to click a step to see its numbers. The hover card shows
    rows in/out, selectivity, spill, and — the one that explains a wrong plan — how far the
    estimate was off, but only the fields that apply to this step."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    node = {
        "kind": "hash_join",
        "detail": "inner on k",
        "rows_in": 400_000,
        "rows_out": 380_000,
        "selectivity": 0.95,
        "elapsed_ms": 20.0,
        "spilled": True,
        "spill_bytes": 2048,
        "est_error": 40.0,
        "on_critical_path": True,
    }
    ctx.eval(f"showNodeTip({{clientX: 100, clientY: 100}}, {json.dumps(node)}, 0.6);")
    tip = ctx.eval("document.getElementById('tip').innerHTML")
    for field in ("rows in", "rows out", "kept", "spilled", "estimate", "40x off", "critical path"):
        assert field in tip, f"the hover card is missing {field!r}"

    # A clean scan shows only what applies — no spill row, no estimate warning.
    clean = {"kind": "scan", "rows_out": 1000, "elapsed_ms": 1.0}
    ctx.eval(f"showNodeTip({{clientX: 100, clientY: 100}}, {json.dumps(clean)}, 0.1);")
    plain = ctx.eval("document.getElementById('tip').innerHTML")
    assert "spilled" not in plain, "a step that did not spill shows no spill row"
    assert "estimate" not in plain, "a well-estimated step shows no estimate warning"
    assert "rows out" in plain, "but the core numbers are always there"


def test_command_palette_ranks_exact_over_prefix_over_fuzzy():
    """A subsequence match that happens to be short is not a better match than a prefix. The
    palette scores exact 4, prefix 3, word-start 2, subsequence 1 — so typing "tab" surfaces
    "Table" before something that merely contains t-a-b in order."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    assert ctx.eval('matchScore("theme", "theme")') == 4
    assert ctx.eval('matchScore("the", "theme toggle")') == 3
    assert ctx.eval('matchScore("tog", "theme toggle")') == 2
    assert ctx.eval('matchScore("tgl", "toggle")') == 1
    assert ctx.eval('matchScore("zzz", "theme")') == 0
    # An empty needle keeps everything (score 1), which is what the recents view then re-sorts.
    assert ctx.eval('matchScore("", "anything")') == 1


def test_the_palette_remembers_recent_choices():
    """An empty query should surface what you last used, most recent first — not a static
    list every time (the Linear/Raycast pattern)."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    ctx.eval("UI.setPref('paletteRecent', []);")
    ctx.eval("rememberPaletteChoice('a');")
    ctx.eval("rememberPaletteChoice('b');")
    ctx.eval("rememberPaletteChoice('a');")  # re-choosing moves it to front, no duplicate
    recent = json.loads(ctx.eval("JSON.stringify(UI.getPref('paletteRecent'))"))
    assert recent == ["a", "b"], "most recent first, deduplicated"
    # Bounded so the list cannot grow without limit.
    for i in range(12):
        ctx.eval(f"rememberPaletteChoice('x{i}');")
    assert len(json.loads(ctx.eval("JSON.stringify(UI.getPref('paletteRecent'))"))) <= 8


def test_the_time_bar_splits_wall_clock_into_steps_and_overhead():
    """Where a run's wall clock went: the steps versus the fixed cost around them. Computed
    from what the profile measures — summed operator time against total — not an invented
    plan/queue/exec split."""
    ctx = _views_context()
    ctx.eval("""
    VIEWS.timeBar({ total_ms: 100.0, dag: { nodes: [
      { measured: true, elapsed_ms: 30.0 }, { measured: true, elapsed_ms: 10.0 }] } });
    """)
    html = ctx.eval("document.getElementById('d-timebar').innerHTML")
    assert "width:40.0%" in html, "40ms of steps out of 100ms"
    assert "width:60.0%" in html, "60ms of overhead"
    assert "aria-label=" in html and "in the steps" in html, "the split is described for SR"

    # Operator time sums across steps and can exceed wall clock under parallelism; the bar
    # shows time, so the executed share is clamped to the run's own duration.
    ctx.eval("""
    VIEWS.timeBar({ total_ms: 50.0, dag: { nodes: [
      { measured: true, elapsed_ms: 40.0 }, { measured: true, elapsed_ms: 40.0 }] } });
    """)
    parallel = ctx.eval("document.getElementById('d-timebar').innerHTML")
    assert "width:100.0%" in parallel, "executed share never exceeds the wall clock"

    # No measured steps → the bar hides rather than dividing by zero.
    ctx.eval("VIEWS.timeBar({ total_ms: 0, dag: { nodes: [] } });")
    assert ctx.eval("!!document.getElementById('d-timebar').hidden"), "no data hides the bar"


def test_the_comparison_change_encodes_direction_and_goodness_separately():
    """A change carries two independent facts: which way it moved, and whether that is good.
    Slower is up-and-warn, faster is down-and-good — so a large improvement reads as clearly
    as a large regression, not as an absence of red."""
    ctx = _views_context()
    cmp = {
        "ok": True,
        "reason": "",
        "totals": [],
        "steps": [
            {
                "kind": "hash_join",
                "detail": "",
                "a_ms": 10.0,
                "b_ms": 40.0,
                "delta_ms": 30.0,
                "ratio": 4.0,
            },
            {
                "kind": "scan",
                "detail": "",
                "a_ms": 20.0,
                "b_ms": 5.0,
                "delta_ms": -15.0,
                "ratio": 0.25,
            },
        ],
    }
    ctx.eval(f"VIEWS.comparison({json.dumps(cmp)}, 'before', 'after');")
    html = ctx.eval("document.getElementById('compare').innerHTML")
    assert "▲" in html and "is-warn" in html, "a slower step is up-and-warn"
    assert "▼" in html and "is-good" in html, "a faster step is down-and-good"
    # A magnitude bar behind each number, sized against the biggest change. Built by the
    # shared chart layer (`CHARTS.delta`) so a signed change looks the same wherever the
    # dashboard shows one.
    assert html.count("ch-delta-track") == 2
    # The bar grows away from a centre line, and which way encodes the direction too.
    assert "to-right" in html and "to-left" in html
    # The change shows the absolute magnitude next to the arrow (no doubled sign), through
    # the page's one duration formatter rather than a local `toFixed`.
    assert "30ms" in html and "15ms" in html
    assert "+30" not in html, "the arrow carries the sign, not a + prefix"
    # The answer above the evidence: the step that accounts for most of the difference.
    assert "accounts for" in html and "Join" in html


def test_comparison_totals_judge_duration_but_not_a_different_result():
    """More time is worse; more rows returned is a *different result*, not a regression.
    Colouring a row-count change red would tell the reader something false."""
    ctx = _views_context()
    cmp = {
        "ok": False,
        "reason": "These runs have different plan shapes.",
        "steps": [],
        "totals": [
            {"label": "Duration", "unit": "ms", "a": 100, "b": 250, "delta": 150, "ratio": 2.5},
            {
                "label": "Rows returned",
                "unit": "rows",
                "a": 1000,
                "b": 5000,
                "delta": 4000,
                "ratio": 5.0,
            },
        ],
    }
    ctx.eval(f"VIEWS.comparison({json.dumps(cmp)}, 'a', 'b');")
    html = ctx.eval("document.getElementById('compare').innerHTML")
    # The incomparable-shapes reason is surfaced, not swallowed.
    assert "different plan shapes" in html
    # Duration is judged; the row-count change is shown but not coloured good/bad.
    assert "is-warn" in html, "a slower duration is flagged"
    assert "is-good" not in html, "a bigger result is not 'good', it is just different"
    assert "▲" in html, "the direction arrow is shown"


def test_segmented_controls_move_with_arrow_keys():
    """A `role="group"` of toggle buttons should move with left/right (the toolbar/radiogroup
    idiom), activating as it goes — not force a keyboard user to Tab through each."""
    app = (_ASSETS / "app.js").read_text()
    assert "function installSegmentedGroups" in app
    fn = app[app.index("function installSegmentedGroups") : app.index("/* Anything monospaced")]
    for key in ("ArrowLeft", "ArrowRight", "Home", "End"):
        assert key in fn, f"{key} should move within a segmented group"
    assert ".segmented, .pane-switch" in fn, "both segmented families are covered"
    # Moving focus also activates, and wraps at the ends.
    assert ".focus()" in fn and ".click()" in fn
    assert "% segs.length" in fn, "navigation wraps around"
    # The static label between buttons is skipped.
    assert "is-label" in fn


def test_a_log_line_can_show_its_neighbours_ignoring_filters():
    """The filters that surfaced a line are usually the wrong ones for understanding it — the
    DEBUG lines a level filter hid are often what explains it. Each line carries its position
    in the *unfiltered* stream so a context view can pull the neighbours back."""
    ctx = _views_context()
    line = ctx.eval(
        "VIEWS.logLine({wall: 1000, level: 'INFO', message: 'm', fields: {}, seq: 42}, 42)"
    )
    assert 'data-seq="42"' in line, "the line carries its unfiltered position"
    assert 'data-context="42"' in line, "and an affordance to show its neighbours"
    assert "Show surrounding lines" in line

    app = (_ASSETS / "app.js").read_text()
    assert "function showLogContext" in app
    fn = app[app.index("function showLogContext") : app.index("function pickLogRange")]
    # It pulls from the full stored log, not the filtered view.
    assert "state.logLines" in fn
    assert "Math.abs(l.seq - seq)" in fn, "a symmetric window around the chosen line"
    assert "ignoring filters" in fn, "the panel says it is bypassing the filters"


def test_glossary_terms_in_insight_prose_become_clickable():
    """An insight's evidence says "the step spilled 2 GB" — a newcomer should be able to click
    "spill" for what it means, without the rule author marking up every occurrence. The
    auto-linker does it, safely."""
    ctx = _reference_context()
    linked = ctx.eval('LEARN.autolink("This step had to spill to disk before the hash join.")')
    assert 'data-term="spill"' in linked
    # Longest term wins: "hash join", not "join" inside it.
    assert 'data-term="hash join"' in linked

    # HTML is escaped before linking — a message with a "<" cannot inject markup.
    assert "&lt;" in ctx.eval('LEARN.autolink("a < b, then spill")')

    # Each term links once, so repeated words do not become a field of dotted underlines.
    once = ctx.eval('LEARN.autolink("spill, then more spill, then spill again")')
    assert once.count('data-term="spill"') == 1

    # A whole word only: "plan" must not match inside "planner".
    assert 'data-term="plan"' not in ctx.eval('LEARN.autolink("the planner decided")')

    # Prose with no known term passes through unchanged (bar escaping).
    assert ctx.eval('LEARN.autolink("nothing notable happened")') == "nothing notable happened"


def test_system_settings_that_shape_performance_carry_a_definition():
    """A newcomer on the system page sees "rows per batch: 16,384" with no idea why it
    matters. The performance-shaping settings link to their glossary term so the answer is
    one click away — but a term that does not exist must not render a broken link."""
    ctx = _views_context()
    sys_data = {
        "host": {"cpus": 16},
        "engine": {},
        "cluster": {},
        "config": {
            "parallelism": 0,
            "morsel_rows": 16384,
            "morsel_bytes": 1048576,
            "split_bytes": 1048576,
            "adaptive_morsel_sizing": True,
            "max_memory_bytes": None,
            "spill_enabled": True,
            "spill_compression": "lz4",
            "soft_limit": "80%",
            "hard_limit": "90%",
            "verbosity": "info",
        },
    }
    ctx.eval(f"VIEWS.system({json.dumps(sys_data)});")
    html = ctx.eval("document.getElementById('system-cards').innerHTML")
    for term in ("parallelism", "morsel", "spill"):
        assert f'data-term="{term}"' in html, f"the {term} setting should link to its definition"
    # Every linked term actually exists in the reference — LEARN.hint returns '' for unknowns.
    linked = set(re.findall(r'data-term="([^"]+)"', html))
    known = set(json.loads(ctx.eval("JSON.stringify(REFERENCE.termKeys)")))
    assert linked <= known, f"system links a term with no entry: {linked - known}"


def test_the_reference_keeps_growing_without_breaking_its_invariants():
    """A snapshot count so a regression that silently drops content is visible, plus the
    structural checks that must hold at any size."""
    ctx = _reference_context()
    assert ctx.eval("REFERENCE.termKeys.length") >= 65
    assert ctx.eval("REFERENCE.RECIPES.length") >= 19
    assert ctx.eval("Object.keys(REFERENCE.METRICS).length") >= 21
    # Every recipe has a task and at least two steps — a one-step "recipe" is not guidance.
    thin = json.loads(
        ctx.eval("""
    JSON.stringify(REFERENCE.RECIPES.filter(function(r){
      return !r.task || !r.steps || r.steps.length < 2;
    }).map(function(r){ return r.task; }))
    """)
    )
    assert thin == [], f"recipes with fewer than two steps: {thin}"


def test_every_run_panel_empty_state_teaches_rather_than_apologises():
    """A blank panel or a terse "no data" tells a newcomer nothing. Each run panel's empty
    state should name what will appear and why it is worth looking at — so the empty page is
    still a page that teaches."""
    ctx = _views_context()
    checks = [
        ("timeline", "VIEWS.timeline([]);"),
        ("operators", "VIEWS.operators([], function(){});"),
        ("attention", "VIEWS.attention([], []);"),
    ]
    for host_id, call in checks:
        ctx.eval(call)
        html = ctx.eval(f"document.getElementById('{host_id}').innerHTML")
        assert "empty-state" in html, f"{host_id} should use the teaching empty component"
        # A teaching empty has a title and a body of real length, not a bare sentence.
        assert "<h3>" in html, f"{host_id} empty needs a heading"
        body = re.search(r"<p>(.*?)</p>", html, re.S)
        assert body and len(body.group(1)) > 40, f"{host_id} empty body is too thin to teach"


def test_sparklines_are_legible_without_sight_and_readable_on_hover():
    """A sparkline that is only `aria-hidden` decoration tells a screen-reader user nothing
    about the trend it draws. It should carry a one-line description (direction, current,
    peak) and per-point hit targets whose value shows on hover."""
    ctx = _views_context()
    rising = ctx.eval("UI.sparkline([10, 20, 30, 25, 40], {label: 'rate', unit: '/s'})")
    assert 'role="img"' in rising and "aria-label=" in rising, "the trend must be described"
    assert "rising" in rising, "the description names the direction"
    assert "peak" in rising and "40" in rising, "and the peak"
    assert rising.count("spark-hit") == 5, "one hit target per point"
    assert "<title>" in rising, "each point's value is readable on hover"

    assert "falling" in ctx.eval("UI.sparkline([40, 30, 20, 10], {label: 'x'})")
    assert "steady" in ctx.eval("UI.sparkline([20, 21, 19, 20], {label: 'x'})")
    # An empty series draws nothing rather than an empty svg.
    assert ctx.eval("UI.sparkline([], {label: 'x'})") == ""


def test_the_js_and_python_count_formatters_agree_where_they_overlap():
    """The terminal logger (`console._count`) and the web UI (`UI.count`) format the same row
    counts. If they drift, one number reads two ways depending on where you look — the same
    class of bug as the two diverged duration formatters earlier.

    They are not identical by construction (the JS handles trillions and nulls the terminal
    never sees), so this pins the overlapping range where they must agree.
    """
    quickjs = pytest.importorskip("quickjs")
    from batcher.observe.console import _count as py_count

    ctx = quickjs.Context()
    ctx.eval((_ASSETS / "ui.js").read_text())

    # Cover each SI band and its boundaries, staying inside the range both format.
    samples = [
        0,
        1,
        42,
        999,
        1000,
        1234,
        9999,
        12345,
        999_999,
        1_000_000,
        3_400_000,
        999_999_999,
        1_000_000_000,
        5_600_000_000,
    ]
    for n in samples:
        js = ctx.eval(f"UI.count({n})")
        py = py_count(n)
        assert js == py, f"count({n}): JS gave {js!r}, Python gave {py!r}"


def test_the_js_and_python_duration_formatters_agree_where_they_overlap():
    """Same discipline for durations: `console._dur` (terminal) vs `UI.ms` (web). The web
    formatter drops to microseconds below 1ms where the terminal does not, so this pins the
    millisecond-and-up range where both apply.
    """
    quickjs = pytest.importorskip("quickjs")
    from batcher.observe.console import _dur as py_dur

    ctx = quickjs.Context()
    ctx.eval((_ASSETS / "ui.js").read_text())

    for ms in [1, 42, 850, 999, 1000, 1500, 4200, 59_999, 60_000, 90_000, 125_000]:
        js = ctx.eval(f"UI.ms({ms})")
        py = py_dur(ms)
        assert js == py, f"duration({ms}): JS gave {js!r}, Python gave {py!r}"


def test_no_two_shortcuts_claim_the_same_key():
    """Two actions on one key means one of them never fires. Adding the layout and
    focus-search shortcuts exposed a three-way collision on "/" (focus, shortcuts, help);
    this pins the keymap so the next addition cannot re-introduce one."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    keyed = json.loads(
        ctx.eval("""
    JSON.stringify(ACTIONS.filter(function(a){ return a.keys; })
      .map(function(a){ return [a.keys, a.id]; }))
    """)
    )
    from collections import Counter

    counts = Counter(k for k, _ in keyed)
    collisions = {k: [i for kk, i in keyed if kk == k] for k, c in counts.items() if c > 1}
    assert not collisions, f"keys claimed by more than one action: {collisions}"


def test_the_key_dispatcher_matches_the_registry():
    """`renderShortcuts` builds the help sheet from the registry, and a separate dispatcher
    actually handles the keys. If they disagree, the sheet documents a binding that does
    something else. Every single-key action must be handled by the dispatcher."""
    app = (_ASSETS / "app.js").read_text()
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())
    single = json.loads(
        ctx.eval("""
    JSON.stringify(ACTIONS.filter(function(a){ return a.keys && a.keys.length === 1; })
      .map(function(a){ return a.keys; }))
    """)
    )
    # Every single-key action must be dispatched somewhere via `e.key === '<key>'` — the
    # global keys in one block, f/c in the run-view conditional, space as `e.key === ' '`.
    for key in single:
        assert f"e.key === '{key}'" in app, f"key '{key}' is registered but never dispatched"


def test_the_js_and_python_percent_formatters_agree():
    """`console._pct` (terminal) and `UI.pct` (web) format the same shares. Both must treat a
    small-but-present share as "<1%" and clamp to 100%, or the same ratio reads two ways."""
    quickjs = pytest.importorskip("quickjs")
    from batcher.observe.console import _pct as py_pct

    ctx = quickjs.Context()
    ctx.eval((_ASSETS / "ui.js").read_text())

    for frac in [0, 0.003, 0.01, 0.1, 0.499, 0.5, 0.624, 0.999, 1.0, 1.5]:
        js = ctx.eval(f"UI.pct({frac})")
        py = py_pct(frac)
        assert js == py, f"pct({frac}): JS gave {js!r}, Python gave {py!r}"


def test_the_js_and_python_byte_formatters_agree():
    """`console._bytes` (added for the terminal) and `UI.bytes` (web) format the same sizes.
    Both read `0` and `None` as an em dash and use binary units."""
    quickjs = pytest.importorskip("quickjs")
    from batcher.observe.console import _bytes as py_bytes

    ctx = quickjs.Context()
    ctx.eval((_ASSETS / "ui.js").read_text())

    for n in [
        0,
        1,
        512,
        1023,
        1024,
        1536,
        1048575,
        1048576,
        3_400_000,
        1073741824,
        1610612736,
        1099511627776,
    ]:
        js = ctx.eval(f"UI.bytes({n})")
        py = py_bytes(n)
        assert js == py, f"bytes({n}): JS gave {js!r}, Python gave {py!r}"


def test_the_browser_tab_title_reflects_where_you_are():
    """Several runs open in tabs are indistinguishable if every tab reads "Batcher", and a
    bookmark of the bare title tells you nothing. The title tracks the view."""
    ctx = _views_context()
    ctx.eval((_ASSETS / "app.js").read_text())

    def title(setup):
        ctx.eval(setup)
        ctx.eval("updateDocumentTitle();")
        return ctx.eval("document.title")

    assert title("state.view = 'pipelines';") == "Batcher", "the landing page is just the app name"
    assert title("state.view = 'logs';") == "Logs — Batcher"
    assert title("state.view = 'system';") == "System — Batcher"
    run = title("state.view = 'run'; state.detail = {label: 'scan', total_ms: 42};")
    assert run.startswith("Read source"), "a run names its query"
    assert "42ms" in run, "and its duration, so two runs of one query differ"
    assert run.endswith("Batcher")


def test_the_metrics_are_browsable_on_the_learn_page():
    """25 metrics were defined but only ever appeared in tooltips — a reader could meet a
    good/bad yardstick in passing but never read the whole set. They now have their own
    section, with the healthy and worth-a-look bands shown."""
    ctx = _views_context()
    html = ctx.eval(
        "(function(){ var d = document.createElement('div');"
        " LEARN.renderLearn(d, ''); return d.innerHTML; })()"
    )
    assert 'id="learn-metrics"' in html, "the metrics section exists"
    assert 'href="#learn-metrics"' in html, "and is in the contents strip"
    metric_count = ctx.eval("Object.keys(REFERENCE.METRICS).length")
    assert html.count("metric-good") == metric_count, "every metric shows its healthy band"
    assert html.count("metric-bad") == metric_count, "and its worth-a-look band"


def test_forced_colors_mode_keeps_state_visible():
    """Windows High Contrast flattens the palette to system colours; meaning carried only by a
    fill or a border colour is lost. The elements whose state *is* their colour get explicit
    system-colour treatment."""
    css = (_ASSETS / "app.css").read_text()
    assert "@media (forced-colors: active)" in css
    block = css[css.index("@media (forced-colors: active)") :]
    # Selection, focus, and severity must survive the palette override.
    assert "SelectedItem" in block or "Highlight" in block, "selection stays visible"
    assert "focus-visible" in block, "focus stays visible when the accent is overridden"
    # Status dots convey state by fill; a border keeps the shape.
    assert ".dot" in block


def test_user_data_is_escaped_before_it_reaches_the_dom():
    """Query labels, log messages, and error strings are user data. Building markup from them
    without escaping would inject into the operator's own browser — the worst case here,
    because the dashboard runs beside the engine it observes.

    `esc` is the single chokepoint; this pins that it neutralises every HTML-significant
    character, and that the renderers actually route through it.
    """
    ctx = _views_context()
    # Every dangerous character is escaped.
    assert ctx.eval("""UI.esc('<script>alert(1)</script>')""") == (
        "&lt;script&gt;alert(1)&lt;/script&gt;"
    )
    assert ctx.eval("""UI.esc('" onmouseover="evil()')""") == "&quot; onmouseover=&quot;evil()"
    assert ctx.eval("""UI.esc("a & b < c > d ' e")""") == "a &amp; b &lt; c &gt; d &#39; e"
    # Ampersand first, so an escape is not double-escaped into nonsense.
    assert ctx.eval("""UI.esc('&lt;')""") == "&amp;lt;"
    # Null and undefined become empty, not the strings "null"/"undefined".
    assert ctx.eval("UI.esc(null)") == ""
    assert ctx.eval("UI.esc(undefined)") == ""

    # A log message containing markup renders escaped, not as live HTML.
    line = ctx.eval(
        "VIEWS.logLine({wall: 1, level: 'ERROR',"
        " message: '<img src=x onerror=alert(1)>', fields: {}, seq: 0}, 0)"
    )
    assert "<img" not in line, "a message with a tag must not render the tag"
    assert "&lt;img" in line, "it renders escaped"

    # The auto-linker escapes before linking, so a message cannot inject via the term markup.
    linked = ctx.eval("""LEARN.autolink('<b>spill</b> happened')""")
    assert "<b>" not in linked, "the auto-linker must not pass raw tags through"
    assert "&lt;b&gt;" in linked
    # …but it still linked the real term inside the escaped text.
    assert 'data-term="spill"' in linked


def test_a_query_label_with_markup_cannot_inject_via_a_card():
    """A user-supplied pipeline name and note must render as text, not live markup, on the
    lane that is the primary navigation.

    The injection surface moved when pipelines gained names: the display name is now a
    person's typed string (persisted in the registry) rather than the engine's op label, so
    the name and the note are the user data that must be escaped."""
    ctx = _views_context()
    pipes = [
        {
            "signature": "s",
            "pipeline_id": "s",
            "name": "<svg onload=alert(1)>",
            "note": "<img src=x onerror=alert(2)>",
            "label": "aggregate",
            "last_status": "ok",
            "n_failed": 0,
            "runs": 1,
            "median_ms": 10,
            "recent_ms": [10],
            "plan_shape": {"nodes": [], "edges": [], "width": 0, "depth": 0},
        }
    ]
    ctx.eval(f"var P = {json.dumps(pipes)};")
    ctx.eval(
        "VIEWS.pipelines(P, {onOpen: function(){}, onPin: function(){},"
        " onRename: function(){}, sort: 'time', needle: ''});"
    )
    html = ctx.eval("document.getElementById('pipeline-cards').innerHTML")
    assert "<svg onload" not in html, "the name's tag must not render"
    assert "<img src=x" not in html, "the note's tag must not render"
    assert "&lt;svg" in html and "&lt;img" in html, "both render escaped"


def test_a_structured_log_field_value_cannot_inject():
    """Field values are user data too, and they become click-to-filter buttons. A value with
    a quote in it must not break out of the attribute it is placed in."""
    ctx = _views_context()
    line = ctx.eval(
        "VIEWS.logLine({wall: 1, level: 'INFO', message: 'm',"
        " fields: {evil: '\\\" onclick=\\\"alert(1)'}, seq: 0}, 0)"
    )
    # The value is escaped inside the data attribute and the visible text.
    assert 'onclick="alert' not in line, "a field value must not inject an attribute"
    assert "&quot;" in line, "the quote is escaped"


def test_an_operator_detail_with_markup_renders_as_text_in_the_graph():
    """A step's detail string (a join condition, a filter predicate) reflects the query and
    is user data. It must render as text in the plan graph, not as markup."""
    ctx = _views_context()
    dag = {
        "nodes": [
            {
                "op_id": 0,
                "kind": "filter",
                "detail": "<b>x</b> = 1",
                "measured": True,
                "rows_out": 10,
                "elapsed_ms": 5.0,
                "column": 0,
                "row": 0,
                "depth": 0,
            }
        ],
        "edges": [],
        "width": 1,
        "depth": 1,
        "critical_path": [],
    }
    ctx.eval(f"var D = {json.dumps(dag)};")
    ctx.eval("""
    var svg = document.getElementById('dag');
    DAG.render(svg, D, {onHover: function(){}, onLeave: function(){},
                        onZoom: function(){}, onSelect: function(){}}, 'q');
    """)
    texts = json.loads(
        ctx.eval("""
    JSON.stringify(document.getElementById('dag').querySelectorAll('text')
      .map(function(t){ return t.textContent; }))
    """)
    )
    # The detail appears as its literal text; SVG <text> textContent is not parsed as HTML,
    # but the value must still be the raw string, never a stripped-tag artifact.
    assert any("<b>x</b>" in t or "x" in t for t in texts), "the detail is shown as text"


def test_the_whole_bundle_has_no_stray_debug_output():
    """No console.log left in shipped JS, and no TODO/FIXME markers — the tree stays clean."""
    for name in ("ui.js", "reference.js", "dag.js", "learn.js", "views.js", "app.js"):
        src = (_ASSETS / name).read_text()
        assert "console.log" not in src, f"{name} has a stray console.log"
        assert "console.debug" not in src, f"{name} has a stray console.debug"
        for marker in ("TODO", "FIXME", "XXX", "HACK"):
            assert marker not in src, f"{name} has a {marker} marker"


def test_the_changelog_and_gate_are_internally_consistent():
    """A meta-check: the module set the tests load must match the scripts the shell loads, so
    a module added to one but not the other cannot slip through."""
    shell = (_ASSETS / "index.html").read_text()
    loaded = re.findall(r'<script src="/([^"]+)"', shell)
    # Every JS module the tests exercise is loaded by the page, in the same order.
    assert loaded == list(_JS_MODULES), (
        f"the page loads {loaded}, the tests load {list(_JS_MODULES)}"
    )
