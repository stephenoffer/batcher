"""The dashboard's plan renderings: the EXPLAIN tree, the optimizer diff, and live telemetry.

These cover the three things the UI now shows that nothing measured before: the plan as
text, what the optimizer changed between the plan you wrote and the plan that ran, and the
distributed/accelerator events the activity store used to drop on the floor.

Every assertion here is about *not inventing*: an unmeasured operator prints no timing, a
missing logical plan reports "not recorded" rather than "no changes", and a single-node run
carries no accelerator panel.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request

import pytest

from batcher._internal import events
from batcher.observe.dag import build_dag, explain_rows, explain_text, plan_diff
from batcher.observe.server import UIServer
from batcher.observe.store import ActivityStore

# --- fixtures ---------------------------------------------------------------


def _scan(source_id: int = 0) -> dict:
    return {"op": "scan", "source_id": source_id}


def _predicate() -> dict:
    return {
        "e": "binary",
        "op": "gt",
        "left": {"e": "col", "name": "amount"},
        "right": {"e": "lit", "value": 100},
    }


@pytest.fixture
def logical() -> dict:
    """A filter sitting above a join — the shape every pushdown rule exists to fix."""
    return {
        "op": "filter",
        "predicate": _predicate(),
        "input": {
            "op": "hash_join",
            "join_type": "inner",
            "left_keys": ["id"],
            "left": _scan(0),
            "right": _scan(1),
        },
    }


@pytest.fixture
def optimized() -> dict:
    """The same query with the filter pushed onto the left scan."""
    return {
        "op": "hash_join",
        "join_type": "inner",
        "left_keys": ["id"],
        "left": {"op": "filter", "predicate": _predicate(), "input": _scan(0)},
        "right": _scan(1),
    }


# --- EXPLAIN ----------------------------------------------------------------


def test_explain_renders_the_tree_in_plan_order(optimized):
    rows = explain_rows(optimized)
    assert [r["kind"] for r in rows] == ["hash_join", "filter", "scan", "scan"]
    # op_id is the pre-order index, the same identity the graph and the metrics use.
    assert [r["op_id"] for r in rows] == [0, 1, 2, 3]


def test_explain_op_ids_match_the_graphs(optimized):
    graph = build_dag(optimized, [])
    by_id = {n["op_id"]: (n["kind"], n["detail"]) for n in graph["nodes"]}
    for row in explain_rows(optimized):
        assert by_id[row["op_id"]] == (row["kind"], row["detail"])


def test_explain_of_an_unmeasured_plan_prints_no_timing(optimized):
    text = explain_text(optimized)
    assert "hash_join" in text and "amount > 100" in text
    assert "ms" not in text


def test_explain_prints_a_measurement_only_where_one_exists(optimized):
    ops = [{"op_id": 0, "measured": True, "elapsed_ms": 4.25, "rows_out": 900}]
    text = explain_text(optimized, ops)
    lines = text.splitlines()
    assert "4.25 ms" in lines[1] and "900 rows" in lines[1]
    # The unmeasured operators below it get no invented zero.
    assert sum("ms" in line for line in lines) == 1


def test_explain_closes_the_spine_under_the_last_child(optimized):
    text = explain_text(optimized)
    # The right-hand scan is the last child of the join, so nothing continues below it.
    assert text.splitlines()[-1].lstrip().startswith("└─")


def test_explain_has_an_ascii_form_for_a_terminal_that_cannot_encode_box_drawing(optimized):
    text = explain_text(optimized, ascii_only=True)
    assert text.isascii()


def test_explain_of_no_plan_is_empty_not_an_error():
    assert explain_rows(None) == []
    assert explain_text(None) == ""


# --- optimizer diff ---------------------------------------------------------


def test_plan_diff_reports_a_pushdown_as_one_primary_change(logical, optimized):
    diff = plan_diff(logical, optimized)
    primary = [c for c in diff["changes"] if c["primary"]]
    assert len(primary) == 1
    assert primary[0]["kind"] == "filter"
    assert primary[0]["change"] == "moved"
    assert "before hash_join" in primary[0]["note"]


def test_plan_diff_keeps_the_knock_on_moves_but_does_not_promote_them(logical, optimized):
    diff = plan_diff(logical, optimized)
    # Every operator the filter passed did move; they are reported, just not as the finding.
    assert len(diff["changes"]) > 1
    assert all(not c["primary"] for c in diff["changes"][1:])


def test_plan_diff_notices_a_removed_operator(logical):
    optimized = logical["input"]  # the filter was proved unnecessary
    diff = plan_diff(logical, optimized)
    removed = [c for c in diff["changes"] if c["change"] == "removed"]
    assert [c["kind"] for c in removed] == ["filter"]
    assert diff["before_ops"] == 4 and diff["after_ops"] == 3


def test_plan_diff_notices_an_added_operator(logical):
    optimized = {"op": "project", "items": ["a"], "input": logical}
    diff = plan_diff(logical, optimized)
    assert [c["kind"] for c in diff["changes"] if c["change"] == "added"] == ["project"]


def test_plan_diff_of_an_untouched_plan_says_so(optimized):
    diff = plan_diff(optimized, optimized)
    assert diff["identical"] is True
    assert diff["changes"] == []
    assert "as written" in diff["summary"]


def test_plan_diff_distinguishes_not_recorded_from_no_changes(optimized):
    missing = plan_diff(None, optimized)
    assert missing["available"] is False
    # The dangerous conflation: "we have nothing to compare" must never read as "nothing
    # changed", which would credit the optimizer with leaving a plan alone it never saw.
    assert missing["identical"] is False


def test_plan_diff_counts_every_operator_kind_on_both_sides(logical, optimized):
    counts = {c["kind"]: c for c in plan_diff(logical, optimized)["counts"]}
    assert counts["scan"]["before"] == counts["scan"]["after"] == 2
    assert counts["filter"]["delta"] == 0


# --- pipeline stages --------------------------------------------------------


def test_stages_group_the_operators_that_stream_together(optimized):
    graph = build_dag(optimized, [])
    by_kind = {}
    for node in graph["nodes"]:
        by_kind.setdefault(node["kind"], []).append(node["stage"])
    # The scans and the filter stream into the join build; the join is a breaker above them.
    assert by_kind["scan"] == [0, 0]
    assert by_kind["filter"] == [0]
    assert by_kind["hash_join"] == [1]
    assert graph["stages"] == 2


# --- live telemetry ---------------------------------------------------------


def _emit(store: ActivityStore, kind: str, query_id: str, name: str = "", **fields) -> None:
    store.handle(events.Event(kind, time.monotonic(), time.time(), query_id, name, fields))


@pytest.fixture
def store_with_a_gpu_job() -> ActivityStore:
    store = ActivityStore()
    _emit(store, events.QUERY_START, "q1", "embed", label="embed", signature="sig")
    _emit(store, events.PARTITION, "q1", "map", op_id=1, total=8, rows=1000)
    _emit(store, events.PARTITION, "q1", "map", op_id=1, total=8, rows=1000)
    _emit(
        store,
        events.GPU,
        "q1",
        device="gpu0",
        util_pct=91.0,
        mem_used_bytes=12 << 30,
        mem_total_bytes=16 << 30,
    )
    _emit(store, events.INFER, "q1", "map", rows=1000, latency_ms=40.0, blocked_ms=2.0)
    _emit(store, events.SKIPPED, "q1", count=3, reason="decode error")
    _emit(store, events.POOL, "q1", "map", size=4, pending=2)
    _emit(
        store,
        events.QUERY_END,
        "q1",
        ok=True,
        total_ms=500.0,
        rows=2000,
        profile={"ops": [], "optimized_ir": _scan(), "logical_ir": _scan()},
    )
    return store


def test_the_store_no_longer_drops_distributed_and_accelerator_events(store_with_a_gpu_job):
    live = store_with_a_gpu_job.query("q1")["live"]
    assert live["partitions"]["done"] == 2
    assert live["partitions"]["total"] == 8
    assert live["gpu"]["gpu0"]["util_pct"] == 91.0
    assert live["inference"]["batches"] == 1
    assert live["skipped"]["total"] == 3
    assert live["pool"] == {"size": 4, "pending": 2}


def test_dropped_rows_are_surfaced_rather_than_lost(store_with_a_gpu_job):
    live = store_with_a_gpu_job.live("q1")
    assert live["skipped"]["by_reason"] == {"decode error": 3}


def test_an_ordinary_single_node_run_carries_no_accelerator_panel():
    store = ActivityStore()
    _emit(store, events.QUERY_START, "q2", "plain", label="plain")
    _emit(store, events.QUERY_END, "q2", ok=True, total_ms=1.0, rows=1)
    assert store.query("q2")["live"] is None


def test_live_defaults_to_the_most_recently_active_job(store_with_a_gpu_job):
    assert store_with_a_gpu_job.live()["query_id"] == "q1"


def test_live_of_a_process_that_never_distributed_is_none():
    assert ActivityStore().live() is None


# --- HTTP -------------------------------------------------------------------


@pytest.fixture
def ui():
    server = UIServer(ActivityStore(), port=0)
    server.start()
    yield server
    server.stop()


def _fetch(server: UIServer, path: str, **kwargs) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(server.url + path, **kwargs)
    with urllib.request.urlopen(request) as response:
        return response.status, dict(response.headers), response.read()


def test_the_api_publishes_its_own_route_index(ui):
    _status, _headers, body = _fetch(ui, "/api")
    paths = {r["path"] for r in json.loads(body)["routes"]}
    assert {"/api/summary", "/api/live", "/api/metrics"} <= paths
    # Every advertised route must actually answer, or the index is a list of promises.
    for path in paths:
        assert _fetch(ui, path)[0] == 200


def test_metrics_are_served_in_both_json_and_prometheus_form(ui):
    _status, headers, body = _fetch(ui, "/metrics")
    assert headers["Content-Type"].startswith("text/plain")
    assert b"batcher_queries_total" in body
    assert "total" in json.loads(_fetch(ui, "/api/metrics")[2])["queries"]


def test_assets_revalidate_instead_of_re_downloading(ui):
    _status, headers, body = _fetch(ui, "/app.js")
    etag = headers["ETag"]
    with pytest.raises(urllib.error.HTTPError) as caught:
        _fetch(ui, "/app.js", headers={"If-None-Match": etag})
    assert caught.value.code == 304
    assert len(body) > 0


def test_a_json_payload_is_compressed_when_the_browser_accepts_it(ui):
    _status, headers, body = _fetch(ui, "/app.css", headers={"Accept-Encoding": "gzip"})
    assert headers["Content-Encoding"] == "gzip"
    assert b"--accent" in gzip.decompress(body)


def test_head_answers_without_a_body(ui):
    status, headers, body = _fetch(ui, "/api/summary", method="HEAD")
    assert status == 200
    assert body == b""
    assert int(headers["Content-Length"]) > 0


def test_the_api_refuses_every_mutating_verb_except_the_one_write(ui):
    # A read route rejects every mutation. The dashboard's one write is `/api/pipeline/meta`,
    # tested on its own below — everything else stays read-only.
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _fetch(ui, "/api/summary", method=method, data=b"")
        assert caught.value.code == 405
        assert caught.value.headers["Allow"] == "GET, HEAD, POST"
    # Even the write route accepts only POST, not the other mutating verbs.
    for method in ("PUT", "DELETE", "PATCH"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _fetch(ui, "/api/pipeline/meta", method=method, data=b"{}")
        assert caught.value.code == 405


def test_every_response_forbids_content_sniffing_and_framing(ui):
    _status, headers, _body = _fetch(ui, "/")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    # The dashboard renders query labels and log lines; a `self`-only script policy is what
    # keeps a stray escaping bug from becoming code execution in the operator's browser.
    assert "script-src 'self'" in headers["Content-Security-Policy"]


# --- the renderers, actually executed ---------------------------------------
# The static checks prove the modules parse and that nothing calls a function that does not
# exist. They cannot prove a renderer produces anything, and a panel that silently renders
# an empty string looks exactly like a panel with nothing to show. These drive the real
# functions against the small DOM in `data/minidom.js` and assert what came out.

_MINIDOM = __import__("pathlib").Path(__file__).parent / "data" / "minidom.js"
_ASSETS = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "python"
    / "batcher"
    / "observe"
    / "assets"
)


@pytest.fixture
def page():
    """A QuickJS context with the DOM double and every renderer loaded."""
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


@pytest.fixture
def shaped_page(page):
    """A page context with a real plan shape exposed as the JS global `PLAN_SHAPE_JS`.

    Built by the actual `plan_shape` backend so the thumbnail is tested against the exact
    shape the server produces, with the critical path marked so the tint is exercised. Not
    autouse: the pure-Python tests in this file must not require the JS engine.
    """
    from batcher.observe.dag import plan_shape

    shape = plan_shape(_agg_ir())
    for node in shape["nodes"]:
        node["on_critical_path"] = True
    page.eval(f"var PLAN_SHAPE_JS = {json.dumps(shape)};")
    return page


def _render(ctx, call: str, host: str) -> str:
    ctx.eval(call)
    return ctx.eval(f"document.getElementById({host!r}).innerHTML")


_EXPLAIN_ROWS = [
    {
        "op_id": 0,
        "depth": 0,
        "kind": "hash_join",
        "detail": "inner on id",
        "last": True,
        "ancestors": [],
        "measured": True,
        "elapsed_ms": 40.0,
        "rows_out": 900,
        "est_rows": 1000.0,
        "spilled": False,
        "algorithm": "hash",
        "provenance": "sketch",
    },
    {
        "op_id": 1,
        "depth": 1,
        "kind": "filter",
        "detail": "amount > 100",
        "last": False,
        "ancestors": [False],
        "measured": True,
        "elapsed_ms": 10.0,
        "rows_out": 900,
        "est_rows": None,
        "spilled": False,
        "algorithm": "",
        "provenance": "",
    },
    {
        "op_id": 2,
        "depth": 2,
        "kind": "scan",
        "detail": "source 0",
        "last": True,
        "ancestors": [False, True],
        "measured": False,
        "elapsed_ms": None,
        "rows_out": None,
        "est_rows": None,
        "spilled": False,
        "algorithm": "",
        "provenance": "",
    },
]


def test_the_explain_tree_renders_every_step_with_its_share(page):
    html = _render(page, f"PLAN.explain('explain', {json.dumps(_EXPLAIN_ROWS)});", "explain")
    assert html.count('<div class="pv-row') == 3
    # The friendly names the rest of the dashboard uses, not the raw IR tags — the graph
    # and this tree must call the same operator the same thing.
    for label in ("Join", "Filter rows", "Read source"):
        assert label in html
    # 40ms of 50ms measured is 80% of the share bar.
    assert "80.0%" in html
    # The unmeasured step says so rather than showing a zero.
    assert "not measured" in html


def test_a_collapsed_subtree_hides_only_what_is_nested_under_it(page):
    html = _render(
        page,
        f"PLAN.explain('explain', {json.dumps(_EXPLAIN_ROWS)}, {{ collapsed: new Set([1]) }});",
        "explain",
    )
    assert html.count('<div class="pv-row') == 2, (
        "the scan under the collapsed filter should be hidden"
    )
    assert "is-collapsed" in html


def test_the_explain_search_marks_a_hit_and_keeps_its_context(page):
    html = _render(
        page,
        f"PLAN.explain('explain', {json.dumps(_EXPLAIN_ROWS)}, {{ needle: 'amount' }});",
        "explain",
    )
    # Marked, not filtered to: the surrounding plan is what makes a hit mean anything.
    assert html.count('<div class="pv-row') == 3
    assert html.count("is-hit") == 1


def test_the_explain_text_form_matches_the_rendered_tree(page):
    text = page.eval(f"PLAN.explainText({json.dumps(_EXPLAIN_ROWS)})")
    assert text.splitlines()[0].startswith("hash_join")
    assert "└─ scan" in text
    assert "(40ms, 900 rows)" in text


def test_the_explain_pane_says_so_when_a_run_has_no_plan(page):
    html = _render(page, "PLAN.explain('explain', []);", "explain")
    assert "No plan recorded" in html


_DIFF = {
    "available": True,
    "identical": False,
    "summary": "The optimizer reordered 1 step.",
    "before_ops": 4,
    "after_ops": 4,
    "counts": [{"kind": "filter", "before": 1, "after": 1, "delta": 0}],
    "changes": [
        {
            "change": "moved",
            "kind": "filter",
            "detail": "amount > 100",
            "op_id": 1,
            "note": "now runs before hash_join",
            "primary": True,
            "before_path": [],
            "after_path": ["hash_join"],
        },
        {
            "change": "moved",
            "kind": "scan",
            "detail": "source 1",
            "op_id": 3,
            "note": "no longer runs before filter",
            "primary": False,
            "before_path": [],
            "after_path": [],
        },
    ],
}


def test_the_diff_leads_with_the_rewrite_and_folds_its_knock_ons(page):
    html = _render(page, f"PLAN.diff('plan-diff', {json.dumps(_DIFF)});", "plan-diff")
    assert "The optimizer reordered 1 step." in html
    assert "now runs before hash_join" in html
    # The consequential move is behind a disclosure, not presented as a second finding.
    assert "knock-on change" in html
    assert "no longer runs before filter" not in html
    assert html.count('<div class="pv-change ') == 1


def test_the_diff_shows_the_knock_ons_when_asked(page):
    html = _render(
        page, f"PLAN.diff('plan-diff', {json.dumps(_DIFF)}, {{ showAll: true }});", "plan-diff"
    )
    assert "no longer runs before filter" in html
    assert html.count('<div class="pv-change ') == 2
    assert "is-secondary" in html


def test_the_diff_distinguishes_not_recorded_from_unchanged(page):
    missing = _render(
        page, "PLAN.diff('plan-diff', { available: false, changes: [] });", "plan-diff"
    )
    assert "was not recorded" in missing
    unchanged = _render(
        page,
        "PLAN.diff('plan-diff', { available: true, identical: true, changes: [], "
        "counts: [], summary: 'The optimizer left the plan as written.' });",
        "plan-diff",
    )
    assert "left the plan as written" in unchanged
    assert "was not recorded" not in unchanged


_MEASURED_DAG = {
    "nodes": [
        {
            "op_id": 0,
            "kind": "aggregate",
            "detail": "by region",
            "depth": 0,
            "stage": 1,
            "measured": True,
            "elapsed_ms": 60.0,
            "rows_out": 3,
            "spilled": False,
            "on_critical_path": True,
            "breaker": True,
        },
        {
            "op_id": 1,
            "kind": "filter",
            "detail": "amount > 1",
            "depth": 1,
            "stage": 0,
            "measured": True,
            "elapsed_ms": 20.0,
            "rows_out": 900,
            "spilled": True,
            "on_critical_path": True,
            "breaker": False,
        },
        {
            "op_id": 2,
            "kind": "scan",
            "detail": "source 0",
            "depth": 2,
            "stage": 0,
            "measured": True,
            "elapsed_ms": 20.0,
            "rows_out": 2000,
            "spilled": False,
            "on_critical_path": False,
            "breaker": False,
        },
    ],
    "critical_path": [0, 1],
}


def test_the_stage_view_groups_the_steps_that_stream_together(page):
    html = _render(page, f"PLAN.stages('stages', {json.dumps(_MEASURED_DAG)});", "stages")
    assert "stage 1" in html and "stage 2" in html
    assert html.count('<button class="pv-stage-row') == 3
    # Says plainly that these are durations, not positions on a clock.
    assert "not</b> placed on a clock" in html or "not placed on a clock" in html


def test_the_stage_view_marks_a_spilled_step(page):
    html = _render(page, f"PLAN.stages('stages', {json.dumps(_MEASURED_DAG)});", "stages")
    assert "is-spilled" in html


def test_the_flame_view_sizes_each_step_by_its_own_time(page):
    html = _render(page, f"PLAN.flame('flame', {json.dumps(_MEASURED_DAG['nodes'])});", "flame")
    assert html.count('<button class="ch-flame-cell') == 3
    # 60 of 100ms is the dominant step, so it lands in the darkest band.
    assert "ch-seq-5" in html
    # And the panel states why a parent is not the sum of its children.
    assert "not</b> the sum" in html


def test_a_run_with_no_timings_gets_an_explanation_not_an_empty_panel(page):
    for call, host in (
        ("PLAN.stages('stages', { nodes: [] });", "stages"),
        ("PLAN.flame('flame', []);", "flame"),
    ):
        html = _render(page, call, host)
        assert "empty-state" in html, f"{host} rendered nothing at all"


def test_the_plan_document_viewer_leads_with_the_operator_tag(page):
    doc = {"op": "filter", "predicate": {"e": "col", "name": "x"}, "input": {"op": "scan"}}
    html = _render(page, f"PLAN.ir('ir', {json.dumps(doc)});", "ir")
    # The `op` tag is what a reader scans for, so it is the node's summary.
    assert "pv-tag" in html and "filter" in html
    assert "pv-key" in html and "predicate" in html


# --- the live view ----------------------------------------------------------


_LIVE = {
    "query_id": "q1",
    "label": "embed",
    "partitions": {
        "done": 4,
        "total": 16,
        "fraction": 0.25,
        "stages": {"map": {"done": 4, "total": 16, "rows": 4000}},
    },
    "rows_per_sec": 1200.0,
    "total_rows": 4000,
    "inference": {"batches": 40, "latency_ms": 35.0, "blocked_ms": 20.0},
    "gpu": {
        "gpu0": {
            "util_pct": 24.0,
            "mem_used_bytes": 15 << 30,
            "mem_total_bytes": 16 << 30,
            "mem_fraction": 0.94,
            "starved": True,
        }
    },
    "pool": {"size": 4, "pending": 2},
    "skipped": {"total": 12, "by_reason": {"decode error": 12}},
    "diagnostics": [
        {"severity": "warning", "code": "gpu-underused", "message": "gpu0 is at 24% utilization"}
    ],
}


def test_the_live_view_reports_partition_progress_with_a_real_denominator(page):
    html = _render(page, f"LIVE.render({json.dumps(_LIVE)}, []);", "live-body")
    assert "of 16" in html and "partitions finished" in html
    assert "live-progress" in html


def test_the_live_view_shows_no_percentage_without_a_total(page):
    live = json.loads(json.dumps(_LIVE))
    live["partitions"] = {"done": 7, "total": None, "fraction": None, "stages": {}}
    html = _render(page, f"LIVE.render({json.dumps(live)}, []);", "live-body")
    assert "no total reported, so no percentage" in html
    assert "live-progress" not in html


def test_the_live_view_calls_out_a_starved_accelerator(page):
    html = _render(page, f"LIVE.render({json.dumps(_LIVE)}, []);", "live-body")
    assert "starved" in html
    # The gauge carries its target bands, so 24% reads as bad rather than merely as 24%.
    assert "ch-gauge-band" in html
    assert "ch-gauge" in html


def test_the_live_view_surfaces_dropped_rows_rather_than_hiding_them(page):
    html = _render(page, f"LIVE.render({json.dumps(_LIVE)}, []);", "live-body")
    assert "rows were dropped" in html
    assert "decode error" in html


def test_the_live_view_names_the_pipeline_as_the_bottleneck_when_workers_wait(page):
    html = _render(page, f"LIVE.render({json.dumps(_LIVE)}, []);", "live-body")
    assert "waiting for input" in html
    assert "upstream" in html


def test_the_live_view_presents_the_engines_own_verdicts_unchanged(page):
    html = _render(page, f"LIVE.render({json.dumps(_LIVE)}, []);", "live-body")
    assert "gpu0 is at 24% utilization" in html
    assert "gpu-underused" in html


def test_an_idle_engine_gets_a_teaching_empty_state(page):
    html = _render(page, "LIVE.render(null, []);", "live-body")
    assert "Nothing is running" in html
    assert "empty-code" in html, "the empty state should carry a runnable snippet"


def test_a_running_query_is_listed_with_what_the_engine_can_actually_report(page):
    running = [
        {
            "query_id": "q9",
            "label": "scan",
            "started_wall": 0,
            "rows_seen": 5000,
            "bytes_seen": 1 << 20,
            "n_stages": 4,
            "n_done": 1,
        }
    ]
    html = _render(page, f"LIVE.render(null, {json.dumps(running)});", "live-body")
    assert "1 of 4 steps" in html
    assert "pulse-dot" in html


# --- the chart layer --------------------------------------------------------


def test_a_time_series_draws_a_zero_based_axis_and_a_crosshair(page):
    page.eval(
        "CHARTS.timeSeries(document.getElementById('throughput'), "
        "[{ key: 'a', label: 'rows', values: [{x:1,y:10},{x:2,y:40},{x:3,y:25}] }], "
        "{ format: 'count' });"
    )
    html = page.eval("document.getElementById('throughput').innerHTML")
    assert "ch-line" in html and "ch-grid" in html
    assert "ch-cross" in html, "every plot ships a crosshair"
    assert "ch-capture" in html, "the hit target is the plot area, not the 2px line"
    # A single series carries no legend: the panel heading already names it.
    assert "ch-legend" not in html


def test_two_series_always_carry_a_legend(page):
    page.eval(
        "CHARTS.timeSeries(document.getElementById('throughput'), ["
        "{ key: 'a', label: 'one', values: [{x:1,y:1},{x:2,y:2}] },"
        "{ key: 'b', label: 'two', values: [{x:1,y:2},{x:2,y:1}] }]);"
    )
    html = page.eval("document.getElementById('throughput').innerHTML")
    assert "ch-legend" in html and "one" in html and "two" in html


def test_a_proportion_bar_folds_the_tail_into_one_labelled_bucket(page):
    parts = [{"label": f"p{i}", "value": 10 - i, "id": f"s{i}"} for i in range(9)]
    html = page.eval(f"CHARTS.proportion({json.dumps(parts)}, {{ max: 4, link: 'pipe' }})")
    assert "5 others" in html
    assert html.count('<span class="ch-prop-seg') == 5
    # The named segments navigate; the "others" bucket has nowhere to go and so takes no
    # place in the focus order.
    assert html.count("data-pipe=") == 8
    assert html.count('tabindex="0"') == 8


def test_a_gauge_without_a_real_maximum_reports_nothing_it_cannot_measure(page):
    html = page.eval("CHARTS.gauge({ value: 5, max: 0, label: 'vram', format: 'bytes' })")
    # No maximum means no fill: a gauge implies a ceiling, and inventing one is the lie.
    assert 'style="width:0.0%"' in html


def test_a_signed_change_encodes_direction_and_goodness_separately(page):
    worse = page.eval("CHARTS.delta(30, 30, { format: 'ms' })")
    better = page.eval("CHARTS.delta(-30, 30, { format: 'ms' })")
    assert "▲" in worse and "is-warn" in worse and "to-right" in worse
    assert "▼" in better and "is-good" in better and "to-left" in better
    assert "-30" not in better, "the arrow carries the sign, not a minus"


def test_a_run_faster_than_the_timer_still_gets_a_verdict(page):
    """`ratio: 0` is a real reading, not a missing one.

    A run that finishes below the timer's resolution divides to exactly zero. Treated as
    falsy, that told the reader "first run — no baseline yet" about a pipeline they had
    just watched run five times, which is the one thing the verdict must never say wrongly.
    """
    run = {
        "query_id": "q1",
        "label": "aggregate",
        "status": "ok",
        "total_ms": 0.0,
        "rows": 3,
        "started_wall": 0,
        "baseline": {
            "runs": 4,
            "median_ms": 12.0,
            "ratio": 0.0,
            "fastest_ms": 0.0,
            "slowest_ms": 30.0,
        },
    }
    page.eval(f"VIEWS.verdict({json.dumps(run)});")
    text = page.eval("document.getElementById('d-verdict').textContent")
    assert "first run" not in text
    assert "faster" in text


def test_a_genuinely_first_run_still_says_so(page):
    run = {
        "query_id": "q1",
        "label": "scan",
        "status": "ok",
        "total_ms": 5.0,
        "rows": 1,
        "started_wall": 0,
        "baseline": None,
    }
    page.eval(f"VIEWS.verdict({json.dumps(run)});")
    assert "first run" in page.eval("document.getElementById('d-verdict').textContent")


# --- pipeline identity: name, id, thumbnail, and the registry -----------------

from batcher.observe.pipelines import PipelineRegistry  # noqa: E402


def _agg_ir() -> dict:
    return {
        "op": "aggregate",
        "group_keys": [{"alias": "region"}],
        "aggregates": [{"func": "sum"}],
        "input": {"op": "filter", "predicate": _predicate(), "input": _scan(0)},
    }


@pytest.fixture
def registered_store(tmp_path):
    """A store whose pipeline registry writes to a tmp file, with one pipeline run twice.

    Two distinct query ids under one signature, so the group genuinely has two runs — the
    same shape a real pipeline has.
    """
    store = ActivityStore(registry=PipelineRegistry(path=tmp_path / "pipelines.json"))
    for i in range(2):
        qid = f"run-{i}"
        _emit(store, events.QUERY_START, qid, "aggregate", label="aggregate", signature="sig-1")
        _emit(
            store,
            events.QUERY_END,
            qid,
            ok=True,
            total_ms=10.0 + i,
            rows=3,
            profile={"ops": [], "optimized_ir": _agg_ir()},
        )
    assert store.pipelines()[0]["runs"] == 2
    return store


def test_a_pipeline_carries_a_stable_id_and_a_plan_shape(registered_store):
    p = registered_store.pipelines()[0]
    # The id is the plan signature — the same identity Kyber keys learned stats on.
    assert p["pipeline_id"] == "sig-1"
    assert p["signature"] == "sig-1"
    # The plan shape is the thumbnail's fingerprint: the operators, laid out, no measurements.
    kinds = [n["kind"] for n in p["plan_shape"]["nodes"]]
    assert kinds == ["aggregate", "filter", "scan"]
    assert all("elapsed_ms" not in n for n in p["plan_shape"]["nodes"])
    assert p["plan_shape"]["edges"]


def test_an_unnamed_pipeline_reports_an_empty_name_not_a_guess(registered_store):
    # The backend does not invent a name — that is the frontend's job from the shape. It
    # reports "" so the two cannot disagree about what the default is.
    assert registered_store.pipelines()[0]["name"] == ""


def test_naming_a_pipeline_persists_across_a_reload(tmp_path):
    path = tmp_path / "pipelines.json"
    store = ActivityStore(registry=PipelineRegistry(path=path))
    _emit(store, events.QUERY_START, "p0", "aggregate", label="aggregate", signature="sig-1")
    _emit(
        store,
        events.QUERY_END,
        "p0",
        ok=True,
        total_ms=5.0,
        rows=1,
        profile={"ops": [], "optimized_ir": _agg_ir()},
    )
    store.pipelines()  # stamps first-seen
    store.set_pipeline_meta("sig-1", name="Nightly rollup", note="watch the spill")

    # A fresh registry reading the same file sees the name — the whole point of "for later".
    reloaded = PipelineRegistry(path=path)
    meta = reloaded.get("sig-1")
    assert meta.name == "Nightly rollup"
    assert meta.note == "watch the spill"
    assert meta.first_seen_wall > 0


def test_the_name_shows_up_in_the_pipeline_listing(registered_store):
    registered_store.set_pipeline_meta("sig-1", name="My rollup")
    assert registered_store.pipelines()[0]["name"] == "My rollup"


def test_setting_a_note_leaves_the_name_intact_and_vice_versa(tmp_path):
    reg = PipelineRegistry(path=tmp_path / "pipelines.json")
    reg.set_meta("sig-1", name="Keep me")
    reg.set_meta("sig-1", note="just a note")
    assert reg.get("sig-1").name == "Keep me"
    assert reg.get("sig-1").note == "just a note"


def test_a_corrupt_registry_file_starts_empty_rather_than_crashing(tmp_path):
    path = tmp_path / "pipelines.json"
    path.write_text("{ this is not json")
    reg = PipelineRegistry(path=path)  # must not raise
    assert reg.all() == {}
    # And a subsequent write repairs it.
    reg.set_meta("sig-1", name="ok")
    assert PipelineRegistry(path=path).get("sig-1").name == "ok"


def test_names_and_notes_are_clamped_to_a_sane_length(tmp_path):
    reg = PipelineRegistry(path=tmp_path / "pipelines.json")
    meta = reg.set_meta("sig-1", name="x" * 5000, note="y" * 5000)
    assert len(meta.name) <= 200
    assert len(meta.note) <= 2000


def test_the_meta_write_endpoint_names_a_pipeline_over_http(tmp_path):
    store = ActivityStore(registry=PipelineRegistry(path=tmp_path / "pipelines.json"))
    _emit(store, events.QUERY_START, "p0", "agg", label="agg", signature="sig-1")
    _emit(
        store,
        events.QUERY_END,
        "p0",
        ok=True,
        total_ms=5.0,
        rows=1,
        profile={"ops": [], "optimized_ir": _scan()},
    )
    server = UIServer(store, port=0)
    server.start()
    try:
        request = urllib.request.Request(
            server.url + "/api/pipeline/meta",
            data=json.dumps({"pipeline_id": "sig-1", "name": "Renamed"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
            assert json.loads(response.read())["name"] == "Renamed"
        listed = json.loads(_fetch_url(server, "/api/pipelines"))
        assert listed["pipelines"][0]["name"] == "Renamed"
    finally:
        server.stop()


def test_the_meta_endpoint_rejects_a_body_with_no_pipeline_id(tmp_path):
    store = ActivityStore(registry=PipelineRegistry(path=tmp_path / "pipelines.json"))
    server = UIServer(store, port=0)
    server.start()
    try:
        request = urllib.request.Request(
            server.url + "/api/pipeline/meta",
            data=json.dumps({"name": "no id here"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 400
    finally:
        server.stop()


def _fetch_url(server, path):
    with urllib.request.urlopen(server.url + path) as response:
        return response.read()


# --- the pipeline thumbnail and name, rendered -------------------------------


def test_the_thumbnail_draws_a_node_per_operator_and_edges_between_them(shaped_page):
    html = shaped_page.eval("DAG.thumbnail(PLAN_SHAPE_JS)")
    assert html.count("tn-node") == 3
    assert html.count("tn-edge") == 2
    # The critical path is the one thing the neutral thumbnail tints.
    assert "is-crit" in html


def test_the_thumbnail_of_no_plan_is_an_empty_placeholder_not_an_error(page):
    html = page.eval("DAG.thumbnail({nodes: [], edges: [], width: 0, depth: 0})")
    assert "is-empty" in html


def test_a_generated_name_reads_as_the_plan_chain(shaped_page):
    name = shaped_page.eval("DAG.pipelineName(PLAN_SHAPE_JS, '')")
    # Execution order, sources first, collapsed — a filter/group over a scan.
    assert name == "Read → Filter → Group"


def test_a_custom_name_wins_over_the_generated_one(shaped_page):
    assert shaped_page.eval("DAG.pipelineName(PLAN_SHAPE_JS, 'My rollup')") == "My rollup"
