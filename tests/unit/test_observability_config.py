"""Central logging configuration and the per-query event-log writer (pure-Python)."""

from __future__ import annotations

import json
import logging
import os
from unittest import mock

import pytest

from batcher._internal import logging as blog
from batcher.config import ObservabilityConfig
from batcher.plan.profile import Decision, ProfileCollector

pytestmark = pytest.mark.unit


def test_get_logger_namespaces_under_batcher():
    assert blog.get_logger().name == "batcher"
    assert blog.get_logger("kyber").name == "batcher.kyber"


def test_configure_sets_level_and_file_handler(tmp_path):
    log_file = tmp_path / "engine.log"
    # Reset the module's applied-state so configure actually runs in this test.
    blog._applied = None
    blog.configure(ObservabilityConfig(log_level="DEBUG", console=False, log_file=str(log_file)))
    logger = blog.get_logger()
    assert logger.level == logging.DEBUG
    blog.get_logger("kyber").debug("hello-from-test")
    for h in logger.handlers:
        h.flush()
    assert "hello-from-test" in log_file.read_text()


def test_json_format_emits_json_records(tmp_path):
    log_file = tmp_path / "engine.json"
    blog._applied = None
    blog.configure(
        ObservabilityConfig(
            log_level="INFO", console=False, log_file=str(log_file), log_format="json"
        )
    )
    blog.get_logger("core").info("structured")
    for h in blog.get_logger().handlers:
        h.flush()
    record = json.loads(log_file.read_text().splitlines()[0])
    assert record["message"] == "structured" and record["logger"] == "batcher.core"


def test_native_tracing_settings_reflect_configured_level():
    blog._applied = None
    blog._native_settings = None
    assert blog.native_tracing_settings() is None  # unconfigured
    blog.configure(ObservabilityConfig(log_level="INFO", console=False, log_format="json"))
    assert blog.native_tracing_settings() == ("INFO", True)


def _collector_with_one_op() -> ProfileCollector:
    c = ProfileCollector()
    c.optimized_ir = {"op": "scan", "source_id": 0}
    c.logical_ir = {"op": "scan", "source_id": 0}
    c.metric_ops = [{"op_id": 0, "kind": "scan", "rows_in": 3, "rows_out": 3, "elapsed_ns": 1000}]
    c.decisions = [Decision("carbonite", "admission", "feasible")]
    return c


def test_event_log_writes_document_and_prunes(tmp_path):
    from batcher.api.terminal.event_log import _prune, write_event_log
    from batcher.config import active_config, set_config

    prev = active_config()
    set_config(
        prev.replace(observability=ObservabilityConfig(event_log=True, event_log_dir=str(tmp_path)))
    )
    try:
        write_event_log(_collector_with_one_op(), total_ms=1.5, rows=3)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        doc = json.loads(files[0].read_text())
        assert doc["rows"] == 3 and doc["ops"][0]["kind"] == "scan"
        assert doc["decisions"][0]["summary"] == "feasible"
        # Pruning keeps at most `max_files`, oldest first. `seq=0` is a reconciling pass,
        # which is what collects documents this process did not write -- here the five
        # planted below, and in a deployment the ones a sibling process left.
        for i in range(5):
            (tmp_path / f"20200101-000000-{i:06d}.json").write_text("{}")
        _prune(tmp_path, max_files=2, wrote="20200101-000000-000004.json", seq=0)
        kept = sorted(p.name for p in tmp_path.glob("*.json"))
        # Two survive, and they are the two newest by name: the planted `2020` documents go
        # oldest-first, and the real one written above carries today's date, so it sorts last.
        assert len(kept) == 2
        assert kept[0] == "20200101-000000-000004.json"
        assert not any(name.startswith("20200101-000000-00000") and name < kept[0] for name in kept)
    finally:
        set_config(prev)


def test_event_log_prunes_incrementally_without_rescanning(tmp_path):
    """Retention holds at the cap across many writes without listing the directory.

    The in-memory window is the whole point of `_prune` (it was 11% of a small query's
    control plane as a scan), so the test asserts *both* halves: the cap is honored, and
    the scan does not happen. Without the second assertion the fast path could regress to
    the slow one and nothing would notice -- the directory would look identical.
    """
    from batcher.api.terminal.event_log import _WRITTEN, _prune

    _WRITTEN.pop(tmp_path, None)
    scans = 0
    real_scandir = os.scandir

    def counting_scandir(path):
        nonlocal scans
        if str(path) == str(tmp_path):
            scans += 1
        return real_scandir(path)

    with mock.patch.object(os, "scandir", counting_scandir):
        for i in range(40):
            name = f"20200101-000000-{i:06d}.json"
            (tmp_path / name).write_text("{}")
            _prune(tmp_path, max_files=5, wrote=name, seq=i + 1)

    assert len(list(tmp_path.glob("*.json"))) == 5
    assert scans == 1  # only the first write seeds the window


def test_event_log_disabled_collector_is_none():
    """With every consumer off, the collector is not built and the query pays nothing.

    The bus is emptied for the duration, because "no consumer" is the whole premise and
    `event_log_collector` deliberately returns a collector whenever *anything* is watching
    — a dashboard, a metrics scrape, an OTel exporter. Without this the assertion holds or
    fails according to whether some earlier test in the process left a sink attached, which
    is a property of the run order rather than of the code under test.
    """
    from batcher._internal import events
    from batcher.api.terminal.event_log import event_log_collector
    from batcher.config import active_config, set_config

    prev = active_config()
    saved_subscribers = events._subscribers
    events._subscribers = ()
    set_config(prev.replace(observability=ObservabilityConfig(event_log=False)))
    try:
        assert event_log_collector() is None
    finally:
        set_config(prev)
        events._subscribers = saved_subscribers
