"""The event log splices pre-serialized IR text into its document instead of re-encoding it.

Both IR fields already exist as JSON on every run — the logical plan serializes its IR to
compute `content_key`, and a cached physical plan memoizes `to_json` — and encoding them a
second time was over half the default-on event log's cost (1.18 of 2.05 ms on TPC-H q8).
What must not change is the *content*: the document read back is the document assembled.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import batcher as bt
from batcher.api.terminal.event_log import _encode
from batcher.config import active_config, config_context
from batcher.plan.profile import ProfileCollector

pytestmark = pytest.mark.unit


def _collector(logical: dict, optimized: dict) -> ProfileCollector:
    col = ProfileCollector()
    col.logical_ir, col.optimized_ir = logical, optimized
    col.logical_ir_json = json.dumps(logical, separators=(",", ":"))
    col.optimized_ir_json = json.dumps(optimized)
    return col


def test_a_spliced_document_parses_to_the_assembled_document():
    logical = {"op": "filter", "input": {"op": "scan", "source_id": 0}, "n": [1, 2.5, None]}
    optimized = {"op": "scan", "source_id": 0, "s": 'quote " and \\ slash'}
    col = _collector(logical, optimized)
    doc = {"query_id": "q", "logical_ir": logical, "ops": [{"x": 1}], "optimized_ir": optimized}

    assert json.loads(_encode(doc, col)) == doc


def test_a_document_holding_only_the_ir_still_encodes():
    logical, optimized = {"op": "scan"}, {"op": "scan", "source_id": 1}
    col = _collector(logical, optimized)
    doc = {"logical_ir": logical, "optimized_ir": optimized}

    assert json.loads(_encode(doc, col)) == doc


def test_text_for_a_different_dict_is_never_spliced():
    # The collector's text describes one IR; the document now holds another. Splicing would
    # write the wrong plan into the record, so the field must be encoded from the document.
    col = _collector({"op": "scan", "source_id": 0}, {"op": "scan", "source_id": 0})
    doc = {"logical_ir": {"op": "limit"}, "optimized_ir": {"op": "sort"}}

    assert json.loads(_encode(doc, col)) == doc


def test_the_written_event_log_carries_the_plan_that_ran(tmp_path, monkeypatch):
    cfg = active_config()
    obs = dataclasses.replace(cfg.observability, event_log=True, event_log_dir=str(tmp_path))
    import batcher.api.terminal.event_log as event_log

    encoded: list[object] = []
    real_dumps = json.dumps

    def spy(obj, *a, **k):
        encoded.append(obj)
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(event_log.json, "dumps", spy)
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]}).filter(bt.col("a") > 1)
    with config_context(cfg.replace(observability=obs)):
        ds.collect()
    assert event_log.flush_event_log(10.0)  # the encode runs on the event-log writer
    monkeypatch.undo()
    # The splice ran: the event log encoded a document, and none it encoded carried an IR.
    docs = [o for o in encoded if isinstance(o, dict) and "query_id" in o]
    assert docs
    assert not any("logical_ir" in d or "optimized_ir" in d for d in docs)

    written = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert written, "no event-log document was written"
    doc = written[-1]
    assert doc["logical_ir"] == ds._plan.to_ir()
    assert doc["optimized_ir"]["op"]  # the lowered IR the engine ran, as a JSON object
