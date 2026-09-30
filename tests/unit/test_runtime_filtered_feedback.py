"""A count a runtime join filter reduced reaches no cardinality correction.

A runtime join filter (`bc_interp::stream::runtime_filter`) removes probe rows the join would
discard, at the scan. Every operator between that scan and the join then counts only the rows
that survived it, which is not its own cardinality: whether a filter sits beneath it depends on
the plan chosen around it. Read as its size, that count taught TPC-H q7 at sf10 a correction the
next plan's filters contradicted, and the plan cache re-planned every few runs between a 76 ms
plan and a 95 ms one. The engine lists those operators (`ExecMetrics::runtime_filtered`), and
Core records them with `n_estimated = 0`, which the correction loop reads as nothing to learn.

The control that makes the end-to-end half mean something: the scan the filter is placed on and
the join it serves are *not* listed, because their own counts are true.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import batcher as bt
from batcher.core import executor
from batcher.core.executor import record_exec_metrics
from batcher.plan.expr_ir import col
from batcher.plan.feedback import OperatorFeedback

pytestmark = pytest.mark.unit


class _Sink:
    def __init__(self) -> None:
        self.rows: list[OperatorFeedback] = []

    def record(self, feedback: OperatorFeedback) -> None:
        self.rows.append(feedback)


def _planned(n: int) -> list:
    props = SimpleNamespace(signature="sig", est_rows_raw=100.0, expr_factor=1.0)
    return [SimpleNamespace(properties=props)] * n


def test_a_listed_operator_is_recorded_with_no_estimate() -> None:
    sink = _Sink()
    ops = [
        {"op_id": 0, "kind": "hash_join", "rows_in": 10, "rows_out": 10},
        {"op_id": 1, "kind": "filter", "rows_in": 10, "rows_out": 4},
        {"op_id": 2, "kind": "scan", "rows_in": 100, "rows_out": 100},
    ]
    doc = json.dumps({"ops": ops, "runtime_filtered": [1]})
    record_exec_metrics(sink, doc, batch_size=16_384, planned=_planned(3))
    by_id = {int(r.op_id): r for r in sink.rows}
    assert by_id[1].n_estimated == 0.0, "a reduced count reached the correction loop"
    assert by_id[1].n_actual == 4, "the measurement itself is still recorded"
    assert by_id[0].n_estimated == 100.0 and by_id[2].n_estimated == 100.0


def test_a_document_without_the_list_records_every_estimate() -> None:
    sink = _Sink()
    ops = [{"op_id": 0, "kind": "filter", "rows_in": 10, "rows_out": 4}]
    record_exec_metrics(sink, json.dumps({"ops": ops}), batch_size=16_384, planned=_planned(1))
    assert sink.rows[0].n_estimated == 100.0


def test_the_engine_lists_the_operators_between_the_filter_and_its_join(monkeypatch) -> None:
    monkeypatch.setenv("BATCHER_RUNTIME_JOIN_FILTER", "force")
    captured: list[list[dict]] = []
    real = executor._record_op_feedback

    def spy(sink, ops, batch_size, planned=()):
        captured.append([dict(op) for op in ops])
        return real(sink, ops, batch_size, planned)

    monkeypatch.setattr(executor, "_record_op_feedback", spy)
    n = 200_000
    fact = bt.from_pydict({"k": [i % 50_000 for i in range(n)], "v": [i % 1_000 for i in range(n)]})
    keys = list(range(0, 50_000, 7))
    dim = bt.from_pydict({"k": keys, "w": [1] * len(keys)})
    got = fact.filter(col("v") < 900).join(dim, on="k").agg(n=col("v").count()).collect()
    want = sum(1 for i in range(n) if i % 1_000 < 900 and i % 50_000 % 7 == 0)
    assert got.to_pydict()["n"] == [want]

    ops = captured[-1]
    listed = {op["kind"] for op in ops if op.get("runtime_filtered")}
    assert "filter" in listed, f"the filter above the filtered scan was not listed: {ops}"
    assert "scan" not in listed and "hash_join" not in listed, f"a true count was listed: {ops}"
