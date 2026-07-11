"""Measured spill *volume* flows from the engine into operator feedback.

`OpMetric.spill_bytes` (Rust) is transcribed to `OperatorFeedback.spill_bytes` (the
magnitude Carbonite sizes spill scratch from) — a `spilled` bool alone cannot tell a
1 GB spill from a 100 GB one.
"""

from __future__ import annotations

import json

import pytest

from batcher.core.executor import record_exec_metrics
from batcher.plan.feedback import OperatorFeedback

pytestmark = pytest.mark.unit


class _Sink:
    def __init__(self) -> None:
        self.rows: list[OperatorFeedback] = []

    def record(self, feedback: OperatorFeedback) -> None:
        self.rows.append(feedback)


def test_spill_bytes_is_transcribed_from_native_metrics():
    sink = _Sink()
    doc = json.dumps(
        {
            "ops": [
                {
                    "op_id": 0,
                    "kind": "aggregate",
                    "rows_in": 1000,
                    "rows_out": 4,
                    "rows_build": 0,
                    "elapsed_ns": 10_000,
                    "cpu_ns": 8_000,
                    "threads": 4,
                    "peak_bytes": 50_000,
                    "result_bytes": 100,
                    "spilled": True,
                    "spill_bytes": 42_000,
                    "backend": "interp",
                }
            ]
        }
    )
    record_exec_metrics(sink, doc, batch_size=16_384)
    assert len(sink.rows) == 1
    fb = sink.rows[0]
    assert fb.spill_bytes == 42_000
    assert fb.algorithm == "spill"  # the bool signal is still there


def test_threads_is_transcribed():
    # The measured worker-thread count reaches feedback (metadata capture), where a
    # per-task num_cpus sizing loop needs it to tell a busy 8-core op from a busy 1-core one.
    sink = _Sink()
    op = {"op_id": 0, "kind": "aggregate", "rows_in": 10, "rows_out": 2, "threads": 8}
    record_exec_metrics(sink, json.dumps({"ops": [op]}), batch_size=16_384)
    assert sink.rows[0].threads == 8


def test_peak_rss_bytes_is_transcribed():
    # The measured process peak-RSS growth (getrusage ru_maxrss) reaches feedback, where
    # Carbonite's learned memory model fits against the true high-water.
    sink = _Sink()
    op = {"op_id": 0, "kind": "sort", "rows_in": 5, "rows_out": 5, "peak_rss_bytes": 9_000}
    doc = json.dumps({"ops": [op]})
    record_exec_metrics(sink, doc, batch_size=16_384)
    assert sink.rows[0].peak_rss_bytes == 9_000


def test_missing_spill_bytes_defaults_to_zero():
    # An older engine that predates the field, or a non-spilling op, reports 0.
    sink = _Sink()
    doc = json.dumps({"ops": [{"op_id": 0, "kind": "filter", "rows_in": 10, "rows_out": 5}]})
    record_exec_metrics(sink, doc, batch_size=16_384)
    assert sink.rows[0].spill_bytes == 0
