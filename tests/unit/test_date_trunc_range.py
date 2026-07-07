"""Plan-shape unit tests for `date_trunc_to_range` (sargable time-series rewrite)."""

from __future__ import annotations

import datetime as dt
import json

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.normalize import date_trunc_to_range
from batcher.plan.logical import Filter


def _t():
    return bt.from_pydict(
        {"ts": [dt.datetime(2024, 6, 10, 8, 30), dt.datetime(2024, 7, 2, 1, 0)], "v": [1, 2]}
    )


def test_rule_registered():
    assert "date_trunc_to_range" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_month_equality_becomes_range():
    plan = _t().filter(col("ts").dt.truncate("month") == dt.datetime(2024, 6, 1))._plan
    out = date_trunc_to_range(plan)
    assert isinstance(out, Filter)
    pred = out.predicate.to_ir()
    assert pred["op"] == "and"
    lo, hi = pred["left"], pred["right"]
    assert lo["op"] == "ge" and lo["left"]["name"] == "ts"
    assert hi["op"] == "lt" and hi["left"]["name"] == "ts"
    # [2024-06-01, 2024-07-01) in micros since epoch.
    assert lo["right"]["value"]["timestamp"] == 1717200000000000
    assert hi["right"]["value"]["timestamp"] == 1719792000000000


def test_full_optimizer_removes_date_trunc():
    plan = _t().filter(col("ts").dt.truncate("day") == dt.datetime(2024, 6, 10))._plan
    ir = json.dumps(Optimizer().optimize(plan).ir)
    assert "date_trunc" not in ir
    assert '"ge"' in ir and '"lt"' in ir


def test_unaligned_literal_left_untouched():
    # A literal not aligned to the unit matches nothing this range would, so the rule
    # must not fire (the engine still evaluates the truncation correctly).
    plan = _t().filter(col("ts").dt.truncate("month") == dt.datetime(2024, 6, 15))._plan
    out = date_trunc_to_range(plan)
    assert "date_trunc" in json.dumps(out.predicate.to_ir())


def test_inequality_left_untouched():
    plan = _t().filter(col("ts").dt.truncate("day") < dt.datetime(2024, 6, 10))._plan
    out = date_trunc_to_range(plan)
    assert "date_trunc" in json.dumps(out.predicate.to_ir())


def test_idempotent():
    plan = _t().filter(col("ts").dt.truncate("hour") == dt.datetime(2024, 6, 10, 8))._plan
    once = date_trunc_to_range(plan)
    assert date_trunc_to_range(once).to_ir() == once.to_ir()
