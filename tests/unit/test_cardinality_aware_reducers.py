"""A keyed aggregate sizes its shuffle reducers by LEARNED output cardinality.

The reduce phase of a distributed aggregate shuffles PARTIAL-aggregated state, whose size
is the group count — not the (far larger) scanned input. Sizing reducers one-per-worker (the
generic shuffle fan-out passed as ``base_reducers``) means a 60M-row → 4-group aggregate opens
``workers x workers`` near-empty Flight streams. Once a run has measured the true output
cardinality (`record_aggregate_cardinality` → `record_execution`), the reducer count collapses
to what the group count needs. Any reducer count is result-correct under the mergeable algebra,
so this only shapes the exchange.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.dist.adaptive_sizing import aggregate_reducer_count

pytestmark = pytest.mark.unit

_BASE = 8  # the generic one-per-worker shuffle fan-out the caller passes in


def _agg_node():
    t = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
    return bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count())._plan


def test_cold_signature_keeps_base_fanout(monkeypatch):
    from batcher.kyber import learned_tuning

    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: None)
    assert aggregate_reducer_count(_agg_node(), _BASE) == _BASE


def test_low_cardinality_collapses_to_one_reducer(monkeypatch):
    from batcher.kyber import learned_tuning

    # A learned 4-group output → a single reducer regardless of the base fan-out.
    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 4.0)
    assert aggregate_reducer_count(_agg_node(), 16) == 1


def test_high_cardinality_stays_capped_at_base(monkeypatch):
    from batcher.kyber import learned_tuning

    # A huge learned output would want many reducers, but never more than the base fan-out.
    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 1e12)
    assert aggregate_reducer_count(_agg_node(), _BASE) == _BASE


def test_never_below_one(monkeypatch):
    from batcher.kyber import learned_tuning

    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 0.0)
    assert aggregate_reducer_count(_agg_node(), 4) >= 1
