"""A keyed aggregate sizes its shuffle reducers by LEARNED output cardinality.

The reduce phase of a distributed aggregate shuffles PARTIAL-aggregated state, whose size
is the group count — not the (far larger) scanned input. Sizing reducers one-per-worker (the
generic shuffle fan-out passed as ``base_reducers``) is wrong in both directions, and the
group count is what corrects it.

Too many, at the low end: a 60M-row → 4-group aggregate opens ``workers x workers``
near-empty Flight streams where one reducer would do.

Too few, at the high end: one reducer per worker pins the reduce fan-out to the *cluster*, so
each reducer's group table grows with the data and eventually spills — wall time growing
faster than the input, on a cluster that has not changed. The mergeable algebra permits any
number of independent merges, and tying that number to the node count is what discards the
property. The reducer count therefore scales ABOVE the worker fan-out when the groups call
for it, bounded only by ``distributed.max_shuffle_partitions`` (the exchange opens
``mappers x reducers`` streams, and it is that product a huge cluster cannot afford).

Any reducer count is result-correct under the mergeable algebra, so this only shapes the
exchange.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import active_config
from batcher.dist.adaptive_sizing import aggregate_reducer_count

pytestmark = pytest.mark.unit

_BASE = 8  # the generic one-per-worker shuffle fan-out the caller passes in


def _agg_node():
    t = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
    return bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count())._plan


def test_cold_signature_keeps_base_fanout(monkeypatch):
    from batcher.kyber import learned_tuning

    # Cold *and* no sources to estimate from: nothing better than the caller's fan-out.
    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: None)
    assert aggregate_reducer_count(_agg_node(), _BASE) == _BASE


def test_cold_signature_falls_back_to_the_estimated_group_count(monkeypatch):
    """A first run at scale is sized by Kyber's cardinality estimate, not by the cluster.

    The run that most needs the sizing is the one the learned store cannot help with, so a
    cold signature reaches for the optimizer's estimate before giving up on `base_reducers`.
    """
    from batcher.dist.adaptive_sizing import sizing
    from batcher.kyber import learned_tuning

    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: None)
    monkeypatch.setattr(sizing, "_estimated_rows", lambda node, sources: 1e9)
    target = active_config().optimizer.target_rows_per_task
    n = aggregate_reducer_count(_agg_node(), _BASE, 1, sources=["a source"])
    assert n == math.ceil(1e9 / target)
    assert n > _BASE


def test_low_cardinality_collapses_to_one_reducer(monkeypatch):
    from batcher.kyber import learned_tuning

    # A learned 4-group output → a single reducer regardless of the base fan-out.
    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 4.0)
    assert aggregate_reducer_count(_agg_node(), 16) == 1


def test_high_cardinality_scales_past_the_worker_fanout(monkeypatch):
    from batcher.kyber import learned_tuning

    # A billion groups needs far more than one reducer per worker: capping it there is what
    # makes each reducer's state grow with the data on a fixed cluster.
    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 1e9)
    target = active_config().optimizer.target_rows_per_task
    n = aggregate_reducer_count(_agg_node(), _BASE)
    assert n > _BASE
    assert n == math.ceil(1e9 / target)


def test_reducer_count_bounds_each_reducers_groups(monkeypatch):
    """The scaling property: groups per reducer stays bounded as the aggregate grows.

    It holds up to the point the stream cap binds, at
    ``max_shuffle_partitions x target_rows_per_task`` groups (~8.2e9 by default). Past that
    the exchange's stream budget is the binding constraint rather than reducer memory, and
    each reducer's state does grow again — the honest limit of this sizing, pinned here so it
    is a known boundary rather than a surprise.
    """
    from batcher.kyber import learned_tuning

    cfg = active_config()
    target = cfg.optimizer.target_rows_per_task
    ceiling = cfg.distributed.max_shuffle_partitions * target
    for groups in (1e7, 1e8, 1e9, float(ceiling)):
        monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, g=groups, **k: g)
        n = aggregate_reducer_count(_agg_node(), _BASE)
        assert groups / n <= target, f"{groups} groups over {n} reducers exceeds the target"


def test_never_above_the_shuffle_partition_cap(monkeypatch):
    from batcher.kyber import learned_tuning

    # The exchange opens mappers x reducers streams, so the count stays capped however many
    # groups the aggregate produces.
    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 1e18)
    cap = active_config().distributed.max_shuffle_partitions
    assert aggregate_reducer_count(_agg_node(), _BASE) == cap


def test_never_below_one(monkeypatch):
    from batcher.kyber import learned_tuning

    monkeypatch.setattr(learned_tuning, "learned_signature_rows", lambda *a, **k: 0.0)
    assert aggregate_reducer_count(_agg_node(), 4) >= 1
