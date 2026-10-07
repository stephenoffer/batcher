"""Live smoke test: real Ibis expressions compiled by Ibis and run by the Batcher bridge.

Skipped unless ``BATCHER_LIVE_IBIS=1`` and ibis-framework is installed (the ``ibis`` extra).
Run: ``BATCHER_LIVE_IBIS=1 pytest tests/integration/live/test_live_ibis.py``.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("BATCHER_LIVE_IBIS"), reason="set BATCHER_LIVE_IBIS=1"),
]


def test_the_documented_subset_runs_through_ibis() -> None:
    pytest.importorskip("ibis")
    import batcher as bt
    from batcher.integrations import ibis as bt_ibis

    s = bt.Session()
    s.register("orders", bt.from_pydict({"id": [1, 2, 3], "g": ["a", "b", "a"], "v": [5, 7, 9]}))
    s.register("labels", bt.from_pydict({"g": ["a", "b"], "label": ["A", "B"]}))
    orders, labels = bt_ibis.table("orders", s), bt_ibis.table("labels", s)

    filtered = orders.filter(orders.v > 6).select("id").order_by("id")
    assert bt_ibis.to_dataset(filtered, s).to_pydict() == {"id": [2, 3]}

    grouped = orders.group_by("g").aggregate(total=orders.v.sum()).order_by("g")
    assert bt_ibis.to_dataset(grouped, s).to_pydict() == {"g": ["a", "b"], "total": [14, 7]}

    joined = orders.join(labels, "g").select("id", "label").order_by("id").limit(2)
    assert bt_ibis.to_dataset(joined, s).to_pydict() == {"id": [1, 2], "label": ["A", "B"]}
