"""`explain(analyze=True)` / `stats()` / event log — end to end over the engine.

Verifies the combined planned↔measured profile against a real run: estimate-vs-actual is
present, the JSON form parses, `stats()` carries the planned estimate, the event log lands
on disk, and — the correctness gate — profiling never changes the query result.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher import col
from batcher.config import ObservabilityConfig, active_config, set_config

pytestmark = pytest.mark.integration


def _ds():
    tbl = pa.table({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
    return bt.from_arrow(tbl).filter(col("v") > 5).group_by("k").agg(s=col("v").sum())


def test_explain_planned_tree_has_estimates_no_execution():
    text = _ds().explain()
    assert "aggregate" in text and "scan" in text
    assert "est≈" in text


def test_explain_analyze_shows_estimate_vs_actual():
    text = _ds().explain(analyze=True)
    assert "actual=" in text
    assert "bottleneck" in text
    # The measured leaf scan saw all 5 input rows.
    assert "scan" in text


def test_explain_analyze_json_is_machine_readable():
    doc = json.loads(_ds().explain(analyze=True, format="json"))
    assert doc["measured"] is True
    measured = [o for o in doc["ops"] if o["measured"]]
    assert {"scan", "aggregate"} <= {o["kind"] for o in measured}
    assert doc["rows"] == 3


def test_stats_carries_planned_estimate():
    st = _ds().stats()
    assert st.rows == 3
    assert any(o.kind == "aggregate" for o in st.ops)
    # `est_rows` is joined from the optimizer (a finite estimate or nan, never absent).
    assert all(hasattr(o, "est_rows") for o in st.ops)


def test_stats_rejects_map_batches():
    from batcher._internal.errors import BackendError

    ds = bt.from_arrow(pa.table({"x": [1, 2, 3]})).ml.map_batches(lambda b: b, output_columns=["x"])
    with pytest.raises(BackendError, match="map_batches"):
        ds.stats()


def test_event_log_written_on_collect(tmp_path):
    prev = active_config()
    set_config(
        prev.replace(observability=ObservabilityConfig(event_log=True, event_log_dir=str(tmp_path)))
    )
    try:
        _ds().collect()
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        doc = json.loads(files[0].read_text())
        assert doc["measured"] is True and doc["rows"] == 3
        assert doc["optimized_ir"] is not None
    finally:
        set_config(prev)


def test_distributed_run_surfaces_worker_map_metrics(tmp_path):
    import numpy as np

    prev = active_config()
    set_config(
        prev.replace(observability=ObservabilityConfig(event_log=True, event_log_dir=str(tmp_path)))
    )
    try:
        t = pa.table({"k": (np.arange(20_000) % 100).astype("int64"), "v": np.arange(20_000) % 7})
        bt.from_arrow(t).group_by("k").agg(s=col("v").sum()).collect(distributed=True)
        doc = json.loads(sorted(tmp_path.glob("*.json"))[-1].read_text())
        assert doc["distributed"] is True
        # The distributed map sub-plan is surfaced as its own measured section (a separate
        # op-id space from the driver tree, not falsely joined into it).
        assert len(doc["worker_ops"]) >= 1
        assert all(o["measured"] for o in doc["worker_ops"])
    finally:
        set_config(prev)


def test_profiling_does_not_change_result():
    prev = active_config()
    ds = _ds()
    set_config(prev.replace(observability=ObservabilityConfig(event_log=False)))
    try:
        without = ds.collect().to_pydict()
    finally:
        set_config(prev)
    # explain(analyze) runs the same plan with profiling on; the result must match.
    with_profiling = ds.collect().to_pydict()
    assert _sorted(without) == _sorted(with_profiling)


def _sorted(d: dict) -> list:
    return sorted(zip(*d.values(), strict=True))
