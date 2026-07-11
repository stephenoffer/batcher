"""`Core measures` must hold on the distributed path, not just single-node.

Kyber calibrates its cost coefficients and Carbonite fits its memory model from the
`OperatorFeedback` Core records. A distributed stage runs its sub-plan inside a Ray worker,
so a worker that calls the unmetered `execute_plan` throws those measurements away — and
for a long time only the disk-aggregate map task was metered. The learning loop was
therefore fit *entirely* from single-node runs, even though the distributed path runs the
largest inputs and is the one that spills.

These tests pin the loop shut: after a distributed sort / join / window, the process-wide
hub must hold feedback for the operator that stage is named after. They assert the
*contract* (a measurement arrived, carrying the facts the models regress on), never a
particular row count or worker count — that would pin the shuffle's shape, not the loop.

The equivalence tests elsewhere already prove distributed == single-node for *results*;
this file makes the same claim for *learned state*.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher import Config, config_context

pytest.importorskip("ray", reason="distributed path requires ray")
pytest.importorskip("batcher._native", reason="native engine not built")

_WORKERS = 2


@pytest.fixture(autouse=True)
def _isolate_metadata_hub():
    """Reset the process-wide hub so an earlier test's stats can't satisfy an assertion."""
    from batcher.core import reset_default_hub

    reset_default_hub()
    yield
    reset_default_hub()


def _kinds() -> set[str]:
    from batcher.core import default_hub

    return set(default_hub().op_stats_by_kind())


def _rows(kind: str) -> list[dict]:
    from batcher.core import default_hub

    return default_hub().op_stats_by_kind().get(kind, [])


def _source(n: int = 4096) -> pa.Table:
    return pa.table(
        {
            "k": [i % 32 for i in range(n)],
            "g": [i % 7 for i in range(n)],
            "v": [float(i % 101) for i in range(n)],
        }
    )


@pytest.mark.integration
def test_distributed_sort_records_operator_feedback():
    got = bt.from_arrow(_source()).sort("v").collect(distributed=True, num_workers=_WORKERS)
    assert got.num_rows == 4096
    assert "sort" in _kinds(), f"the distributed sort learned nothing; hub saw {_kinds()}"
    assert all(r["t_op_ms"] >= 0.0 for r in _rows("sort"))


@pytest.mark.integration
def test_distributed_join_records_build_side_and_peak():
    left = bt.from_arrow(_source())
    right = bt.from_arrow(pa.table({"k": list(range(32)), "label": [f"L{i}" for i in range(32)]}))
    got = left.join(right, on="k").collect(distributed=True, num_workers=_WORKERS)
    assert got.num_rows == 4096
    assert "hash_join" in _kinds(), f"the distributed join learned nothing; hub saw {_kinds()}"
    rows = _rows("hash_join")
    # `n_build` and `m_peak_bytes` are exactly the two facts the learned memory model
    # regresses on. A distributed join reporting neither cannot inform spilling.
    assert any(r["n_build"] > 0 for r in rows), "no build-side row count reached the hub"
    assert any(r["m_peak_bytes"] > 0 for r in rows), "no peak working set reached the hub"


@pytest.mark.integration
def test_distributed_shuffle_join_records_feedback_and_matches_single_node():
    """The co-partition reduce is the hash-join breaker; a broadcast join never reaches it.

    Forcing `broadcast_max_bytes` down routes the same query through the shuffle path, so
    this covers the reducer that carries the join's `rows_build` and `peak_bytes`.
    """
    base = Config()
    forced = dataclasses.replace(
        base, optimizer=dataclasses.replace(base.optimizer, broadcast_max_bytes=1)
    )
    left = bt.from_arrow(_source())
    right = bt.from_arrow(pa.table({"k": list(range(32)), "label": [f"L{i}" for i in range(32)]}))
    with config_context(forced):
        got = left.join(right, on="k").collect(distributed=True, num_workers=_WORKERS)
    # Snapshot before the single-node comparison run — it records into the same
    # process-wide hub and would otherwise double the build-row count.
    rows = _rows("hash_join")
    expected = left.join(right, on="k").collect()
    assert got.num_rows == expected.num_rows == 4096
    assert rows, f"the distributed shuffle join learned nothing; hub saw {_kinds()}"
    # Co-partitioning splits both sides across reducers, so the build rows sum to the 32
    # distinct keys. That is what proves these came from the shuffle reduce rather than a
    # broadcast probe, where every task sees all 32.
    assert sum(r["n_build"] for r in rows) == 32
    assert sum(r["n_actual"] for r in rows) == 4096


@pytest.mark.integration
def test_distributed_window_records_operator_feedback():
    ds = bt.from_arrow(_source()).with_columns(s=bt.col("v").sum().over("g"))
    got = ds.collect(distributed=True, num_workers=_WORKERS)
    assert got.num_rows == 4096
    assert "window" in _kinds(), f"the distributed window learned nothing; hub saw {_kinds()}"


@pytest.mark.integration
def test_distributed_distinct_records_operator_feedback():
    ds = bt.from_arrow(_source()).select("k", "g").distinct()
    got = ds.collect(distributed=True, num_workers=_WORKERS)
    assert got.num_rows > 0
    # DISTINCT rides the aggregate shuffle, so what the workers measure is its map sub-plan.
    # The claim is only that the stage fed the loop at all.
    assert _kinds(), "the distributed distinct learned nothing at all"


@pytest.mark.integration
def test_stage_row_count_reads_every_result_shape():
    """The adaptive loop must see a distributed stage's size, not only a collected table."""
    from batcher.api.adaptive import _stage_row_count
    from batcher.io.source import InMemorySource

    batch = pa.record_batch([pa.array([1, 2, 3])], names=["a"])
    assert _stage_row_count(pa.table({"a": [1, 2, 3]})) == 3
    assert _stage_row_count(InMemorySource([batch])) == 3

    class _Materialized:  # what a distributed stage parks
        def row_count(self):
            return 4096

    class _Streaming:  # unbounded: no exact count to check an estimate against
        def row_count(self):
            return None

    assert _stage_row_count(_Materialized()) == 4096
    assert _stage_row_count(_Streaming()) is None
    assert _stage_row_count(object()) is None


@pytest.mark.integration
def test_distributed_adaptive_records_a_flip(monkeypatch):
    """`learned_adaptive_helps` must be reachable on the distributed path.

    A distributed stage parks a `MaterializedSource`, so the old `isinstance(result,
    pa.Table)` accuracy check never ran and `flipped` stayed `False` forever — adaptivity
    could not learn from the very shapes that most need it. Forcing a wildly wrong estimate
    makes a flip mandatory *if* the check runs at all.
    """
    import batcher.api.adaptive as adaptive

    monkeypatch.setattr(adaptive, "_estimate_rows", lambda *a, **k: 999_999_999)
    recorded: dict[str, bool] = {}
    original = adaptive._record_adaptive_flip
    monkeypatch.setattr(
        adaptive,
        "_record_adaptive_flip",
        lambda hub, plan, flipped: (
            recorded.__setitem__("flipped", flipped),
            original(hub, plan, flipped),
        )[1],
    )

    fact = bt.from_arrow(_source(200_000))
    dim = bt.from_arrow(pa.table({"k": list(range(32)), "lbl": [f"d{i}" for i in range(32)]}))
    staged = fact.filter(bt.col("v") < 3.0).group_by("k").agg(s=bt.col("v").sum())
    got = staged.join(dim, on="k").collect(distributed=True, num_workers=_WORKERS, adaptive=True)

    assert got.num_rows == 32
    assert recorded.get("flipped") is True, "the distributed stage's size never reached the check"


@pytest.mark.integration
def test_metering_degrades_when_the_engine_lacks_the_metered_entry_point(monkeypatch):
    """A worker must never fail a query in order to collect a statistic."""
    import batcher._native as nat

    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import execute_metered

    monkeypatch.delattr(nat, "execute_plan_metered", raising=False)
    batch = pa.record_batch([pa.array([1, 2, 3])], names=["a"])
    out, metrics = execute_metered(
        '{"op": "scan", "source_id": 0}', [[batch]], active_config().engine_config_json()
    )
    assert sum(b.num_rows for b in out) == 3
    assert metrics == ""


@pytest.mark.integration
def test_record_worker_metrics_survives_a_malformed_document():
    from batcher.core import default_hub
    from batcher.dist.executors.ray_runtime import record_worker_metrics

    hub = default_hub()
    before = sum(len(v) for v in hub.op_stats_by_kind().values())
    record_worker_metrics(hub, ["", "not json", '{"ops": []}'])
    assert sum(len(v) for v in hub.op_stats_by_kind().values()) == before
