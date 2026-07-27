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
def test_distributed_adaptive_records_its_route():
    """The staged-vs-one-shot bandit must be fed by the distributed path too.

    This is the same guarantee an earlier flip-counter test pinned, restated for the
    mechanism that replaced it. A distributed stage parks a `MaterializedSource` rather
    than collecting a table, and the previous accuracy check keyed on `isinstance(result,
    pa.Table)` — so adaptivity could not learn from the very shapes that most need it. The
    route bandit reads wall time at the `collect` seam, which has no such shape dependence;
    what this asserts is that the distributed run reaches it at all.
    """
    from batcher import core
    from batcher.kyber.learned_tuning import learned_adaptive_route
    from batcher.kyber.signature import plan_signature

    fact = bt.from_arrow(_source(200_000))
    dim = bt.from_arrow(pa.table({"k": list(range(32)), "lbl": [f"d{i}" for i in range(32)]}))
    staged = fact.filter(bt.col("v") < 3.0).group_by("k").agg(s=bt.col("v").sum())
    query = staged.join(dim, on="k")

    hub = core.default_hub()
    sig = plan_signature(query._plan)
    for _ in range(2):
        got = query.collect(distributed=True, num_workers=_WORKERS, adaptive=True)
        assert got.num_rows == 32

    assert learned_adaptive_route(hub, sig) is not None, (
        "the distributed adaptive run never reached the route bandit"
    )


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
