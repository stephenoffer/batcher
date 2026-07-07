"""Learned per-family memory sizing — Carbonite consumes measured `m_peak_bytes`.

The headline gap these cover: `OperatorFeedback.m_peak_bytes` is recorded to the hub
but was never consumed — admission / spill / reservation / per-task grants all sized
from the PLAN estimate alone. `carbonite.memory.learned.LearnedMemoryModel` fits a
per-operator-family bytes-per-input-row figure from those measured peaks, and the
sizing decisions blend the plan estimate toward it. Each test proves both halves of the
contract: the tuned parameter moves when the hub has measured stats, and a cold store
falls back to the exact plan-only behavior. The result-invariance test proves the
lever these decisions drive (in-memory vs the out-of-core spill route) yields a
byte-identical result either way.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import Config, col, config_context
from batcher.carbonite import ResourceManager
from batcher.carbonite.base import ResourceContext
from batcher.carbonite.memory.learned import learned_memory_model
from batcher.carbonite.policies import BudgetingAdmission, DefaultSchedulingPolicy
from batcher.config import active_config
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds

# Default per-row assumption the plan estimate uses; a measured wider footprint scales up.
_ROW_BYTES = active_config().optimizer.row_bytes


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _seed(
    hub: MetadataHub, kind: str, bytes_per_row: float, *, n: int = 30, rows: int = 1000
) -> None:
    """Record `n` measured peaks for `kind` so its learned bytes-per-row == `bytes_per_row`."""
    for _ in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind=kind,
                n_actual=rows // 10,
                t_op_ms=1.0,
                m_peak_bytes=int(bytes_per_row * rows),
                selectivity=0.1,
                batch_size=16384,
                n_input=rows,
            )
        )


def _plan(kind: str, m_max_bytes: int) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(1),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=m_max_bytes, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


# --- the learned model itself ------------------------------------------------


def test_bytes_per_row_learned_from_measured_peaks():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=256.0)
    model = learned_memory_model(hub)
    # Plan-op kind "Aggregate" (LogicalPlan class name) resolves the native "aggregate" stats.
    assert model.bytes_per_row("Aggregate") == 256.0
    assert model.max_bytes_per_row() == 256.0


def test_cold_store_is_pass_through():
    model = learned_memory_model(_hub())  # no feedback recorded
    assert model.bytes_per_row("Aggregate") is None
    assert model.max_bytes_per_row() is None
    # A pass-through model returns the plan estimate untouched.
    assert model.blend_peak("Aggregate", 1_000_000) == 1_000_000
    assert learned_memory_model(None).blend_peak("Aggregate", 777) == 777


def test_blend_scales_toward_measured_and_clamps():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)  # 4x the assumed width
    model = learned_memory_model(hub)
    # measured peak = 4x plan; alpha=0.5 → halfway between plan (1x) and measured (4x) = 2.5x.
    assert model.blend_peak("Aggregate", 1_000_000) == 2_500_000
    # An unsized op (plan estimate 0) has nothing to rescale — stays 0.
    assert model.blend_peak("Aggregate", 0) == 0


def test_min_samples_floor():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=256.0, n=3)  # below the sample floor
    assert learned_memory_model(hub).bytes_per_row("Aggregate") is None


# --- decisions that consume the model ---------------------------------------


def test_estimated_bytes_blends_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)
    plan = _plan("Aggregate", 1_000_000)
    assert ResourceManager().estimated_bytes(plan) == 1_000_000  # cold: plan only
    assert ResourceManager(hub=hub).estimated_bytes(plan) == 2_500_000  # warm: blended up


def test_should_spill_uses_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 8)  # 8x → blended 4.5x
    # A plan the cold manager keeps in memory (estimate under budget) spills once the
    # learned peak (much larger) is blended in — Carbonite protects from a real OOM.
    cfg = Config().replace(memory=active_config().memory)
    with config_context(cfg):
        rm_cold = ResourceManager()
        budget = rm_cold._hard_budget()
        # size the plan so cold estimate fits but the 4.5x blend does not
        plan = _plan("Aggregate", int(budget * 0.5))
        assert rm_cold.should_spill(plan) is False
        assert ResourceManager(hub=hub).should_spill(plan) is True


def test_should_spill_cold_never_spills_unsized():
    assert ResourceManager(hub=_hub()).should_spill(_plan("Aggregate", 0)) is False


def test_admission_blends_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)
    model = learned_memory_model(hub)
    # Envelope = 3 MB. Plan estimate 1 MB fits; the 2.5 MB blended peak still fits →
    # feasible. A plan whose blend crosses the envelope becomes infeasible.
    ctx_cold = ResourceContext(config=active_config(), envelope_bytes=None)
    ctx_warm = ResourceContext(config=active_config(), envelope_bytes=None, memory_model=model)
    adm = BudgetingAdmission(available_bytes=3_000_000, soft_limit=1.0)
    plan = _plan("Aggregate", 1_500_000)  # cold fits (1.5<3); blended 3.75M exceeds 3M
    assert adm.validate(plan, ctx_cold).feasible is True
    assert adm.validate(plan, ctx_warm).feasible is False


def test_scheduling_envelope_memory_from_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)
    model = learned_memory_model(hub)
    plan = _plan("Aggregate", 4_000_000)
    pol = DefaultSchedulingPolicy()
    cold = pol.envelope(
        plan,
        ResourceContext(config=active_config()),
        requested_workers=1,
        available_bytes=1 << 40,
    )
    warm = pol.envelope(
        plan,
        ResourceContext(config=active_config(), memory_model=model),
        requested_workers=1,
        available_bytes=1 << 40,
    )
    # The per-task grant tracks the blended (larger) peak, so a distributed worker is
    # right-sized to what the family really used rather than the plan guess.
    assert warm.memory_bytes > cold.memory_bytes


# --- result-invariance -------------------------------------------------------


def _rows(tbl: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in tbl.to_pylist())


def test_spill_route_is_result_invariant():
    # The lever learned should_spill selects between (in-memory vs out-of-core) must
    # yield a byte-identical result — that is what makes learned spill sizing safe.
    t = pa.table({"k": [i % 11 for i in range(6000)], "v": list(range(6000))})

    def q(**kw):
        return (
            bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=col("v").count()).collect(**kw)
        )

    in_memory = q()
    spilled = q(spill=True, num_partitions=8)
    assert _rows(in_memory) == _rows(spilled)
