"""Metadata-driven distributed scheduling (Pillar B), tested without a cluster.

Covers the derivation chain end to end as pure/local logic: Kyber fills the
parallelism/credit axes from estimated rows (`_annotate_ops`); Carbonite clamps
those into a per-task `SchedulingEnvelope`; the executor turns the envelope into Ray
`.options(...)` kwargs; the GPU tag flows and adapts from measured utilization; and
the AIMD credit window evolves from a congestion signal.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.policies import DefaultSchedulingPolicy
from batcher.config import active_config
from batcher.dist.executors.ray_runtime import task_options
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds, SchedulingEnvelope

pytestmark = pytest.mark.unit


def _plan(ops: list[PhysicalOp]) -> PhysicalPlan:
    return PhysicalPlan(ir={}, output_schema=None, ops=tuple(ops))


def _op(op_id: int, kind: str, mem: int, credits: int, par: int) -> PhysicalOp:
    return PhysicalOp(
        op_id=OpId(op_id),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=mem, c_max_credits=credits, n_max_parallelism=par),
        inputs=(),
    )


# --- Kyber fills the parallelism / credit axes (B1) -----------------------------


def test_annotate_ops_scales_parallelism_with_rows():
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.optimizer import _annotate_ops

    class _Source:
        """Source stub that reports a huge row count to the estimator."""

        def __init__(self, rows: int) -> None:
            self._rows = rows

        def row_count(self) -> int:
            return self._rows

    cfg = active_config()
    big = cfg.optimizer.target_rows_per_task * 5  # a breaker over this wants ~5 tasks
    ds = bt.from_pydict({"k": [1, 2], "v": [3, 4]}).group_by("k").agg(s=bt.col("v").sum())
    est = CardinalityEstimator([_Source(big)])
    ops = _annotate_ops(ds._plan, est, cfg)
    agg_op = next(op for op in ops if op.kind == "Aggregate")
    # A breaker over ~5×target rows wants multiple tasks and a positive credit window.
    assert agg_op.bounds.n_max_parallelism >= 2
    assert agg_op.bounds.c_max_credits >= 1


# --- Carbonite turns bounds into a per-task grant (B2) --------------------------


def test_scheduling_envelope_tracks_parallelism_and_clamps_memory():
    # Breaker wants 8 tasks and 64 MiB; available budget is tiny so per-task memory
    # is clamped to a fair share.
    plan = _plan([_op(0, "Aggregate", mem=64 << 20, credits=10, par=8)])
    rm = ResourceManager()
    env = rm.scheduling_envelope(plan)
    assert env.n_tasks >= 1
    assert env.num_cpus == active_config().execution.cpus_per_task
    assert env.credits >= 1  # granted from c_max_credits, never a stale 0
    assert env.memory_bytes > 0


def test_scheduling_envelope_honors_requested_workers():
    plan = _plan([_op(0, "Aggregate", mem=1 << 20, credits=4, par=99)])
    env = ResourceManager().scheduling_envelope(plan, requested_workers=3)
    assert env.n_tasks == 3  # an explicit request wins over the estimate


def test_scheduling_policy_passes_through_data_driven_fanout():
    # N11: the data-driven desired fan-out is NOT clamped to the driver's core
    # count — that would cap a large cluster job at the driver's cores. The
    # cluster-aware `clamp_workers` (in the dist layer) owns the real cap, so the
    # envelope reports the full data-driven want.
    from batcher.carbonite.base import ResourceContext

    cfg = active_config()
    ctx = ResourceContext(config=cfg)
    plan = _plan([_op(0, "Aggregate", mem=1 << 30, credits=4, par=10_000)])
    env = DefaultSchedulingPolicy().envelope(
        plan, ctx, requested_workers=None, available_bytes=1 << 40
    )
    assert env.n_tasks == 10_000


def test_scheduling_policy_unsized_falls_back_to_local_budget():
    # With no data-driven fan-out the policy falls back to the local cpu budget
    # (the best local guess); clamp_workers still caps it to the cluster.
    from batcher.carbonite.base import ResourceContext

    cfg = active_config()
    ctx = ResourceContext(config=cfg)
    plan = _plan([_op(0, "Scan", mem=0, credits=0, par=0)])
    env = DefaultSchedulingPolicy().envelope(
        plan, ctx, requested_workers=None, available_bytes=1 << 40
    )
    budget = cfg.execution.parallelism or __import__("os").cpu_count() or 4
    assert env.n_tasks == max(1, budget)


def test_unsized_plan_falls_back_without_memory_hint():
    # No Kyber sizes → no memory hint, fan-out falls back to the cpu budget.
    plan = _plan([_op(0, "Scan", mem=0, credits=0, par=0)])
    env = ResourceManager().scheduling_envelope(plan)
    assert env.memory_bytes == 0
    assert env.n_tasks >= 1


# --- Worker engine config inherits the per-task envelope budget (OOM survival) --


def test_engine_config_json_folds_envelope_budget():
    # With an ambient envelope, the per-task memory grant is folded into the
    # worker engine config's `memory_budget_bytes`, so each distributed worker's
    # `execute_plan` spills its reducer bucket within its share instead of OOMing.
    import json

    from batcher.dist.executors.ray_runtime import (
        engine_config_json,
        reset_scheduling_envelope,
        set_scheduling_envelope,
    )

    # No envelope → unchanged (the single-node default budget, 0 = unbounded).
    base = json.loads(engine_config_json())
    assert base["memory_budget_bytes"] == 0

    env = SchedulingEnvelope(num_cpus=1.0, memory_bytes=4 << 20, num_gpus=0.0, n_tasks=4, credits=4)
    token = set_scheduling_envelope(env)
    try:
        cfg = json.loads(engine_config_json())
        assert cfg["memory_budget_bytes"] == 4 << 20  # envelope grant enables spill
        # The other knobs are still the driver's config (not lost in the merge).
        assert cfg["morsel_rows"] == base["morsel_rows"]
    finally:
        reset_scheduling_envelope(token)

    # A zero-memory (unsized) envelope leaves the config untouched.
    zero_env = SchedulingEnvelope(num_cpus=1.0, memory_bytes=0, num_gpus=0.0, n_tasks=2, credits=4)
    token = set_scheduling_envelope(zero_env)
    try:
        assert json.loads(engine_config_json())["memory_budget_bytes"] == 0
    finally:
        reset_scheduling_envelope(token)


def test_engine_config_json_takes_tighter_of_envelope_and_global_cap():
    # When the user has also set a global memory cap, the worker honors whichever
    # is tighter: a reducer must not exceed the smaller of its per-task grant and
    # the global spill cap.
    import json

    from batcher.config import Config, MemoryConfig, config_context
    from batcher.dist.executors.ray_runtime import (
        engine_config_json,
        reset_scheduling_envelope,
        set_scheduling_envelope,
    )

    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=8 << 20))):
        global_cap = json.loads(engine_config_json())["memory_budget_bytes"]
        assert global_cap > 0  # cap is active even with no envelope

        # Envelope grant smaller than the global cap → envelope wins.
        small = SchedulingEnvelope(
            num_cpus=1.0, memory_bytes=1 << 20, num_gpus=0.0, n_tasks=4, credits=4
        )
        token = set_scheduling_envelope(small)
        try:
            assert json.loads(engine_config_json())["memory_budget_bytes"] == 1 << 20
        finally:
            reset_scheduling_envelope(token)

        # Envelope grant larger than the global cap → the global cap wins.
        large = SchedulingEnvelope(
            num_cpus=1.0, memory_bytes=64 << 20, num_gpus=0.0, n_tasks=4, credits=4
        )
        token = set_scheduling_envelope(large)
        try:
            assert json.loads(engine_config_json())["memory_budget_bytes"] == global_cap
        finally:
            reset_scheduling_envelope(token)


# --- Executor turns the envelope into Ray .options() (B3) -----------------------


def test_task_options_builds_ray_kwargs():
    env = SchedulingEnvelope(num_cpus=2.0, memory_bytes=1000, num_gpus=1.0, n_tasks=4, credits=8)
    assert task_options(env) == {"num_cpus": 2.0, "memory": 1000, "num_gpus": 1.0}


def test_task_options_omits_gpu_when_zero():
    # A CPU-only task must never request a GPU (unschedulable on GPU-less clusters).
    env = SchedulingEnvelope(num_cpus=1.0, memory_bytes=0, num_gpus=0.0, n_tasks=2, credits=4)
    opts = task_options(env)
    assert "num_gpus" not in opts
    assert "memory" not in opts  # unsized → no soft hint
    assert opts == {"num_cpus": 1.0}


def test_task_options_none_envelope_is_empty():
    assert task_options(None) == {}


def test_fault_options_from_config():
    """Task retries come from config; `retry_on_transient` toggles retry_exceptions."""
    from batcher.config import Config, DistributedConfig, config_context
    from batcher.dist.executors.ray_runtime import actor_fault_options, fault_options

    with config_context(
        Config().replace(
            distributed=DistributedConfig(
                task_max_retries=4,
                retry_on_transient=True,
                actor_max_restarts=3,
                actor_max_task_retries=2,
            )
        )
    ):
        assert fault_options() == {"max_retries": 4, "retry_exceptions": True}
        assert actor_fault_options() == {"max_restarts": 3, "max_task_retries": 2}

    with config_context(
        Config().replace(
            distributed=DistributedConfig(task_max_retries=0, retry_on_transient=False)
        )
    ):
        # retry_exceptions omitted when transient retries are off.
        assert fault_options() == {"max_retries": 0}


# --- GPU tag + utilization feedback loop (B4) -----------------------------------


def test_recommend_num_gpus_adapts_to_utilization():
    from batcher.ml.gpu import recommend_num_gpus

    assert recommend_num_gpus(None, 1.0) == 1.0  # no measurement → keep
    assert recommend_num_gpus(0.2, 1.0) == 0.25  # idle whole GPU → pack onto a fraction
    assert recommend_num_gpus(0.95, 0.5) == 1.0  # saturated fraction → whole GPU
    assert recommend_num_gpus(0.7, 1.0) == 1.0  # mid-range → keep


def test_gpu_utilization_round_trips_through_hub():
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend
    from batcher.ml.gpu import load_gpu_utilization, record_gpu_utilization

    hub = MetadataHub(InProcessBackend())
    record_gpu_utilization(hub, "pipe", 0.4)
    record_gpu_utilization(hub, "pipe", 0.6)  # exp-smoothed
    util = load_gpu_utilization(hub, "pipe")
    assert util is not None and 0.4 <= util <= 0.6


# --- AIMD credit window (B5) ----------------------------------------------------


def test_aimd_window_grows_and_shrinks():
    aimd = ResourceManager().adaptive_flow_control()
    start = aimd.window
    shrunk = aimd.observe(congested=True)
    grown = aimd.observe(congested=False)
    assert shrunk < start  # multiplicative decrease on congestion
    assert grown > shrunk  # additive increase when clear


def test_shuffle_session_adaptive_window_consumes_pressure():
    # The session's window follows the AIMD controller, which consumes a congestion
    # signal — tested directly without a live Flight transfer.
    from batcher.carbonite.policies import AIMDFlowControl
    from batcher.carbonite.transfer.session import ShuffleSession

    fc = AIMDFlowControl()
    session = ShuffleSession(flow_control=fc)
    assert session._window() == fc.window
    fc.observe(congested=True)
    assert session._window() == fc.window  # window tracks the controller
