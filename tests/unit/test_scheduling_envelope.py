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


def test_task_options_carries_fractional_cpu():
    # A CPU-light stage asks for <1 CPU so Ray packs more than one per core.
    env = SchedulingEnvelope(num_cpus=0.5, memory_bytes=0, num_gpus=0.0, n_tasks=4, credits=4)
    assert task_options(env) == {"num_cpus": 0.5}


# --- Fractional CPU: per-operator share → envelope → clamp ----------------------


def _cpu_op(op_id: int, kind: str, cpu: float) -> PhysicalOp:
    return PhysicalOp(
        op_id=OpId(op_id),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(
            m_max_bytes=0, c_max_credits=0, n_max_parallelism=0, c_cpu_shares=cpu
        ),
        inputs=(),
    )


def test_envelope_cpu_is_dominant_operator_share():
    from batcher.carbonite.base import ResourceContext

    cfg = active_config()
    ctx = ResourceContext(config=cfg)
    # A pure scan→filter plan asks for the CPU-light fraction (packs tighter).
    light = _plan([_cpu_op(0, "Scan", 0.5), _cpu_op(1, "Filter", 0.5)])
    env = DefaultSchedulingPolicy().envelope(
        light, ctx, requested_workers=None, available_bytes=1 << 40
    )
    assert env.num_cpus == 0.5
    # Add a breaker (full core) and the dominant share pulls back to 1.0.
    heavy = _plan([_cpu_op(0, "Scan", 0.5), _cpu_op(1, "Aggregate", 1.0)])
    env = DefaultSchedulingPolicy().envelope(
        heavy, ctx, requested_workers=None, available_bytes=1 << 40
    )
    assert env.num_cpus == 1.0


def test_kyber_annotates_cpu_light_with_fraction():
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.optimizer import _annotate_ops

    cfg = active_config()
    ds = bt.from_pydict({"k": [1, 2], "v": [3, 4]}).filter(bt.col("v") > 0)
    ops = _annotate_ops(ds._plan, CardinalityEstimator([]), cfg)
    filt = next(op for op in ops if op.kind == "Filter")
    scan = next(op for op in ops if op.kind == "Scan")
    assert filt.bounds.c_cpu_shares == cfg.execution.cpu_share_io
    assert scan.bounds.c_cpu_shares == cfg.execution.cpu_share_io


def test_recommend_num_cpus_adapts_to_utilization():
    from batcher.kyber.cpu_shares import recommend_num_cpus

    # No measurement → keep the static prior.
    assert recommend_num_cpus(None, 0.5) == 0.5
    assert recommend_num_cpus(0.0, 0.5) == 0.5
    # CPU-bound family → near a whole core (overrides a light prior).
    assert recommend_num_cpus(0.95, 0.5) == 0.95
    # IO-bound family → floored at cpu_share_min (never an unschedulable sliver).
    assert recommend_num_cpus(0.05, 1.0) == active_config().execution.cpu_share_min
    # Never exceeds a whole core.
    assert recommend_num_cpus(1.5, 0.5) == active_config().execution.cpus_per_task


def test_load_cpu_utilization_medians_by_kind():
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend
    from batcher.plan.feedback import OperatorFeedback
    from batcher.plan.ids import OpId

    hub = MetadataHub(InProcessBackend())
    n = active_config().optimizer.cost_calibration_min_samples
    for _ in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(0),
                kind="filter",
                n_actual=100,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                cpu_utilization=0.9,
            )
        )
    # An unmeasured (0.0) row must not pull the learned value down.
    hub.record(
        OperatorFeedback(
            op_id=OpId(1),
            kind="filter",
            n_actual=1,
            t_op_ms=1.0,
            m_peak_bytes=0,
            selectivity=1.0,
            batch_size=1,
            cpu_utilization=0.0,
        )
    )
    from batcher.kyber.cpu_shares import load_cpu_utilization

    util = load_cpu_utilization(hub)
    assert util["filter"] == 0.9
    # A kind with too few samples is absent (keeps its static prior).
    assert "scan" not in util


def test_annotate_ops_overrides_static_cpu_with_learned():
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.optimizer import _annotate_ops

    cfg = active_config()
    ds = bt.from_pydict({"k": [1, 2], "v": [3, 4]}).filter(bt.col("v") > 0)
    # Cold start: Filter keeps the CPU-light prior.
    cold = _annotate_ops(ds._plan, CardinalityEstimator([]), cfg)
    filt_cold = next(op for op in cold if op.kind == "Filter")
    assert filt_cold.bounds.c_cpu_shares == cfg.execution.cpu_share_io
    # Learned: a CPU-bound filter (regex-heavy) measured at 0.95 → near a whole core.
    warm = _annotate_ops(ds._plan, CardinalityEstimator([]), cfg, {"filter": 0.95})
    filt_warm = next(op for op in warm if op.kind == "Filter")
    assert filt_warm.bounds.c_cpu_shares == 0.95


def test_learned_cpu_flows_through_optimizer_into_envelope():
    # The whole adaptive CPU loop, end to end through the *real* Optimizer and
    # ResourceManager: recorded utilization → load_cpu_utilization (inside optimize)
    # → _annotate_ops c_cpu_shares → envelope.num_cpus. Proves the class→IR-tag bridge
    # and the wiring actually fire (not just the pieces in isolation).
    import batcher as bt
    from batcher.carbonite import ResourceManager
    from batcher.kyber import optimize
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend
    from batcher.plan.feedback import OperatorFeedback
    from batcher.plan.ids import OpId

    cfg = active_config()
    hub = MetadataHub(InProcessBackend())
    # A CPU-bound filter family, measured well above its static 0.5 prior.
    for _ in range(cfg.optimizer.cost_calibration_min_samples):
        hub.record(
            OperatorFeedback(
                op_id=OpId(0),
                kind="filter",
                n_actual=1,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1,
                cpu_utilization=0.9,
            )
        )
    ds = bt.from_pydict({"k": [1, 2, 3], "v": [4, 5, 6]}).filter(bt.col("v") > 0)
    phys = optimize(ds._plan, sources=ds._sources, hub=hub)
    filt = next(op for op in phys.ops if op.kind == "Filter")
    assert filt.bounds.c_cpu_shares == 0.9, "learned utilization overrode the static prior"
    # And it reaches the per-task grant Carbonite hands the distributed executor.
    env = ResourceManager().scheduling_envelope(phys)
    assert env.num_cpus >= 0.9


def test_record_exec_metrics_computes_cpu_utilization():
    import json

    from batcher.core import record_exec_metrics
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    elapsed = 1_000_000  # 1 ms wall
    threads = 8  # the engine's reported live pool size — NOT a guessed host count
    cpu = int(0.5 * elapsed * threads)  # half the allocated cores kept busy
    doc = {
        "ops": [
            {
                "op_id": 0,
                "kind": "filter",
                "rows_in": 10,
                "rows_out": 5,
                "elapsed_ns": elapsed,
                "cpu_ns": cpu,
                "threads": threads,
                "peak_bytes": 0,
                "spilled": False,
                "backend": "interp",
            }
        ]
    }
    record_exec_metrics(hub, json.dumps(doc), batch_size=16384)
    row = hub.op_stats_by_kind()["filter"][0]
    assert abs(row["cpu_utilization"] - 0.5) < 1e-9
    # A document without cpu_ns (older engine) records 0.0 (unmeasured), never crashes.
    doc["ops"][0].pop("cpu_ns")
    record_exec_metrics(hub, json.dumps(doc), batch_size=16384)
    assert hub.op_stats_by_kind()["filter"][1]["cpu_utilization"] == 0.0


# --- Fractional GPU rides the same float plumbing as CPU ------------------------


def test_fractional_gpu_round_trips_through_options_and_bundle():
    from batcher.dist.executors.ray_runtime import _bundle

    env = SchedulingEnvelope(num_cpus=0.5, memory_bytes=0, num_gpus=0.25, n_tasks=2, credits=4)
    # Ray scheduling kwargs carry the fractional GPU verbatim (so 4 actors pack a GPU).
    assert task_options(env) == {"num_cpus": 0.5, "num_gpus": 0.25}
    # The placement-group bundle reserves the same fractional GPU + CPU.
    assert _bundle(env) == {"CPU": 0.5, "GPU": 0.25}


def test_clamp_workers_packs_fractional_cpu(monkeypatch):
    ray = pytest.importorskip("ray")
    from batcher.dist.executors import ray_runtime

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray_runtime, "cluster_topology", lambda: {"nodes": 1, "cpus": 8.0, "gpus": 0.0}
    )
    # 0.5 CPU/task → 16 tasks fit on 8 cores; 1.0 reproduces today's count; 2.0 → 4.
    assert ray_runtime.clamp_workers(16, 0.5) == 16
    assert ray_runtime.clamp_workers(20, 0.5) == 16  # over-subscribed → clamp to 16
    assert ray_runtime.clamp_workers(8, 1.0) == 8
    assert ray_runtime.clamp_workers(10, 1.0) == 8  # the pre-existing 1-CPU behavior
    assert ray_runtime.clamp_workers(8, 2.0) == 4


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
