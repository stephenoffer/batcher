"""Scheduling: turn Kyber's per-operator bounds into a per-Ray-task resource envelope.

Worker fan-out, per-task memory, per-task CPU, and placement preference all come from what
the plan says the data is, rather than from `os.cpu_count()`. `dist` consumes the envelope
and reconciles it against the live cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.hardware import available_cpu_count
from batcher.carbonite.memory.estimator import learned_plan_peak
from batcher.plan.resource import SchedulingEnvelope

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.plan.physical import PhysicalPlan

__all__ = ["DefaultSchedulingPolicy"]

# Absolute ceiling on the worker fan-out this policy will *ask* for.
#
# `n_max_parallelism` is Kyber's `rows / target_rows_per_task`, and Kyber stamps an
# `unknown_rows` placeholder (~1e12) on any operator it could not size — which turns into
# a request for millions of Ray tasks. `dist.clamp_workers` reduces the figure to live
# cluster capacity and so hides the consequence on a real cluster, but the number still
# travels through the envelope: it is logged, it is compared, and it divides the per-task
# memory grant (`peak // n_tasks`) down to the morsel floor, which is how an un-sized plan
# ends up asking Ray for a million tasks of one morsel each.
#
# 100k is far above any real *narrow-row* fan-out (a 10k-core cluster at ten tasks per core)
# and far below the placeholder, so it normally only catches the pathological case.
#
# **It is no longer unreachable by a legitimate plan.** Since Kyber's desired parallelism
# became byte-aware (`annotate._desired_parallelism`), a genuinely wide relation asks for
# tasks in proportion to its bytes: a petabyte of 1080p video frames at 256 MiB per task
# wants around a million. Clamping there is still the right *default* — a fan-out that large
# is a shuffle with 10^12 fragments and needs a different plan, not more tasks — but the
# clamp now costs oversized tasks rather than merely trimming nonsense, so the cap being hit
# is a signal about the plan and not only about the estimate. Recorded rather than raised:
# what the ceiling should be on a real fleet is a measurement, and `dist.clamp_workers`
# reduces the figure to live cluster capacity downstream in any case.
_MAX_TASK_FANOUT = 100_000


def _healthy_devices(gpu_count: int) -> int:
    """Devices that passed the health verdicts, or `gpu_count` when health checking is off.

    Absent telemetry reports `None` and is treated as "no opinion" rather than as an unhealthy
    fleet: the alternative takes a cluster offline the day `pynvml` stops being installed.
    """
    from batcher.carbonite.accel.health import schedulable_device_count
    from batcher.config import active_config

    if not active_config().accelerator.health.enabled:
        return gpu_count
    healthy = schedulable_device_count()
    return gpu_count if healthy is None else max(1, healthy)


class DefaultSchedulingPolicy:
    """Derive a per-Ray-task `SchedulingEnvelope` from Kyber's per-operator bounds.

    This is where worker fan-out stops being a blind `os.cpu_count()` and starts
    tracking the data: a breaker's `n_max_parallelism` (≈ rows / target-rows-per-task)
    sets the desired task count, clamped to the machine's cpu budget. Per-task memory
    is the dominant breaker's footprint split across those tasks (each holds one
    partition's share), clamped to a fair slice of the live budget so a soft Ray
    `memory=` hint never over-asks. `num_cpus` is the configured per-task share; GPUs
    are 0 here (the GPU map/inference path sets its own `num_gpus`). Credits are filled
    by the manager from its flow-control policy.
    """

    @staticmethod
    def gpu_envelope(
        *,
        num_gpus: float,
        n_tasks: int,
        gpu_count: int,
        accelerator_type: str | None = None,
    ) -> SchedulingEnvelope:
        """Budget a GPU map/inference stage against the GPUs that actually exist.

               The relational `envelope` below is the CPU shuffle grant and correctly requests no
               GPU. This closes the gap that GPU demand previously reached Ray *only* as the raw
               `map_batches(num_gpus=)` tag, so Carbonite — the layer that decides feasibility —
               never saw it, and an infeasible request hung instead of erroring. The grant is
               clamped to inventory: `gpu_count / num_gpus` tasks can hold a GPU at once, and a
               cluster reporting no GPUs gets no GPU grant (the stage runs on CPU rather than
               pending forever). Fractional `num_gpus` is preserved — packing four 0.25-GPU
               actors onto one device is the point of a fractional request.

               Inventory is not the only ceiling, and neither is the count of devices that exist.
               With `accelerator.health` enabled the grant is also clamped to the devices that are
               *schedulable*: one reporting uncorrectable ECC errors returns wrong tensors, and one
               the driver has clamped contributes a fraction of a healthy device while drawing most
               of its power. Granting against the raw count places work on both.
        Where a power budget is configured
               (`accelerator.energy.power_budget_watts`) it is frequently the tighter one: a rack of
               sixteen 700 W devices needs more than eleven kilowatts of device power alone, more
               than one rack circuit delivers, and exceeding it does not fail — the driver clamps
               every device in the zone, which reads as the whole rack getting slower. Protecting
               against that is exactly Carbonite's job, so the grant is clamped to the devices the
               budget can power as well as to the ones that exist.

               There is deliberately no VRAM envelope yet: the analogue of the RAM grant needs a
               `gpu_memory_bytes` field on `SchedulingEnvelope` in the neutral `plan` layer, and
               that contract change belongs in the same commit as its consumer.

               Args:
                   num_gpus: GPUs requested per task, fractional allowed.
                   n_tasks: The desired worker fan-out before GPU clamping.
                   gpu_count: GPU devices the cluster/host reports.
                   accelerator_type: The fleet's device model, used to price the power budget.
                       `None` or an unrecognized model skips the power clamp rather than guessing
                       a draw, so an unknown fleet is granted exactly what inventory allows.

               Returns:
                   A `SchedulingEnvelope` whose GPU grant is feasible against the inventory and,
                   where one is configured, against the power budget.
        """
        if num_gpus <= 0 or gpu_count <= 0:
            # No GPU visible (or none asked for): grant none. Asking for a GPU the
            # cluster does not have makes the task permanently unschedulable.
            return SchedulingEnvelope(num_gpus=0.0, n_tasks=max(1, n_tasks))
        from batcher.carbonite.accel.power import devices_within_budget

        devices = min(gpu_count, devices_within_budget(accelerator_type, gpu_count))
        devices = min(devices, _healthy_devices(gpu_count))
        concurrent = max(1, int(devices / num_gpus))
        return SchedulingEnvelope(
            num_gpus=num_gpus,
            n_tasks=max(1, min(n_tasks, concurrent)),
            accelerator_type=accelerator_type,
        )

    def envelope(
        self,
        plan: PhysicalPlan,
        ctx: ResourceContext,
        *,
        requested_workers: int | None,
        available_bytes: int,
    ) -> SchedulingEnvelope:
        cfg = ctx.config
        # Local fallback only — used when the plan carries no data-driven fan-out.
        # NOT a clamp on the data-driven want: this envelope is consumed only by the
        # distributed path, where the *cluster*-aware `clamp_workers` owns the real
        # cap. Clamping the desired fan-out to the driver's core count here would
        # cap a 100-node job at the driver's cores (the bug N11 fixes).
        cpu_budget = max(1, cfg.execution.parallelism or available_cpu_count())

        # Desired parallelism: the widest breaker request (≈ rows / target-rows). An
        # explicit user `requested_workers` always wins; an unsized/streaming plan
        # (no breaker estimate) falls back to the local cpu budget. The data-driven
        # `desired` is passed through un-clamped — `clamp_workers` reduces it to live
        # cluster capacity downstream.
        desired = max((op.bounds.n_max_parallelism for op in plan.ops), default=0)
        if requested_workers and requested_workers > 0:
            n_tasks = requested_workers
        elif desired > 0:
            n_tasks = desired
        else:
            n_tasks = cpu_budget
        # An explicit user request is honored as given; a *derived* fan-out is held under
        # the sanity ceiling, because the derivation multiplies through Kyber's
        # unknown-cardinality placeholder when a plan could not be sized.
        n_tasks = max(1, n_tasks if requested_workers else min(n_tasks, _MAX_TASK_FANOUT))

        # Per-task memory: the dominant breaker split across tasks, never below one
        # morsel and never above a fair share of the live budget. 0 (no hint) when
        # Kyber could not size the plan. Blended toward the measured peak (learned from
        # `m_peak_bytes`) when available, so each distributed worker gets a right-sized
        # grant instead of one sized from the plan guess; cold families pass through.
        peak = learned_plan_peak(plan, ctx.memory_model)
        # The configured value, not `morsel_rows * row_bytes`. The two agree only at the
        # defaults (16,384 x 64 == 1 MiB); `ResourceManager.recommended_config` rewrites
        # `morsel_rows` (from the learned per-row width) and `morsel_bytes` (from the
        # pressure factor) *independently*, so after any adaptive resize the derivation
        # drifts from the real morsel — and this value is the per-task memory floor below.
        morsel_bytes = max(1, cfg.execution.morsel_bytes)
        if peak <= 0:
            # Kyber could not size the plan — a cold start, an unbounded source, or a
            # shape its estimator abstained on. Granting 0 here leaves each worker's
            # spill budget unbounded (the engine makes no memory pool for a 0 budget),
            # so a large unknown-size distributed query OOMs instead of spilling — the
            # one case a regular user would have to rescue by hand (`num_workers` /
            # `max_memory_bytes`). Carbonite protects even when Kyber can't measure:
            # fall back to a conservative fair share of the budget so the worker SPILLS
            # and the query "just works" (survives). It sharpens to the real footprint
            # once a run measures it; a query that fits under the share never spills, so
            # the only cost is mild over-spill on an un-estimable large query — far
            # better than an OOM. A no-op when the budget is unknown (`available_bytes`
            # <= 0, e.g. test stubs), preserving the old "no hint" behavior there.
            memory_bytes = (
                max(morsel_bytes, int(available_bytes * cfg.memory.soft_limit) // n_tasks)
                if available_bytes > 0
                else 0
            )
        else:
            # A task holds one partition of the dominant breaker, so `peak // n_tasks` is
            # already its share — the *data* is divided by the task count exactly once.
            per_task = max(morsel_bytes, peak // n_tasks)
            # Ray's `memory=` is a **reservation**: the scheduler only places a task on a
            # node with that much free, and packs against it. Under-reporting it therefore
            # over-packs the node and OOMs. The old clamp took `min(per_task,
            # available_bytes // n_tasks)`, dividing one *machine's* budget by the
            # *cluster-wide* fan-out — so a 100-task job asked for 1/100th of a node for
            # every task, and Ray stacked all hundred onto one node.
            #
            # The only legitimate ceiling is a single node's usable memory: asking for more
            # than a node has makes the task permanently unschedulable. Clamp there, and
            # nowhere else, so the hint stays what the task actually needs.
            node_capacity = (
                max(morsel_bytes, int(available_bytes * cfg.memory.soft_limit))
                if available_bytes > 0
                else per_task
            )
            memory_bytes = min(per_task, node_capacity)

        # Per-task CPU: the dominant operator's share (a task runs a whole plan
        # partition, so its heaviest op sets the core need). A pure scan→filter→write
        # plan asks <1 CPU and packs tighter; any breaker pulls it back to a full core.
        # Falls back to the configured default for an unsized plan (no bounds).
        #
        # Floored at `cpu_share_min` for the same reason that floor exists on the learned
        # path: a plan whose operators all declare `c_cpu_shares = 0` (an annotator that
        # abstained, a hand-built plan) would otherwise request `num_cpus=0`, which Ray
        # treats as "needs no CPU" and will pack without limit onto one node — the exact
        # oversubscription the concurrency limiter exists to prevent, arriving through
        # the scheduler instead.
        num_cpus = max(
            (op.bounds.c_cpu_shares for op in plan.ops),
            default=cfg.execution.cpus_per_task,
        )
        num_cpus = max(num_cpus, cfg.execution.cpu_share_min)

        # Placement-strategy preference (resolved against the live cluster in `dist`).
        # A small-shuffle breaker prefers PACK — co-locate the workers, no cross-node
        # shuffle — but only when the gang plausibly fits one node: a fan-out wider than
        # a node's cores cannot PACK, so it stays SPREAD. Carbonite has no live topology;
        # `cpu_budget` (≈ one machine's cores) is the feasibility proxy, and `dist` makes
        # the final call (it can downgrade SPREAD→PACK on a single-node cluster).
        prefers_local = any(op.bounds.prefers_locality for op in plan.ops)
        placement_strategy = "PACK" if prefers_local and n_tasks <= cpu_budget else "SPREAD"

        # This is the *relational* (CPU shuffle) grant — `num_gpus` is 0 here, the GPU
        # map/inference path sets its own. Record the intent to keep this fleet off GPU
        # nodes; `dist` enforces it only when the live cluster has CPU-only capacity to
        # host the fleet (a no-op on a homogeneous cluster), so a CPU shuffle never
        # steals an inference stage's GPU-node cores on a mixed cluster.
        return SchedulingEnvelope(
            num_cpus=num_cpus,
            memory_bytes=int(memory_bytes),
            num_gpus=0.0,
            n_tasks=n_tasks,
            credits=cfg.flow_control.default_credits,
            placement_strategy=placement_strategy,
            prefer_cpu_only_nodes=True,
        )
