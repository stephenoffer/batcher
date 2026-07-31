"""The resource contracts Kyber annotates, Carbonite validates, and `dist` schedules against.

Kyber sizes each physical operator (`ResourceBounds`); Carbonite answers whether the plan fits
(`FeasibilityVerdict`) and derives the per-task grant (`SchedulingEnvelope`) the distributed
executor turns into Ray options. Every one is a plain frozen dataclass in the neutral layer, so
the three subsystems exchange them without importing each other.

Kept apart from `hardware`, which describes the machine being planned *for* rather than the
demand a plan places on it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FeasibilityVerdict", "ResourceBounds", "SchedulingEnvelope"]


@dataclass(frozen=True, slots=True)
class ResourceBounds:
    """R = (M_max, C_max, N_max, CPU) for one physical operator.

    * `m_max_bytes`     — peak memory envelope the operator may use.
    * `c_max_credits`   — max in-flight RecordBatch credits (flow-control bound).
    * `n_max_parallelism` — max concurrent morsels/workers for the operator.
    * `c_cpu_shares`    — CPU shares one task running this operator needs. A
      CPU-heavy breaker (hash/sort) saturates a core (`1.0`); a CPU-light,
      IO/decode-bound streaming op asks for a fraction so more tasks pack per node.
    * `prefers_locality` — whether the operator's shuffle is small enough that
      co-locating its workers (PACK) beats spreading them (SPREAD). Set by Kyber from
      the estimated shuffle volume; consumed by Carbonite to pick a placement strategy
      preference. A pure plan property — the live cluster decides the final strategy.
    """

    m_max_bytes: int
    c_max_credits: int
    n_max_parallelism: int
    c_cpu_shares: float = 1.0
    prefers_locality: bool = False


@dataclass(frozen=True, slots=True)
class FeasibilityVerdict:
    """Carbonite's answer to "can this plan run within these bounds?"."""

    feasible: bool
    binding_constraint: str | None = None  # "memory" | "credits" | "parallelism" | None
    suggested_bounds: ResourceBounds | None = None
    # The operator whose demand binds the constraint, as a plain ``"kind#id"`` string (e.g.
    # ``"Aggregate#3"``), or `None` when nothing was sizable. `binding_constraint` says
    # *which resource* ran out; this says *who ran it out*, which is the actionable half —
    # a user told "this query will spill" can do nothing, while one told the plan's third
    # aggregate is the breaker knows which step to reshape. A string, not an op reference,
    # so the verdict stays a plain data contract that can cross the Ray boundary.
    binding_op: str | None = None
    # Whether this verdict rests on a *guess* rather than a measurement or a proof — set
    # when the operator that binds the constraint carries `Provenance.DEFAULT`. An
    # advisory infeasibility should still steer the plan toward its out-of-core path, but
    # it must never *fail* a query: a plan Kyber could not size may well fit. The
    # conductor honours the routing and suppresses the error.
    advisory: bool = False


@dataclass(frozen=True, slots=True)
class SchedulingEnvelope:
    """Per-task scheduling grant Carbonite derives from a plan's `ResourceBounds`.

    The plain-int payload the distributed executor turns into Ray scheduling hints
    (`.options(num_cpus=, memory=, num_gpus=)`) plus the worker/reducer fan-out and
    the shuffle credit window. It lives in the neutral `plan` layer so Kyber and
    Carbonite can both name it and `dist` can receive it without any layer importing
    another — and nothing live (a policy, a pool) ever crosses the Ray boundary.

    * `num_cpus`     — CPU shares requested per task (Ray default is an implicit 1).
    * `memory_bytes` — heap bytes requested per task (a soft Ray scheduling hint).
    * `num_gpus`     — GPUs requested per task; `0.0` for the CPU relational path,
                       `>0` (incl. fractional) for GPU-tagged map/inference tasks.
    * `n_tasks`      — worker/reducer fan-out, derived from estimated rows (replaces
                       a blind `os.cpu_count()`), clamped to the machine's budget.
    * `credits`      — initial shuffle credit window (flow-control bound).
    * `placement_strategy` — preferred Ray placement-group strategy for the worker
                       fleet (`SPREAD | PACK | STRICT_PACK | STRICT_SPREAD`). A
                       *preference* derived from the plan; the distributed executor
                       resolves it against the live cluster (e.g. downgrades SPREAD to
                       PACK on a tiny cluster where spreading buys nothing).
    * `prefer_cpu_only_nodes` — keep this (relational) fleet off GPU nodes when CPU-only
                       nodes can host it, so a CPU shuffle never steals an inference
                       stage's GPU-node cores. `dist` turns it into a node-label selector
                       against the live topology; a no-op on a homogeneous cluster.
    * `gpu_collective` — the GPU stage's UDF runs its own multi-GPU collective (NCCL/etc.)
                       internally, so `dist` gang-schedules its actors co-located
                       (STRICT_PACK). Batcher never touches a tensor — the Arrow contract
                       at operator boundaries is unchanged; only placement is affected.
    * `inflight_depth` — per-actor submit-ahead depth for a GPU/inference actor pool: how
                       many partitions one actor may have in flight at once. `1` is the
                       one-at-a-time default; `>1` keeps a GPU fed across the
                       dispatch/gather round-trip. Set by the conductor from measured GPU
                       utilization; consumed only by the `dist` actor-pool driver.
    """

    num_cpus: float = 1.0
    memory_bytes: int = 0
    num_gpus: float = 0.0
    n_tasks: int = 1
    # A conservative default window (matches the engine's `DEFAULT_CREDITS`) so a
    # default-constructed envelope never starts a shuffle at a 1-batch serialized
    # window. The scheduling policy overrides this from `FlowControlConfig`.
    credits: int = 4
    # Optional GPU model to pin tasks/actors to (a `ray.util.accelerators` name such
    # as `"NVIDIA_A100"`); `None` lets Ray pick any GPU. Passed straight to
    # `.options(accelerator_type=...)` for GPU map/inference stages.
    accelerator_type: str | None = None
    # Custom Ray resources requested per task, as an immutable `((name, amount), ...)`.
    #
    # `num_gpus` covers only what Ray reports as the `GPU` resource (NVIDIA, AMD, Intel,
    # MetaX). Every other accelerator is a *custom resource*: `TPU`, `neuron_cores`
    # (Trainium/Inferentia), `HPU` (Gaudi), `NPU`. Without this field they are unreachable,
    # so a TPU stage requests `num_gpus` on a node that has none and pends forever. Kept
    # generic rather than one field per vendor so it equally carries a resource an operator
    # defined on their own on-prem cluster.
    #
    # A tuple, not a dict, because this dataclass is frozen *and hashable* — a dict field
    # would make `hash(envelope)` raise. `dist` converts it back at the Ray boundary.
    resources: tuple[tuple[str, float], ...] = ()
    # Scheduling hints resolved entirely in the `dist` layer (never serialized to the
    # JSON IR / FFI). Defaults preserve today's behavior: SPREAD, no node-class
    # preference, no collective co-location.
    placement_strategy: str = "SPREAD"
    prefer_cpu_only_nodes: bool = False
    gpu_collective: bool = False
    inflight_depth: int = 1
