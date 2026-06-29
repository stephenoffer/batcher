"""Resource contracts between Kyber (optimizer) and Carbonite (resource manager).

Kyber annotates each physical operator with the resources it expects to need
(`ResourceBounds`); Carbonite validates the plan against the cluster/machine and
returns a `FeasibilityVerdict`. If infeasible, the verdict carries a counter-offer
that Kyber can re-plan around (e.g. force a spill-friendly join) — closing the
optimizer↔resource loop without either layer importing the other.
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
    # Scheduling hints resolved entirely in the `dist` layer (never serialized to the
    # JSON IR / FFI). Defaults preserve today's behavior: SPREAD, no node-class
    # preference, no collective co-location.
    placement_strategy: str = "SPREAD"
    prefer_cpu_only_nodes: bool = False
    gpu_collective: bool = False
