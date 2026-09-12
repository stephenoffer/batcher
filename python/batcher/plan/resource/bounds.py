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

__all__ = [
    "CAPACITY_ANY",
    "CAPACITY_ON_DEMAND",
    "CAPACITY_PREFERENCES",
    "CAPACITY_SPOT",
    "FeasibilityVerdict",
    "ResourceBounds",
    "SchedulingEnvelope",
]

#: The three things a plan can say about the capacity its tasks should land on, and the
#: vocabulary of `SchedulingEnvelope.capacity_preference`.
#:
#: Named here, in the neutral layer, because Carbonite chooses one and `dist` translates it,
#: and neither may import the other. The *meaning* of each — which node labels satisfy it,
#: what happens when the preferred market is full — belongs to `dist`, which is the only
#: layer that can see a live fleet; this is only the set of words.
CAPACITY_ANY = "any"
CAPACITY_SPOT = "spot"
CAPACITY_ON_DEMAND = "on_demand"
CAPACITY_PREFERENCES = frozenset({CAPACITY_ANY, CAPACITY_SPOT, CAPACITY_ON_DEMAND})


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
    * `materializes` — whether the operator is a **pipeline breaker**: it holds the
      relation rather than streaming a morsel of it. Kyber already decides this to size
      `m_max_bytes` (a breaker is budgeted `rows x width`, a streaming operator one
      morsel), and publishing it is what lets Carbonite tell the two apart without
      restating the vocabulary — which it cannot import, since the subsystems are
      independent. It is the difference between work a preemption costs a *resubmission*
      and work a preemption costs the *stage*: a stateless partition re-derives from its
      durable descriptor, a breaker's accumulated state does not. `False` is the safe
      default here for the same reason it is everywhere else in this file — an operator
      nobody sized reads as streaming, which asks for less rather than more.
    """

    m_max_bytes: int
    c_max_credits: int
    n_max_parallelism: int
    c_cpu_shares: float = 1.0
    prefers_locality: bool = False
    materializes: bool = False


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
    * `capacity_preference` — which market the fleet's tasks should land on when the cluster
                       offers more than one: `"any"` (state nothing), `"spot"` (the work
                       re-derives from durable inputs, so a reclamation costs a resubmission),
                       or `"on_demand"` (the work holds state a reclamation would destroy). A
                       *preference*, like `placement_strategy`: `dist` resolves it against the
                       live fleet's market labels and emits nothing at all unless the fleet is
                       genuinely mixed, so it is a no-op on every single-market cluster.
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
    capacity_preference: str = "any"
    gpu_collective: bool = False
    inflight_depth: int = 1
    # Per-worker grants for a fleet whose nodes are NOT the same size, parallel to each other
    # and `n_tasks` long. Empty — the default, and what every homogeneous cluster and every
    # explicitly-sized fan-out produces — means the fleet is uniform and `num_cpus` /
    # `memory_bytes` describe every worker, which is what this envelope meant before these
    # existed.
    #
    # They exist because a mixed fleet has no single right answer and forcing one makes the
    # whole cluster behave like its weakest member: a uniform grant is sized so *every* node
    # can host it, so a 96-core/206 GB node next to a 4-core/8.6 GB one ran 24 four-core
    # workers on a 228 MB budget each. See `plan.resource.fleet_plan`, which computes these.
    #
    # `num_cpus` and `memory_bytes` stay populated alongside them, and mean what they always
    # did: `num_cpus` is set to the fleet's *smallest* grant, so a placement bound or an
    # oversubscription check reading the scalar reads a figure no worker undercuts, and
    # `memory_bytes` remains Carbonite's own per-task estimate rather than any node's share.
    # Anything sized per worker must read `slot_cpus`/`slot_memory_bytes`.
    worker_cpus: tuple[float, ...] = ()
    worker_memory_bytes: tuple[int, ...] = ()
    # Cores a worker may COMPUTE over, where that differs from the cores it RESERVES above.
    #
    # The two are the same number on almost every path and were one field for a long time,
    # which is right until a reservation is deliberately thinned. Both fills hold a core back
    # on every node (`executor._headroom_grant`, `fleet_plan`'s `node_reserve_cores`) so the
    # query's own plain Ray tasks have somewhere to run — a *scheduling* reserve, and the
    # right one. But the same figure also sizes each worker's rayon pool
    # (`ray_runtime.lifecycle.engine_config_json`), so the reserve was silently taking a
    # thread as well as a slot: on a 96-core node that is 1%, and on a **4-core node it is
    # 25% of the machine, permanently**, which is what a wide fleet of small instances is.
    #
    # `capacity._FLEET_TASK_HEADROOM_MAX` already states the principle these carry — "Ray's
    # CPU figure is a *reservation*, and the worker's real width comes from the
    # `EngineConfig`" — and it was true of the in-bundle headroom and false of the fill's.
    # Empty/zero means "no difference", which is every fleet that does not thin.
    compute_cpus: float = 0.0
    worker_compute_cpus: tuple[float, ...] = ()

    def slot_compute_cpus(self, index: int | None = None) -> float:
        """Cores worker `index` may compute over — its reservation unless one was held back.

        Args:
            index: The worker's position in the fleet, or `None` for the uniform figure.

        Returns:
            That worker's compute width, falling back to its reservation.
        """
        reserved = self.slot_cpus(index)
        if index is not None and index < len(self.worker_compute_cpus):
            return max(reserved, self.worker_compute_cpus[index])
        # `max`, not the bare figure: a later step may *raise* the reservation
        # (`dist.executor`'s `_even_cpu_share` branch does), and a width recorded before that
        # would then quietly cap the worker below the cores it now holds.
        return max(reserved, self.compute_cpus)

    def slot_cpus(self, index: int | None = None) -> float:
        """Cores granted to worker `index`, falling back to the uniform `num_cpus`.

        Args:
            index: The worker's position in the fleet, or `None` to ask for the uniform grant.

        Returns:
            That worker's core grant.
        """
        if index is not None and index < len(self.worker_cpus):
            return self.worker_cpus[index]
        return self.num_cpus

    def slot_memory_bytes(self, index: int | None = None) -> int:
        """Heap bytes granted to worker `index`, falling back to the uniform `memory_bytes`.

        Args:
            index: The worker's position in the fleet, or `None` to ask for the uniform grant.

        Returns:
            That worker's memory budget.
        """
        if index is not None and index < len(self.worker_memory_bytes):
            return self.worker_memory_bytes[index]
        return self.memory_bytes
