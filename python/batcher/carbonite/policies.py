"""Bootstrap resource policies — the permissive single-node defaults.

Each class implements one `carbonite.base` policy `Protocol` and reproduces the
behavior the bootstrap `ResourceManager` has today: everything is feasible,
nothing spills, every credit is granted, and the memory envelope is permissive.
These are the seam's default occupants; real policies replace them by being
constructed in their place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from batcher._internal.hardware import available_cpu_count
from batcher.carbonite.memory.estimator import peak_operator_bytes
from batcher.carbonite.memory.pressure import total_memory_bytes
from batcher.config import Config, active_config
from batcher.plan.physical import PhysicalOp
from batcher.plan.resource import FeasibilityVerdict, ResourceBounds, SchedulingEnvelope
from batcher.plan.stats import Provenance

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.metadata import MetadataHub
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "AIMDFlowControl",
    "BudgetingAdmission",
    "DefaultSchedulingPolicy",
    "StaticCreditFlowControl",
    "credit_ceiling",
    "load_shuffle_window",
    "record_shuffle_window",
]

# Learned-parameter namespace for the converged AIMD credit window, keyed by a shuffle
# channel's stable signature. One smoothed integer window per signature.
_SHUFFLE_WINDOW_NS = "carbonite.shuffle_window"


class BudgetingAdmission:
    """Real admission: reject a plan whose dominant materializing operator would
    not fit the memory envelope, returning a spill-friendly counter-offer.

    Conservative by construction so it never fails a legitimate query: it budgets
    only operators with a *known* size (Kyber leaves unknown-size operators at
    `m_max_bytes == 0`), compares against a soft fraction of physical RAM, and uses
    the single dominant breaker (operators materialize one at a time in a linear
    pipeline) rather than over-summing. With no bounds emitted, it abstains.
    """

    def __init__(
        self, available_bytes: int | None = None, *, soft_limit: float | None = None
    ) -> None:
        # Optional explicit overrides (used by tests / a standalone policy). When
        # left unset, `validate` reads the unified envelope + soft limit from the
        # `ResourceContext` the manager threads in, so admission budgets against the
        # *same* figure as spill and reserve (and the live config, not a stale one).
        self._available = available_bytes
        self._soft = soft_limit

    def validate(self, plan: PhysicalPlan, ctx: ResourceContext) -> FeasibilityVerdict:
        if not plan.ops:
            return FeasibilityVerdict(feasible=True)  # no annotations → abstain
        available = self._available
        if available is None:
            available = (
                ctx.envelope_bytes if ctx.envelope_bytes is not None else total_memory_bytes()
            )
        soft = self._soft if self._soft is not None else ctx.config.memory.soft_limit
        envelope = int(available * soft)
        # Cross-query admission (C56): subtract what concurrent queries already hold
        # against the shared buffer pool, so N queries that each individually fit the
        # envelope are not all admitted into a collective OOM.
        if self._available is None:
            from batcher.carbonite.memory.pool import current_process_pool

            pool = current_process_pool()
            if pool is not None:
                envelope = max(0, envelope - pool.used)
        # The envelope can never be smaller than one morsel: the engine must hold at
        # least a single morsel to make any progress, and a *streaming* operator's whole
        # footprint is one morsel (`min(morsel_rows·width, morsel_bytes)`). Flooring here
        # keeps a streaming/tiny plan feasible under a sub-morsel budget (it would
        # otherwise be rejected as infeasible with "no out-of-core path", since a
        # streaming op has nothing to spill) — a no-op for any realistic budget, which is
        # orders of magnitude larger than a morsel. A genuine breaker that materializes
        # more than this floor still exceeds it and routes to the spill path.
        envelope = max(envelope, ctx.config.execution.morsel_bytes)
        # Blend each operator's plan estimate toward its measured peak (learned from
        # `m_peak_bytes`) before taking the dominant breaker, so admission budgets
        # against what the family really used — admitting a query the plan over-sized
        # (avoiding a needless spill route) and catching one the plan under-sized
        # (avoiding an OOM). Cold families pass through unchanged.
        model = ctx.memory_model
        if model is not None:
            peak = model.plan_peak(plan.ops)
        else:
            peak = max((op.bounds.m_max_bytes for op in plan.ops), default=0)
        if peak <= envelope:
            return FeasibilityVerdict(feasible=True)
        # Over budget: offer the envelope as the per-operator bound so the engine can
        # re-plan with a spill-friendly strategy instead of OOMing.
        #
        # `plan/physical.py` promises that "Carbonite reads provenance to decide how
        # defensively to budget", and this is where it must: the byte figure above is only
        # as trustworthy as the cardinality it was derived from. When the operator that
        # binds the constraint was sized from a pure Selinger guess, the verdict is
        # *advisory* — it still routes the plan out-of-core, but the conductor will not
        # fail a query on it. Rejecting on a guess breaks the admission contract that a
        # guess never fails a legitimate query.
        return FeasibilityVerdict(
            feasible=False,
            binding_constraint="memory",
            suggested_bounds=ResourceBounds(
                m_max_bytes=envelope, c_max_credits=0, n_max_parallelism=0
            ),
            advisory=_binding_op_is_a_guess(plan.ops),
        )


def _binding_op_is_a_guess(ops: Sequence[PhysicalOp]) -> bool:
    """Whether the operator whose memory binds admission was sized from a pure guess.

    The binding operator is the one holding the plan's peak envelope. `Provenance.DEFAULT`
    means its row count came from a Selinger constant with nothing measured behind it — a
    number that can be wrong by orders of magnitude in either direction. Every stronger
    provenance (a proof, a footer, a sketch, or a past measurement) is trusted.

    Args:
        ops: The plan's annotated operators.

    Returns:
        True when the peak-memory operator's cardinality is an unmeasured guess.
    """
    sized = [op for op in ops if op.bounds.m_max_bytes > 0]
    if not sized:
        return True  # nothing was sizable; any verdict over it is a guess
    binding = max(sized, key=lambda op: op.bounds.m_max_bytes)
    return binding.properties.provenance is Provenance.DEFAULT


def credit_ceiling(config: Config, effective_morsel_bytes: int | None = None) -> int:
    """The upper bound on a shuffle channel's credit window (count *and* bytes).

    The count ceiling (`default_credits x credit_ceiling_factor`) is further capped
    so the window's *bytes* (`credits x morsel_bytes`) never exceed
    `credit_byte_budget` — bounding a channel's buffered memory regardless of row
    width (C53). Always >= 1.

    `effective_morsel_bytes` overrides the config `morsel_bytes` when a channel's real
    per-batch size is known to be wider than the configured target — the learned-row-width
    case (embeddings/blobs), where the assumed `morsel_bytes` under-counts the buffered
    bytes and a fast producer would run the window well past `credit_byte_budget`.
    """
    fc = config.flow_control
    count_ceiling = fc.default_credits * fc.credit_ceiling_factor
    morsel_bytes = max(1, effective_morsel_bytes or config.execution.morsel_bytes)
    byte_ceiling = max(1, fc.credit_byte_budget // morsel_bytes)
    return max(1, min(count_ceiling, byte_ceiling))


def _learned_channel_morsel_bytes(ctx: ResourceContext) -> int | None:
    """A channel's effective per-batch bytes from the learned row width, or `None`.

    The credit→bytes conversion assumes a `morsel_bytes`-sized batch; a workload whose
    rows proved *wide* anywhere (the learned `max_bytes_per_row`) fills a `morsel_rows`
    batch to far more than that, so its real buffered footprint per credit is larger.
    Returning `max(morsel_bytes, width x morsel_rows)` lets `credit_ceiling` hand out
    fewer credits for wide-row shuffles, keeping buffered memory within budget. `None`
    (cold model / narrow rows) leaves the conversion at the configured `morsel_bytes`.
    """
    model = ctx.memory_model
    if model is None:
        return None
    width = model.max_bytes_per_row()
    if width is None or width <= 0:
        return None
    ex = ctx.config.execution
    return max(ex.morsel_bytes, int(width * max(1, ex.morsel_rows)))


class StaticCreditFlowControl:
    """Credit-window flow control: clamp the requested window to a memory-safe band.

    This is the Carbonite authority that replaces the engine's hardcoded
    `DEFAULT_CREDITS`: one credit = one in-flight `RecordBatch` slot, so the window
    directly bounds a shuffle channel's buffered memory. The window comes from
    `FlowControlConfig`: a non-positive request (operator with no `c_max_credits`
    estimate) gets `default_credits`; a positive request is clamped into
    `[1, credit_ceiling(config)]` (a count *and* byte bound) so neither a stale zero
    stalls the channel nor an over-large estimate (or a wide-row morsel) lets a fast
    producer run unbounded.
    """

    def grant(self, requested: int, ctx: ResourceContext) -> int:
        fc = ctx.config.flow_control
        ceiling = credit_ceiling(ctx.config, _learned_channel_morsel_bytes(ctx))
        if requested <= 0:
            return min(fc.default_credits, ceiling)
        return min(max(requested, 1), ceiling)


# AIMD's multiplicative-decrease factor must lie strictly inside (0, 1): at 1.0 the
# congested branch stops decreasing (the window never backs off), and above 1.0 it grows
# on congestion. The floor keeps a decrease from collapsing the window to the floor in one
# round, which would serialize the shuffle.
_MIN_AIMD_BETA = 0.1
_MAX_AIMD_BETA = 0.95


class AIMDFlowControl:
    """Adaptive credit window via AIMD (additive-increase / multiplicative-decrease).

    The static policy fixes the window; this one *adapts* it from observed
    backpressure, the TCP-style control law the architecture specifies. It starts at
    the config default window and, per round, `observe`s whether the channel was
    congested: a congested round cuts the window by `aimd_beta` (relieve memory
    pressure fast), an uncongested round grows it by `aimd_alpha` (pipeline deeper
    while memory is plentiful). The window is always clamped to the same memory-safe
    band `[1, default_credits x credit_ceiling_factor]` the static policy uses.

    Stateful — hold one per adaptive channel. `grant` ignores its `requested`
    argument because the controller, not the caller, owns the evolving window.
    """

    def __init__(self, config: Config | None = None, *, initial_window: int | None = None) -> None:
        cfg = config or active_config()
        fc = cfg.flow_control
        self._alpha = max(1, fc.aimd_alpha)
        # A multiplicative *decrease* requires 0 < beta < 1. A misconfigured `beta >= 1`
        # would make the congested branch *grow* the window — the opposite of AIMD, and
        # an unstable control law (the window would only ever increase, congested or not).
        # Clamped rather than raised: flow control must never fail a query on a tunable.
        self._beta = min(max(fc.aimd_beta, _MIN_AIMD_BETA), _MAX_AIMD_BETA)
        self._floor = 1
        self._ceiling = credit_ceiling(cfg)  # count + byte bound (C53)
        # A recurring shuffle warm-starts at the window its past runs converged to
        # (`initial_window`, learned per shuffle signature) instead of re-climbing from
        # `default_credits` every time — the AIMD control law still governs from there,
        # so the window a channel actually uses is unchanged, only its starting point.
        start = fc.default_credits if initial_window is None else initial_window
        self._window: float = float(min(max(start, self._floor), self._ceiling))

    @property
    def window(self) -> int:
        """The current credit window (clamped to the band)."""
        return self._clamp(self._window)

    def grant(self, requested: int, ctx: ResourceContext) -> int:  # noqa: ARG002
        return self.window

    def observe(self, *, congested: bool) -> int:
        """Update the window from one round's congestion signal; return the new window.

        `congested` is true when the round hit backpressure (e.g. the producer ran
        the window full, or memory pressure was high): cut multiplicatively. Else the
        consumer kept up with headroom to spare: grow additively.
        """
        if congested:
            self._window = max(self._floor, self._window * self._beta)
        else:
            self._window = min(self._ceiling, self._window + self._alpha)
        return self.window

    def _clamp(self, w: float) -> int:
        return int(max(self._floor, min(self._ceiling, w)))


def load_shuffle_window(hub: MetadataHub | None, signature: str) -> int | None:
    """The learned converged credit window for a shuffle `signature`, or `None` if unseen.

    Best-effort: any read failure (or a cold store) yields `None`, so the channel starts
    at the configured default. Only the *starting* window is affected — flow control still
    governs the window it actually uses — so this is purely a warm-start, never a result
    or a correctness change.
    """
    if hub is None:
        return None
    try:
        value = hub.get_keyed_param(_SHUFFLE_WINDOW_NS, signature)
    except Exception:  # pragma: no cover - metadata must never break a query
        return None
    return int(value) if value is not None else None


def record_shuffle_window(
    hub: MetadataHub | None, signature: str, window: int, config: Config | None = None
) -> None:
    """Persist a shuffle channel's converged credit `window`, exp-smoothed across runs.

    Best-effort and non-raising (mirrors `ml.gpu.record_gpu_utilization`): a recurring
    shuffle's window is smoothed toward each run's converged value so the next run
    warm-starts near it. Records nothing for a non-positive window."""
    if hub is None or window <= 0:
        return
    try:
        alpha = (config or active_config()).optimizer.learning_smoothing_alpha
        prior = hub.get_keyed_param(_SHUFFLE_WINDOW_NS, signature)
        smoothed = window if prior is None else alpha * window + (1.0 - alpha) * float(prior)
        hub.put_keyed_param(_SHUFFLE_WINDOW_NS, signature, round(smoothed))
    except Exception:  # pragma: no cover - metadata must never break a query
        pass


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
        n_tasks = max(1, n_tasks)

        # Per-task memory: the dominant breaker split across tasks, never below one
        # morsel and never above a fair share of the live budget. 0 (no hint) when
        # Kyber could not size the plan. Blended toward the measured peak (learned from
        # `m_peak_bytes`) when available, so each distributed worker gets a right-sized
        # grant instead of one sized from the plan guess; cold families pass through.
        model = ctx.memory_model
        peak = model.plan_peak(plan.ops) if model is not None else peak_operator_bytes(plan)
        morsel_bytes = max(1, cfg.execution.morsel_rows * cfg.optimizer.row_bytes)
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
        num_cpus = max(
            (op.bounds.c_cpu_shares for op in plan.ops),
            default=cfg.execution.cpus_per_task,
        )

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
