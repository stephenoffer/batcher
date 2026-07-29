"""Physical-plan annotation — the `ResourceBounds` Kyber hands Carbonite.

Kyber decides; Carbonite protects. This module is the hand-off: it walks the optimized
logical plan and tags each operator with its estimated rows, memory envelope, desired
parallelism, credit window and CPU share, plus the feedback keys Core echoes back
(`signature`, `est_rows_raw`, `expr_factor`). Neither subsystem imports the other — the
bounds travel on the `PhysicalPlan`.

Split out of `optimizer` because it is annotation, not rule driving: the optimizer decides
*what the plan is*, this decides *what it will cost to run*.
"""

from __future__ import annotations

import math

from batcher._internal.logging import note_suppressed
from batcher.config import Config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel
from batcher.kyber.cpu_shares import class_ir_tag, recommend_num_cpus
from batcher.plan.ids import OpId
from batcher.plan.logical import LogicalPlan
from batcher.plan.physical import PhysicalOp, PlanProperties
from batcher.plan.resource import ResourceBounds
from batcher.plan.visitor import children, walk

__all__ = ["annotate_ops"]


# Memory-budgeting model (consumed by Carbonite admission). Materializing
# operators ("breakers") hold ~all their rows; streaming operators hold ~one morsel.
# The tunables (row footprint, morsel size, unknown-size threshold) live in `Config`.
# `AsofJoin` materializes a sorted side to search; `RangeJoin` needs both sides whole
# (a match is a function of the global sort order on each axis — `bc_interp::ops::joins`
# says so in as many words) and gathers its whole output; `WatermarkStreamJoin` holds a
# window buffer and `WatermarkDedup` a seen-key set. All four hold far more than a morsel,
# and omitting them budgeted each at ~one morsel — an *under*-estimate of peak memory, which
# is the direction that lets Carbonite over-admit and OOM. `RangeJoin` was the sharpest case
# precisely because it is the one operator here whose output is super-linear in its inputs:
# a 50,000 x 50,000 inequality join estimated at 833M rows was admitted against a 393 KB
# envelope. Being listed here only ever raises the envelope from a morsel to `rows x width`.
_BREAKER_KINDS = frozenset(
    {
        "Aggregate",
        "Sort",
        "Distinct",
        "Join",
        "AsofJoin",
        "RangeJoin",
        "Window",
        "WatermarkStreamJoin",
        "WatermarkDedup",
    }
)

# CPU-light, IO/decode-bound streaming operators: one task running these waits on IO
# more than it saturates a core, so it asks for a fractional CPU share (`cpu_share_io`)
# and the cluster packs more than one per core. Breakers (hash/sort) and anything not
# listed here keep the full `cpus_per_task` share — an unknown op never under-requests.
_CPU_LIGHT_KINDS = frozenset(
    {"Scan", "Filter", "Project", "Limit", "Sample", "RowId", "Union", "Unnest", "Unpivot"}
)


def _is_fixed_count_sample(node: LogicalPlan) -> bool:
    """Whether `node` is the materializing form of `Sample`.

    `Sample` is the one operator whose two forms sit on opposite sides of this line, so it
    cannot be settled by kind alone the way `_BREAKER_KINDS` settles the rest:

    - a **fraction** sample keeps a row iff a seeded hash of it falls under the fraction —
      a per-row predicate holding nothing, genuinely one morsel;
    - a **fixed-count** `sample(n=)` keeps the `n` smallest-hash rows of the whole relation.
      `bc_interp::ops::reshape::sample_n_batches` calls it "a breaker: it must see all rows",
      holding a size-`n` heap of row encodings.

    Budgeted by kind, the second was handed one morsel however large `n` was — a
    `sample(n=100_000)` was admitted against 131 KB. The estimator already produces exactly
    the right figure for it (`min(n, input)` rows *is* the heap size), so recognizing the
    form is the whole fix.
    """
    from batcher.plan.logical import Sample

    return isinstance(node, Sample) and node.n is not None


def _cpu_share(
    kind: str, cpu_light: float, cpu_heavy: float, learned_cpu: dict[str, float], config: Config
) -> float:
    """Per-task CPU share for an operator of `kind`.

    The static per-kind prior (a CPU-light streaming op asks for a fraction; breakers
    and unknown ops keep the full share) is overridden by the measured CPU utilization
    of this operator family when a prior run recorded one — so `num_cpus` adapts to
    how CPU-bound the operator actually is.
    """
    base = cpu_light if kind in _CPU_LIGHT_KINDS else cpu_heavy
    tag = class_ir_tag(kind)
    return recommend_num_cpus(learned_cpu.get(tag) if tag else None, base, config)


def annotate_ops(
    plan: LogicalPlan,
    estimator: CardinalityEstimator,
    config: Config,
    cost_model: CostModel,
    cpu_util: dict[str, float] | None = None,
) -> tuple[PhysicalOp, ...]:
    """Tag each operator with its estimated rows + memory envelope for Carbonite.

    Kyber measures; Carbonite protects: these per-operator `ResourceBounds` are what
    the admission policy checks a plan's feasibility against, without either layer
    importing the other (the bounds travel on the `PhysicalPlan`).

    `cpu_util` is the learned per-kind CPU utilization (from prior runs); when a kind
    has a measurement it overrides the static CPU-share prior, so the per-task
    `num_cpus` request adapts to how CPU-bound each operator family actually is.

    Each op also carries the feedback keys Core echoes back (see `PlanProperties`).
    """
    learned_cpu = cpu_util or {}
    row_bytes = config.optimizer.row_bytes
    morsel_rows = config.execution.morsel_rows
    morsel_bytes = max(1, config.execution.morsel_bytes)
    target_rows = max(1, config.optimizer.target_rows_per_task)
    fc = config.flow_control
    credit_ceiling = max(1, fc.default_credits * fc.credit_ceiling_factor)
    cpu_heavy = config.execution.cpus_per_task
    cpu_light = config.execution.cpu_share_io
    # At/above this, a cardinality is a placeholder (unknown source size), not a real
    # estimate — such operators are left unbudgeted so a guess never fails a real query.
    unknown_rows = config.optimizer.cardinality.unknown_rows
    ops: list[PhysicalOp] = []
    try:
        nodes = list(walk(plan))
        for i, node in enumerate(nodes):
            est = estimator.estimate(node)
            rows = est.rows
            kind = type(node).__name__
            materializes = kind in _BREAKER_KINDS or _is_fixed_count_sample(node)
            known = 0.0 <= rows < unknown_rows
            # Byte-true width: learned per-column widths when measured, else the flat
            # `row_bytes` default (so a cold-start envelope is unchanged). A column of
            # wide payloads (blobs, embeddings) now inflates the envelope correctly.
            width = estimator.row_width(node, row_bytes)
            if not known:
                mem = 0  # unknown size — don't budget (never fail a real query on a guess)
            elif materializes:
                mem = int(rows * width)  # materialized state
            else:
                # streaming: ~one morsel in flight, byte-bounded.
                mem = min(int(morsel_rows * width), morsel_bytes)
            # Desired parallelism: a breaker wants enough tasks that each handles
            # ~`target_rows` of the data it *shuffles* — its input volume, not its
            # (possibly tiny) grouped output. Streaming ops inherit the pipeline's
            # width (0 = unset). Carbonite clamps the request to the cpu budget.
            prefers_local = False
            if known and materializes:
                # `known` gates the *node's* estimate, but a child can still carry the
                # placeholder (a join over an unbound source whose parent estimate a rule
                # collapsed). Summing it blind makes `in_rows` the 1e12 magnitude, `n_par`
                # explode, and `prefers_local` a decision about garbage. Fall back to the
                # node's own known estimate unless every child is itself known.
                child_rows = [estimator.estimate(c).rows for c in children(node)]
                usable = [r for r in child_rows if 0.0 <= r < unknown_rows]
                in_rows = sum(usable) if len(usable) == len(child_rows) else rows
                in_rows = in_rows or rows
                n_par = max(1, math.ceil(in_rows / target_rows))
                # A breaker whose shuffle volume is small enough to keep node-local prefers
                # PACK over SPREAD: co-locating its few workers avoids a cross-node shuffle that
                # buys nothing. Large shuffles keep SPREAD so the network load distributes; dist
                # makes the final call. This is a *network* threshold (`locality_max_bytes`),
                # deliberately separate from the *cache*-sized broadcast threshold the two used
                # to share — an L3-derived broadcast size must not drag this placement choice.
                prefers_local = int(in_rows * width) <= config.optimizer.locality_max_bytes
            else:
                n_par = 0
            # Desired credit window: enough in-flight batch slots to cover one task's
            # partition of the materialized state, clamped to the configured ceiling.
            if n_par > 0 and mem > 0:
                partition_bytes = mem / n_par
                c_max = max(1, min(credit_ceiling, math.ceil(partition_bytes / morsel_bytes)))
            else:
                c_max = 0  # no estimate → Carbonite supplies the default window
            c_cpu = _cpu_share(kind, cpu_light, cpu_heavy, learned_cpu, config)
            ops.append(
                PhysicalOp(
                    op_id=OpId(i),
                    kind=kind,
                    backend="native",
                    algorithm="",
                    bounds=ResourceBounds(
                        m_max_bytes=mem,
                        c_max_credits=c_max,
                        n_max_parallelism=n_par,
                        c_cpu_shares=c_cpu,
                        prefers_locality=prefers_local,
                    ),
                    inputs=(),
                    properties=PlanProperties(
                        est_rows=rows,
                        # Publish the width the envelope above was sized with, so a consumer
                        # never has to invert `m_max_bytes` by the flat default to recover it.
                        row_size=width,
                        provenance=est.provenance,
                        signature=estimator.signature_of(node),
                        est_rows_raw=estimator.reportable_estimate(node),
                        expr_factor=cost_model.expr_factor(node),
                    ),
                )
            )
    except Exception as exc:
        note_suppressed("kyber", "annotate resource bounds", exc)
        return ()  # estimation unavailable (e.g. unbound sources) → Carbonite abstains
    return tuple(ops)
