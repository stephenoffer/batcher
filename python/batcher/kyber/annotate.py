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


def _resident_bytes(
    node: LogicalPlan, rows: float, width: float, estimator, morsel_rows: int = 0
) -> int:
    """Bytes a materializing operator actually holds — which is not always its output.

    For most breakers the two coincide: an aggregate's state is its groups, a sort's is
    its rows, a distinct's is its distinct set. **A hash join is the exception, and it is
    the most common operator in an analytic plan.** Its resident state is the hash table
    over its *build* side (the right input, by Batcher's convention), while its output is
    the probe side fanned out by the match rate — and in a star schema those differ by the
    fan-out ratio, in the direction that over-budgets.

    Measured on a 100,000-row fact joined to a 100-row dimension: the cost model sizes the
    table at 1,600 bytes and this used to hand Carbonite **2,400,000** — 1,500x. That
    figure is not advisory. It is what admission checks feasibility against, what the spill
    decision reads, and what the distributed per-task memory grant is derived from, so the
    single most common join shape in analytics was systematically pushed toward spilling
    and toward a rejected admission for a hash table that fits in a cache line's worth of
    pages. `cost.py` has always had this right (`mem=build_bytes`); the two now agree.

    A **top-N** is the same shape one level down: a fused `Sort` + `Limit` holds a heap of
    `limit` rows, not the relation, which is the entire reason to fuse them.

    Args:
        node: The materializing operator.
        rows: Its estimated output rows.
        width: Its estimated output row width in bytes.
        estimator: The shared cardinality estimator, for sizing a join's build side.
        morsel_rows: The configured morsel size, for sizing an aggregate's live partial
            state (see `_aggregate_resident_bytes`). `0` skips that refinement.

    Returns:
        The operator's resident state in bytes.
    """
    from batcher.plan.logical import Aggregate, Distinct, Join, Sort

    if isinstance(node, Join):
        build = estimator.estimate(node.right).rows
        return int(max(0.0, build) * estimator.row_width(node.right, width))
    if isinstance(node, Sort) and node.limit:
        return int(min(float(node.limit), rows) * width)
    if isinstance(node, (Aggregate, Distinct)):
        return _aggregate_resident_bytes(node, rows, width, estimator, morsel_rows)
    return int(rows * width)


def _aggregate_resident_bytes(
    node: LogicalPlan, rows: float, width: float, estimator, morsel_rows: int
) -> int:
    """An aggregate's (or `DISTINCT`'s) resident state, which is **not** its groups when the
    key is wide.

    `DISTINCT` takes the same rule because it *is* an all-columns group-by, run by the same
    `partial -> combine` path over the same per-morsel tables. Measured on a 24 M-row
    `distinct(k)` under a 537 MB envelope: budgeted at its 2 M distinct rows (16 MB) it stayed
    in memory and peaked at 1.6 GB.

    The parallel aggregate is `partial -> combine -> finalize`: every morsel builds its own
    group table and all of them are live when `combine` merges them. So what it holds is the
    sum of the per-morsel partials, and the reduction that decides how big those are is the
    one *within a morsel* — not the global one.

    That distinction is the whole defect. `GROUP BY k` over 24 M rows into 2 M groups reduces
    12:1 globally, so budgeting the output gave 2 M x 16 B = **32 MB**. But a 16,384-row
    morsel of a 2 M-group key space holds ~16,300 distinct keys, so it reduces by nothing:
    every input row survives into a partial, and the live partial state is the size of the
    input again. Measured peak for that query was ~2.4 GB against a 537 MB envelope, with the
    query never routed out of core because the estimate said 32 MB.

    `agg_par` states this exact asymmetry about *CPU* — "when grouping does not reduce, the
    pre-aggregation is pure overhead" — and it was never applied to memory.

    The per-morsel group count is `min(morsel_rows, ndv)`, so the partial state is
    `input_rows x (that / morsel_rows) x width`. Both ends come out right: a wide key gives
    the factor 1 and the full input, and `GROUP BY flag` over three groups gives
    `3 / 16,384` — negligible, exactly as it should be, so a reducing group-by's envelope is
    unchanged.

    The larger of the two readings wins, because a wide aggregate holds its partials *and*
    eventually its output, and the output dominates only when grouping reduces.
    """
    out_bytes = int(rows * width)
    if morsel_rows <= 0 or rows <= 0:
        return out_bytes
    inputs = list(children(node))
    if not inputs:
        return out_bytes
    in_rows = estimator.estimate(inputs[0]).rows
    if in_rows <= 0:
        return out_bytes
    # Distinct keys a single morsel can hold: bounded by the morsel and by the key space.
    per_morsel_groups = min(float(morsel_rows), rows)
    partial_bytes = int(in_rows * (per_morsel_groups / morsel_rows) * width)
    return max(out_bytes, partial_bytes)


def _streaming_bytes(
    node: LogicalPlan, width: float, morsel_rows: int, morsel_bytes: int, estimator=None
) -> int:
    """Bytes a non-materializing operator holds in flight.

    A genuine streaming operator holds about one morsel, and a morsel is byte-bounded
    (`carbonite.policies.morsel`), so `min(rows x width, morsel_bytes)` is right for it.

    **A row-*expanding* operator is not byte-bounded, and that was measured rather than
    assumed.** Exploding 4,000 rows of a `fixed_size_list<float32, 768>` produces a *single*
    output batch of 3,072,000 rows — 12 MB against a 1 MiB morsel budget, 11.7x it — and an
    `unpivot` of 4,000 rows over 20 columns produces one batch of 80,000. Both emit a whole
    morsel's fan-out in one go and nothing re-cuts it. At a full 16,384-row morsel the explode
    is ~100 MB from a 1 MB input, budgeted at 65 KB. So the fan-out multiplies the in-flight
    rows and the morsel byte cap does **not** apply: capping there reports the budget rather
    than the operator.

    Stated as a property of the *fan-out* rather than of a node type, so it covers `Unnest`,
    `Unpivot`, and any future expander without a list to keep in sync — and so a `Filter`,
    `Project`, or `Limit`, which cannot expand, is byte-capped exactly as before. The fan-out
    comes from the estimator's own propagated counts (output rows over input rows), so it is
    exact wherever the type or the operator proves it (a `fixed_size_list`'s length, an
    unpivot's column count) and carries any learned correction otherwise — one source of
    truth rather than a second rule that could drift from the cardinality estimate.

    **A `map_batches` carrying an explicit `batch_size` is not that operator.** It re-batches
    its input to exactly that many rows regardless of the morsel it was handed, so what it
    holds is `batch_size x width` and the morsel byte cap does not apply. That is the shape
    of every inference stage: `kyber/gpu/sizing.py` seeds a batch size from the device's VRAM
    headroom, tens of thousands of rows for a light model, and on a decoded image column
    those rows are hundreds of kilobytes each. Budgeted at one morsel, a stage really holding
    gigabytes was reported to Carbonite as holding one megabyte — an under-count in the
    direction that lets admission accept a query the node cannot run, on precisely the
    pipeline this engine is built for.

    Only the *input* batch is charged. A UDF's output is arbitrary Python and Batcher cannot
    size it; claiming a multiple here would be inventing a number inside a memory bound.

    Args:
        node: The streaming operator.
        width: Its estimated output row width in bytes.
        morsel_rows: The configured rows per morsel.
        morsel_bytes: The configured bytes per morsel.
        estimator: The shared cardinality estimator, for an explode's fan-out.

    Returns:
        The operator's in-flight bytes.
    """
    from batcher.plan.logical import MapBatches

    batch_size = getattr(node, "batch_size", None) if isinstance(node, MapBatches) else None
    if batch_size:
        return int(max(1, batch_size) * width)
    fanout = _fanout(node, estimator) if estimator is not None else 1.0
    if fanout > 1.0:
        return int(morsel_rows * fanout * width)
    return min(int(morsel_rows * width), morsel_bytes)


def _fanout(node: LogicalPlan, estimator) -> float:
    """Output rows per input row for `node`, from the estimator's own propagated counts.

    At least 1.0: an operator the estimator believes *shrinks* its input is not a reason to
    budget below one morsel (a filter still reads a whole one), and a fan-out of zero would
    report an operator holding nothing. `1.0` for anything with no single input.
    """
    inp = getattr(node, "input", None)
    if inp is None:
        return 1.0
    try:
        in_rows = estimator.estimate(inp).rows
        out_rows = estimator.estimate(node).rows
    except Exception as exc:  # pragma: no cover - budgeting must never break a plan
        # Falling back to 1.0 budgets every operator below this one at a single morsel.
        # That is the right *behaviour*, but it is indistinguishable from a plan that
        # genuinely does not fan out, so an estimator broken here would quietly cap
        # parallelism forever. Trace it the way the annotate loop at the bottom of this
        # module already traces its own abstention.
        note_suppressed("kyber", "estimate operator fan-out", exc)
        return 1.0
    if in_rows <= 0 or out_rows <= 0:
        return 1.0
    return max(1.0, out_rows / in_rows)


def _desired_parallelism(in_rows: float, width: float, target_rows: int, target_bytes: int) -> int:
    """Tasks a breaker wants, from the rows it shuffles **and** how wide they are.

    A row target alone assumes a row width, and the shipped `target_rows_per_task` of four
    million assumes the flat `row_bytes` of 64 — a sensible 256 MiB per task, and the figure
    it was plainly tuned for. On anything wider it sizes tasks that cannot exist:

    | column                      | width     | bytes per task at 4M rows |
    |-----------------------------|-----------|---------------------------|
    | two `int64` keys            | 16 B      | 64 MB                     |
    | 768-dim `float32` embedding | 3 KiB     | 12 GB                     |
    | 224x224x3 `uint8` image     | 147 KiB   | **602 GB**                |
    | one 1080p RGB frame         | 5.9 MiB   | **25 TB**                 |

    A multimodal pipeline was therefore fanned out as though every row were sixteen bytes,
    and each of its tasks was asked to hold hundreds of gigabytes — the OOM that arrives at
    scale rather than in the small test.

    **This is not a new rule, it is the rule the rest of the engine already follows.**
    `api/tuning/decisions.py::auto_num_partitions` and `dist/executors/map.py` both take the
    larger of the row- and byte-derived counts against `target_bytes_per_task`, and
    `docs/deep-dives/distributed-scheduling.md` documents that as how a stage is sized —
    naming video frames and embeddings as the reason. Kyber's `n_max_parallelism`, which is
    what `SchedulingEnvelope.n_tasks` is actually derived from, was the one place that still
    counted only rows. So the two answers to "how many tasks" disagreed on exactly the data
    the documented one was written for.

    The two demands combine with `max`, never `min`: the byte term can only ask for *more*
    parallelism, so a relation no wider than the flat default gets exactly the fan-out it
    got before and no structured plan is re-shaped by this.

    Args:
        in_rows: Rows the breaker shuffles (its input volume, not its output).
        width: Estimated bytes per row.
        target_rows: `optimizer.target_rows_per_task`.
        target_bytes: `optimizer.target_bytes_per_task`.

    Returns:
        The desired task count, at least 1.
    """
    by_rows = math.ceil(in_rows / max(1, target_rows))
    by_bytes = math.ceil(in_rows * max(0.0, width) / max(1, target_bytes))
    return max(1, by_rows, by_bytes)


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
    target_bytes = max(1, config.optimizer.target_bytes_per_task)
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
        # `walk` is pre-order over an immutable tree, so a node's position in it is its
        # `OpId`. Recording that mapping is what lets `inputs` carry the plan's *shape*
        # alongside its per-operator numbers.
        #
        # It was hardcoded empty, and `carbonite/memory/estimator.py` names the consequence
        # in its own docstring: with no tree, a plan's envelope can only be the largest
        # single breaker, and a bushy plan holds more than one at once. On a four-way bushy
        # join with three hash tables sized 18.2 / 9.1 / 9.1 MB, the honest concurrent
        # reading is 27.4 MB and the `max` reports 18.2 — a 1.5x under-count, in the
        # direction that over-admits and OOMs. That docstring ends "Populating `inputs` is
        # Kyber's to do"; this is that.
        #
        # Keyed by `id()` and read within this loop, so no freed node's reused address can
        # be observed: `nodes` holds a strong reference to every one of them throughout.
        op_id_of = {id(n): OpId(i) for i, n in enumerate(nodes)}
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
                mem = _resident_bytes(node, rows, width, estimator, morsel_rows)
            else:
                # streaming: ~one morsel in flight, byte-bounded.
                mem = _streaming_bytes(node, width, morsel_rows, morsel_bytes, estimator)
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
                n_par = _desired_parallelism(in_rows, width, target_rows, target_bytes)
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
                    inputs=tuple(op_id_of[id(c)] for c in children(node)),
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
