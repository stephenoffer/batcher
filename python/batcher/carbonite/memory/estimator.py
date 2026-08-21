"""Per-operator memory estimation — what envelope a plan needs to run in memory.

Kyber annotates each physical operator with a `ResourceBounds` carrying its estimated
peak memory (`m_max_bytes`) and, since it began populating `PhysicalOp.inputs`, the plan's
shape. Carbonite consumes both: on a linear plan the engine materializes one pipeline
breaker at a time, so the footprint is the largest breaker rather than the sum; on a bushy
plan a join's build side stays resident while the probe side runs, so several are alive at
once and the largest-single reading under-counts. `peak_operator_bytes` walks the schedule
that distinguishes them, and `OperatorMemoryEstimator` returns its answer as the envelope
the admission check and the spill decision reason about.

Everything here is a *rule*, used by more than one caller, and the callers must agree:
a plan admitted against one envelope and granted another is a query admitted into a
budget it was never given. So admission, the estimator, and the distributed grant all
call `learned_plan_peak` and `binding_operator` rather than re-deriving them.

This replaces the permissive bootstrap estimator. It stays conservative: operators
Kyber could not size (`m_max_bytes == 0`) contribute nothing, so a query is never
pushed to spill on a guess — only on an estimate the optimizer actually produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.plan.resource import ResourceBounds

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "OperatorMemoryEstimator",
    "binding_operator",
    "learned_plan_peak",
    "peak_contributors",
    "peak_operator_bytes",
]


def peak_operator_bytes(plan: PhysicalPlan) -> int:
    """The plan's peak in-memory footprint: the most bytes resident at any one moment.

    On a **linear** pipeline that is the dominant breaker, because the engine materializes
    one at a time and summing them would double-count memory never live together.

    On a **bushy** plan it is not, and the difference is the difference between admitting a
    query and OOMing it. A hash join's build side stays resident for as long as the probe
    side runs, so a join of two joins has three hash tables alive at once. With three sized
    at 18.2 / 9.1 / 9.1 MB, the largest-single reading is 18.2 and the honest concurrent one
    is 27.4 — a 1.5x under-count, in the one direction a safety bound must not fail.

    So the walk is now the schedule, not a `max`. For a join, the build side is materialized
    first (its own subtree peaks and then collapses to the table), and that table is
    resident while the probe subtree runs:

        peak(join)  = max(peak(build), resident(join) + peak(probe))
        peak(unary) = max(peak(input), resident(node))

    A unary breaker takes the `max` because its input pipeline has finished and released by
    the time its own state is full — which is exactly why the linear case is unchanged and
    every existing linear-plan envelope is byte-for-byte what it was.

    Falls back to the `max` over all operators when `inputs` is empty, which is what a
    hand-built `PhysicalPlan` (and every test double) carries. That fallback is the previous
    behavior exactly, so an unwired plan is never *worse* off than before.

    Args:
        plan: The annotated physical plan.

    Returns:
        Peak concurrent bytes, `0` when nothing in the plan could be sized.
    """
    return _peak(plan)[0]


def peak_contributors(plan: PhysicalPlan) -> tuple:
    """The operators whose footprints *add up to* `peak_operator_bytes(plan)`.

    On a linear pipeline this is the single dominant breaker, which is what the envelope
    always was. On a bushy plan the peak is a **sum** over operators alive at the same
    moment, and knowing which ones is what keeps the rest of Carbonite honest about it.

    Admission is the caller that needs it. Its contract is that a plan sized from a *guess*
    may route a query out-of-core but must never fail it, and it enforced that by reading
    the provenance of the largest single operator. Once the envelope became a sum, that
    became the wrong operator to ask: a peak of an EXACT 18 MB table plus a guessed 9 MB one
    is a guess, and reading only the larger term would fail a legitimate query on the
    strength of the smaller one.

    Args:
        plan: The annotated physical plan.

    Returns:
        The contributing `PhysicalOp`s, largest first; empty when nothing is sized.
    """
    _, ids = _peak(plan)
    by_id = {int(getattr(op, "op_id", -1)): op for op in plan.ops}
    ops = [by_id[i] for i in ids if i in by_id]
    return tuple(sorted(ops, key=lambda op: op.bounds.m_max_bytes, reverse=True))


def _peak(plan: PhysicalPlan, size_of=None) -> tuple[int, frozenset[int]]:
    """`(peak bytes, contributing operator ids)` for `plan`.

    `size_of` overrides each operator's byte figure — the seam the learned blend uses, so
    "how big is this operator" has one answer and the schedule walk has one implementation.
    """
    size_of = size_of or (lambda op: int(op.bounds.m_max_bytes))
    if not plan.ops:
        return 0, frozenset()
    # `getattr` for the same reason the rest of this module uses it: a bare test double
    # carrying only `bounds` is a supported shape, and an estimator is never the right
    # place to fail a query.
    if not any(getattr(op, "inputs", ()) for op in plan.ops):
        return _flat_peak(plan, size_of)  # no tree — the pre-`inputs` reading, unchanged
    by_id = {int(op.op_id): op for op in plan.ops}
    walked = [_subtree_peak(r, by_id, set(), size_of) for r in _roots(plan)]
    return max(walked, default=_flat_peak(plan, size_of))


def _flat_peak(plan: PhysicalPlan, size_of) -> tuple[int, frozenset[int]]:
    """The largest single operator and its id — the reading from before the tree existed."""
    sized = [op for op in plan.ops if size_of(op) > 0]
    if not sized:
        return 0, frozenset()
    top = max(sized, key=size_of)
    # `getattr` for the same reason the rest of this module uses it: a bare double carrying
    # only `bounds` is a supported shape, and it is the shape that reaches this branch —
    # a plan with no `inputs` is exactly a hand-built one. An id nothing can resolve simply
    # yields no contributors, and the caller falls back to the operator it already named.
    return int(size_of(top)), frozenset({int(getattr(top, "op_id", -1))})


def _roots(plan: PhysicalPlan) -> list[int]:
    """Operator ids nothing else consumes — the plan's output(s).

    `annotate_ops` walks pre-order from one root, so in practice there is exactly one; the
    plural is what keeps this honest against a plan shape that has more.
    """
    consumed = {int(i) for op in plan.ops for i in getattr(op, "inputs", ())}
    return [int(op.op_id) for op in plan.ops if int(op.op_id) not in consumed]


def _subtree_peak(op_id: int, by_id: dict, seen: set[int], size_of) -> tuple[int, frozenset[int]]:
    """Peak concurrent bytes of the subtree rooted at `op_id`, and who contributes them.

    `seen` guards against a cycle. A `PhysicalPlan` is built from an immutable tree so it
    cannot contain one, but this figure gates admission for every query in the process and
    an unbounded recursion here would hang the planner rather than mis-size it.
    """
    op = by_id.get(op_id)
    if op is None or op_id in seen:
        return 0, frozenset()
    seen = seen | {op_id}
    resident = int(size_of(op))
    mine = frozenset({op_id}) if resident > 0 else frozenset()
    children = [_subtree_peak(int(c), by_id, seen, size_of) for c in getattr(op, "inputs", ())]
    if len(children) < 2:
        return max([(resident, mine), *children])
    # A binary breaker holds its build side's state while the probe side runs. Batcher
    # builds on the *right*, which `annotate_ops` records as the second input, so the
    # first input is the one still streaming underneath the resident table.
    probe = children[0]
    build = max(children[1:])
    concurrent = (resident + probe[0], mine | probe[1])
    return max(build, concurrent)


def learned_plan_peak(plan: PhysicalPlan, model) -> int:
    """The plan's memory envelope, blended toward measured reality when a model exists.

    Every Carbonite memory decision starts here: admission's fit check, the estimator's
    envelope, and the distributed per-task grant all need the same number, and they must
    agree — a plan admitted against one figure and granted against another is a query
    admitted into a budget it was never given.

    Each operator's plan estimate is folded toward what its family really used
    (`LearnedMemoryModel.blend_peak`), and cold families pass through unchanged, so on a
    cold store this is exactly `peak_operator_bytes`.

    **The blend is applied per operator and the schedule is walked over the result**, rather
    than by taking the model's own flat `plan_peak`. That distinction is the whole point: a
    flat `max` over blended operators is the pre-`inputs` reading, so routing the warm path
    through it would have made the concurrent-peak walk apply *only on a cold store* — that
    is, only until the engine learns anything, which is exactly when it stops being the
    normal case. A correction that quietly switches itself off once a system is warm is
    worse than one that was never written, because the tests that cover the cold path stay
    green.

    Args:
        plan: The annotated physical plan.
        model: A `LearnedMemoryModel`, or `None` on a cold store.

    Returns:
        The envelope in bytes; `0` when nothing in the plan could be sized.
    """
    return _peak(plan, _blender(model, plan))[0]


def _blender(model, plan: PhysicalPlan):
    """A `PhysicalOp -> bytes` sizer that folds the plan estimate toward measured reality.

    `None` when there is no model, which makes every walk use the plan's own estimate.
    Never raises: a model that cannot size an operator (a bare test double with no `kind`)
    falls back to the plan estimate rather than failing a query inside a memory guard.

    The plan is threaded through so each operator's **input** rows can be resolved from its
    children's estimates and handed to the model. That is the basis the learned per-row
    footprint was fitted against; without it the model has to recover a row count by
    dividing the estimate by its width, which recovers the *output* count and rescales the
    measurement by the operator's selectivity. A model that does not accept the basis (a
    test double with the older signature) is called exactly as before.
    """
    if model is None:
        return None
    basis_of = getattr(model, "est_basis_rows", None)
    by_id = {getattr(op, "op_id", None): op for op in plan.ops} if basis_of is not None else {}

    def size_of(op) -> int:
        planned = int(op.bounds.m_max_bytes)
        try:
            if basis_of is None:
                return int(model.blend_peak(getattr(op, "kind", ""), planned, _row_size(op)))
            return int(
                model.blend_peak(
                    getattr(op, "kind", ""),
                    planned,
                    _row_size(op),
                    input_rows=basis_of(op, by_id),
                )
            )
        except Exception:  # pragma: no cover - a memory guard never raises
            return planned

    return size_of


def _row_size(op) -> float | None:
    """The per-row width the plan sized this operator with, or `None` if it published none.

    `annotate` publishes it precisely so a consumer never has to invert `m_max_bytes` by the
    flat `optimizer.row_bytes` default — which is wrong by `row_size / row_bytes` for any
    operator whose rows are not the assumed 64 bytes wide. `getattr` throughout because a
    bare test double carrying only `bounds` is a supported shape here.
    """
    props = getattr(op, "properties", None)
    width = getattr(props, "row_size", None) if props is not None else None
    # NaN is `PlanProperties`'s *unset* default, not a width of zero.
    if width is None or width != width or width <= 0:
        return None
    return float(width)


def binding_operator(plan: PhysicalPlan):
    """The largest operator *contributing to* the plan's envelope, or `None`.

    Every Carbonite memory decision reduces the plan to one number — its peak — and then
    reports that number with nothing attached. An operator reading "this query will spill"
    has no way back from the figure to the operator that produced it, which is the only
    actionable part: it names which join, aggregate, or sort to reshape.

    "Contributing to" is load-bearing and was not always true. While the envelope was a
    plain `max` this was simply the largest sized operator, and the two coincided. Once the
    envelope became a *sum* over co-resident operators, the largest single operator can sit
    in a branch the peak does not come from at all — a build subtree whose own peak lost to
    the concurrent reading on the other side — and naming it would point a reader at an
    operator that is not why their query does not fit. It falls back to the largest sized
    operator when nothing contributes, which is the pre-tree behavior exactly.

    This is the one implementation of the rule. Three call sites re-derived the `max`
    locally — admission's provenance check, the estimator, the scheduling grant — and
    admission went on re-deriving it after the other two were folded in here, while this
    docstring already claimed the triplication was gone. That is how a "deduplicated"
    helper keeps a surviving copy free to drift from it: the check that matters is whether
    anything still spells the `max` out, not whether a helper exists to spell it once.

    Args:
        plan: The annotated physical plan.

    Returns:
        The `PhysicalOp` to name, or `None` when nothing is sized.
    """
    contributors = peak_contributors(plan)
    if contributors:
        return contributors[0]  # `peak_contributors` returns them largest first
    sized = [op for op in plan.ops if op.bounds.m_max_bytes > 0]
    if not sized:
        return None
    return max(sized, key=lambda op: op.bounds.m_max_bytes)


class OperatorMemoryEstimator:
    """Estimates a plan's memory envelope from Kyber's per-operator bounds.

    The envelope's `m_max_bytes` is the dominant breaker (`peak_operator_bytes`);
    the credit and parallelism fields carry the same conservative defaults the
    bootstrap used so the flow-control and scheduling sides are unaffected until
    they grow their own estimates.

    When a `LearnedMemoryModel` is present on the context (the hub has measured
    `m_peak_bytes` for this operator family), each operator's plan estimate is
    *blended* toward that measured reality before the dominant breaker is taken —
    so admission, spill, and reserve all size against what the query really used,
    not the plan guess alone. Cold families pass through unchanged, so on a cold
    store the envelope equals the plan's own dominant breaker exactly.
    """

    def envelope(self, plan: PhysicalPlan, ctx: ResourceContext) -> ResourceBounds:
        fc = ctx.config.flow_control
        peak = learned_plan_peak(plan, ctx.memory_model)
        # Credits and parallelism take the plan's own widest request when Kyber emitted
        # one, falling back to the configured defaults for an unsized plan. They used to be
        # the configured constants unconditionally, which made the returned envelope a
        # description of the *config* rather than of the plan for two of its three fields —
        # so any consumer reading them (rather than only `m_max_bytes`) would have been
        # told a 200-way shuffle wanted the default 4-way parallelism.
        #
        # `getattr` rather than attribute access for the same reason `plan_peak` uses it:
        # a bare-sized bounds object (a test double carrying only `m_max_bytes`) is a
        # supported shape, and an estimator is never the right place to fail a query.
        return ResourceBounds(
            m_max_bytes=peak,
            c_max_credits=_widest(plan, "c_max_credits") or fc.default_credits,
            n_max_parallelism=_widest(plan, "n_max_parallelism")
            or (ctx.config.execution.parallelism or 0),
        )


def _widest(plan: PhysicalPlan, field: str) -> int:
    """The largest `bounds.<field>` across `plan`'s operators; `0` when none declare it."""
    return max((int(getattr(op.bounds, field, 0) or 0) for op in plan.ops), default=0)
