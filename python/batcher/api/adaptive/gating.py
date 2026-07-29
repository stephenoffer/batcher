"""Whether to run adaptively, and how far to trust an estimate (control plane, `api`).

The seam: this module holds every *decision about* adaptivity — never the adaptivity
itself. It resolves ``adaptive="auto"``, builds the same `CardinalityEstimator` Kyber
uses so the gate reads the optimizer's own numbers, judges whether a stage's measured
size matched its estimate, and folds that outcome back into the learned tuner. It runs
before and between stages; the stage loop in `staging` runs them. Keeping the two apart
means the gate stays pure and unit-testable without executing a query.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.api.adaptive.plan_surgery import joins, walk
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Scan, is_streamable
from batcher.plan.stats import Provenance

__all__ = ["record_adaptive_route", "resolve_adaptive"]


def resolve_adaptive(
    adaptive: bool | str,
    plan: LogicalPlan,
    sources: list[Source],
    hub,
    *,
    distributed: bool = False,
) -> bool:
    """Resolve ``adaptive="auto"`` to a concrete on/off decision.

    ``"auto"`` (the default) turns stage-by-stage re-optimization on *only* when it
    could change a downstream decision: a join whose operand is produced by a pipeline
    breaker the loop can materialize, and whose size is a pure estimate (a Selinger
    guess, `Provenance.DEFAULT`). That is exactly when measuring the real cardinality
    flips a build-side / broadcast / join-order choice. A plan whose join inputs are
    confidently sized — from source statistics, sketches, or a prior run — gains nothing
    from the extra per-stage materialization, so it stays on the cheaper one-shot path
    (zero adaptive overhead). An explicit ``True``/``False`` always wins.

    When `distributed`, ``"auto"`` ALSO turns it on for a shape the one-shot dispatcher
    cannot route at all — a join whose operand already spans two sources (every 3+-table
    star/snowflake query), which used to raise `PlanError`. There staging is not an
    optimization but the only distributed path. Explicit ``adaptive=False`` still wins.
    """
    if adaptive != "auto":
        return bool(adaptive)
    if distributed:
        from batcher.dist import requires_staging

        if requires_staging(plan):
            return True
    # The size floor is a precondition, not one vote among several, and it is checked before
    # anything learned. `plan_signature` deliberately normalizes literals so statistics
    # generalize across runs — which also makes it **scale-blind**: the same query over sf1
    # and sf10 shares a signature. A route measured where staging pays would then be replayed
    # at interactive scale, where it cannot: measured on TPC-H sf1, replaying sf10's routes
    # took q8 from 18.8 ms to 181.9 ms and q2 from 11.2 ms to 123.2 ms. Nothing keyed by
    # signature may decide a question about size.
    if not _large_enough(plan, sources, hub):
        return False
    # Above the floor, measured cost decides once both routes have been tried, because this is
    # a cost question and the structural heuristic below cannot answer it. That heuristic fires
    # on nearly every multi-join query at scale, and staging is not the ~20-40 ms of control
    # plane it was priced against: the loop runs one breaker per stage, so it materializes every
    # join separately and gives up both operator fusion and the streaming executor's width.
    # Measured at sf10, that cost q8 4.1x, q17 6.3x, q9 3.3x and q3 3.1x against the one-shot
    # plan for the identical result.
    #
    # The heuristic is not simply inverted, because which route wins is not a constant of the
    # plan: staging is the only distributed route for some shapes, and it is what earns the
    # statistics a cold shape has not learned yet. `learned_adaptive_route` measures both and
    # minimizes regret. Staging only re-plans equivalent algebra, so the arms return the
    # identical relation and the choice is result-invariant.
    #
    # But an arm is only worth exploring if it could win, and `staged` cannot win a plan whose
    # join operands are ALREADY confidently sized: measuring a cardinality the optimizer
    # already knows exactly changes no decision, so staging can only add its own cost. UCB1
    # gives every offered arm a turn and its evidence expires, so offering it anyway means
    # re-paying that cost forever — the same regret `sort_merge` was withheld from the
    # build-side bandit for, and for the same reason. Measured on `lineitem ⋈ orders` at sf10
    # (both scans EXACT-sized): the converged one-shot route runs 132 ms, and the periodic
    # staged exploration 283-470 ms, on a query where the two arms cannot differ in what they
    # learn. So the structural question is asked FIRST and gates the bandit, rather than being
    # the cold-start fallback the bandit overrides once it has a verdict.
    if not _adaptive_would_help(plan, sources, hub):
        return False
    route = _learned_adaptive_route(plan, hub)
    if route is not None:
        return route == "staged"
    return True


def _large_enough(plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Whether the query is big enough for stage-by-stage re-optimization to pay at all.

    A join, and total scan rows clearing `_ADAPTIVE_MIN_INPUT_ROWS`. Both read EXACT source
    row counts, so this separates scales without depending on the guessed operand size the
    rest of the gate is about.
    """
    if not joins(plan):
        return False
    estimator = _build_estimator(sources, hub)
    return _total_input_rows(plan, estimator) >= _ADAPTIVE_MIN_INPUT_ROWS


def _learned_adaptive_route(plan: LogicalPlan, hub) -> str | None:
    """The measured-cheaper route for `plan` (`staged`/`one_shot`), or `None` cold."""
    if hub is None:
        return None
    try:
        from batcher.kyber.learned_tuning import learned_adaptive_route
        from batcher.kyber.signature import plan_signature

        return learned_adaptive_route(hub, plan_signature(plan))
    except Exception as exc:  # pragma: no cover - a learned read must never break routing
        note_suppressed("api", "read learned adaptive-routing verdict", exc)
        return None


def record_adaptive_route(hub, plan: LogicalPlan, staged: bool, wall_ms: float) -> None:
    """Fold one query's measured wall time into the staged-vs-one-shot bandit. Best-effort."""
    if hub is None or wall_ms <= 0.0:
        return
    try:
        from batcher.kyber.learned_tuning import record_adaptive_route as _record
        from batcher.kyber.signature import plan_signature

        _record(hub, plan_signature(plan), "staged" if staged else "one_shot", wall_ms)
    except Exception as exc:  # pragma: no cover - recording must never break a query
        note_suppressed("api", "record adaptive-routing outcome", exc)


# Below this total input-row count, stage-by-stage re-optimization is not worth its
# cost. Adaptive re-opt trades a per-stage materialize + re-plan (~20-40ms of control
# plane) for a better downstream join/build-side choice — a win only when the data is
# large enough that a mis-estimated plan would cost *more* than that overhead. At
# interactive / dev scale (a few million rows, the whole query well under a second) the
# one-shot plan is already fast and the re-plan is pure overhead. The gate reads EXACT
# source row counts, so it separates scales cleanly (TPC-H sf1≈9M off, sf10≈90M on)
# without ever depending on the guessed operand size it is trying to protect against.
_ADAPTIVE_MIN_INPUT_ROWS = 20_000_000


def _total_input_rows(plan: LogicalPlan, estimator) -> float:
    """Sum of every `Scan`'s estimated rows — the query's total input size.

    Scan estimates come straight from EXACT source statistics (footer/catalog row
    counts), so this is a trustworthy size gauge even when downstream operand sizes are
    only guessed.
    """
    total = 0.0
    for node in walk(plan):
        if isinstance(node, Scan):
            total += estimator.estimate(node).rows
    return total


def _adaptive_would_help(plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Whether any join has a breaker-produced operand whose size is not yet trustworthy.

    The size floor this used to check itself now sits in `_large_enough`, ahead of the
    learned router, because it has to bind that too — see `resolve_adaptive`.

    Two conditions have to hold for an operand to justify staging, and the second one is
    the correction. It must be breaker-produced, so the loop can actually materialize it
    and measure something. And its size must be genuinely unknown.

    Provenance alone answers the second badly, because it describes where a number came
    from and not whether that number was right. `Provenance.DEFAULT` is sticky: the
    one-shot path never records an intermediate operator's measured cardinality against
    the operand's signature, so a shape can be estimated to within a percent of actual
    forever and still read as a guess. Measured at sf10, TPC-H q5's operands land within
    1.0x of actual and carried the default label anyway, which fired this gate on every
    run and put the query on a route that costs it. The label was standing in for evidence
    the hub already had.

    So the label now only opens the question, and the measured q-error history closes it.
    An operand whose signature has a run of observations that never crossed the
    re-optimization threshold (`kyber.estimate_is_reliable`) is treated as confidently
    sized whatever its provenance says, because a stage boundary placed there would have
    had nothing to correct. A cold hub knows nothing, returns `False` from that check, and
    the gate behaves exactly as it did before any history existed.
    """
    plan_joins = joins(plan)
    if not plan_joins:
        return False
    estimator = _build_estimator(sources, hub)
    return any(
        not is_streamable(operand)
        and estimator.estimate(operand).provenance >= Provenance.DEFAULT
        and not _estimate_has_held_up(operand, hub)
        for join in plan_joins
        for operand in (join.left, join.right)
    )


def _estimate_has_held_up(operand: LogicalPlan, hub) -> bool:
    """Whether `operand`'s shape has a measured history of accurate size estimates."""
    if hub is None:
        return False
    try:
        from batcher.kyber import estimate_is_reliable
        from batcher.kyber.signature import plan_signature

        return estimate_is_reliable(hub, plan_signature(operand))
    except Exception as exc:  # pragma: no cover - a learned read must never break routing
        note_suppressed("api", "read operand q-error reliability", exc)
        return False


def _build_estimator(sources: list[Source], hub):
    """A `CardinalityEstimator` configured exactly as Kyber's, for the confidence gate."""
    from batcher.api.orchestration import collect_source_stats
    from batcher.config import active_config
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator

    cfg = active_config()
    learned = load_learned_stats(hub) if hub is not None else {}
    return CardinalityEstimator(
        sources,
        learned,
        cfg.optimizer.cardinality,
        source_stats=collect_source_stats(sources, hub),
    )


def _estimate_rows(node: LogicalPlan, sources: list[Source], hub) -> int:
    """The optimizer's pre-execution row estimate for `node` over `sources` (0 on error).

    Built from the same `CardinalityEstimator` Kyber uses, over the *current* sources
    (which include any exact-sized intermediates spliced in by earlier stages).
    """
    try:
        return int(_build_estimator(sources, hub).estimate(node).rows)
    except Exception:
        return 0


def _estimate_accurate(actual: int, estimate: int, reopt_error: float) -> bool:
    """Whether `actual` and `estimate` agree within a factor of `1 + reopt_error`.

    The **symmetric q-error**, not a relative error normalized by the estimate: the latter is
    bounded by 1 for any over-estimate, so it called every over-estimate accurate — and an
    over-estimate is exactly what this loop exists to catch. Error is multiplicative, so the
    band is too. A positive estimate that produced nothing is a total miss.

    Zero against zero is the one case the ratio cannot express, and it is a *perfect*
    estimate, not a miss: the optimizer predicted an empty intermediate and got one, so the
    residual plan re-plans to the same shape. Calling it inaccurate forced a re-optimization
    pass — and another pipeline break — on exactly the query whose estimates were right.
    """
    if estimate <= 0 and actual <= 0:
        return True
    if estimate <= 0 or actual <= 0:
        return False
    return max(actual / estimate, estimate / actual) <= 1.0 + reopt_error
