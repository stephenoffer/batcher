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
from batcher.api.adaptive.plan_surgery import BREAKERS, joins, walk
from batcher.api.source_stats import build_estimator
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

    A join, and total scan input clearing either the row floor **or** the byte floor. Both
    read EXACT source row counts, so this separates scales without depending on the guessed
    operand size the rest of the gate is about; the byte term additionally reads the scan's
    width, which is a property of the schema and of the source's own measurements rather
    than of an estimate.

    The floor is **per stage**, not per query, and that is the whole point of it. What
    staging costs is not a constant of the query, it is a constant of each *cut*: one
    materialize, one re-plan, and the operator fusion and streaming width given up at that
    boundary. A plan with one breaker-produced operand pays that once; a snowflake with six
    pays it six times. A single flat number cannot separate those, so it was set high enough
    for the worst of them — which is why adaptivity was off for essentially every query
    below 20M rows, including the cheap two-breaker shapes where it costs almost nothing.

    Scaling the floor by the number of breakers the loop would cut at fixes both ends: a
    two-breaker plan now qualifies at half the old floor, and a six-breaker plan needs half
    again more than the old floor before it is allowed to try — which is the direction the
    measured sf10 regressions point (q8 4.1x, q17 6.3x, q9 3.3x, q3 3.1x are all
    many-breaker shapes; see `resolve_adaptive`).

    Args:
        plan: The logical plan being routed.
        sources: The plan's bound inputs.
        hub: The metadata hub, or `None`.

    Returns:
        Whether the query clears the size floor.
    """
    if not joins(plan):
        return False
    estimator = build_estimator(sources, hub)
    rows, in_bytes = _total_input_size(plan, estimator)
    stages = _stage_count(plan)
    return (
        rows >= _ADAPTIVE_MIN_ROWS_PER_STAGE * stages
        or in_bytes >= _ADAPTIVE_MIN_BYTES_PER_STAGE * stages
    )


def _stage_count(plan: LogicalPlan) -> int:
    """How many stages the loop would cut `plan` into — its pipeline-breaker count.

    `staging` runs one breaker per stage (`lowest_breaker`, then splice, then repeat), so
    the breaker count is what the per-stage cost multiplies. It is an upper bound rather
    than the exact number: the loop skips a breaker whose output size is already known
    exactly, which measured as 17 of 51 across the TPC-H shapes. Erring high is the safe
    direction here — it asks a complicated plan to be larger before staging it — and the
    exact count is not available without running the loop, which is the thing being decided.

    Never below 1, so the floor is a floor even for a plan the walk finds nothing in.
    """
    return max(1, sum(1 for node in walk(plan) if isinstance(node, BREAKERS)))


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


# Input rows required *per stage the loop would cut* before stage-by-stage re-optimization
# is worth its cost. Adaptive re-opt trades a per-stage materialize + re-plan (~20-40 ms of
# control plane, plus the fusion and streaming width given up at that boundary) for a better
# downstream join/build-side choice — a win only when the data is large enough that a
# mis-estimated plan would cost *more* than that overhead.
#
# This was a flat 20,000,000 for the whole query, and the flatness was the defect rather
# than the number. One cut costs about a thirtieth of what a query over 10M rows costs; six
# cuts cost six times that. A single threshold has to be set for the worst shape it will see,
# so it was, and the consequence was that the loop never engaged on anything below 20M rows
# — the great majority of queries, including the cheap two-breaker shapes where a cut is
# nearly free. "The adaptive moat is off for most queries" was a fair description.
#
# The per-stage number is chosen to hold the old floor **fixed at the shape it was
# calibrated on**. Every regression recorded against staging — q8 4.1x, q17 6.3x, q9 3.3x,
# q3 3.1x at sf10 — is a many-breaker query, so 20M was in effect the right answer for a
# four-cut plan. 4 x 5M is that same 20M. What changes is everything either side of it:
#
#   breakers   old floor   new floor
#   2          20M          10M      <- the cheap shape, now reachable
#   4          20M          20M      <- unchanged, the calibration point
#   6          20M          30M      <- the shapes that measurably lost, now stricter
#
# The floor still reads EXACT source row counts, so it separates scales without ever
# depending on the guessed operand size it is there to protect against. And it remains only
# one of several conditions: `_adaptive_would_help` still requires a join with a
# breaker-produced operand whose size is genuinely unknown, and above the floor the learned
# route bandit measures both arms and can turn staging back off for a shape where it loses.
_ADAPTIVE_MIN_ROWS_PER_STAGE = 5_000_000

# ...and the same per-stage floor stated in bytes, because a row count assumes a row width.
#
# The rationale above is about *work*: re-optimization pays when a mis-estimated plan would
# cost more than the ~20-40 ms re-plan. Rows are a proxy for work, and the proxy holds only
# while a row is the ~64 bytes `optimizer.row_bytes` assumes. Across the modality range it
# inverts at both ends: 20M rows of two `int64` keys is 320 MB, which the row gate turns
# adaptation ON for, while 1M rows of decoded 224x224x3 images is **150 GB**, which it turns
# it OFF for. The single most expensive query class in the engine was the one class the
# adaptive loop never ran on.
#
# Derived from the row floor rather than added as a second independent knob, so there is one
# place that says how big "big" is; and the two gates are combined with OR, so a query clears
# whichever of the two suits its shape.
_ADAPTIVE_MIN_BYTES_PER_STAGE = _ADAPTIVE_MIN_ROWS_PER_STAGE * 64


def _total_input_size(plan: LogicalPlan, estimator) -> tuple[float, float]:
    """`(rows, bytes)` summed over every `Scan` — the query's total input size.

    Scan estimates come straight from EXACT source statistics (footer/catalog row counts),
    so this is a trustworthy size gauge even when downstream operand sizes are only guessed.
    The width beside them is the scan's own — its column types, or a width the source
    measured — and never a guessed intermediate.

    Args:
        plan: The logical plan.
        estimator: The shared cardinality estimator.

    Returns:
        Total input rows and total input bytes.
    """
    from batcher.config import active_config

    row_bytes = active_config().optimizer.row_bytes
    rows = 0.0
    nbytes = 0.0
    for node in walk(plan):
        if isinstance(node, Scan):
            scan_rows = estimator.estimate(node).rows
            rows += scan_rows
            nbytes += scan_rows * estimator.row_width(node, row_bytes)
    return rows, nbytes


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
    estimator = build_estimator(sources, hub)
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


def _estimate_rows(node: LogicalPlan, sources: list[Source], hub) -> int:
    """The optimizer's pre-execution row estimate for `node` over `sources` (0 on error).

    Built from the same `CardinalityEstimator` Kyber uses, over the *current* sources
    (which include any exact-sized intermediates spliced in by earlier stages).
    """
    try:
        return int(build_estimator(sources, hub).estimate(node).rows)
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
