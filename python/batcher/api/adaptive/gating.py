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

__all__ = ["resolve_adaptive"]


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
    # The cold heuristic (a join over a breaker-produced, guessed-size operand) fires on
    # the first run. Once history exists, ALSO enable for any signature whose measured
    # re-optimization actually *flipped* a plan often enough — a shape whose estimates the
    # loop keeps correcting is worth the per-stage cost even if the structural gate misses
    # it. Gating adaptivity only trades planning overhead, never the result.
    return _adaptive_would_help(plan, sources, hub) or _learned_adaptive_helps(plan, hub)


def _learned_adaptive_helps(plan: LogicalPlan, hub) -> bool:
    """Whether the hub has learned that stage-by-stage re-opt historically helps `plan`."""
    if hub is None:
        return False
    try:
        from batcher.kyber.learned_tuning import learned_adaptive_helps
        from batcher.kyber.signature import plan_signature

        return learned_adaptive_helps(hub, plan_signature(plan))
    except Exception as exc:  # pragma: no cover - a learned read must never break routing
        note_suppressed("api", "read learned adaptive-routing verdict", exc)
        return False


def _record_adaptive_flip(hub, plan: LogicalPlan, flipped: bool) -> None:
    """Fold this adaptive run's flip outcome into the learned adaptive gate. Best-effort."""
    if hub is None:
        return
    try:
        from batcher.kyber.learned_tuning import record_adaptive_flip
        from batcher.kyber.signature import plan_signature

        record_adaptive_flip(hub, plan_signature(plan), flipped)
    except Exception as exc:  # pragma: no cover - recording must never break a query
        note_suppressed("api", "record adaptive-routing flip", exc)
        return


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
    """Whether any join has a breaker-produced operand whose size is only guessed —
    *and* the total input is large enough for re-optimization to pay for itself."""
    plan_joins = joins(plan)
    if not plan_joins:
        return False
    estimator = _build_estimator(sources, hub)
    if _total_input_rows(plan, estimator) < _ADAPTIVE_MIN_INPUT_ROWS:
        return False  # small inputs: the one-shot plan is already fast (see threshold note)
    return any(
        not is_streamable(operand) and estimator.estimate(operand).provenance >= Provenance.DEFAULT
        for join in plan_joins
        for operand in (join.left, join.right)
    )


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
