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

__all__ = ["aligned_claims", "record_adaptive_route", "resolve_adaptive"]


def aligned_claims(plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Whether the aligned executor claims `plan`, which is why the gate does not stage it.

    Asked of the *optimized* plan, which is what the executor sees: before projection
    pushdown every scan reads its whole table, so TPC-H q3 priced a 25 GB broadcast of
    `customer` it never reads (1.5 GB once pruned) and was staged. `optimize_logical` is
    memoized, so this costs one lookup. `orchestration.stages` asks the same question when
    the executor then declines, so the two must be one function: asked of different plans,
    they disagreed on TPC-H q22 and its decline raised instead of falling back to staging.
    """
    from batcher import kyber
    from batcher.dist.executors.aligned import aligned_route

    return aligned_route(kyber.optimize_logical(plan, sources=sources, hub=hub), sources)


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

        # A plan the aligned executor will run whole is not staged: a boundary between two
        # tables stored in join-key order is exactly the exchange their layout removes.
        if aligned_claims(plan, sources, hub):
            return False
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
    # A plan that can stream its largest input whole needs no stage boundary to bound its
    # memory, and a boundary is exactly what makes it expensive: the loop cuts at every join,
    # so TPC-H sf100 q9 materialized `lineitem JOIN orders` — 600M rows — as its first stage,
    # where streaming `lineitem` through all five joins holds only their build sides.
    if _streams_whole(plan, sources, hub):
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
    # Cold, the one-shot route runs first. Staging used to, on the argument that it earns the
    # statistics a cold shape lacks, but the one-shot route earns them too (it seeds distinct
    # counts and records every operator's feedback the same way), and it converges in fewer
    # runs: at TPC-H sf10, forced one-shot, 20 of 22 queries are within 10% of their steady
    # time by the fifth run, where the staged start kept q5 at 170-290 ms for five runs before
    # trying one-shot and settling at 101. The bandit still explores staging once one-shot has a
    # settled sample, so a shape where staging wins (q7, 76 against 90 ms) still finds it.
    route = _learned_adaptive_route(plan, hub)
    return route == "staged"


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
    rows, in_bytes, stages = _input_size(plan, sources, hub)
    return (
        rows >= _ADAPTIVE_MIN_ROWS_PER_STAGE * stages
        or in_bytes >= _ADAPTIVE_MIN_BYTES_PER_STAGE * stages
    )


def _input_size(plan: LogicalPlan, sources: list[Source], hub) -> tuple[float, float, int]:
    """`(input rows, input bytes, stage count)` for `_large_enough`, memoized.

    The *measurement* is cached rather than the verdict, so the floors it is compared with
    are read fresh on every call and can never be answered from a stale threshold.
    """
    key = _size_key(plan, sources, hub)
    hit = _INPUT_SIZES.get(key) if key is not None else None
    if hit is not None:
        return hit
    estimator = build_estimator(sources, hub)
    rows, in_bytes = _total_input_size(plan, estimator)
    size = (rows, in_bytes, _stage_count(plan))
    if key is not None:
        _INPUT_SIZES[key] = size
        while len(_INPUT_SIZES) > _INPUT_SIZES_MAX:
            _INPUT_SIZES.pop(next(iter(_INPUT_SIZES)))
    return size


# `_input_size` results, keyed on everything the measurement reads. Building an estimator and
# sizing every scan cost ~1.8 ms profiled on each execution of a small TPC-H join, to reach
# the same numbers every time. Insertion-ordered and trimmed oldest-first.
_INPUT_SIZES: dict[tuple, tuple[float, float, int]] = {}
_INPUT_SIZES_MAX = 1024


def _size_key(plan: LogicalPlan, sources: list[Source], hub) -> tuple | None:
    """What `_input_size` depends on, or `None` when any part of it cannot be keyed.

    The plan's content, each source's data-stable key (the one the learned statistics are
    filed under, so a changed source is a different key), and the hub and learning generation,
    since a scan without exact statistics is sized from learned ones, and the configured row
    width the byte figure falls back to. An unkeyable source means no memo, never a stale answer.
    """
    from batcher.config import active_config
    from batcher.kyber import learning
    from batcher.plan.source_stats import source_stats_key

    keys = []
    for source in sources:
        k = source_stats_key(source)
        if k is None:
            return None
        keys.append(k)
    row_bytes = active_config().optimizer.row_bytes
    return (plan.content_key(), tuple(keys), id(hub), learning.generation(), row_bytes)


def _streams_whole(plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Whether `plan` looks like one the chunked path streams whole (`orchestration.chunked`).

    Its largest input — by the estimator's exact row count and the scan's own width — can read
    itself in chunks, is past the chunked path's size threshold, is scanned exactly once, and no
    right or full join sits above it. This mirrors the engine's `chunkable` on the plan as
    written; when the optimized plan turns out not to be chunkable after all, the query runs
    one-shot, which the size floor's own measurements show beats staging on these shapes.
    """
    from collections import Counter

    from batcher.api.orchestration.chunked import chunk_worthy
    from batcher.plan.logical import Join

    scans = [n for n in walk(plan) if isinstance(n, Scan)]
    if not scans:
        return False
    counts = Counter(n.source_id for n in scans)
    sized = _scan_sizes(plan, scans, sources, hub)
    driving = max(sized, key=sized.__getitem__)
    if (
        counts[driving] != 1
        or not chunk_worthy(int(sized[driving]))
        or driving >= len(sources)
        or not callable(getattr(sources[driving], "iter_chunks", None))
    ):
        return False
    return not any(isinstance(n, Join) and n.join_type in ("right", "full") for n in walk(plan))


def _scan_sizes(
    plan: LogicalPlan, scans: list[Scan], sources: list[Source], hub
) -> dict[int, float]:
    """Each scanned source's estimated bytes (rows x width), memoized like `_input_size`.

    The same measurement-not-verdict memo, under the same key, for the same reason: building
    an estimator and sizing every scan was ~2 ms of a warm TPC-H sf1 join on every execution,
    to reach the numbers it reached last time. `chunk_worthy` and the shape checks that turn
    the sizes into a verdict still run on every call.
    """
    from batcher.config import active_config

    key = _size_key(plan, sources, hub)
    hit = _SCAN_SIZES.get(key) if key is not None else None
    if hit is not None:
        return hit
    estimator = build_estimator(sources, hub)
    row_bytes = active_config().optimizer.row_bytes
    sized = {
        n.source_id: estimator.estimate(n).rows * estimator.row_width(n, row_bytes) for n in scans
    }
    if key is not None:
        _SCAN_SIZES[key] = sized
        while len(_SCAN_SIZES) > _INPUT_SIZES_MAX:
            _SCAN_SIZES.pop(next(iter(_SCAN_SIZES)))
    return sized


#: `_scan_sizes` results, keyed and bounded exactly as `_INPUT_SIZES` is. Read-only values.
_SCAN_SIZES: dict[tuple, dict[int, float]] = {}


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


def record_adaptive_route(
    hub, plan: LogicalPlan, staged: bool, wall_ms: float, misses_before: int
) -> None:
    """Fold one query's measured wall time into the staged-vs-one-shot bandit. Best-effort.

    `misses_before` is `kyber.plan_cache.misses()` read as the run started. A one-shot run
    that derived any plan rather than replaying it is not recorded, because it timed learning
    rather than the route: TPC-H q10 at sf10 runs its one-shot plan at 183, 149, 143 ms while
    the plan converges on the 105 ms it keeps, against 125 ms staged, and a bandit fed those
    first runs settled on staging in roughly half of the processes it ran in.

    A staged run is recorded either way. One-shot is the route a query starts on and the one
    whose steady state matters; staging is explored, and exploring it costs what it costs. Held
    to the same rule, staging's stage plans took four runs to settle before the bandit would
    compare the arms at all, so the exploration landed four slow runs on every staged-eligible
    query: TPC-H sf10 q5 ran 444, 280, 186, 171 ms against a 101 ms one-shot.
    """
    if hub is None or wall_ms <= 0.0:
        return
    try:
        from batcher.kyber import plan_cache
        from batcher.kyber.learned_tuning import record_adaptive_route as _record

        if not staged and plan_cache.misses() != misses_before:
            return
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
    # The estimator half is memoized on what it reads, as `_input_size` is: it built an
    # estimator and sized every join operand on each execution of the same plan, ~5 ms of a
    # 25 ms JOB query to reach the answer the previous run reached. The q-error half is not:
    # recording an operator's feedback moves no learning generation, so a memoized verdict
    # would go on staging a plan whose history had since shown staging corrects nothing.
    # It reads the hub's history for the few operands left, which is cheap.
    key = _size_key(plan, sources, hub)
    unsized = _WOULD_HELP.get(key) if key is not None else None
    if unsized is None:
        unsized = _unsized_operands(plan_joins, sources, hub)
        if key is not None:
            _WOULD_HELP[key] = unsized
            while len(_WOULD_HELP) > _INPUT_SIZES_MAX:
                _WOULD_HELP.pop(next(iter(_WOULD_HELP)))
    return any(not _estimate_has_held_up(operand, hub) for operand in unsized)


# `_unsized_operands` results, keyed and trimmed exactly like `_INPUT_SIZES`.
_WOULD_HELP: dict[tuple, tuple[LogicalPlan, ...]] = {}


def _unsized_operands(plan_joins, sources: list[Source], hub) -> tuple[LogicalPlan, ...]:
    """The join operands a stage boundary could measure whose size is only a default guess."""
    estimator = build_estimator(sources, hub)
    return tuple(
        operand
        for join in plan_joins
        for operand in (join.left, join.right)
        if not is_streamable(operand)
        and estimator.estimate(operand).provenance >= Provenance.DEFAULT
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
