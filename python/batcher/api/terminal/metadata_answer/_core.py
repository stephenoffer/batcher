"""Metadata-first terminal resolution.

Answer a terminal (`count` / `is_empty` / a keyless aggregate) from the sources'
declared `SourceStatistics` (footers, manifests, catalogs) *before* running the
engine. Each helper consults Kyber's answerability decision and returns ``None`` when
the answer is not provably exact, so the caller executes normally — a returned value
is gated on `Provenance.EXACT`, hence identical to the executed result. Purely an
optimisation; lives beside the materializing terminals in `terminal` (which call it)
rather than in the `orchestration` conductor, keeping that file within budget. It
draws `collect_source_stats` from `orchestration` lazily (at call time) so the two
modules never form an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "global_count_plan",
    "metadata_all_null",
    "metadata_approx_n_unique",
    "metadata_count",
    "metadata_empty_table",
    "metadata_has_nulls",
    "metadata_is_empty",
    "metadata_learned_quantile",
    "metadata_max",
    "metadata_min",
    "metadata_n_unique",
    "metadata_null_count",
]


def global_count_plan(plan: LogicalPlan) -> LogicalPlan:
    """Wrap `plan` in a keyless `COUNT(*)` aggregate (one output row, column ``n``).

    Counting result rows this way — rather than materializing them and taking the length
    — lets projection pushdown read only the columns the plan's filters/keys touch, and a
    `COUNT(*)` directly over a `Filter` fuses into a single `count_if` pass.
    """
    from batcher.plan.expr_ir.constructors import count
    from batcher.plan.logical import Aggregate, AggregateSpec

    return Aggregate(plan, (), (AggregateSpec("n", count()),))


def _metadata_answerable(plan: LogicalPlan, sources: list[Source]) -> bool:
    """Whether a metadata-only answer may even be *attempted* for this plan.

    These helpers are a pure optimization — `None` means "execute normally" — so they
    must never be tried on a plan the stats machinery can't handle: an unbounded
    (streaming) source has no finite answer, and a `map_batches`/UDF pipeline is
    opaque to the IR (`to_ir` is intentionally unsupported), so propagating stats
    through it would raise. Guarding here keeps `count()`/`is_empty()`/an aggregate
    over an ML pipeline runnable instead of crashing in the fast path.
    """
    from batcher import core
    from batcher.io.source import is_bounded

    if any(not is_bounded(s) for s in sources):
        return False
    return not core.has_map_batches(plan)


def _has_row_limiter(node: LogicalPlan) -> bool:
    """Whether `node`'s subtree contains a row-limiting operator (`Limit`, a top-N `Sort`,
    or a per-partition top-N `QUALIFY` / ``distinct(subset)``).

    A `LIMIT` (or a `Sort` folded into a top-N with a limit) restricts *which* rows survive,
    and that choice is invisible to whole-relation source statistics — a column's footer/ndv
    min, max, distinct count, or null count describes every row, not the surviving subset. So
    a scalar-column or keyless-aggregate metadata answer over a limited subtree would silently
    report the whole-relation value (e.g. `min` over `t ORDER BY x DESC LIMIT 2` reading the
    global minimum) and disagree with execution. Detecting a limiter here forces execution.

    A per-partition top-N is the same hazard in a different shape: ``distinct(subset)`` and a
    ``QUALIFY <rank> <= k`` lower to `Filter(<rank> <= k)` over a `Window` with a ranking
    function, which the optimizer fuses into a row-dropping `rank_limit` window. It keeps only
    the top rows per partition, so a non-key column's whole-relation min/max/null-count/ndv
    describes rows the result no longer contains (e.g. `min(x)` over ``distinct(["g"],
    keep="first", order_by="y")`` reads the global minimum, not the survivors'). It too must
    force execution — see `_has_rank_reduction`.

    A plain `Sort` with no limit is row- and stat-preserving, so it is not a limiter.
    """
    from batcher.plan.logical import Limit, Sort
    from batcher.plan.visitor import walk

    for n in walk(node):
        if isinstance(n, Limit):
            return True
        if isinstance(n, Sort) and n.limit is not None:
            return True
    return _has_rank_reduction(node)


def _has_rank_reduction(node: LogicalPlan) -> bool:
    """Whether the subtree keeps only some rows per key (a keyed dedup / QUALIFY top-N).

    Two shapes do this. A **keyed** `Distinct` keeps one row per key, so a non-key column
    ends up holding a subset of its values and the whole-relation min/max/null-count no
    longer describes the output. A ``QUALIFY <rank> <= k`` lowers to `Filter(<rank> <= k)`
    over a `Window` with a single ranking function (`row_number`/`rank`/`dense_rank`), which
    the `qualify_to_partition_topn` rule then fuses into a row-dropping `rank_limit` window;
    either form of that pair (pre- or post-fusion) drops rows the same way.

    A **whole-row** `Distinct` is not here and must not be: it preserves the value set of
    every column, so min/max stay answerable from the source statistics (only the null count
    goes, which its own stats already drop).

    `distinct(subset)` used to be a `Filter` over a ranking `Window` and so was caught by the
    second shape alone. It is now its own operator, and nothing about the window shape would
    have found it — the answer would have come back from the footer statistic, without
    executing and without an error.
    """
    from batcher.plan.expr_ir.walk import referenced_columns
    from batcher.plan.ir_tags import WINDOW_RANKING
    from batcher.plan.logical import Distinct, Filter, Window
    from batcher.plan.visitor import children

    def visit(n: LogicalPlan, filtered_cols: frozenset[str]) -> bool:
        if isinstance(n, Distinct) and n.keys:
            return True
        if isinstance(n, Window):
            if n.rank_limit is not None:
                return True
            ranking = {f.alias for f in n.functions if f.func in WINDOW_RANKING}
            if ranking & filtered_cols:
                return True
        if isinstance(n, Filter):
            filtered_cols = filtered_cols | referenced_columns(n.predicate)
        return any(visit(child, filtered_cols) for child in children(n))

    return visit(node, frozenset())


def _dedups_nulls(node: LogicalPlan) -> bool:
    """Whether the subtree collapses duplicate rows in a way a whole-relation null count can't see.

    A `Union(distinct=True)` deduplicates across branches, so any repeated null row folds to a
    single row — yet the source statistics sum each branch's null count (correct for `UNION
    ALL`, an over-count under `UNION`). `answer_null_count` reads that summed count as `EXACT`
    and reports it, disagreeing with execution (e.g. two branches each with one null give a
    metadata null count of 2 where the deduplicated result holds 1). A plain `Distinct` node is
    already handled (its stats drop the null count), and `distinct(subset)` is caught by
    `_has_rank_reduction`; the distinct `Union` is the remaining shape, so `null_count` must
    decline over it and execute.
    """
    from batcher.plan.logical import Union
    from batcher.plan.visitor import walk

    return any(isinstance(n, Union) and n.distinct for n in walk(node))


def _source_stats(sources: list[Source], precomputed: list | None) -> list:
    """The caller's already-collected source stats, else collect them now."""
    if precomputed is not None:
        return precomputed
    from batcher import core
    from batcher.api.orchestration import collect_source_stats

    return collect_source_stats(sources, core.default_hub())


def metadata_count(
    plan: LogicalPlan, sources: list[Source], source_stats: list | None = None
) -> int | None:
    """The metadata-only result row count, or None if not provably exact."""
    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core, kyber

    try:
        stats = _source_stats(sources, source_stats)
        hub = core.default_hub()
        # The general structural count first (whole-relation exact rows); then the
        # filtered-count shapes it can't make exact (WHERE col IS NULL → null_count,
        # WHERE col > max → 0, …) answered directly from the child's EXACT column stats.
        result = kyber.answer_count(plan, sources, stats, hub)
        if result is not None:
            return result
        return kyber.answer_filter_count(plan, sources, stats, hub)
    except Exception as exc:  # the metadata shortcut must never break a runnable query
        note_suppressed("api", "answer count() from metadata", exc)
        return None


def metadata_is_empty(
    plan: LogicalPlan, sources: list[Source], source_stats: list | None = None
) -> bool | None:
    """Whether the result is empty from metadata, or None if not provably known."""
    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core, kyber

    try:
        stats = _source_stats(sources, source_stats)
        hub = core.default_hub()
        # Structural emptiness first (EXACT zero rows); then the filtered-emptiness
        # shapes it can't prove (WHERE col > max, WHERE col = <out-of-range>, …).
        result = kyber.answer_is_empty(plan, sources, stats, hub)
        if result is not None:
            return result
        return kyber.answer_filter_is_empty(plan, sources, stats, hub)
    except Exception as exc:  # the metadata shortcut must never break a runnable query
        note_suppressed("api", "answer is-empty from metadata", exc)
        return None


def _has_structural_empty(plan: LogicalPlan) -> bool:
    """Cheap O(nodes) test for a plan that could be *provably* empty from metadata.

    A `Limit(_, 0)` (explicit or the canonical empty marker other rules fold to) is the
    one shape whose emptiness the pre-check proves and the engine could not skip for
    free. Everything else metadata could prove empty — a contradiction filter, an
    always-false predicate, an empty-side join — the execution optimizer already folds
    to an empty marker itself, so executing returns the same zero rows scan-free; only
    the tiny engine-call setup is forgone, never a scan. Gating the (~60ms at small
    scale) full re-optimization behind this keeps the scan-free win for `limit(0)` while
    a normal join/aggregate/sort — which planning already dominates — pays nothing.
    """
    from batcher.plan.logical import Limit
    from batcher.plan.visitor import walk

    return any(isinstance(node, Limit) and node.n == 0 for node in walk(plan))


def metadata_empty_table(
    plan: LogicalPlan, sources: list[Source], source_stats: list | None = None
) -> pa.Table | None:
    """An empty result table (correct schema, zero rows) when metadata proves the plan
    empty, else None — a scan-free answer for a `limit(0)` / provably-empty subtree.

    Gated on a schema inferable without execution (`available_schema`), so it never
    triggers a zero-row run; an un-inferable schema returns None and the caller executes
    (which also yields the empty result). A cheap structural pre-gate (`_has_structural_
    empty`) skips the full metadata re-optimization for the overwhelmingly common
    non-empty plan — the execution path folds any residual emptiness itself.
    """
    inferred = plan.available_schema()
    if inferred is None or not _has_structural_empty(plan):
        return None
    return inferred.arrow.empty_table() if metadata_is_empty(plan, sources, source_stats) else None


def _scalar_answer(kyber_fn, column: str, plan: LogicalPlan, sources, source_stats):
    """Run one EXACT-gated scalar column shortcut (`min`/`max`/`null_count`/…), or None.

    Shares the guard/collect/never-raise scaffolding the count and aggregate shortcuts
    use: it is skipped for a plan the stats machinery can't handle, reuses any
    already-collected source stats, and swallows every error so a runnable query is
    never broken by the optimisation. `kyber_fn` is the matching `answer_*` decision.

    A row-limiter below the plan (`LIMIT` / top-N `Sort`) makes a whole-relation column
    statistic the wrong answer (it describes every row, not the surviving subset), so the
    shortcut declines and the query executes (see `_has_row_limiter`).
    """
    if not _metadata_answerable(plan, sources) or _has_row_limiter(plan):
        return None
    from batcher import core

    try:
        stats = _source_stats(sources, source_stats)
        return kyber_fn(column, plan, sources, stats, core.default_hub())
    except Exception as exc:  # the metadata shortcut must never break a runnable query
        note_suppressed("api", "answer a scalar aggregate from metadata", exc)
        return None


def metadata_min(plan: LogicalPlan, sources: list[Source], column: str, source_stats=None):
    """Exact `min(column)` from metadata, or None to execute."""
    from batcher.kyber.metadata_answer import answer_min

    return _scalar_answer(answer_min, column, plan, sources, source_stats)


def metadata_max(plan: LogicalPlan, sources: list[Source], column: str, source_stats=None):
    """Exact `max(column)` from metadata, or None to execute."""
    from batcher.kyber.metadata_answer import answer_max

    return _scalar_answer(answer_max, column, plan, sources, source_stats)


def metadata_null_count(
    plan: LogicalPlan, sources: list[Source], column: str, source_stats=None
) -> int | None:
    """Exact null count of `column` from metadata, or None to execute."""
    from batcher.kyber.metadata_answer import answer_null_count

    # A distinct `Union` collapses duplicate null rows the summed source null count can't see,
    # so the metadata count would over-report (see `_dedups_nulls`); execute instead.
    if _dedups_nulls(plan):
        return None
    return _scalar_answer(answer_null_count, column, plan, sources, source_stats)


def metadata_n_unique(
    plan: LogicalPlan, sources: list[Source], column: str, source_stats=None
) -> int | None:
    """Exact distinct count of `column` from an EXACT ndv, or None to execute."""
    from batcher.kyber.metadata_answer import answer_n_unique

    return _scalar_answer(answer_n_unique, column, plan, sources, source_stats)


def metadata_has_nulls(
    plan: LogicalPlan, sources: list[Source], column: str, source_stats=None
) -> bool | None:
    """Whether `column` has any null, from metadata, or None to execute."""
    from batcher.kyber.metadata_answer import answer_has_nulls

    return _scalar_answer(answer_has_nulls, column, plan, sources, source_stats)


def metadata_all_null(
    plan: LogicalPlan, sources: list[Source], column: str, source_stats=None
) -> bool | None:
    """Whether every value of `column` is null, from metadata, or None to execute."""
    from batcher.kyber.metadata_answer import answer_all_null

    return _scalar_answer(answer_all_null, column, plan, sources, source_stats)


def metadata_approx_n_unique(
    plan: LogicalPlan, sources: list[Source], column: str, source_stats=None
) -> int | None:
    """Approximate distinct count of `column` from a sketch ndv, or None to execute.

    Explicitly approximate (accepts a SKETCH-provenance HLL ndv the exact `n_unique`
    path rejects), so it only ever backs an `approx_*` terminal.
    """
    if not _metadata_answerable(plan, sources) or _has_row_limiter(plan):
        return None
    from batcher import core
    from batcher.kyber.metadata_answer import approx_count_distinct

    try:
        stats = _source_stats(sources, source_stats)
        return approx_count_distinct(column, plan, sources, stats, core.default_hub())
    except Exception as exc:  # the metadata shortcut must never break a runnable query
        note_suppressed("api", "answer approx_n_unique from metadata", exc)
        return None


def metadata_learned_quantile(
    column: str, q: float, sources: Sequence[object] = ()
) -> float | None:
    """Approximate quantile `q` of `column` from the hub's learned grid, or None.

    A pure hub lookup (no plan estimation): explicitly approximate, backed by the
    `__column_quantiles__` boundaries a past run measured. None when nothing has been
    learned for `column`, so the caller streams an exact-ish sketch instead.

    `sources` identifies *whose* column this is. Learned column statistics are qualified by
    source key — a bare name identifies nothing, since two tables both have an `id` — so
    without it the lookup can only ever match the legacy unqualified shape and misses. Only
    a single-source plan can be attributed unambiguously; anything else stays `None` and
    streams, which is the safe direction.
    """
    from batcher import core
    from batcher.kyber.metadata_answer import answer_learned_quantile
    from batcher.plan.source_stats import source_stats_key

    try:
        key = source_stats_key(sources[0]) if len(sources) == 1 else None
        return answer_learned_quantile(column, q, core.default_hub(), key)
    except Exception as exc:  # the metadata shortcut must never break a runnable query
        note_suppressed("api", "answer a quantile from learned stats", exc)
        return None
