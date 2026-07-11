"""Metadata-first resolution of a *keyless aggregate* terminal.

Split from `metadata_answer` on a responsibility seam: this module owns the one shape
whose answer is an aggregate row — `SELECT count(*) / min / max / count(distinct) / avg /
sum … FROM t [WHERE …]` with no GROUP BY. When every output is exactly derivable from EXACT
source statistics (footer/manifest bounds, or an immutable in-memory source's lazily-cached
per-column ndv/mean/sum/value-counts), it returns the one-row answer scan-free — batcher's
learned-metadata moat, a query that gets cheaper the more it runs.

Layer: `api/terminal`, control plane. It decides *whether* a metadata answer exists and
reads it from stats; it never touches a row. Shares the never-raise `_metadata_answerable` /
`_source_stats` scaffolding with `metadata_answer` (imported one-way from there).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.api.terminal.metadata_answer._core import _metadata_answerable, _source_stats

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["is_global_aggregate", "metadata_aggregate_table"]


# Aggregate functions whose result can *ever* be derived from EXACT source
# statistics (row count, footer min/max, exact distinct count). Any other function
# — `sum`, `mean`, `stddev`, … — needs the actual values, so a metadata answer is
# structurally impossible and the (non-trivial) `answer_aggregate` rewrite+estimate
# would only burn ~0.5ms to return `None`. Skipping it when an aggregate is provably
# non-derivable is the dominant fixed cost on a small global-aggregate query (e.g.
# `SELECT sum(x) FROM t`). Conservative: a func in this set is only a *candidate* —
# `answer_aggregate` stays the EXACT-gated authority on whether it truly answers.
_METADATA_DERIVABLE_AGGS = frozenset(
    {
        "count",
        "count_star",
        "count_distinct",
        "min",
        "max",
        # `sum` is derivable when a catalog records an EXACT total; `bool_and`/`bool_or`
        # are derivable from an EXACT min/max on a boolean column. These are only
        # *candidates* — `answer_aggregate` stays the EXACT-gated authority and returns
        # None (→ execute) whenever the child stats can't actually derive them.
        "sum",
        # `mean` (SQL `avg`) is derivable from an EXACT recorded average — an immutable
        # in-memory source computes and caches it per column (`InMemorySource.column_mean`).
        "mean",
        "bool_and",
        "bool_or",
    }
)


def is_global_aggregate(plan: LogicalPlan) -> bool:
    """Whether `plan` is a keyless aggregate, optionally behind output projection(s).

    `SELECT count(*) AS n FROM t` and `ds.agg(n=col(...).count())` both lower to a
    `Project(Aggregate(...))` — the projection just names/forwards the aggregate's
    output — so the bare-`Aggregate` check would miss them and force a full scan of a
    query the footer can answer. `answer_aggregate` propagates stats through the
    projection and is EXACT-gated, so this widened structural guard is safe: it only
    decides *whether to attempt* the metadata answer, never the answer itself.

    Also requires that *every* output aggregate is metadata-derivable in principle
    (`_METADATA_DERIVABLE_AGGS`): a `sum`/`mean`/… aggregate can never be answered
    from stats, so attempting the (non-trivial) metadata rewrite for it is pure waste.
    """
    from batcher.plan.logical import Aggregate, Project

    node = plan
    while isinstance(node, Project):
        node = node.input
    if not isinstance(node, Aggregate) or node.group_keys:
        return False
    return all(spec.agg.func in _METADATA_DERIVABLE_AGGS for spec in node.aggregates)


def _unanswerable_over_memory(plan: LogicalPlan, sources: list[Source]) -> bool:
    """Whether a global aggregate is provably non-metadata-answerable over in-memory data.

    An in-memory relation carries only its row count — no per-column min/max, null, or
    distinct statistics — so the *only* aggregate exactly derivable from it is an
    unfiltered ``COUNT(*)``. A `min`/`max`/`sum`/`count(col)`/… aggregate, or any
    row-level `Filter` (which no in-memory zonemap can prune to an exact answer), always
    falls through to execution. The (optimize-heavy) `answer_aggregate` probe is then
    pure per-query overhead — skip it. File sources keep the full probe (footer/manifest
    stats may still answer a min/max or a partition-pruned count).
    """
    from batcher.io.source import InMemorySource
    from batcher.plan.logical import Aggregate, Filter, Project
    from batcher.plan.visitor import children

    if not sources or not all(isinstance(s, InMemorySource) for s in sources):
        return False
    node = plan
    while isinstance(node, Project):
        node = node.input
    # `count_star` needs only the row count; `min`/`max`/`count_distinct` are answered from
    # the source's EXACT per-column bounds/ndv (`InMemorySource.statistics` / `.column_ndv`),
    # so those keyless aggregates are worth the metadata probe. `sum`/`avg`/`count(col)`/…
    # have no in-memory metadata to derive them, so skip the (optimize-heavy) probe.
    _metadata_derivable = {"count_star", "min", "max", "count_distinct", "mean", "sum"}
    if not isinstance(node, Aggregate) or any(
        spec.agg.func not in _metadata_derivable for spec in node.aggregates
    ):
        return True
    stack: list[LogicalPlan] = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Filter):
            return True
        stack.extend(children(n))
    return False


def _enrich_count_distinct_ndv(plan: LogicalPlan, sources: list[Source], stats: list) -> list:
    """Fill in EXACT per-column ndv for a keyless ``COUNT(DISTINCT col)`` over a single
    in-memory source, so ``answer_aggregate`` can derive ``count_distinct(col) = col.ndv``.

    The ndv is computed lazily by the source (one distinct pass per column, cached) — only
    the column a count-distinct references pays it, so the cheap `MIN`/`MAX`/`COUNT(*)`
    answers never do. The caller only reaches here for an *unfiltered* aggregate (the gate
    skips any `Filter`), so the source's distinct count is exactly the query's answer. A
    computed/renamed input is left alone — its `col` won't resolve on the source, so it
    simply falls through to execution."""
    import dataclasses

    from batcher.io.source import InMemorySource
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Aggregate, Project
    from batcher.plan.stats import ColumnStat, Provenance

    if len(sources) != 1 or not isinstance(sources[0], InMemorySource) or not stats or not stats[0]:
        return stats
    src, source_stat = sources[0], stats[0]
    node = plan
    while isinstance(node, Project):
        node = node.input
    if not isinstance(node, Aggregate):
        return stats
    ndv_cols = {
        spec.agg.input.name
        for spec in node.aggregates
        if spec.agg.func == "count_distinct" and isinstance(spec.agg.input, Col)
    }
    mean_cols = {
        spec.agg.input.name
        for spec in node.aggregates
        if spec.agg.func == "mean" and isinstance(spec.agg.input, Col)
    }
    sum_cols = {
        spec.agg.input.name
        for spec in node.aggregates
        if spec.agg.func == "sum" and isinstance(spec.agg.input, Col)
    }
    columns = dict(source_stat.columns)
    changed = False
    for name in ndv_cols:
        existing = columns.get(name)
        if existing is not None and existing.ndv is not None:
            continue
        ndv = src.column_ndv(name)
        if ndv is None:
            continue
        columns[name] = (
            dataclasses.replace(existing, ndv=ndv, provenance=Provenance.EXACT)
            if existing is not None
            else ColumnStat(ndv=ndv, provenance=Provenance.EXACT)
        )
        changed = True
    for name in mean_cols:
        existing = columns.get(name)
        if existing is not None and existing.mean is not None:
            continue
        mean = src.column_mean(name)
        if mean is None:
            continue
        columns[name] = (
            dataclasses.replace(existing, mean=mean, provenance=Provenance.EXACT)
            if existing is not None
            else ColumnStat(mean=mean, provenance=Provenance.EXACT)
        )
        changed = True
    for name in sum_cols:
        existing = columns.get(name)
        if existing is not None and existing.total_sum is not None:
            continue
        total = src.column_sum(name)
        if total is None:
            continue
        columns[name] = (
            dataclasses.replace(existing, total_sum=total, provenance=Provenance.EXACT)
            if existing is not None
            else ColumnStat(total_sum=total, provenance=Provenance.EXACT)
        )
        changed = True
    if not changed:
        return stats
    return [dataclasses.replace(source_stat, columns=columns), *stats[1:]]


def metadata_aggregate_table(
    plan: LogicalPlan, sources: list[Source], source_stats: list | None = None
) -> pa.Table | None:
    """One-row result of a global aggregate from metadata, or None to execute.

    Returns a single-row Arrow table when the plan is a keyless aggregate (optionally
    behind output projections) whose every output is exactly derivable from source
    statistics (e.g. `count(*)`, `min`/`max` over footer bounds). The cheap structural
    guard runs first so non-aggregate collects pay nothing.
    """
    if not is_global_aggregate(plan):
        return None
    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core, kyber

    try:
        stats = _source_stats(sources, source_stats)
        # A keyless `COUNT(*)` over a `Filter` (`WHERE col = v` / `col <> v`, …): the filter
        # defeats the whole-relation row-count shortcut, but the EXACT-gated filter-count
        # layer can still derive the surviving count from a learned per-value count. Try it
        # before the in-memory gate rejects the plan for carrying a filter.
        filtered = _answer_filtered_count_star(plan, sources, stats)
        if filtered is not None:
            return filtered
        if _unanswerable_over_memory(plan, sources):
            return None
        stats = _enrich_count_distinct_ndv(plan, sources, stats)
        answer = kyber.answer_aggregate(plan, sources, stats, core.default_hub())
        if answer is None:
            # `answer_aggregate` estimates the *rewritten* plan, where `count_distinct` has
            # been lowered to a `count` over a distinct sub-plan it can't derive. Retry on the
            # ORIGINAL node, whose `count_distinct(col)` derives directly from the (enriched)
            # EXACT ndv — the min/max/count path is unchanged, this only adds count-distinct.
            answer = _answer_keyless_aggregate_direct(plan, sources, stats)
    except Exception:  # the metadata shortcut must never break a runnable query
        return None
    if answer is None:
        return None
    return pa.table({alias: [value] for alias, value in answer.items()})


def _answer_filtered_count_star(
    plan: LogicalPlan, sources: list[Source], stats: list
) -> pa.Table | None:
    """One-row `COUNT(*)` table for a keyless single `count_star` over a `Filter`, or None.

    The plain aggregate path can't reach this: `is_global_aggregate` admits it, but the
    filter makes the whole-relation row count non-EXACT so `answer_aggregate` gives up. Here
    we defer to `answer_filter_count` over the aggregate's *input* (the filtered relation),
    which derives an EXACT surviving count from metadata (a learned per-value count for
    `col = v`/`col <> v`, footer bounds for a provably-empty range, …). Restricted to a
    pass-through output projection — every `Projection` a bare `Col` — so mapping the single
    surviving count to the single output column is exact; any computed projection
    (`COUNT(*) * 2`) falls through to the estimator-based path instead.
    """
    from batcher import core, kyber
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Aggregate, Filter, Project

    node = plan
    while isinstance(node, Project):
        if any(not isinstance(p.expr, Col) for p in node.items):
            return None  # a computed projection — not a bare forward of the count
        node = node.input
    if not isinstance(node, Aggregate) or node.group_keys or len(node.aggregates) != 1:
        return None
    if node.aggregates[0].agg.func != "count_star":
        return None
    probe = node.input
    while isinstance(probe, Project):
        probe = probe.input
    if not isinstance(probe, Filter):
        return None  # no filter — the plain unfiltered count path already handles it
    count = kyber.answer_filter_count(node.input, sources, stats, core.default_hub())
    if count is None:
        return None
    cols = plan.available_columns()
    if len(cols) != 1:
        return None
    return pa.table({cols[0]: [count]})


def _answer_keyless_aggregate_direct(plan: LogicalPlan, sources: list[Source], stats: list):
    """Answer a keyless aggregate from the ORIGINAL (un-rewritten) plan's EXACT estimate.

    Only used when the rewriting `answer_aggregate` gives up (e.g. `count_distinct`, lowered
    by the optimizer to a `count` over a distinct it can't derive). The estimator carries the
    aggregate's exact output through any pass-through output projection, so every result
    column must be EXACT-derivable — else None (execute)."""
    from batcher.kyber.stats.estimator import StatsEstimator
    from batcher.plan.stats import Provenance

    rstats = StatsEstimator(sources, {}, source_stats=stats, exact_first=True).estimate(plan)
    answer: dict = {}
    for name in plan.available_columns():
        col = rstats.columns.get(name)
        if col is None or col.provenance is not Provenance.EXACT:
            return None
        answer[name] = col.min
    return answer
