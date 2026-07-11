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

if TYPE_CHECKING:
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
    except Exception:  # the metadata shortcut must never break a runnable query
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
    except Exception:  # the metadata shortcut must never break a runnable query
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
    """
    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core

    try:
        stats = _source_stats(sources, source_stats)
        return kyber_fn(column, plan, sources, stats, core.default_hub())
    except Exception:  # the metadata shortcut must never break a runnable query
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
    if not _metadata_answerable(plan, sources):
        return None
    from batcher import core
    from batcher.kyber.metadata_answer import approx_count_distinct

    try:
        stats = _source_stats(sources, source_stats)
        return approx_count_distinct(column, plan, sources, stats, core.default_hub())
    except Exception:  # the metadata shortcut must never break a runnable query
        return None


def metadata_learned_quantile(column: str, q: float) -> float | None:
    """Approximate quantile `q` of `column` from the hub's learned grid, or None.

    A pure hub lookup (no plan estimation): explicitly approximate, backed by the
    `__column_quantiles__` boundaries a past run measured. None when nothing has been
    learned for `column`, so the caller streams an exact-ish sketch instead.
    """
    from batcher import core
    from batcher.kyber.metadata_answer import answer_learned_quantile

    try:
        return answer_learned_quantile(column, q, core.default_hub())
    except Exception:  # the metadata shortcut must never break a runnable query
        return None
