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

__all__ = ["metadata_aggregate_table", "metadata_count", "metadata_is_empty"]


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
        return kyber.answer_count(plan, sources, stats, core.default_hub())
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
        return kyber.answer_is_empty(plan, sources, stats, core.default_hub())
    except Exception:  # the metadata shortcut must never break a runnable query
        return None


def is_global_aggregate(plan: LogicalPlan) -> bool:
    """Whether `plan` is a keyless aggregate, optionally behind output projection(s).

    `SELECT count(*) AS n FROM t` and `ds.agg(n=col(...).count())` both lower to a
    `Project(Aggregate(...))` — the projection just names/forwards the aggregate's
    output — so the bare-`Aggregate` check would miss them and force a full scan of a
    query the footer can answer. `answer_aggregate` propagates stats through the
    projection and is EXACT-gated, so this widened structural guard is safe: it only
    decides *whether to attempt* the metadata answer, never the answer itself.
    """
    from batcher.plan.logical import Aggregate, Project

    node = plan
    while isinstance(node, Project):
        node = node.input
    return isinstance(node, Aggregate) and not node.group_keys


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
        answer = kyber.answer_aggregate(plan, sources, stats, core.default_hub())
    except Exception:  # the metadata shortcut must never break a runnable query
        return None
    if answer is None:
        return None
    return pa.table({alias: [value] for alias, value in answer.items()})
