"""Answer terminals from metadata alone — Kyber's metadata-first decision layer.

Some terminal queries don't need the engine at all: `ds.limit(n).count()` is
`min(n, count(child))`; `count()` over a global aggregate is `1`; `min(x)` /
`max(x)` over a Parquet scan are in the footer; an empty source `is_empty()` is
known from its row count. This module decides *whether* such a query is provably
answerable from metadata and, if so, returns the answer. The conductor
(`api.terminal`) calls these before executing and falls back to a full run when
they return `None` — so a metadata answer is only ever an optimisation, never a
risk to correctness.

The firewall: every answer is gated on `Provenance.EXACT` end to end. An
approximate statistic (an HLL distinct count, a Postgres `reltuples` estimate, a
byte-truncated string bound) never answers an exact terminal — it only informs
cost or powers an explicitly-named `approx_*` terminal. Kyber *decides*; it never
executes or measures.
"""

from __future__ import annotations

from typing import Any

from batcher.config import Config
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.stats import StatsEstimator
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import Aggregate, LogicalPlan
from batcher.plan.stats import Provenance, RelStats

__all__ = [
    "answer_aggregate",
    "answer_count",
    "answer_is_empty",
    "approx_count_distinct",
]


def _root_stats(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None,
    hub: MetadataHub | None,
    config: Config | None,
) -> tuple[LogicalPlan, RelStats]:
    """Rewrite the plan, then estimate its root with an EXACT-first estimator.

    Rewrites run through the optimizer (so pruning/algebra apply); the final
    estimate uses an `exact_first` estimator so a provably-exact structural count
    is never shadowed by a learned (weaker-provenance) measurement from a past
    run — the difference between answering from metadata and falling back to
    execution.
    """
    optimizer = Optimizer(config=config, sources=sources, hub=hub, source_stats=source_stats)
    rewritten = optimizer.logical_rewrite(plan)
    learned = load_learned_stats(hub) if hub is not None else {}
    estimator = StatsEstimator(sources, learned, source_stats=source_stats, exact_first=True)
    return rewritten, estimator.estimate(rewritten)


def answer_count(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> int | None:
    """Exact result row count from metadata, or None if not provably exact."""
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    return int(stats.rows) if stats.rows_exact else None


def answer_is_empty(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> bool | None:
    """Whether the result is empty, from metadata, or None if not provably known."""
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    return (stats.rows == 0) if stats.rows_exact else None


def answer_aggregate(
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> dict[str, Any] | None:
    """The one-row result of a *global* aggregate, from metadata, or None.

    Returns `{alias: value}` only when the plan's root is a keyless `Aggregate`
    and **every** output aggregate is exactly derivable from the child's EXACT
    column stats (e.g. `count(*)`, `min`/`max` over footer bounds,
    `count_distinct` over an exact distinct count). If any output is not
    derivable, returns None so the caller executes — a partial answer is never
    returned.
    """
    rewritten, stats = _root_stats(plan, sources, source_stats, hub, config)
    if not isinstance(rewritten, Aggregate) or rewritten.group_keys:
        return None
    answer: dict[str, Any] = {}
    for spec in rewritten.aggregates:
        col = stats.columns.get(spec.alias)
        if col is None or col.provenance is not Provenance.EXACT:
            return None  # at least one output isn't exactly derivable → execute
        answer[spec.alias] = col.min  # constant column: min == max == the value
    return answer


def approx_count_distinct(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> int | None:
    """Approximate distinct count of `column` from a sketch (HLL) ndv, or None.

    Opt-in and explicitly approximate: it accepts a SKETCH-provenance ndv (which
    the exact `count_distinct` path rejects), so it must only back an
    `approx_count_distinct` terminal, never `n_unique()`/`count_distinct()`.
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    ndv = stats.column(column).ndv
    return int(ndv) if ndv is not None else None
