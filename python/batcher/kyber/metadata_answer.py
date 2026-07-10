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
from batcher.kyber.learning import QUANTILES_KEY, load_learned_stats
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.stats import StatsEstimator
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import Aggregate, LogicalPlan
from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = [
    "answer_aggregate",
    "answer_all_null",
    "answer_count",
    "answer_has_nulls",
    "answer_is_empty",
    "answer_learned_quantile",
    "answer_max",
    "answer_min",
    "answer_n_unique",
    "answer_null_count",
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


def _exact_col(stats: RelStats, column: str) -> ColumnStat | None:
    """`column`'s `ColumnStat` iff its whole bundle is `Provenance.EXACT`, else None.

    The single firewall for every scalar-column shortcut: a column carried through a
    filter/limit keeps its min/max as *bounds* but downgrades away from `EXACT`, so
    gating here means a value-answer is only ever produced when it is provably correct.
    """
    stat = stats.columns.get(column)
    if stat is None or stat.provenance is not Provenance.EXACT:
        return None
    return stat


def answer_min(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> Any | None:
    """The scalar `min(column)` from an EXACT footer/manifest bound, or None.

    The scalar analog of the aggregate `min` path. Returns None (execute) when the
    bound is not EXACT (e.g. after a filter, or a byte-truncated string bound) or is
    absent (an all-null / empty column, whose true `min` is SQL NULL — a full run
    returns that correctly).
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    return None if stat is None else stat.min


def answer_max(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> Any | None:
    """The scalar `max(column)` from an EXACT footer/manifest bound, or None.

    Mirror of `answer_min`; None (execute) unless the upper bound is provably exact.
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    return None if stat is None else stat.max


def answer_null_count(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> int | None:
    """Exact null count of `column` (`count(*) - count(column)`) from metadata, or None.

    Needs both an EXACT row count and an EXACT per-column null count (a Parquet/ORC
    footer records the latter per row group). Any weaker input → None (execute).
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    if not stats.rows_exact or stat is None or stat.null_count is None:
        return None
    return int(stat.null_count)


def answer_n_unique(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> int | None:
    """Exact distinct count of `column` from an EXACT `ndv`, or None (execute).

    The exact analog of `approx_count_distinct`: it accepts only an `EXACT` distinct
    count, never a sketch (an HLL/Parquet estimate is approximate). Footers seldom
    carry an exact `ndv`, so this fires on constant/derived-exact columns and sources
    that record a true distinct count; otherwise it falls back to a real run.
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    if stat is None or stat.ndv is None:
        return None
    return int(stat.ndv)


def answer_has_nulls(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> bool | None:
    """Whether `column` contains any null, from an EXACT null count, or None (execute)."""
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    if stat is None or stat.null_count is None:
        return None
    return stat.null_count > 0


def answer_all_null(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> bool | None:
    """Whether every value of `column` is null, from EXACT row + null counts, or None.

    True only for a non-empty relation whose null count equals its row count (an empty
    relation is *not* reported all-null). Needs both counts EXACT; else None (execute).
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    if not stats.rows_exact or stat is None or stat.null_count is None:
        return None
    return stats.rows > 0 and int(stat.null_count) == int(stats.rows)


def answer_learned_quantile(
    column: str,
    q: float,
    hub: MetadataHub | None = None,
) -> float | None:
    """Approximate quantile `q` of `column` from the hub's learned quantile grid, or None.

    EXPLICITLY approximate — it reads the `__column_quantiles__` KLL boundaries a past
    run measured (SKETCH/HISTOGRAM provenance), so it must only ever back an
    `approx_*` terminal, never an exact one. Returns None when no grid has been learned
    for `column` (the caller then streams an exact-ish sketch instead).
    """
    learned = load_learned_stats(hub) if hub is not None else {}
    grid = learned.get(QUANTILES_KEY, {}).get(column)
    if not grid:
        return None
    return _value_at_quantile(float(q), grid.get("probs", []), grid.get("values", []))


def _value_at_quantile(q: float, probs: list[float], values: list[float]) -> float | None:
    """Interpolate the value at probability `q` from ascending (`probs`, `values`)
    quantile boundaries — the inverse of the estimator's `_fraction_below`. None if the
    boundaries are unusable."""
    if len(probs) != len(values) or len(values) < 2:
        return None
    q = max(0.0, min(1.0, q))
    if q <= probs[0]:
        return float(values[0])
    if q >= probs[-1]:
        return float(values[-1])
    for i in range(len(probs) - 1):
        lo, hi = probs[i], probs[i + 1]
        if lo <= q <= hi:
            if hi == lo:
                return float(values[i])
            return float(values[i] + (q - lo) / (hi - lo) * (values[i + 1] - values[i]))
    return None
