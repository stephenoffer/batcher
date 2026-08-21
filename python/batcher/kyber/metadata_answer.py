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

import itertools
from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.config import Config
from batcher.kyber.column_tables import QUANTILES_KEY, columns_for
from batcher.kyber.learning import load_learned_stats
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.stats import StatsEstimator
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import Aggregate, LogicalPlan, Project
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
    "exact_null_count",
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
    # `optimize_logical` is the *memoized* form of this exact call. Constructing a fresh
    # `Optimizer` and calling `logical_rewrite` directly bypassed the plan cache entirely, so
    # every `.count()`, `.is_empty()`, `.min()` — and every `Facts` construction, which the
    # shortcut layer builds per accessor — paid a full optimizer run, join-order search
    # included. That flatly contradicted `shortcuts.facts`'s own claim that "a namespace that
    # answers thirty questions about a dataset pays for the plan analysis once".
    rewritten = optimize_logical(plan, config, sources, hub, source_stats)
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

    Returns `{alias: value}` only when `plan` is a keyless `Aggregate` and **every** output
    is exactly derivable from the child's EXACT column stats (e.g. `count(*)`, `min`/`max`
    over footer bounds, `count_distinct` over an exact distinct count). If any output is not
    derivable, returns None so the caller executes — a partial answer is never returned.

    The *rewritten* root may be either the `Aggregate` itself or a `Project` of constants a
    rule already folded it into; both are read the same way (see below).
    """
    if not isinstance(plan, Aggregate) or plan.group_keys:
        return None  # this answers a *global* aggregate; the caller's plan must be one
    rewritten, stats = _root_stats(plan, sources, source_stats, hub, config)

    # The rewrite may have already answered the question. A rule that folds a keyless
    # aggregate to constants reads the *same* EXACT statistics this path does, and leaves a
    # `Project` of literals where the `Aggregate` was — so insisting the root still be an
    # `Aggregate` would refuse to answer precisely the plans the optimizer understood best,
    # and send them to the engine to re-derive a constant. Either shape is accepted; the
    # answer is read from the root's column statistics in exactly the same way.
    if isinstance(rewritten, Aggregate) and not rewritten.group_keys:
        aliases = [spec.alias for spec in rewritten.aggregates]
    elif isinstance(rewritten, Project):
        aliases = [item.alias for item in rewritten.items]
    else:
        return None
    # The answer below reads each output's `min` as *the* value, which is only what a column
    # holds when the relation is one row. A keyless aggregate is one row by definition, so
    # this is free on the shape that reaches here — but the `Project` branch above accepts a
    # root the rewrite produced rather than one this function built, and a `Project` over a
    # multi-row relation would hand back the minimum of a column instead of its aggregate. A
    # bad rewrite would then surface as a silently wrong scalar rather than as a plan bug, on
    # the one path in this module whose whole contract is that it never risks correctness.
    if not stats.rows_exact or int(stats.rows) != 1:
        return None
    if not aliases:
        return None  # nothing to answer; an empty dict would read as "answered"

    answer: dict[str, Any] = {}
    for alias in aliases:
        col = stats.columns.get(alias)
        if col is None or col.provenance is not Provenance.EXACT:
            return None  # at least one output isn't exactly derivable → execute
        answer[alias] = col.min  # constant column: min == max == the value
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


def _has_float_column(columns: set[str], sources: list) -> bool:
    """Whether any of `columns` is a floating-point column in some source's schema."""
    import pyarrow as pa

    for src in sources:
        try:
            schema = src.schema()
        except Exception as exc:  # pragma: no cover - a source that cannot describe itself
            note_suppressed("kyber", "read a source schema", exc)
            continue
        for name in columns:
            idx = schema.get_field_index(name)
            if idx >= 0 and pa.types.is_floating(schema.field(idx).type):
                return True
    return False


def nan_aware_bounds(sources: list, source_stats: list | None) -> bool:
    """Whether **every** source declares that its bounds rank NaN the way SQL does.

    The one gate that decides whether a float column's upper bound (and anything derived
    from it) may answer a query. A source sets `bounds_include_nan` only when it computed
    its own bounds over the real values and recorded NaN as the maximum; a footer-derived
    source cannot, and leaves it False. Unknown or missing statistics count as *not*
    NaN-aware, so the safe answer is the default.
    """
    if source_stats is None or len(source_stats) != len(sources):
        return False
    return bool(sources) and all(
        stat is not None and getattr(stat, "bounds_include_nan", False) for stat in source_stats
    )


def _bound_cannot_answer(column_or_expr: Any, sources: list, source_stats: list | None) -> bool:
    """Whether a **`max`** over this column may NOT be answered from a stored bound.

    A float bound is not a sound answer for `max` unless the source that produced it ranks
    NaN the way SQL does — because the usual producers of a bound deliberately exclude NaN:

    * the KLL quantile sketch drops NaN on `add` ("it has no place in an ordered sketch"),
      which is right for quantiles and fatal for a bound; and
    * the Parquet spec omits NaN from a column's min/max statistics.

    But SQL's total order — the one our own `ORDER BY` uses — makes NaN the *greatest*
    value, so `max(f)` over a column containing a NaN **is** NaN. Answering from such a
    bound returns the largest non-NaN value instead, and the query silently disagrees with
    what executing it would produce. The bound cannot represent the answer, and nothing in
    the stats says whether a NaN was dropped, so the only sound move is to execute.

    A source that computes its bounds itself (an immutable in-memory relation) *does* record
    NaN as the max, and says so via `bounds_include_nan` — its float bound is then a sound
    answer and this returns False. Everything else is gated.

    `min` is deliberately **not** gated at all: because NaN is the greatest value, a dropped
    NaN can never have been the minimum, so a float `min` bound is sound whatever produced
    it. (An all-NaN column has no bound at all, so it falls through to execution rather than
    answering wrongly.) Integers, strings, and temporals have no NaN and answer from metadata
    for both.
    """
    from batcher.plan.expr_ir.walk import referenced_columns

    if nan_aware_bounds(sources, source_stats):
        return False
    if isinstance(column_or_expr, str):
        columns = {column_or_expr}
    else:
        columns = referenced_columns(column_or_expr)
    return _has_float_column(columns, sources)


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
    if _bound_cannot_answer(column, sources, source_stats):
        return None
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    stat = _exact_col(stats, column)
    return None if stat is None else stat.max


def exact_null_count(stats: RelStats, column: str) -> int | None:
    """`column`'s null count iff it is provably exact, else None — the one gate for nulls.

    Deliberately **not** `_exact_col`. That gate asks whether the column's whole statistics
    bundle is EXACT, which is the right question for a *bound* and the wrong one for a null
    count: a Parquet footer records the null count exactly for every type, but a string
    column's min/max may be writer-truncated, so the bundle is DEFAULT and a bundle-gated
    answer threw the exact null count away with the inexact bounds. `null_count_is_exact`
    reads the null count's own provenance, so the two facts are trusted independently.
    """
    stat = stats.columns.get(column)
    if stat is None or stat.null_count is None or not stat.null_count_is_exact:
        return None
    return int(stat.null_count)


def answer_null_count(
    column: str,
    plan: LogicalPlan,
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> int | None:
    """Exact null count of `column` (`count(*) - count(column)`) from metadata, or None.

    Needs an EXACT per-column null count — which a Parquet/ORC footer records per row group,
    for every column type. Any weaker input → None (execute).
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    return exact_null_count(stats, column)


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
    # `_exact_col` vouches for the *bundle*; the ndv carries its own tag, because a Parquet
    # column now holds a measured (SKETCH) distinct count alongside its exact bounds. Only an
    # exact ndv may answer an exact `n_unique`.
    if stat is None or stat.ndv is None or not stat.ndv_is_exact:
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
    nulls = exact_null_count(stats, column)
    return None if nulls is None else nulls > 0


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
    nulls = exact_null_count(stats, column)
    if not stats.rows_exact or nulls is None:
        return None
    return stats.rows > 0 and nulls == int(stats.rows)


def answer_learned_quantile(
    column: str,
    q: float,
    hub: MetadataHub | None = None,
    source_key: str | None = None,
) -> float | None:
    """Approximate quantile `q` of `column` from the hub's learned quantile grid, or None.

    EXPLICITLY approximate — it reads the `__column_quantiles__` KLL boundaries a past
    run measured (SKETCH/HISTOGRAM provenance), so it must only ever back an
    `approx_*` terminal, never an exact one. Returns None when no grid has been learned
    for `column` (the caller then streams an exact-ish sketch instead).

    Resolved through `learning.columns_for`, like every other learned column map. Reading
    `grid[column]` directly could never hit: a bare column name identifies nothing (two
    tables both have an `id`), so `record_column_stats` qualifies every entry as
    `source_key\\x1f column` and refuses to write a source it cannot key. An unqualified
    lookup therefore missed 100% of the time and this shortcut always fell through to the
    streaming TDigest. `columns_for` still honors the legacy unqualified shape, so a hub
    persisted by an older build keeps resolving.
    """
    learned = load_learned_stats(hub) if hub is not None else {}
    grid = columns_for(learned, QUANTILES_KEY, source_key).get(column)
    if not grid:
        return None
    return _value_at_quantile(float(q), grid.get("probs", []), grid.get("values", []))


def _value_at_quantile(q: float, probs: list[float], values: list[float]) -> float | None:
    """Interpolate the value at probability `q` from ascending (`probs`, `values`)
    quantile boundaries — the inverse of the estimator's `_fraction_below`. None if the
    boundaries are unusable."""
    if len(probs) != len(values) or len(values) < 2:
        return None
    # The bracketing search below assumes an **ascending** grid; a corrupted or mis-merged
    # learned grid would otherwise fall through it and interpolate nonsense rather than
    # decline. This function only backs the `approx_*` terminals, so declining is free.
    if any(b < a for a, b in itertools.pairwise(probs)):
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
