"""Answer a per-column *summary* from metadata — Kyber's EXACT-gated describe layer.

A `describe()` / `summary()` terminal wants a per-column stats snapshot — count, null
count, min, max, distinct count. When those come from a Parquet/ORC footer they are
already known, so the whole snapshot can be produced without scanning a row. This module
builds that snapshot from EXACT column statistics, filling **only** the entries it can
prove and omitting the rest so the caller executes for whatever is missing.

The firewall matches the sibling `metadata_answer` module: an entry is included only when
its statistic is `Provenance.EXACT` end to end (an exact footer row count, null count, or
min/max bound; an EXACT distinct count). A filtered or otherwise weakened column carries
no EXACT entry, so `answer_column_summary` returns `None` there and the caller runs the
real describe. The explicitly-approximate `approx_column_summary` additionally exposes an
`approx_n_unique` from a SKETCH (HLL) distinct count — clearly separated from the exact
`n_unique`, never silently answering it.
"""

from __future__ import annotations

from typing import Any

from batcher.config import Config
from batcher.kyber.metadata_answer import _root_stats
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import LogicalPlan
from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = [
    "answer_column_summary",
    "approx_column_summary",
]


def answer_column_summary(
    plan: LogicalPlan,
    columns: list[str],
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> dict[str, dict[str, Any]] | None:
    """EXACT per-column summary `{col: {count, null_count, min, max, n_unique}}`, or None.

    Each column's entry carries only the facets provably derivable from EXACT stats: the
    non-null `count` (rows - null_count, needs EXACT rows + null count), the `null_count`,
    the `min`/`max` footer bounds, and `n_unique` (an EXACT distinct count — footers
    seldom record one, so it is usually omitted). A column with no EXACT facet is left out;
    if no column has any, returns None so the caller runs the real describe.
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    result: dict[str, dict[str, Any]] = {}
    for col in columns:
        entry = _exact_entry(stats, col)
        if entry:
            result[col] = entry
    return result or None


def approx_column_summary(
    plan: LogicalPlan,
    columns: list[str],
    sources: list,
    source_stats: list | None = None,
    hub: MetadataHub | None = None,
    config: Config | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Per-column summary with the EXACT facets plus an `approx_n_unique`, or None.

    Extends `answer_column_summary` with an explicitly-approximate `approx_n_unique` drawn
    from a SKETCH (HLL) distinct count — the count the exact `n_unique` path rejects. It is
    a separate key, so an approximate distinct count never masquerades as the exact one.
    """
    _, stats = _root_stats(plan, sources, source_stats, hub, config)
    result: dict[str, dict[str, Any]] = {}
    for col in columns:
        entry = _exact_entry(stats, col)
        stat = stats.columns.get(col)
        # An entry holding *only* `approx_n_unique` is intentional — surfacing the sketch for
        # a column with no exact facets is the whole point of this variant. But the count must
        # be genuinely approximate *and* genuinely measured, which is two conditions rather
        # than one. When the ndv is exact, `_exact_entry` already reported it as `n_unique`,
        # and repeating it here would label an exact value approximate. And when it is only a
        # DEFAULT *bound* — which is what a scan publishes when nothing measured the column,
        # typically the row count itself — reporting it as an approximate distinct count
        # dresses a guess as a measurement. `approx_*` promises a sketch, not a placeholder.
        if stat is not None and stat.ndv is not None and _is_measured_sketch(stat):
            entry["approx_n_unique"] = int(stat.ndv)
        if entry:
            result[col] = entry
    return result or None


def _is_measured_sketch(stat: ColumnStat) -> bool:
    """Whether `stat.ndv` is an approximate count something actually measured.

    `HISTOGRAM`/`SKETCH` are measurements of the data (KLL, HLL); `LEARNED` is a prior from a
    past run and `DEFAULT` is an unconstrained bound. Only the first pair may be published as
    `approx_n_unique`, and `EXACT` belongs to `n_unique` instead.
    """
    tag = stat.ndv_provenance if stat.ndv_provenance is not None else stat.provenance
    return tag in (Provenance.HISTOGRAM, Provenance.SKETCH)


def _exact_entry(stats: RelStats, col: str) -> dict[str, Any]:
    """The EXACT-derivable summary facets for one column (possibly empty).

    Populated only from a `Provenance.EXACT` column bundle: `null_count` (and the derived
    non-null `count` when the row count is also EXACT), `min`, `max`, and an EXACT `ndv` as
    `n_unique`. An absent facet (e.g. `min` of an all-null column, whose true min is SQL
    NULL) is simply omitted — never guessed.
    """
    entry: dict[str, Any] = {}
    stat = stats.columns.get(col)
    if stat is None or stat.provenance is not Provenance.EXACT:
        return entry
    if stat.null_count is not None:
        entry["null_count"] = int(stat.null_count)
        if stats.rows_exact:
            entry["count"] = int(stats.rows - stat.null_count)
    if stat.min is not None:
        entry["min"] = stat.min
    if stat.max is not None:
        entry["max"] = stat.max
    # `ndv_is_exact`, not the bundle tag: a Parquet column carries EXACT bounds *and* a
    # measured (SKETCH) HLL distinct count, so gating on the bundle alone reported an
    # approximate `n_unique` as exact — the one thing `plan.stats` says every answer path
    # must read `ndv_is_exact` to prevent. The sketch is still surfaced, as the explicitly
    # approximate `approx_n_unique` in `approx_column_summary`.
    if stat.ndv is not None and stat.ndv_is_exact:
        entry["n_unique"] = int(stat.ndv)
    return entry
