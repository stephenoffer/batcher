"""Teach the source statistics the facets a source can compute but has not been asked for.

An immutable in-memory relation knows its own exact distinct count, sum, and average — it
just does not compute them eagerly, because each is an O(rows) pass and most queries want
none of them. `InMemorySource` therefore exposes them as *lazy, memoized* per-column
accessors, and this module lifts the ones a caller actually needs into the `SourceStatistics`
the estimator reads.

That laziness is the whole economics of the learned-metadata moat: the first
``COUNT(DISTINCT x)`` pays for the distinct pass, and every subsequent query that needs it —
`n_unique`, `is_unique`, `is_key`, `duplicate_count`, a join-cardinality estimate — reads the
cached answer for free. A footer-backed source is left alone (it has no such accessor); only
an in-memory one is enriched, and only for the columns and facets asked for.

Layer: `api/terminal`, control plane. It moves an already-computed statistic into the record
the optimizer reads; it computes nothing itself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from batcher.plan.stats import ColumnStat, Provenance

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = ["enrich_in_memory"]

# facet name → (the source accessor that computes it, the `ColumnStat` field it fills)
_FACETS = {
    "ndv": ("column_ndv", "ndv"),
    "mean": ("column_mean", "mean"),
    "total": ("column_sum", "total_sum"),
}


def enrich_in_memory(
    sources: list[Source],
    stats: list,
    *,
    ndv: Iterable[str] = (),
    mean: Iterable[str] = (),
    total: Iterable[str] = (),
) -> list:
    """Fill in the EXACT ndv / mean / sum an in-memory source can compute for these columns.

    Returns a new stats list (the input is never mutated) with each requested facet attached
    at `Provenance.EXACT` — sound because the relation is immutable, so a value computed from
    it once is true of it forever. A facet already present is left as it is; a column the
    source cannot resolve (a computed or renamed one) is skipped, and the query simply falls
    back to execution for it.

    Only a *single-source* plan is enriched. With two sources a bare column name does not
    identify a column — both tables may have an `id` — and filing one table's distinct count
    where the other's is read is precisely the statistics-collision bug the source-keying
    discipline exists to prevent.

    Args:
        sources: The plan's bound sources.
        stats: Their `SourceStatistics`, index-aligned with `sources`.
        ndv: Columns whose exact distinct count is wanted.
        mean: Columns whose exact average is wanted.
        total: Columns whose exact sum is wanted.

    Returns:
        The statistics list, with the computable facets attached.
    """
    from batcher.io.source import InMemorySource

    wanted = {"ndv": set(ndv), "mean": set(mean), "total": set(total)}
    if not any(wanted.values()):
        return stats
    if len(sources) != 1 or not isinstance(sources[0], InMemorySource) or not stats or not stats[0]:
        return stats

    source, declared = sources[0], stats[0]
    columns = dict(declared.columns)
    changed = False
    for facet, names in wanted.items():
        accessor, field = _FACETS[facet]
        for name in names:
            if _fill(source, columns, name, accessor, field):
                changed = True
    if not changed:
        return stats
    return [dataclasses.replace(declared, columns=columns), *stats[1:]]


def _fill(
    source: Any, columns: dict[str, ColumnStat], name: str, accessor: str, field: str
) -> bool:
    """Attach one facet of one column, returning whether anything changed.

    The facet is marked `Provenance.EXACT` because the source computed it from the real
    values of an immutable relation — the same standard a Parquet footer's null count meets.
    """
    existing = columns.get(name)
    if existing is not None and getattr(existing, field) is not None:
        return False  # already known — never recompute
    value = getattr(source, accessor)(name)
    if value is None:
        return False  # this column's type has no such statistic — fall back to execution
    if existing is None:
        columns[name] = ColumnStat(**{field: value}, provenance=Provenance.EXACT)
    else:
        columns[name] = dataclasses.replace(existing, **{field: value})
    return True
