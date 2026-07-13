"""Per-source statistics collection for the conductor.

The conductor collects each bound source's `SourceStatistics` at plan-build time and
threads it into Kyber's estimator (zone-map pruning, cardinality). This module owns that
collection — the session cache keyed by data-stable source identity, the resident-source
fast path that skips the O(rows) zone-map scan when the plan carries no predicate to use
it, and the write-side persistence that lets a footerless format read back exact stats it
was written with.

It lives in `api` (like `orchestration`, which re-exports the public names) because it
composes `io` (source statistics), `metadata` (the persisted-stats store), and `core`
(sketching a written result) — the conductor's privilege. Split out of `orchestration` so
each stays within the module-size budget; the import path is preserved by re-export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.native import engine
from batcher.io.source import Source

if TYPE_CHECKING:
    from batcher.metadata.hub import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "collect_source_stats",
    "column_bounds_needed",
    "invalidate_source_stats",
    "persist_written_source_stats",
]

# Session cache of per-source statistics, keyed by source identity (see collect_source_stats).
_SOURCE_STATS_CACHE: dict[str, object] = {}


def collect_source_stats(
    sources: list[Source], hub: MetadataHub | None, *, need_columns: set[str] | None = None
) -> list:
    """Per-source `SourceStatistics`, from the source itself or the metadata cache.

    A source's own `statistics()` (footer/manifest/catalog) is authoritative for
    the file as it exists now. When a source declares none (a footerless CSV/JSON),
    fall back to statistics Batcher persisted when it *wrote* that path — but marked
    advisory (`exact_rows=False`), since the file may have changed since: cached
    stats sharpen cost and cardinality, they never answer an exact `count()`.

    A **resident** (in-memory) source has no footer, so its `statistics()` *scans every
    row* to build the per-column min/max zone maps (~46 ms at 10M rows, per column — the
    single biggest cost of a fresh-source query). Those bounds only feed predicate pruning
    and range-selectivity, so `need_columns` (the column names whose bounds the plan can
    actually use — see `column_bounds_needed`) restricts the scan to exactly those: an
    empty set keeps only the cheap exact row count, and a small set over a wide relation
    scans a handful of columns instead of all of them. `None` (the default, for callers
    that cannot narrow it) keeps the full scan. Column NDV, which join ordering uses, is
    learned separately and lazily (`seed_column_ndv`), so it is unaffected. File sources
    always take the full path — their footer stats are already cheap.
    """
    from dataclasses import replace

    from batcher.io.source import source_statistics
    from batcher.metadata.source_stats_store import load_source_stats

    out = []
    for s in sources:
        # Footer/manifest statistics are stable for a source's (immutable) file set, but a
        # source's `statistics()` re-reads + re-processes every row-group footer on each
        # call — ~9s for a 100-file TPC-H sf100 read, paid PER QUERY and dwarfing the actual
        # distributed run. Memoize by source identity for the session. The memo is only
        # sound while the path's contents do not change: a column's min/max is a zone map
        # the optimizer uses to prune predicates and join sides, so a *stale* entry yields a
        # wrong answer, not merely a slower plan. `invalidate_source_stats` drops the entry
        # whenever Batcher rewrites the path. Sources without an identity are not cached.
        # In-memory `identity()` is only shape-based, so different data collides; keep its
        # stats out of the shared cache (it self-memoizes). File identities are data-stable.
        stable = getattr(s, "stable_stats_identity", True)
        ident = _source_identity(s)
        if stable and ident and ident in _SOURCE_STATS_CACHE:
            out.append(_SOURCE_STATS_CACHE[ident])
            continue
        narrowed = _resident_subset_stats(s, need_columns) if need_columns is not None else None
        if narrowed is not None:
            # A row-count + only-needed-columns view; not cached, so a later query needing
            # a different column's bounds on the same source still computes them.
            out.append(narrowed)
            continue
        stats = source_statistics(s)
        if stats is None and hub is not None:
            cached = load_source_stats(hub, ident)
            stats = replace(cached, exact_rows=False) if cached is not None else None
        if stable and ident:
            _SOURCE_STATS_CACHE[ident] = stats
        out.append(stats)
    return out


def _resident_subset_stats(source: Source, need_columns: set[str]):
    """Row-count + bounds-for-`need_columns`-only `SourceStatistics` for a resident source.

    Returns `None` — so the caller takes the full `statistics()` path — for a non-resident
    source, one without a cheap row count, or one that cannot compute per-column bounds on
    demand (no `column_bounds`). A resident source computes the exact row count (cheap) and
    the min/max of only the requested columns, skipping the O(rows) pass over every other
    column that the plan's predicates never read.
    """
    if not getattr(source, "resident", False):
        return None
    row_count = getattr(source, "row_count", None)
    if not callable(row_count):
        return None
    rc = row_count()
    if rc is None:
        return None
    from batcher.plan.source_stats import SourceStatistics

    column_bounds = getattr(source, "column_bounds", None)
    if need_columns and not callable(column_bounds):
        return None  # can't narrow — let the caller take the full path
    have = set(source.schema().names) if hasattr(source, "schema") else set()
    columns = {}
    for name in need_columns & have:
        stat = column_bounds(name)
        if stat is not None:
            columns[name] = stat
    return SourceStatistics(row_count=rc, columns=columns)


def column_bounds_needed(plan: LogicalPlan) -> set[str]:
    """The column names whose min/max bounds the plan could consume, from its predicates.

    Only a `Filter` (zone-map pruning + range selectivity) reads a source's column bounds
    on the execution path; a plain group-by / aggregate / sort / join never does (join
    ordering uses NDV, learned separately). So the needed set is the union of every filter
    predicate's referenced columns — empty for a filter-free plan (skip the scan entirely),
    and just the predicate columns for a filter over a wide relation. Computing a column
    the optimizer does not read only wastes work; omitting one only forgoes pruning (never
    changes a result), so a predicate-columns superset is exactly right.
    """
    from batcher.plan.expr_ir import referenced_columns
    from batcher.plan.logical import Filter
    from batcher.plan.visitor import walk

    needed: set[str] = set()
    for node in walk(plan):
        if isinstance(node, Filter):
            needed |= referenced_columns(node.predicate)
    return needed


def invalidate_source_stats(path: str, fmt: str) -> None:
    """Drop the session's cached statistics for a path Batcher has just rewritten.

    A column's min/max is a zone map the optimizer prunes predicates and join sides with,
    and some terminals are answered from statistics without executing — so serving an
    entry that describes a *previous* version of a path yields a wrong answer, not a slow
    plan. Every copy-on-write pattern (`write.merge`, `ds.scd.*`) rewrites a path it reads.
    """
    _SOURCE_STATS_CACHE.pop(f"{fmt}:{path}", None)


def _source_identity(source: Source) -> str:
    identity_fn = getattr(source, "identity", None)
    return identity_fn() if callable(identity_fn) else ""


def persist_written_source_stats(table: pa.Table, path: str, fmt: str) -> None:
    """Persist a freshly-written result's statistics for a future read of `path`.

    Keyed by the read-side identity (`<fmt>:<path>`), so a later `read.<fmt>(path)`
    over a footerless format still finds an exact row count and per-column distinct
    estimates. Best-effort; never breaks a write.
    """
    from batcher import core
    from batcher.metadata.source_stats_store import save_source_stats
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    try:
        from batcher.config import active_config

        cols = table.schema.names
        ndv, _quants, _bytes = core.column_statistics(table.to_batches(), cols)
        index_on = active_config().optimizer.build_bloom_index
        blooms = _build_bloom_index(table, cols) if index_on else {}
        columns = {
            name: ColumnStat(
                ndv=float(ndv[name]) if ndv.get(name) else None,
                provenance=Provenance.SKETCH,
                bloom=blooms.get(name),
            )
            for name in cols
            if ndv.get(name) or blooms.get(name)
        }
        stats = SourceStatistics(
            row_count=table.num_rows, byte_size=table.nbytes, columns=columns, exact_rows=True
        )
        save_source_stats(core.default_hub(), f"{fmt}:{path}", stats)
    except Exception:  # pragma: no cover - persistence must never break a write
        pass


def _build_bloom_index(table: pa.Table, cols: list[str]) -> dict[str, bytes]:
    """A per-column membership bloom for each indexable (int/text) column — the
    data-skipping index `zonemap_prune_filter` consults for equality/`IN`. Built in
    Rust over the result already in memory; unindexable columns yield no entry."""
    nat = engine()
    batches = table.to_batches()
    out: dict[str, bytes] = {}
    for i, name in enumerate(cols):
        bloom = nat.build_column_bloom(batches, i, max(1, table.num_rows))
        if bloom is not None:
            out[name] = bloom
    return out
