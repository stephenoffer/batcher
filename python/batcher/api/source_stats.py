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

import weakref
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher._internal.native import engine
from batcher.io.source import Source
from batcher.plan.ir_tags import RUNNING_AGGREGATES

if TYPE_CHECKING:
    from batcher.metadata.hub import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "build_estimator",
    "collect_source_stats",
    "column_bounds_needed",
    "invalidate_source_stats",
    "persist_written_source_stats",
]

#: The aggregate functions whose *input column's* statistics a rule reads, and therefore the only
#: ones whose input is worth fetching. Each has a named consumer:
#:
#:   `min`/`max`  -> `global_min_max_from_exact_bounds` (the bound *is* the answer) and
#:                   `min_max_of_constant_column`;
#:   `sum`/`mean` -> `sum_of_constant_column` / `mean_of_constant_column`, which need min == max;
#:   `count`      -> `count_of_non_null_column`, which turns `count(x)` into `count(*)` once the
#:                   column's null count is known to be an exact zero.
#:
#: Kept to that list on purpose rather than "every aggregate" — see `column_bounds_needed`, where
#: fetching a column for one aggregate was found to change what an `approx_*` terminal on the *same*
#: column returns.

# Session cache of per-source statistics, keyed by source identity (see collect_source_stats).
_SOURCE_STATS_CACHE: dict[str, object] = {}

# Per-instance memo of the *narrowed* resident view (`_resident_subset_stats`), keyed by the
# source object itself and, within it, by the exact set of columns whose bounds were asked for.
#
# The shared `_SOURCE_STATS_CACHE` above deliberately excludes resident sources, because an
# in-memory `identity()` is shape-based and two different relations of the same shape collide
# on it. Keying on the *object* has no such failure mode: an `InMemorySource` holds a fixed
# list of Arrow batches from construction, so its statistics cannot change, and two distinct
# relations are two distinct keys however alike their schemas are. The nested key is the
# requested column set, which preserves the property the un-memoized version had — a later
# query needing a *different* column's bounds still computes them.
#
# It is weak-keyed so an entry dies with its source; a benchmark or a notebook that builds a
# relation per iteration must not accumulate their statistics forever.
#
# The "keyed on the object" argument rests on the lookup being by *identity*, and a
# `WeakKeyDictionary` looks up by `==`. That is identity here because no source class defines
# `__eq__` (`InMemorySource` is the only `resident` one and does not), so the default
# identity comparison applies. A source that gave itself value equality would silently make
# two relations share an entry — which for zone-map bounds is a wrong answer, not a slow plan.
#
# What it removes is quadratic in the wrong variable. The view is rebuilt on every `collect()`,
# and building it costs one `ColumnStat` per column of the *source* — not per column the query
# reads. On ClickBench's 105-column `hits` that is 105 objects per query, then re-digested by
# the plan cache and re-derived by the estimator, for a query naming one column.
_RESIDENT_SUBSET_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def build_estimator(sources: list[Source], hub: MetadataHub | None):
    """A `CardinalityEstimator` configured exactly as the optimizer configures its own.

    Anything in the conductor that wants to *reason* about a plan's size before executing
    it — the adaptive gate, the common-subplan analysis — has to read the same numbers
    Kyber will plan with, or it is deciding against a different query than the one that
    runs. That means the same learned corrections, the same cardinality config, and the
    same per-source statistics, which is three couplings and exactly the kind of thing
    that drifts when it is spelled out twice.

    Args:
        sources: The plan's bound inputs.
        hub: The metadata hub, or `None` for a cold estimator with no learned corrections.

    Returns:
        The estimator, ready to `estimate(node)`.
    """
    from batcher.config import active_config
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator

    cfg = active_config()
    learned = load_learned_stats(hub) if hub is not None else {}
    return CardinalityEstimator(
        sources,
        learned,
        cfg.optimizer.cardinality,
        source_stats=collect_source_stats(sources, hub),
    )


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
        # The memo key carries the source's content *version*, not just its identity.
        # Keyed on identity alone it survives the path being rewritten — and this memo
        # holds zone maps, so a stale entry prunes against bounds the data no longer has
        # and returns wrong rows rather than a slow plan. `invalidate_source_stats` covers
        # Batcher's own writes; nothing covered an external writer, which is the ordinary
        # case (the upstream job, a Spark run, a compaction). A source that cannot supply
        # a version cheaply keeps the identity-only key it had before.
        key = _cache_key(s, ident)
        if stable and key and key in _SOURCE_STATS_CACHE:
            out.append(_SOURCE_STATS_CACHE[key])
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
        if stable and key:
            _SOURCE_STATS_CACHE[key] = stats
        out.append(stats)
    return out


def _resident_subset_stats(source: Source, need_columns: set[str]):
    """Row-count + bounds-for-`need_columns`-only `SourceStatistics` for a resident source.

    Returns `None` — so the caller takes the full `statistics()` path — for a non-resident
    source, one without a cheap row count, or one that cannot compute per-column bounds on
    demand (no `column_bounds`). A resident source computes the exact row count (cheap) and
    the min/max of only the requested columns, skipping the O(rows) pass over every other
    column that the plan's predicates never read.

    What is narrowed away is the **bounds pass**, not the column. Every other column still
    reports the facts Arrow already knows — its null count and its average width — because
    those are buffer-field reads, not a scan. Narrowing them away too meant a plan whose
    predicates name no columns at all, a `group_by` or a plain scan, received no column
    statistics whatsoever and priced every row from a type prior: a `group_by` over a column
    of 2 KB documents was sized at the 36-byte string prior, 56x under, while the identical
    source under a `filter` reported the true width.

    The result is memoized per (source instance, requested column set) — see
    `_RESIDENT_SUBSET_CACHE` for why that is sound where the identity-keyed session cache is
    not, and for the per-query cost it removes on a wide relation.
    """
    if not getattr(source, "resident", False):
        return None
    row_count = getattr(source, "row_count", None)
    if not callable(row_count):
        return None
    # Per-instance memo (`_RESIDENT_SUBSET_CACHE`): the answer is a pure function of the
    # source's fixed batches and the requested column set, so recomputing it once per
    # `collect()` rebuilds an object identical to the one it just discarded.
    memo_key = frozenset(need_columns)
    try:
        by_columns = _RESIDENT_SUBSET_CACHE.setdefault(source, {})
    except TypeError:  # not weak-referenceable — compute it every time, as before
        by_columns = None
    if by_columns is not None and memo_key in by_columns:
        return by_columns[memo_key]
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
    cheap_stat = getattr(source, "column_cheap_stat", None)
    if callable(cheap_stat):
        for name in have - columns.keys():
            stat = cheap_stat(name)
            if stat is not None:
                columns[name] = stat
    stats = SourceStatistics(row_count=rc, columns=columns)
    if by_columns is not None:
        by_columns[memo_key] = stats
    return stats


def column_bounds_needed(plan: LogicalPlan) -> set[str]:
    """The column names whose per-column statistics the plan could consume.

    Four kinds of operator read a source's column statistics on the execution path, and every
    one of them must be listed here or the rules that depend on it go quietly blind:

    * a `Filter`, for zone-map pruning, range selectivity, and the sargable transposition that
      needs a column's range to prove its arithmetic cannot wrap — the union of its predicate's
      referenced columns;
    * a **`Join`**, for the disjointness proof. If the two sides' key ranges do not overlap,
      no pair can be equal and an inner/semi join emits nothing — provable from four numbers,
      with neither side read (`join_disjoint_keys_to_empty`, `no_match_join_to_preserved_side`);
    * an **`Aggregate`**, whose global `MIN`/`MAX` *is* an exact bound
      (`global_min_max_from_exact_bounds`) — so the whole scan collapses to a literal — and whose
      group key is droppable when the column is constant (`drop_constant_group_key`), and whose
      `count(x)` becomes `count(*)` once the column has a known-zero null count
      (`count_of_non_null_column`). Only the inputs of the aggregates that read them
      (`RUNNING_AGGREGATES`), deliberately: see the caveat below;
    * a **`Sort`**, whose key is droppable when the column is constant
      (`prune_constant_sort_keys`), when an earlier key is already unique
      (`prune_sort_keys_after_unique_key`), or when the relation holds one row
      (`skip_sort_of_single_row`).

    The join half used to be missing, and the omission was self-concealing: the rules were
    written, tested, and correct, but the bounds they needed were never *fetched*, so on a real
    query they had nothing to reason about and a join whose key ranges provably cannot overlap
    ran a full shuffle. A rule that cannot see is indistinguishable from a rule that is absent.

    The aggregate and sort halves were missing the same way and for longer, with a sharper
    symptom: `SELECT min(x), max(x) FROM t` over a resident relation has *no* filter and *no*
    join, so this returned the empty set, the narrowing then dropped every column's min/max, and
    the rule that exists to answer the query from metadata declined and scanned instead. The
    guard against a fifth recurrence is
    `tests/unit/test_source_stats_needed.py::test_every_metadata_shortcut_still_fires_when_narrowed`,
    which drives each shortcut through the *narrowed* statistics the execution path really passes.

    Computing a column the optimizer does not read only wastes a little footer work; omitting one
    only forgoes an optimization, so a superset is *almost* always right.

    **The one exception, and the reason the aggregate half names only `min`/`max`.** Requesting a
    column's bounds materializes its whole `ColumnStat`, quantile grid included, and the
    `kyber.shortcuts.approx` family answers from a sketch **when one exists** — `None` there means
    "nothing measured yet", not "not provable". So making a sketch available can change what an
    `approx_*` terminal returns: `ds.approx_percentile("x", 50)` over `[1, 2, 3, 4]` falls back to
    exact `2.5` with no grid and answers the grid's `2.0` with one. Both are within that API's
    contract, but which one a query gets should not depend on whether an unrelated rule wanted
    bounds. Requesting every aggregate's input column did exactly that, so this asks only for what
    the consumer above actually reads.
    """
    from batcher.plan.expr_ir import Col, referenced_columns
    from batcher.plan.logical import Aggregate, AsofJoin, Filter, Join, Sort
    from batcher.plan.visitor import walk

    needed: set[str] = set()
    for node in walk(plan):
        if isinstance(node, Filter):
            needed |= referenced_columns(node.predicate)
        elif isinstance(node, Join):
            needed |= set(node.left_keys) | set(node.right_keys)
        elif isinstance(node, AsofJoin):
            # The `on` key is an ordering column (a range bound prunes the right side), and the
            # `by` keys are equi-keys like a hash join's.
            needed |= {node.left_on, node.right_on}
            needed |= set(node.left_by) | set(node.right_by)
        elif isinstance(node, Aggregate):
            for key in node.group_keys:
                needed |= referenced_columns(key.expr)
            for spec in node.aggregates:
                if spec.agg.func in RUNNING_AGGREGATES and spec.agg.input is not None:
                    needed |= referenced_columns(spec.agg.input)
        elif isinstance(node, Sort):
            needed |= {k.expr.name for k in node.keys if isinstance(k.expr, Col)}
    return needed


def invalidate_source_stats(path: str, fmt: str) -> None:
    """Drop the session's cached statistics for a path Batcher has just rewritten.

    A column's min/max is a zone map the optimizer prunes predicates and join sides with,
    and some terminals are answered from statistics without executing — so serving an
    entry that describes a *previous* version of a path yields a wrong answer, not a slow
    plan. Every copy-on-write pattern (`write.merge`, `ds.scd.*`) rewrites a path it reads.

    The cache is keyed by the source's `identity()`, which for a lakehouse table carries a
    version suffix (``delta:/t@7``) — so popping the bare ``fmt:path`` matched nothing and
    this never once fired for one. Every key *for this path* is dropped instead, whatever
    version it names.
    """
    prefix = f"{fmt}:{path}"
    for key in [k for k in _SOURCE_STATS_CACHE if k == prefix or k.startswith(prefix + "@")]:
        _SOURCE_STATS_CACHE.pop(key, None)


def _cache_key(source: Source, identity: str) -> str | None:
    """The memo key for `source`: its identity qualified by its content version."""
    if not identity:
        return None
    version_fn = getattr(source, "stats_version", None)
    if version_fn is None:
        return identity
    try:
        version = version_fn()
    except Exception:  # a source that cannot version itself keeps the identity-only key
        return identity
    return identity if version is None else f"{identity}@{version}"


def _source_identity(source: Source) -> str:
    """A source's identity, through the neutral spelling every layer shares.

    Kyber reads back what this writes (a learned read throughput is keyed on it) and the two
    cannot import each other, so the definition lives in `plan` and both call it. Two
    spellings of a key is a store that silently never hits.
    """
    from batcher.plan.source_stats import source_identity

    return source_identity(source)


def _value_bearing_columns_to_redact(table: str, columns: list[str]) -> set[str]:
    """Columns whose *values* must not be persisted into the shared statistics store.

    Some statistics are cardinalities — a row count, a null count, a distinct estimate.
    Those describe the shape of the data and leak nothing about it. Others carry the data:
    `min`/`max` are literally two values out of the column, and a **bloom filter is a
    membership oracle** — holding one for an `ssn` column lets anyone test whether a
    specific SSN is present, without ever reading the table.

    The `MetadataHub` those land in is not necessarily private. Its backends include Redis
    and object storage, which is the whole point (a learned-stats store shared across a
    fleet), so anything written there is readable by everyone with hub access — including
    principals the catalog would never let read the column itself.

    So under an active `security()` block, a column that is masked or invisible to the
    running principal keeps only its cardinalities. Outside one, nothing changes: an
    ungoverned deployment behaves exactly as before.

    Args:
        table: The governed table name — the bare path, which is what `_binding.table_name`
            keys on and what a policy author writes before the table has ever been read.
        columns: The column names about to have statistics persisted.

    Returns:
        The subset of `columns` whose value-bearing statistics must be dropped.
    """
    from batcher.api.security._context import current_security

    context = current_security()
    if context is None:
        return set()
    if not context.catalog.governs(table):
        return set()
    visible = set(context.catalog.visible_columns(table, columns, context.principal))
    redact = {c for c in columns if c not in visible}
    redact |= {
        c for c in columns if context.catalog.mask_for(table, c, context.principal) is not None
    }
    return redact


def persist_written_source_stats(table: pa.Table, path: str, fmt: str) -> None:
    """Persist a freshly-written result's statistics for a future read of `path`.

    Keyed by the read-side identity (`<fmt>:<path>`), so a later `read.<fmt>(path)`
    over a footerless format still finds an exact row count and per-column distinct
    estimates. Best-effort; never breaks a write.

    Value-bearing statistics for governed columns are dropped first — see
    `_value_bearing_columns_to_redact`.
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
        # A bloom over a governed column is a membership oracle for its values, so it must
        # not reach a shared statistics store. The distinct *count* is a cardinality and
        # stays — the optimizer still gets to order joins on this table.
        redacted = _value_bearing_columns_to_redact(path, list(cols))
        columns = {
            name: ColumnStat(
                ndv=float(ndv[name]) if ndv.get(name) else None,
                provenance=Provenance.SKETCH,
                bloom=None if name in redacted else blooms.get(name),
            )
            for name in cols
            if ndv.get(name) or (blooms.get(name) and name not in redacted)
        }
        stats = SourceStatistics(
            row_count=table.num_rows, byte_size=table.nbytes, columns=columns, exact_rows=True
        )
        save_source_stats(core.default_hub(), f"{fmt}:{path}", stats)
    except Exception as exc:  # pragma: no cover - persistence must never break a write
        note_suppressed("api", "persist source statistics", exc)


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
