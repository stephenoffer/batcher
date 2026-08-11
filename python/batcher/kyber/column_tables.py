"""The learned per-column statistics tables — their schema, their keys, and their bound.

Four measured statistics travel from Core to the estimator through the MetadataHub, and all
four have the same storage shape: **one** store entry per statistic, holding a flat
`{source ⟂ column: value}` map. The shape is not incidental. The estimator resolves a whole
source's columns at once, so it reads the map whole; splitting it per column would turn one
read into one per column of every scanned relation.

That makes the map's size the size of a read, a write, and the JSON crossing the backend —
which is why the cap in `_TABLE_MAX` exists rather than being a precaution. Nothing here is
a plan decision; this is the store's schema, so it lives beside the writer that owns it
(`kyber.learning`) and is imported by the estimator that reads it, rather than being
restated as literals on both sides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "AVG_BYTES_KEY",
    "CARDINALITY_CORRECTION_KEY",
    "MCV_KEY",
    "NDV_KEY",
    "QUANTILES_KEY",
    "ROW_BYTES_KEY",
    "STATS_NAMESPACE",
    "columns_for",
    "merge_column_table",
    "qualify",
]

#: The learned-parameter namespace these tables live in, alongside the per-signature entries.
STATS_NAMESPACE = "kyber.stats"

# Reserved keys inside the stats namespace. Everything else in the namespace is keyed by
# a plan signature; these hold the cross-signature column state the `StatsEstimator` reads.
NDV_KEY = "__column_ndv__"  # per-column distinct counts
QUANTILES_KEY = "__column_quantiles__"  # per-column quantile grids
AVG_BYTES_KEY = "__column_avg_bytes__"  # per-column average byte widths
# Per-column byte widths measured the *cheap* way, for every column a query touched rather
# than only the ones a statistic is sketched for.
#
# It is a separate table from `AVG_BYTES_KEY` and must stay one, because that key carries a
# second job: `api.terminal._metadata.learn_column_stats` uses *the presence of an average
# byte width* as its "already sketched" marker, deliberately and for a documented reason.
# Writing a width there for a column nothing sketched would mark it done and cost that column
# its quantiles and most-common-values forever. So the cheap widths live here, the marker
# keeps meaning what it meant, and the estimator reads the sketched width first.
ROW_BYTES_KEY = "__column_row_bytes__"
MCV_KEY = "__column_mcv__"  # per-column most-common-values (skew)
# Derived, not stored: `load_learned_stats` folds the measured q-error history into
# `{signature: correction_factor}` under this key.
CARDINALITY_CORRECTION_KEY = "__cardinality_correction__"
# Derived, not stored: `load_learned_stats` folds `metadata.udf_stats` into
# `{udf_identity: seconds_per_row}` under this key, so the cost model can price a
# `map_batches` by what Core measured its `fn` to cost rather than as a trivial column map.
# Keyed by UDF identity, not by plan signature — the cost of a callable is a property of the
# callable, and the same `fn` under two different plans costs the same per row.
UDF_ROW_SECONDS_KEY = "__udf_row_seconds__"

# Column statistics are keyed by **source, then column** — never by column name alone.
#
# A bare column name does not identify a column. Two tables both have an `id`, a `key`, a
# `date`; a flat `{name: stat}` map merges them, so whichever table was measured last
# silently answers for every other table with a column of that name — process-wide, for
# every join and group-by estimate that reads it. This repo already learned that lesson on
# the *row* side (see `StatsEstimator._estimate_uncached`: every `Scan` shares the
# signature `["scan"]`, so one table's measured 5M rows became a 1,000-row table's
# estimate, and a pruned MERGE sized its join at 2.4 TB). The column maps had the same
# defect and this is the qualifier that closes it.
#
# The key stays a flat string — `f"{source}\x1f{column}"` — so the stored shape is still
# `dict[str, value]` and every backend, the generation-bump check, and the merge logic are
# untouched. `\x1f` (ASCII unit separator) cannot occur in a column name.
_SOURCE_SEP = "\x1f"

# Cap on the entries each table retains. Nothing bounded them, and the key space is one entry
# per (source, column) *ever seen*: a workload over dated partitions or per-tenant files adds
# a fresh source identity every run, so a served deployment grew the map forever and paid the
# whole of it back — parsed, copied, compared, re-serialized — on every query over any source.
#
# The caps differ by an order of magnitude because the payloads do. An ndv or a width is one
# float; a quantile grid is a list of them and an MCV table is a value-to-frequency map, so
# the same entry count costs far more in each.
#
# They are also deliberately *not* as large as they could be. The cap is the size of the copy
# and the JSON a write pays, so a huge cap does not make the table free — it makes the write
# path slow instead of the read path unbounded. These sit an order of magnitude above the
# columns a realistic deployment's hot sources have, which keeps eviction rare, while keeping
# the whole-table write inside a millisecond.
_TABLE_MAX: dict[str, int] = {
    NDV_KEY: 20_000,
    AVG_BYTES_KEY: 20_000,
    # Sized with `AVG_BYTES_KEY`: one float per column, and it covers *every* column a
    # source has rather than the sketched subset, so it fills faster on a wide table.
    ROW_BYTES_KEY: 20_000,
    QUANTILES_KEY: 5_000,
    MCV_KEY: 5_000,
}
_DEFAULT_TABLE_MAX = 20_000


def qualify(source_key: str, column: str) -> str:
    """The store key for `column` **of `source_key`** (see `_SOURCE_SEP`).

    Args:
        source_key: The source's data-stable identity.
        column: The column's name within that source.

    Returns:
        The flat key the learned tables store the measurement under.
    """
    return f"{source_key}{_SOURCE_SEP}{column}"


def columns_for(learned: dict[str, Any], stat_key: str, source_key: str | None) -> dict[str, Any]:
    """The `{column: value}` slice of a learned column map that describes `source_key`.

    Entries written *unqualified* (no separator) are treated as applying to every source.
    That is the legacy shape — a hub persisted by an older build, or a test that seeds the
    map directly — and a source-qualified entry always wins over it. Nothing on the live
    path writes unqualified any more (`record_column_stats` requires a source key), so the
    fallback is a compatibility shim, not a way back into the collision.

    Args:
        learned: The loaded `kyber.stats` namespace.
        stat_key: Which table to slice (`NDV_KEY`, `QUANTILES_KEY`, ...).
        source_key: The source whose columns are wanted, or `None` for the legacy entries.

    Returns:
        Column name to its learned value for this source.
    """
    table = learned.get(stat_key) or {}
    prefix = f"{source_key}{_SOURCE_SEP}" if source_key is not None else None
    out: dict[str, Any] = {}
    qualified: dict[str, Any] = {}
    for key, value in table.items():
        if _SOURCE_SEP not in key:
            out[key] = value  # legacy: unqualified, applies to any source
        elif prefix is not None and key.startswith(prefix):
            qualified[key[len(prefix) :]] = value
    out.update(qualified)  # a measurement of *this* source beats a legacy global one
    return out


def merge_column_table(
    hub: MetadataHub,
    stat_key: str,
    fresh: dict[str, Any],
    existing: dict[str, Any] | None = None,
) -> None:
    """Fold `fresh` into the learned column table at `stat_key`, bounded and least-recent-out.

    Three things happen here that the plain read-copy-update-write did not do.

    **A no-op re-measure costs O(fresh), not O(table).** The measurement loop re-measures the
    same columns of the same source on every query over it, arriving at values already
    stored. The write was already elided downstream, but only after copying the whole map,
    updating it, and comparing it entry by entry against the stored one. Checking the handful
    of fresh keys first skips all of that.

    **The table is bounded.** See `_TABLE_MAX`.

    **Eviction prefers the least recently *written*.** Every key being written moves to the
    end of the insertion order, so the front of the table is what no measurement has touched
    in the longest time. That is a heuristic, not a true LRU, and the gap is worth naming: a
    column measured constantly at an unchanged value takes the early-out and so never moves,
    and can eventually be evicted. What that costs is one re-measure — which the query path
    performs anyway — plus, for the distinct-count table, one plan-cache generation bump. It
    is not worth defeating the early-out to avoid, because doing so would put a whole-table
    copy and re-serialize on the steady-state path of every query against a full table.

    Args:
        hub: The metadata hub.
        stat_key: The reserved key naming the table (`NDV_KEY`, `QUANTILES_KEY`, ...).
        fresh: The newly measured `{qualified_column: value}` entries.
        existing: The stored table, when the caller has already read it.
    """
    if existing is None:
        existing = hub.get_keyed_param(STATS_NAMESPACE, stat_key) or {}
    if all(name in existing and existing[name] == v for name, v in fresh.items()):
        return  # nothing measured that is not already stored
    cap = _TABLE_MAX.get(stat_key, _DEFAULT_TABLE_MAX)
    table = dict(existing)
    for name, value in fresh.items():
        table.pop(name, None)  # re-insert at the end: eviction is least-recently-written
        table[name] = value
    if len(table) > cap:
        for name in list(table)[: len(table) - cap]:
            del table[name]
    hub.put_keyed_param(STATS_NAMESPACE, stat_key, table)
