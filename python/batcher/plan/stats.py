"""`plan.stats` — the neutral statistics algebra shared across every layer.

Batcher is metadata-first: wherever a query can be answered (or a plan pruned)
from statistics *without touching a row*, it should be. That requires one shared
vocabulary for "what do we know about a relation, and how much do we trust it",
usable by Kyber (which propagates and consumes stats), Core (which measures
them), Carbonite (which budgets from them), and the API (which answers terminals
from them). Because those subsystems must not import one another, the vocabulary
lives here in the neutral `plan` layer.

Two record types share one trust scale:

  - `RelStats`     — what the estimator *propagates* through a plan: a relation's
                     row count + per-column `ColumnStat`, each tagged with
                     `Provenance`.
  - `ColumnStat`   — min/max/null_count/ndv/sum for one column.

The single most important distinction is `Provenance.EXACT` vs everything else:
an EXACT statistic is provably correct without execution, so a terminal answered
from it (e.g. `count()`) is guaranteed to equal the executed answer. Any other
provenance may only *inform* cost/cardinality or power an explicitly-named
`approx_*` terminal — it must never silently answer an exact query.

`SourceStatistics` (what a connector declares) lives in the sibling
`plan.source_stats` module and bridges into a `RelStats` for a `Scan` leaf.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ColumnStat", "Provenance", "RelStats", "weakest"]


class Provenance(enum.IntEnum):
    """How a statistic was obtained — ordered strongest-trust first.

    Declared as an `IntEnum` so trust composes with `max`: combining statistics
    of differing provenance yields the *weakest* (largest) of them. Only
    `EXACT` is safe to answer a query from without execution; the rest inform
    cost, cardinality, and pruning, or power opt-in `approx_*` terminals.
    """

    EXACT = 0  # provably correct without execution (footer/manifest, or exact-from-exact inputs)
    HISTOGRAM = 1  # KLL / TDigest / DDSketch quantile sketch measured from data
    SKETCH = 2  # HLL distinct / Count-Min frequency measured from data (approximate)
    LEARNED = 3  # learned prior from a past run, keyed by plan signature
    DEFAULT = 4  # Selinger heuristic / an unconstrained guess

    @property
    def is_exact(self) -> bool:
        """True iff a value with this provenance may answer an exact terminal."""
        return self is Provenance.EXACT

    def __str__(self) -> str:
        # Lowercase name for human-readable explain output and telemetry strings
        # (IntEnum's default `str` is the integer value, which is useless here).
        return self.name.lower()


def weakest(*provenances: Provenance) -> Provenance:
    """The least-trusted of the given provenances — the *only* combiner.

    Deriving a statistic from inputs of mixed provenance must route through this
    function; no call site may hand-set `EXACT` on a derived facet. That single
    rule is the firewall against a mislabelled-EXACT statistic answering a query
    incorrectly. Empty input is treated as fully unknown (`DEFAULT`).
    """
    if not provenances:
        return Provenance.DEFAULT
    return max(provenances)


@dataclass(frozen=True, slots=True)
class ColumnStat:
    """Per-column statistics with a single trust tag.

    Every field is optional — a connector or operator fills only what it knows.
    `provenance` applies to the whole bundle; a column carried through a filter,
    for instance, keeps its `min`/`max` as valid *bounds* but downgrades
    provenance away from `EXACT` because the filter may have dropped the
    extremes.
    """

    min: Any | None = None
    max: Any | None = None
    null_count: float | None = None
    ndv: float | None = None  # number of distinct values
    total_sum: float | None = None  # only when a catalog/format records it; enables exact sum()
    provenance: Provenance = Provenance.DEFAULT
    # A serialized membership bloom over the column's values (a `BloomIndex`), for
    # data-skipping an equality/`IN` predicate the way min/max skip a range. It
    # survives row-shrinking ops: removing rows never adds a value, so "absent from
    # the bloom" still proves absence in any subset — independent of `provenance`
    # (the bloom is consulted only to prove *absence*, never to answer a value).
    bloom: bytes | None = None

    def downgrade(self, floor: Provenance) -> ColumnStat:
        """Return a copy whose provenance is weakened to at least `floor`.

        Used by row-shrinking operators (filter, limit, join) that preserve the
        *values* as bounds but can no longer vouch for them as exact extremes. The
        bloom is preserved — it stays a sound absence proof over any subset.
        """
        return ColumnStat(
            min=self.min,
            max=self.max,
            null_count=self.null_count,
            ndv=self.ndv,
            total_sum=self.total_sum,
            provenance=weakest(self.provenance, floor),
            bloom=self.bloom,
        )


@dataclass(frozen=True, slots=True)
class RelStats:
    """A relation's statistics as propagated through a plan.

    `rows`/`provenance` drive the row-count shortcut (a terminal `count()` is
    answerable iff `rows_exact`); `columns` carries per-column `ColumnStat` for
    aggregate (`min`/`max`/`sum`/`count_distinct`) and pruning shortcuts;
    `sorted_by` records a physical ordering an order-preserving operator can
    carry, letting a redundant `Sort` be elided. By contract it lists only a
    *canonical* ascending, nulls-last column prefix — the one ordering a producer
    and a consumer can compare unambiguously (a descending or nulls-first key is
    simply not recorded).
    """

    rows: float
    provenance: Provenance
    columns: Mapping[str, ColumnStat] = field(default_factory=dict)
    sorted_by: tuple[str, ...] = ()

    @property
    def rows_exact(self) -> bool:
        """True iff `rows` is provably correct without execution."""
        return self.provenance.is_exact

    def column(self, name: str) -> ColumnStat:
        """`ColumnStat` for `name`, or an empty (all-unknown) one if absent."""
        return self.columns.get(name, ColumnStat())
