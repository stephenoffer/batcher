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
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ColumnStat", "Provenance", "RelStats", "ambiguous_float_bound", "weakest"]


def ambiguous_float_bound(value: Any) -> bool:
    """Whether reasoning about this bound with Python's comparisons could contradict the engine.

    The engine does not order floats the way Python does, and the two disagree in exactly two
    places:

    * **NaN** — the engine's comparisons follow arrow-rs's *total* order, which ranks NaN above
      every number; Python's `>` ranks it nowhere. (And a key path goes further: `bc_runtime::
      keys` canonicalizes every NaN to one value, so a *join* matches NaN to NaN while a
      comparison need not.)
    * **zero** — the engine separates `-0.0` from `0.0` in a comparison (`-0.0 < 0.0` on that
      total order), while Python calls them equal; and the key paths canonicalize them *back*
      together, so a join matches `-0.0` to `0.0` that a `BETWEEN` would have filtered apart.

    Any rewrite that reasons about a float bound with a Python comparison — folding a predicate
    to FALSE, pushing a `BETWEEN` onto a join side, proving two key ranges disjoint — is
    therefore unsound on such a bound, and must decline. Declining costs a scan; not declining
    costs a row. This is the single definition, shared by every such rule, so they cannot drift
    apart on the question of when a float bound may be trusted.

    (The underlying engine divergence is that its scalar float comparisons follow the total
    order rather than IEEE, so `WHERE f = 0.0` misses `-0.0` and disagrees with DuckDB. That
    is a separately-owned question; this guard is sound under either semantics.)

    Args:
        value: A recorded `min` or `max` bound.

    Returns:
        ``True`` if the bound is a float NaN or a float zero, and so cannot be reasoned from.
    """
    return isinstance(value, float) and (math.isnan(value) or value == 0.0)


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
    mean: float | None = None  # recorded average of the non-null values; enables avg()/mean()
    provenance: Provenance = Provenance.DEFAULT
    # A serialized membership bloom over the column's values (a `BloomIndex`), for
    # data-skipping an equality/`IN` predicate the way min/max skip a range. It
    # survives row-shrinking ops: removing rows never adds a value, so "absent from
    # the bloom" still proves absence in any subset — independent of `provenance`
    # (the bloom is consulted only to prove *absence*, never to answer a value).
    bloom: bytes | None = None
    # The three *measured* distributional statistics, carried here rather than in a
    # side map keyed by bare column name. That distinction is the whole point: a
    # relation's statistics must travel **with the relation**, because a column name
    # alone does not identify a column — two tables both have an `id`, and a global
    # `{name: stat}` map silently lets one table's measurement answer for the other's.
    #
    #   `quantiles`  — an ascending quantile grid `{"probs": [...], "values": [...]}`
    #                  (a KLL sketch), for interpolating range selectivity.
    #   `mcv`        — most-common-values `{str(value): frequency}` (Misra-Gries), which
    #                  sharpens equality selectivity on a skewed column far past `1/ndv`.
    #   `avg_bytes`  — measured average width, which makes memory/broadcast sizing
    #                  byte-true for wide columns (strings, embeddings, blobs).
    quantiles: Mapping[str, list[float]] | None = None
    mcv: Mapping[str, float] | None = None
    avg_bytes: float | None = None
    # The distinct count's *own* provenance, when it differs from the bundle's.
    #
    # `provenance` describes the bundle, and that single tag is a real constraint: a Parquet
    # footer gives EXACT min/max/null_count but **no distinct count**, so the only ndv a
    # columnar source can ever have is a measured (HLL) one. With one shared tag, attaching
    # it would tag it EXACT and let an approximate count answer `count_distinct` — so it was
    # refused, and every Parquet column therefore reached the optimizer with **no ndv at
    # all**. Join cardinality then fell back to `max(|L|, |R|)`, every join in a query looked
    # the same size, and join ordering was blind on precisely the workload that matters:
    # TPC-H q9 applied its 5%-selective `part` filter *last* and ran 5.8x slower than DuckDB.
    #
    # Giving the ndv its own tag lets an approximate distinct count ride alongside exact
    # bounds. `ndv_is_exact` is the gate every answer path reads, so a sketch ndv still can
    # never answer an exact `count_distinct` — it only ever informs cost and cardinality.
    ndv_provenance: Provenance | None = None
    # The null count's *own* provenance — the same lesson, learned in the other direction.
    #
    # The bundle's single tag was refusing an **exact** statistic because it sat next to an
    # inexact one. A Parquet footer records every column's null count exactly, whatever the
    # type — but a *string* column's min/max may be byte-truncated by the writer, so the whole
    # bundle was tagged `DEFAULT` and the exact null count went with it. The consequence was
    # quiet and large: `n_null("name")`, `has_nulls("name")`, `null_count()`, `count(name)`, and
    # `dq.not_null("name")` all fell back to a full scan on precisely the columns most real
    # tables are made of. The footer knew the answer; the trust model could not express it.
    #
    # `null_count_is_exact` is the gate every null-answer path now reads, so an exact null count
    # rides alongside untrustworthy bounds — exactly as `ndv_provenance` lets a sketched distinct
    # count ride alongside exact ones.
    null_count_provenance: Provenance | None = None

    @property
    def ndv_is_exact(self) -> bool:
        """True iff `ndv` may answer an exact `count_distinct` (never for a sketch)."""
        tag = self.ndv_provenance if self.ndv_provenance is not None else self.provenance
        return tag.is_exact

    @property
    def null_count_is_exact(self) -> bool:
        """True iff `null_count` may answer an exact question, whatever the bounds are worth."""
        tag = (
            self.null_count_provenance
            if self.null_count_provenance is not None
            else self.provenance
        )
        return tag.is_exact

    def downgrade(self, floor: Provenance) -> ColumnStat:
        """Return a copy whose provenance is weakened to at least `floor`.

        Used by row-shrinking operators (filter, limit, join) that preserve the
        *values* as bounds but can no longer vouch for them as exact extremes. The
        bloom is preserved — it stays a sound absence proof over any subset. So are
        the measured distributional stats: dropping rows can only *shrink* a column's
        support, so its quantile grid and top values remain the best description of it
        we have — and they are only ever read to *estimate*, never to answer.
        """
        return ColumnStat(
            min=self.min,
            max=self.max,
            null_count=self.null_count,
            ndv=self.ndv,
            total_sum=self.total_sum,
            mean=self.mean,
            provenance=weakest(self.provenance, floor),
            bloom=self.bloom,
            quantiles=self.quantiles,
            mcv=self.mcv,
            avg_bytes=self.avg_bytes,
            ndv_provenance=weakest(
                self.ndv_provenance if self.ndv_provenance is not None else self.provenance,
                floor,
            ),
            null_count_provenance=weakest(
                self.null_count_provenance
                if self.null_count_provenance is not None
                else self.provenance,
                floor,
            ),
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
