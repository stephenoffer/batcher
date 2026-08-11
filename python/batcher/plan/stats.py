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

import datetime
import enum
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

__all__ = [
    "AXIS_DATE",
    "AXIS_DATETIME",
    "AXIS_NUMERIC",
    "ColumnStat",
    "Provenance",
    "RelStats",
    "SortOrder",
    "ambiguous_float_bound",
    "arrow_ordinal_axis",
    "as_sort_orders",
    "mismatched_exactness",
    "orderings_satisfy",
    "ordinal_with_axis",
    "weakest",
]

# --------------------------------------------------------------------------- #
# The ordinal axis: one number line per column kind
# --------------------------------------------------------------------------- #
# Core *measures* a column's quantile grid from raw Arrow values; Kyber *consults* it
# with a Python literal from the predicate. Those two are only comparable if both name
# the same number line, and they did not: a `date32` grid is stored in epoch days while
# `datetime.date.toordinal()` counts from year 1 (a 719,163-day offset), and a
# `timestamp[us]` grid is in epoch microseconds while `datetime.timestamp()` is in
# epoch seconds *of the local zone*. Either mismatch puts every predicate literal far
# outside the grid, so a range filter interpolates to ~0 selectivity and the join order
# built on it collapses. Both sides now go through this module.
AXIS_NUMERIC = "numeric"  # the value itself
AXIS_DATE = "epoch_day"  # days since 1970-01-01 (Arrow `date32`'s own unit)
AXIS_DATETIME = "epoch_second"  # seconds since the epoch, UTC

_EPOCH_DATE = datetime.date(1970, 1, 1)

#: Divisor turning an Arrow temporal column's raw ordinal into its axis, by type id.
_ARROW_TEMPORAL_AXES: dict[str, tuple[str, float]] = {
    "date32[day]": (AXIS_DATE, 1.0),
    "date64[ms]": (AXIS_DATE, 86_400_000.0),
    "timestamp[s]": (AXIS_DATETIME, 1.0),
    "timestamp[ms]": (AXIS_DATETIME, 1e3),
    "timestamp[us]": (AXIS_DATETIME, 1e6),
    "timestamp[ns]": (AXIS_DATETIME, 1e9),
}


def ordinal_with_axis(value: Any) -> tuple[str, float] | None:
    """`(axis, position)` of a scalar on its number line, or None if it has no linear order.

    Numbers (and `Decimal`) sit on `AXIS_NUMERIC` as themselves. A `date` sits on
    `AXIS_DATE` at its epoch-day offset and a `datetime` on `AXIS_DATETIME` at its
    epoch-second offset, both matching what Arrow stores — a naive `datetime` is read as
    UTC, because that is how Arrow reads a timestamp with no zone. `bool` is excluded:
    `True` would otherwise interpolate as `1.0`.

    Two values are only comparable when their axes agree, so callers must check the axis
    rather than assume it.

    Args:
        value: The scalar to place.

    Returns:
        The axis name and the position on it, or None for a value with no linear order.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return AXIS_NUMERIC, float(value)
    if isinstance(value, datetime.datetime):
        stamp = value if value.tzinfo is not None else value.replace(tzinfo=datetime.UTC)
        return AXIS_DATETIME, stamp.timestamp()
    if isinstance(value, datetime.date):
        return AXIS_DATE, float((value - _EPOCH_DATE).days)
    return None


def arrow_ordinal_axis(arrow_type: Any) -> tuple[str, float] | None:
    """`(axis, divisor)` placing an Arrow column's raw values on their axis, or None.

    Dividing a measured statistic by `divisor` moves it from the column's storage unit
    onto `axis`, where a predicate literal placed by `ordinal_with_axis` can meet it. A
    non-temporal column needs no conversion and reports `(AXIS_NUMERIC, 1.0)`; a type with
    no linear order (string, list, struct) reports None.

    Args:
        arrow_type: The column's `pyarrow.DataType`.

    Returns:
        The axis name and the divisor, or None for a type with no linear order.
    """
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return None
    if pa.types.is_date(arrow_type) or pa.types.is_timestamp(arrow_type):
        # `timestamp[us, tz=UTC]` and `timestamp[us]` share a storage unit and epoch.
        key = str(arrow_type).split(",")[0].strip()
        if not key.endswith("]"):
            key += "]"
        return _ARROW_TEMPORAL_AXES.get(key)
    if (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_decimal(arrow_type)
    ):
        return AXIS_NUMERIC, 1.0
    return None


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


def mismatched_exactness(bound: Any, literal: Any) -> bool:
    """Whether comparing these two in Python is *more precise* than the engine's comparison.

    A `Decimal` bound against a `float` literal is the case, and it is a silent wrong-answer
    path rather than a missed optimization. Python compares the two exactly: it widens the
    float to the rational it actually represents, so ``Decimal("999999999.990") ==
    999999999.99`` is `False`, because the nearest double to 999999999.99 is
    999999999.9900000095367431640625.

    The engine does not. A decimal column compared against a float literal is promoted to
    Float64 on both sides (`coerce_numeric`, matching DuckDB's DOUBLE-dominates-DECIMAL rule),
    and there the two are the same double and compare equal.

    So a rewrite that reasons about such a pair with Python's comparison proves things the
    engine disagrees with. That is not hypothetical: it folded ``price = Decimal("999999999.99")``
    over a `decimal(20,3)` column to an empty relation, and the query returned no rows while
    executing the same filter returned one. Decimal literals reach the plan as floats (the IR
    has no decimal literal), so every money predicate written the exact way is on this path.

    Two bounds of the same kind are fine, and so is a decimal against an integer — the engine
    widens the integer into the decimal exactly, which is what Python does too.

    Args:
        bound: A recorded `min` or `max` bound.
        literal: The predicate's literal, to be compared against that bound.

    Returns:
        ``True`` if one side is a `Decimal` and the other a `float`, so a Python comparison
        between them cannot be trusted to predict the engine.
    """
    kinds = (isinstance(bound, Decimal), isinstance(literal, Decimal))
    floats = (isinstance(bound, float), isinstance(literal, float))
    return (kinds[0] and floats[1]) or (kinds[1] and floats[0])


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


@dataclass(frozen=True, slots=True, order=True)
class SortOrder:
    """One key of a physical ordering: a column, a direction, and where its nulls sit.

    This is the vocabulary a producer and a consumer of an ordering both speak.
    `RelStats.sorted_by` is a tuple of these, and a `Sort` is redundant exactly when its
    own keys are a prefix of what its input already delivers — compared key for key,
    *including direction and null placement*, because `x ASC` and `x DESC` are different
    orderings and neither satisfies the other.

    Recording the direction is what makes the ordering property useful on real queries.
    A prefix restricted to ascending, nulls-last keys cannot describe ``ORDER BY ts DESC``,
    so the single most common ordered shape in analytics — a recent-first feed, a top-N
    over a timestamp, a descending lakehouse sort key — delivered no ordering at all and
    every consumer of the property was blind to it.

    Null placement is part of the ordering and not a detail: `nulls_first` decides where
    the null rows land, so two orderings differing only there interleave their rows
    differently. It is compared exactly, with one sound relaxation available through
    `orderings_satisfy` for a column proven to hold no nulls, where the two coincide.
    """

    column: str
    descending: bool = False
    nulls_first: bool = False


def as_sort_orders(spec: Iterable[SortOrder | str]) -> tuple[SortOrder, ...]:
    """Normalize an ordering written as column names, `SortOrder`s, or a mix.

    A bare column name means ascending, nulls-last — the default a connector that only
    knows "this file is sorted by `k`" is asserting.

    Args:
        spec: The ordering keys.

    Returns:
        The ordering as `SortOrder` keys.
    """
    return tuple(SortOrder(k) if isinstance(k, str) else k for k in spec)


def orderings_satisfy(
    have: Sequence[SortOrder],
    want: Sequence[SortOrder],
    *,
    non_nullable: Collection[str] = (),
) -> bool:
    """Whether a relation delivering `have` is already ordered as `want` requires.

    An ordering satisfies a requirement when it is a **prefix-extension** of it: rows
    sorted by ``(a, b)`` are also sorted by ``(a,)``, never the reverse. Each key must
    match on column *and* direction.

    `non_nullable` names columns proven to contain no nulls, where `nulls_first` is
    unobservable — there is no null row whose position could distinguish the two
    spellings — so the requirement is satisfied whichever way either side spells it.
    Without that relaxation a `Sort` written with the API's default null placement fails
    to match a source that declared the opposite, and the two describe the same row order.

    Args:
        have: The ordering the relation delivers.
        want: The ordering the consumer requires.
        non_nullable: Columns proven free of nulls.

    Returns:
        True iff no further sort is needed.
    """
    if len(want) > len(have):
        return False
    for got, need in zip(have, want, strict=False):
        if got.column != need.column or got.descending != need.descending:
            return False
        if got.nulls_first != need.nulls_first and need.column not in non_nullable:
            return False
    return True


@dataclass(frozen=True, slots=True)
class RelStats:
    """A relation's statistics as propagated through a plan.

    `rows`/`provenance` drive the row-count shortcut (a terminal `count()` is
    answerable iff `rows_exact`); `columns` carries per-column `ColumnStat` for
    aggregate (`min`/`max`/`sum`/`count_distinct`) and pruning shortcuts;
    `sorted_by` records a physical ordering an order-preserving operator can
    carry, letting a redundant `Sort` be elided. It is a prefix of `SortOrder`
    keys, each naming a column, a direction, and its null placement, so a
    descending or nulls-first ordering is carried as faithfully as an ascending one.
    """

    rows: float
    provenance: Provenance
    columns: Mapping[str, ColumnStat] = field(default_factory=dict)
    sorted_by: tuple[SortOrder, ...] = ()

    def non_null_columns(self) -> frozenset[str]:
        """Columns this relation proves hold no nulls — the `non_nullable` set for ordering.

        Only an EXACT zero null count proves it; an estimate does not, and an unknown
        count certainly does not. The answer is one-sided, so an unproven column simply
        keeps the stricter null-placement comparison.

        Returns:
            The names of columns proven free of nulls.
        """
        return frozenset(
            name
            for name, stat in self.columns.items()
            if stat.null_count == 0 and stat.null_count_is_exact
        )

    @property
    def rows_exact(self) -> bool:
        """True iff `rows` is provably correct without execution."""
        return self.provenance.is_exact

    def column(self, name: str) -> ColumnStat:
        """`ColumnStat` for `name`, or an empty (all-unknown) one if absent."""
        return self.columns.get(name, ColumnStat())
