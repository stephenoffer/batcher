"""`plan.source_stats` — what a connector declares about a source, cheaply.

A `SourceStatistics` is the metadata a connector can produce *without scanning
data*: the row count and byte size it already reads from a Parquet/ORC footer or
a lakehouse manifest, the per-column min/max/null/distinct those footers carry,
the columns the source is physically ordered by, and its partition keys. The
connectors live in `io/`, but the contract is neutral and lives here so Kyber's
estimator can consume it without importing `io` (which the layer rules forbid):
the conductor (`api`/`core`) collects per-source statistics at plan-build time
and threads them into the estimator alongside the sources themselves.

`to_relstats()` bridges a source's declared statistics into the `RelStats` a
`Scan` leaf starts from. The `exact_rows` flag is the gate between a footer/
manifest count (exact — may answer `count()`) and an estimate such as Postgres
`reltuples` or Mongo `estimatedDocumentCount` (informs cost only).
"""

from __future__ import annotations

import itertools
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field

from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = ["SourceStatistics", "source_stats_key"]


# Per-instance serials for shape-keyed sources, and the counter that issues them.
#
# Weak keys, so an entry dies with its source and the table cannot grow without bound; a
# monotonic counter, so a serial is never reused even though an address is. `id()` was the
# previous key and is the bug: CPython hands the next object the address of the one just
# freed, which for a transient in-memory frame is immediate.
_SERIALS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_NEXT_SERIAL = itertools.count(1)
_SERIAL_LOCK = threading.Lock()


def _instance_serial(source: object) -> int:
    """A process-unique, never-reused serial for `source`, allocated on first use.

    Falls back to `id()` for a source that cannot be weak-referenced, which is exactly the
    pre-existing behavior — a collision there is no worse than it already was, and refusing
    to key such a source at all would silently stop it learning anything.
    """
    try:
        with _SERIAL_LOCK:
            serial = _SERIALS.get(source)
            if serial is None:
                serial = next(_NEXT_SERIAL)
                _SERIALS[source] = serial
            return serial
    except TypeError:  # not weak-referenceable
        return id(source)


def source_stats_key(source: object) -> str | None:
    """The key a source's *statistics* are stored under, or `None` if it has none.

    A statistic must be attributed to the source it was measured from — a bare column name
    identifies nothing, since two tables both have an `id` — so every learned column
    statistic is qualified by this key. It is the same discipline the plan cache applies
    (`kyber.plan_cache._source_keys`), and for the same reason:

      - A source with a **data-stable** identity (a file path, a table URI) is keyed by it,
        so what one run measures the next run reads back.
      - A source whose identity is only *shape*-based — in-memory batches, keyed by schema
        and row count — is keyed by a **per-instance serial** instead. Its `identity()` is
        documented to collide across different relations of the same shape, and keying
        statistics on it would re-create the very collision this exists to prevent. Such
        data has no cross-run life anyway, so a process-local key loses nothing.

        The serial is not `id()`, and that distinction is the whole of `_instance_serial`.
        CPython reuses an address the moment an object is freed, and a transient frame is
        freed immediately — so four in-memory sources created in sequence produced **one**
        statistics key between them, each reading and overwriting the others' distinct
        counts, most-common-values and quantile grids. Nothing failed, because a statistic
        never changes a result; the plans were simply built from another relation's data.
      - A source that cannot key itself at all gets `None`: its statistics are simply not
        learned, which is strictly better than filing them where another table reads them.

    Args:
        source: A bound input source.

    Returns:
        The stable key to qualify this source's statistics with, or `None`.
    """
    identity = getattr(source, "identity", None)
    if not callable(identity):
        return None
    prefix = _tenant_prefix()
    if not getattr(source, "stable_stats_identity", True):
        return f"{prefix}obj:{_instance_serial(source)}"
    try:
        return f"{prefix}id:{identity()}"
    except Exception:  # pragma: no cover - a source that cannot key itself
        return None


def _tenant_prefix() -> str:
    """``"<tenant>/"`` inside a `tenant()` block, else ``""``.

    Learned statistics include column `min`/`max` — real values out of real columns — and
    the `MetadataHub` they live in may be a Redis or object-storage backend shared across
    a whole fleet. Unqualified, one tenant's measured bounds are read back by every other
    tenant's optimizer, which is a value leak dressed up as a cost estimate.

    Empty outside a tenant scope, so an un-tenanted deployment keys exactly as before and
    keeps every statistic it has already learned.
    """
    try:
        from batcher.config import active_config

        tenant_id = active_config().tenant.tenant_id
    except Exception:  # pragma: no cover - config unavailable during early import
        return ""
    return f"{tenant_id}/" if tenant_id else ""


@dataclass(frozen=True, slots=True)
class SourceStatistics:
    """Statistics a connector knows about a source without reading its rows.

    All fields are optional; a connector fills what its format/catalog exposes.
    `exact_rows` distinguishes a footer/manifest row count (exact) from a
    catalog estimate (e.g. `reltuples`) — only an exact count may answer a
    terminal. Per-column `ColumnStat` provenance is set by the connector
    (`EXACT` for numeric footer min/max, weaker for byte-truncated string
    bounds or sketch-derived distincts).
    """

    row_count: int | None = None
    byte_size: int | None = None
    columns: Mapping[str, ColumnStat] = field(default_factory=dict)
    # Columns the source is physically sorted by — ascending, nulls-last (the
    # canonical ordering `RelStats.sorted_by` consumes for redundant-sort removal).
    sorted_by: tuple[str, ...] = ()
    partition_keys: tuple[str, ...] = ()
    exact_rows: bool = True
    # How many physical blocks the source is stored in (Parquet row groups, ORC
    # stripes) — the granularity a zone-map prune actually skips at, so a shortcut
    # can report what a scan *would* have read without reading it.
    row_group_count: int | None = None
    # Whether this source's float min/max bounds account for NaN.
    #
    # This is the difference between a bound that can answer `max(f)` and one that
    # cannot. SQL's total order — the one our own `ORDER BY` uses — makes NaN the
    # *greatest* value, but the Parquet spec omits NaN from column statistics, so a
    # footer's `max` is the largest **non-NaN** value and answering `max(f)` from it
    # silently disagrees with executing the query. A source that computes its own
    # bounds over the real values (an immutable in-memory relation) records NaN as
    # the max and may say so here; a footer-derived one leaves this False and every
    # float value-fact falls back to execution. Never set it unless the bounds were
    # produced by an ordering that ranks NaN highest.
    bounds_include_nan: bool = False
    # Whether `byte_size` measures the rows' own **content** rather than their stored
    # encoding — the difference between a figure the width estimator may trust and one it
    # must not.
    #
    # A media, text, or binary listing knows the answer outright: one row *is* one file, so
    # `byte_size / row_count` is the size of a row, full stop. Nothing in the schema can
    # supply that — `column_bytes` sees a `binary` column and returns its 36-byte prior,
    # which for a directory of 200 MB videos is wrong by six orders of magnitude.
    #
    # A **columnar** source is the opposite case, and it is why this is opt-in rather than
    # inferred. A Parquet footer's `total_byte_size` is the encoded, row-group-padded stored
    # size, which measures something related to but distinct from the materialized Arrow
    # width — and the per-column type sum is already a good model of that width. Blending
    # the two there is not a sharpening but a re-tuning, and it was measured as one: taking
    # the footer figure as a floor on TPC-H sf1 moved the type-derived width from 88 to
    # 142 B/row and pushed dimension build sides past the broadcast threshold, taking q9
    # from 55.8 ms to 127.9 (0.84x to 1.60x against DuckDB) with ten other queries slower.
    # The width was *more accurate* and the plans were worse, because the threshold is tuned
    # against the estimate. So a columnar connector leaves this False and the width estimator
    # keeps its type-derived answer; re-tuning the threshold against a sharper width is a
    # separate, benchmark-driven change.
    content_byte_size: bool = False

    def is_empty(self) -> bool:
        """True iff the source is known to contain zero rows."""
        return self.row_count == 0

    def to_relstats(self, *, default_rows: float) -> RelStats:
        """Bridge to the `RelStats` a `Scan` leaf starts from.

        Row provenance is `EXACT` when the count is known and exact, `SKETCH`
        when known but estimated, `DEFAULT` (with `default_rows`) when unknown.
        Column stats are carried through with the provenance the connector set.
        """
        if self.row_count is None:
            rows: float = default_rows
            prov = Provenance.DEFAULT
        elif self.exact_rows:
            rows = float(self.row_count)
            prov = Provenance.EXACT
        else:
            rows = float(self.row_count)
            prov = Provenance.SKETCH
        return RelStats(
            rows=rows,
            provenance=prov,
            columns=dict(self.columns),
            sorted_by=self.sorted_by,
        )
