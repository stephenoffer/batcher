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

from collections.abc import Mapping
from dataclasses import dataclass, field

from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = ["SourceStatistics", "source_stats_key"]


def source_stats_key(source: object) -> str | None:
    """The key a source's *statistics* are stored under, or `None` if it has none.

    A statistic must be attributed to the source it was measured from — a bare column name
    identifies nothing, since two tables both have an `id` — so every learned column
    statistic is qualified by this key. It is the same discipline the plan cache applies
    (`kyber.plan_cache._source_keys`), and for the same reason:

      - A source with a **data-stable** identity (a file path, a table URI) is keyed by it,
        so what one run measures the next run reads back.
      - A source whose identity is only *shape*-based — in-memory batches, keyed by schema
        and row count — is keyed by **object identity** instead. Its `identity()` is
        documented to collide across different relations of the same shape, and keying
        statistics on it would re-create the very collision this exists to prevent. Such
        data has no cross-run life anyway, so a process-local key loses nothing.
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
    if not getattr(source, "stable_stats_identity", True):
        return f"obj:{id(source)}"
    try:
        return f"id:{identity()}"
    except Exception:  # pragma: no cover - a source that cannot key itself
        return None


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
