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

__all__ = ["SourceStatistics"]


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
