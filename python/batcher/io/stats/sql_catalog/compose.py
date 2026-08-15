"""Composing every catalog probe into the one `SourceStatistics` a connector returns."""

from __future__ import annotations

from batcher.io.stats.sql_catalog.columns import catalog_column_stats
from batcher.io.stats.sql_catalog.constraints import _merge_column_stats, constraint_column_stats
from batcher.io.stats.sql_catalog.counts import catalog_byte_size, catalog_row_count
from batcher.io.stats.sql_catalog.probes import RunRows, RunScalar
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat

__all__ = ["sql_statistics"]


def sql_statistics(
    dialect: str,
    table: str,
    *,
    run_scalar: RunScalar,
    run_rows: RunRows | None = None,
) -> SourceStatistics | None:
    """Everything a SQL catalog can cheaply state about `table`, as one `SourceStatistics`.

    Composes the row count, the on-disk byte size, and (where available) both kinds of
    per-column statistic — the *sampled* ones `ANALYZE` recorded and the *declared* ones the
    schema enforces — into a single record for Kyber's estimator. Where the two describe the
    same column the declared fact replaces the sampled one, because it is exact and current
    rather than an estimate from the last analyze (`_merge_column_stats`).

    Returns None only when the catalog yields nothing at all — a count without column stats
    still sharpens cardinality, and column stats without a count still sharpen selectivity,
    so either alone is worth returning.

    Args:
        dialect: The catalog dialect (see `dialect_for_driver`).
        table: The unqualified table name.
        run_scalar: Runs a single-value catalog query.
        run_rows: Runs a multi-row catalog query, for per-column stats. Omit when the
            connector cannot cheaply run one; row count and byte size still apply.

    Returns:
        The composed statistics, or None when the catalog yields nothing.
    """
    base = catalog_row_count(run_scalar, dialect, table)
    byte_size = catalog_byte_size(run_scalar, dialect, table)
    rows = base.row_count if base else None
    columns: dict[str, ColumnStat] = {}
    if run_rows is not None:
        columns = _merge_column_stats(
            catalog_column_stats(run_rows, dialect, table, rows),
            constraint_column_stats(
                run_rows,
                dialect,
                table,
                rows,
                rows_exact=bool(base and base.exact_rows),
            ),
        )
    if base is None and byte_size is None and not columns:
        return None
    if base is None:
        # No row count, but a byte size and/or column stats are still worth carrying.
        return SourceStatistics(byte_size=byte_size, columns=columns, exact_rows=False)
    import dataclasses

    return dataclasses.replace(base, byte_size=byte_size, columns=columns)
