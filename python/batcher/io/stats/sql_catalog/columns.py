"""Per-column statistics from a dialect's sampled statistics catalog.

Three dialects expose one cheap enough to read at plan time, and they carry very different
amounts. Postgres (with Redshift and CockroachDB, which are wire-compatible) is the richest:
``pg_stats`` records, per analyzed column, the null fraction, the distinct-value estimate,
the most-common values with their frequencies, and a histogram of bucket bounds, so a
Postgres table reaches Kyber with the same shape a Parquet footer supplies. Oracle carries
the distinct count, the null count and the average width. MySQL carries one figure — an
index's cardinality — which is a distinct count for the column that index leads.

Everything here is a **sample** — ``ANALYZE`` reads a fraction of the table — so every
facet is tagged `SKETCH`/`HISTOGRAM` and none may answer an exact query. The *declared*
statistics that can are in the sibling `constraints` module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from batcher.io.stats.sql_catalog.probes import RunRows, _at, _to_float
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["catalog_column_stats"]


# ---------------------------------------------------------------------------
# Dispatch: one catalog query per dialect, one row parser per catalog shape
# ---------------------------------------------------------------------------


def catalog_column_stats(
    run_rows: RunRows, dialect: str, table: str, row_count: int | None
) -> dict[str, ColumnStat]:
    """Per-column `ColumnStat` from a dialect's statistics catalog, or an empty dict.

    Three catalogs are read, and they differ in how much they carry.

    **Postgres** (and its wire-compatible kin, Redshift and CockroachDB) is the richest:
    ``pg_stats`` records, per analyzed column, the null fraction, the distinct-value
    estimate, the most-common values with frequencies, and a histogram of bucket bounds.
    **Oracle**'s ``all_tab_col_statistics`` carries the distinct count, the null count and
    the average column length, but no frequency or histogram detail cheap enough to read
    here. **MySQL** has no column-statistics view at all, but
    ``information_schema.STATISTICS`` records an index's ``CARDINALITY``, which for the
    *leading* column of an index is an estimate of that column's distinct values — the one
    facet MySQL can supply, and the one join ordering most needs.

    The Postgres facets map onto `ColumnStat` as follows:

      - ``null_frac`` x `row_count` -> ``null_count`` (SKETCH, a sampled estimate).
      - ``n_distinct`` → ``ndv``. Postgres encodes a *ratio* as a negative number
        (``-1`` = every value distinct), which is resolved against `row_count`.
      - ``most_common_vals`` / ``most_common_freqs`` → ``mcv`` (Misra-Gries-shaped),
        sharpening equality selectivity far past ``1/ndv`` on a skewed column.
      - ``histogram_bounds`` → ``quantiles`` (an even quantile grid), for interpolating
        range selectivity.

    Every facet is `SKETCH`/`HISTOGRAM` provenance, so none can answer an exact query;
    they only inform cost and cardinality. Array columns (``most_common_vals``,
    ``histogram_bounds``) arrive as driver-specific text and are parsed tolerantly —
    a column whose arrays don't parse still contributes its scalar null/ndv facets.

    Args:
        run_rows: Runs a catalog query and returns its rows, each a positional sequence.
        dialect: The catalog dialect. Dialects with no column catalog yield an empty map
            without running a query.
        table: The unqualified table name.
        row_count: The table's row count, used to resolve ratios into absolute counts.

    Returns:
        A ``{column_name: ColumnStat}`` mapping (possibly empty).
    """
    if dialect in ("postgres", "redshift"):
        return _rows_to_stats(run_rows, _PG_STATS_QUERY.format(table=table), _pg_row, row_count)
    if dialect == "oracle":
        return _rows_to_stats(run_rows, _ORACLE_QUERY.format(table=table), _oracle_row, row_count)
    if dialect == "mysql":
        return _rows_to_stats(run_rows, _MYSQL_QUERY.format(table=table), _mysql_row, row_count)
    return {}


_PG_STATS_QUERY = (
    "SELECT attname, null_frac, n_distinct, most_common_vals, "
    "most_common_freqs, histogram_bounds FROM pg_stats WHERE tablename = '{table}'"
)

_ORACLE_QUERY = (
    "SELECT column_name, num_distinct, num_nulls, avg_col_len "
    "FROM all_tab_col_statistics WHERE table_name = UPPER('{table}')"
)

#: An index's `CARDINALITY` estimates the distinct values of its **leading** column only —
#: for a later column it counts distinct *prefixes*, which is a different and larger number.
#: `SEQ_IN_INDEX = 1` is what restricts it to the column the figure actually describes.
#: `MAX` collapses the several indexes a column may lead, taking the freshest-looking
#: estimate rather than an arbitrary one.
_MYSQL_QUERY = (
    "SELECT column_name, MAX(cardinality) FROM information_schema.STATISTICS "
    "WHERE table_name = '{table}' AND seq_in_index = 1 GROUP BY column_name"
)


def _rows_to_stats(
    run_rows: RunRows,
    sql: str,
    parse: Any,
    row_count: int | None,
) -> dict[str, ColumnStat]:
    """Run `sql` and fold each row through `parse` into a `{column: ColumnStat}` map.

    Tolerant at both levels: a query that fails yields an empty map, and a single row that
    cannot be parsed is skipped rather than losing the columns beside it.
    """
    try:
        rows = run_rows(sql)
    except Exception:
        return {}
    out: dict[str, ColumnStat] = {}
    for row in rows or ():
        parsed = parse(row, row_count)
        if parsed is not None:
            name, stat = parsed
            out[name] = stat
    return out


def _oracle_row(row: Sequence[Any], _row_count: int | None) -> tuple[str, ColumnStat] | None:
    """One ``all_tab_col_statistics`` row -> ``(column_name, ColumnStat)``.

    ``low_value``/``high_value`` are deliberately not read: Oracle stores them RAW-encoded
    in a per-datatype internal format, so decoding them is a type-directed byte parse that
    would produce a *wrong bound* rather than no bound whenever it got a type wrong — and a
    bound is what a prune is decided from.
    """
    if not row or row[0] is None:
        return None
    ndv = _to_float(_at(row, 1))
    nulls = _to_float(_at(row, 2))
    width = _to_float(_at(row, 3))
    if ndv is None and nulls is None and width is None:
        return None
    return str(row[0]), ColumnStat(
        null_count=nulls,
        ndv=ndv,
        avg_bytes=width,
        provenance=Provenance.SKETCH,
        ndv_provenance=Provenance.SKETCH if ndv is not None else None,
        null_count_provenance=Provenance.SKETCH if nulls is not None else None,
    )


def _mysql_row(row: Sequence[Any], _row_count: int | None) -> tuple[str, ColumnStat] | None:
    """One ``information_schema.STATISTICS`` row -> ``(column_name, ColumnStat)``.

    A cardinality of 0 is discarded rather than recorded: MySQL reports it for a table that
    has never been analyzed as well as for a genuinely empty one, and a zero distinct count
    would make every equality on the column estimate zero rows.
    """
    if not row or row[0] is None:
        return None
    ndv = _to_float(_at(row, 1))
    if not ndv or ndv <= 0:
        return None
    return str(row[0]), ColumnStat(
        ndv=ndv, provenance=Provenance.SKETCH, ndv_provenance=Provenance.SKETCH
    )


def _pg_row(row: Sequence[Any], row_count: int | None) -> tuple[str, ColumnStat] | None:
    """One ``pg_stats`` row -> ``(column_name, ColumnStat)``, or None if unusable."""
    if not row or row[0] is None:
        return None
    name = str(row[0])
    null_frac = _to_float(_at(row, 1))
    n_distinct = _to_float(_at(row, 2))
    null_count = null_frac * row_count if null_frac is not None and row_count is not None else None
    ndv = _resolve_ndv(n_distinct, row_count)
    mcv = _pg_mcv(_at(row, 3), _at(row, 4))
    quantiles = _pg_histogram(_at(row, 5))
    if null_count is None and ndv is None and mcv is None and quantiles is None:
        return None
    return name, ColumnStat(
        null_count=null_count,
        ndv=ndv,
        mcv=mcv,
        quantiles=quantiles,
        provenance=Provenance.SKETCH,
        # Every catalog column stat is a sampled estimate — never let one answer an
        # exact null_count()/count_distinct(). Both facets carry their own SKETCH tag.
        ndv_provenance=Provenance.SKETCH if ndv is not None else None,
        null_count_provenance=Provenance.SKETCH if null_count is not None else None,
    )


def _resolve_ndv(n_distinct: float | None, row_count: int | None) -> float | None:
    """Postgres ``n_distinct`` -> an absolute distinct-value estimate.

    Postgres records a positive number as the estimated distinct count directly, and a
    negative number as *minus the ratio* of distinct values to rows (``-1`` means every
    value is distinct, ``-0.5`` means half are) — a form that survives the table growing.
    Resolving the ratio needs the row count; without it, only a positive (absolute)
    figure is usable.
    """
    if n_distinct is None or n_distinct == 0.0:
        return None
    if n_distinct > 0:
        return n_distinct
    if row_count is None:
        return None
    return -n_distinct * row_count


def _pg_mcv(vals: Any, freqs: Any) -> dict[str, float] | None:
    """``most_common_vals`` + ``most_common_freqs`` -> ``{str(value): frequency}``."""
    values = _parse_pg_array(vals)
    frequencies = _parse_pg_array(freqs)
    if not values or not frequencies:
        return None
    mcv: dict[str, float] = {}
    for value, freq in zip(values, frequencies, strict=False):
        f = _to_float(freq)
        if f is not None:
            mcv[str(value)] = f
    return mcv or None


def _pg_histogram(bounds: Any) -> dict[str, list[float]] | None:
    """``histogram_bounds`` -> an even quantile grid ``{"probs": …, "values": …}``.

    Postgres stores ``N+1`` bucket boundaries that partition the column into ``N``
    equi-depth buckets, so boundary ``i`` sits at cumulative probability ``i/N`` — an
    ascending quantile grid, exactly the shape `ColumnStat.quantiles` interpolates range
    selectivity from. Non-numeric histograms (text/date bounds) yield None here; their
    range selectivity falls back to the default rather than a mis-parsed grid.
    """
    parsed = _parse_pg_array(bounds)
    if not parsed or len(parsed) < 2:
        return None
    values = [_to_float(v) for v in parsed]
    if any(v is None for v in values):
        return None
    n = len(values) - 1
    probs = [i / n for i in range(len(values))]
    return {"probs": probs, "values": [v for v in values if v is not None]}


def _parse_pg_array(value: Any) -> list[Any] | None:
    """A Postgres array column into a Python list, tolerant of how a driver returns it.

    A driver may hand back a real list (psycopg with array support) or the raw
    ``{a,b,c}`` text form. Both are handled; anything else yields None so a column with an
    unparseable array still contributes its scalar facets.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    body = text[1:-1]
    if not body:
        return []
    return [_unquote(part) for part in body.split(",")]


def _unquote(token: str) -> str:
    """Strip the optional double-quotes Postgres wraps a text array element in."""
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token
