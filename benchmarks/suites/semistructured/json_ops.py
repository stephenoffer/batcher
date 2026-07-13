"""JSON path-extraction cases over a nested-document event log (dataset ``json``).

Every row is one JSON document in a Utf8 ``payload`` column; each case parses it and pulls
fields out by path, then aggregates — the parse-bound shape of real semistructured analytics.
Batcher is exercised through all three surfaces it exposes:

- **Dataset + ``.json`` expression accessor** — ``json-groupby1``, ``json-project5``,
  ``json-array`` (the last also tests array-index paths, ``$.tags[0]``).
- **SQL ``json_extract_string`` / ``->>``** — ``json-filter-agg``, ``json-groupby-sql``.

Competitors use their native JSON paths: DuckDB / Spark ``json_extract_string`` /
``get_json_object`` (fanned as SQL), Polars ``str.json_path_match`` (expression API), Daft
``jq`` (where its JSON-text output is numeric, so it stays comparable — the string-keyed
group-bys it sits out rather than emit quoted keys). PyArrow has no JSON path and shows
``n/a``. The correctness gate checks every engine agrees before any timing is trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import EngineQueries, suite

if TYPE_CHECKING:
    import pyarrow as pa

    from context import Context

js = suite("semistructured", dataset="json")

# SQL engines whose JSON extractor is spelled `json_extract_string` (DuckDB family + Batcher
# SQL, which lowers it to the `.json` accessor). Spark spells it `get_json_object`.
_JSON_STR = "json_extract_string"


def _sql(ctx: Context, duck_sql: str, *, batcher: bool) -> EngineQueries:
    """Fan a DuckDB-dialect JSON query across the SQL engines in the lineup.

    Spark's extractor name differs (``get_json_object``); Batcher speaks the DuckDB form
    verbatim and is included only when this case is testing the SQL surface.
    """
    runners = ctx.sql_runners()
    per: dict[str, str] = {
        "duckdb": duck_sql,
        "duckdb_arrow": duck_sql,
        "spark": duck_sql.replace(_JSON_STR, "get_json_object"),
    }
    if batcher:
        per["batcher"] = duck_sql
    return {n: (lambda run=runners[n], q=q: run(q)) for n, q in per.items() if n in runners}


def _with(ctx: Context, fns: EngineQueries, **natives) -> EngineQueries:
    """Attach native (non-SQL-fanout) engine callables that are in the active lineup."""
    active = set(ctx.names())
    for name, fn in natives.items():
        if fn is not None and name in active:
            fns[name] = fn
    return fns


def _bt(ctx: Context):
    """The Batcher ``Dataset`` handle over the shared ``events`` table."""
    return ctx.handle("events", "batcher")


def _pl(ctx: Context):
    """The Polars handle, or ``None`` when Polars is not in the lineup."""
    return ctx.handle("events", "polars") if "polars" in ctx.names() else None


# --------------------------------------------------------------------------- #
# json-groupby1 — one string field extracted, grouped, counted
# --------------------------------------------------------------------------- #
@js.case("json-groupby1")
def json_groupby1(ctx: Context) -> EngineQueries:
    """GROUP BY user.country, COUNT(*) — a single path extract feeding a group-by."""
    duck = (
        f"SELECT {_JSON_STR}(payload,'$.user.country') AS country, COUNT(*) AS n "
        "FROM events GROUP BY 1"
    )

    def batcher() -> pa.Table:
        from batcher import col

        return (
            _bt(ctx)
            .with_columns(country=col("payload").json.extract_string("$.user.country"))
            .group_by("country")
            .agg(n=col("country").count())
            .collect()
        )

    def polars() -> pa.Table:
        import polars as pl

        df = _pl(ctx)
        return (
            df.select(pl.col("payload").str.json_path_match("$.user.country").alias("country"))
            .group_by("country")
            .agg(pl.len().alias("n"))
            .to_arrow()
        )

    return _with(ctx, _sql(ctx, duck, batcher=False), batcher=batcher, polars=polars)


# --------------------------------------------------------------------------- #
# json-project5 — five fields extracted (3 group keys + 2 measures)
# --------------------------------------------------------------------------- #
@js.case("json-project5")
def json_project5(ctx: Context) -> EngineQueries:
    """GROUP BY country, tier, os with SUM(value), SUM(items), COUNT — five path extracts.

    The multi-extract stressor: every one of the five fields is genuinely used, so the
    document is parsed five times per row on any engine that re-parses per field.
    """
    duck = (
        f"SELECT {_JSON_STR}(payload,'$.user.country') AS country, "
        f"{_JSON_STR}(payload,'$.user.tier') AS tier, "
        f"{_JSON_STR}(payload,'$.device.os') AS os, "
        f"SUM(CAST({_JSON_STR}(payload,'$.event.value') AS DOUBLE)) AS s, "
        f"SUM(CAST({_JSON_STR}(payload,'$.event.items') AS BIGINT)) AS isum, "
        "COUNT(*) AS n FROM events GROUP BY 1, 2, 3"
    )

    def batcher() -> pa.Table:
        from batcher import col

        return (
            _bt(ctx)
            .with_columns(
                country=col("payload").json.extract_string("$.user.country"),
                tier=col("payload").json.extract_string("$.user.tier"),
                os=col("payload").json.extract_string("$.device.os"),
                value=col("payload").json.extract_float("$.event.value"),
                items=col("payload").json.extract_int("$.event.items"),
            )
            .group_by("country", "tier", "os")
            .agg(s=col("value").sum(), isum=col("items").sum(), n=col("country").count())
            .collect()
        )

    def polars() -> pa.Table:
        import polars as pl

        df = _pl(ctx)
        return (
            df.select(
                pl.col("payload").str.json_path_match("$.user.country").alias("country"),
                pl.col("payload").str.json_path_match("$.user.tier").alias("tier"),
                pl.col("payload").str.json_path_match("$.device.os").alias("os"),
                pl.col("payload").str.json_path_match("$.event.value").cast(pl.Float64).alias("v"),
                pl.col("payload").str.json_path_match("$.event.items").cast(pl.Int64).alias("i"),
            )
            .group_by("country", "tier", "os")
            .agg(pl.col("v").sum().alias("s"), pl.col("i").sum().alias("isum"), pl.len().alias("n"))
            .to_arrow()
        )

    return _with(ctx, _sql(ctx, duck, batcher=False), batcher=batcher, polars=polars)


# --------------------------------------------------------------------------- #
# json-array — nested array-index path ($.tags[0])
# --------------------------------------------------------------------------- #
@js.case("json-array")
def json_array(ctx: Context) -> EngineQueries:
    """GROUP BY tags[0], COUNT(*) — an array-index path extract."""
    duck = f"SELECT {_JSON_STR}(payload,'$.tags[0]') AS tag, COUNT(*) AS n FROM events GROUP BY 1"

    def batcher() -> pa.Table:
        from batcher import col

        return (
            _bt(ctx)
            .with_columns(tag=col("payload").json.extract_string("$.tags[0]"))
            .group_by("tag")
            .agg(n=col("tag").count())
            .collect()
        )

    def polars() -> pa.Table:
        import polars as pl

        df = _pl(ctx)
        return (
            df.select(pl.col("payload").str.json_path_match("$.tags[0]").alias("tag"))
            .group_by("tag")
            .agg(pl.len().alias("n"))
            .to_arrow()
        )

    return _with(ctx, _sql(ctx, duck, batcher=False), batcher=batcher, polars=polars)


# --------------------------------------------------------------------------- #
# json-filter-agg — filter on an extracted string, aggregate an extracted number
# (Batcher via its SQL surface; Daft competes here, its output being numeric)
# --------------------------------------------------------------------------- #
@js.case("json-filter-agg")
def json_filter_agg(ctx: Context) -> EngineQueries:
    """WHERE event.type = 'purchase': SUM(event.value), COUNT(*) — extract-filter + extract-sum."""
    duck = (
        f"SELECT SUM(CAST({_JSON_STR}(payload,'$.event.value') AS DOUBLE)) AS s, COUNT(*) AS n "
        f"FROM events WHERE {_JSON_STR}(payload,'$.event.type') = 'purchase'"
    )

    def polars() -> pa.Table:
        import polars as pl

        df = _pl(ctx)
        return (
            df.filter(pl.col("payload").str.json_path_match("$.event.type") == pl.lit("purchase"))
            .select(
                pl.col("payload")
                .str.json_path_match("$.event.value")
                .cast(pl.Float64)
                .sum()
                .alias("s"),
                pl.len().alias("n"),
            )
            .to_arrow()
        )

    def daft() -> pa.Table:
        import daft

        df = ctx.handle("events", "daft")
        typ = daft.col("payload").jq(".event.type")  # jq syntax; JSON text: '"purchase"'
        val = daft.col("payload").jq(".event.value").cast(daft.DataType.float64())
        out = df.where(typ == daft.lit('"purchase"')).agg(
            val.sum().alias("s"), daft.col("payload").count().alias("n")
        )
        return out.to_arrow()

    daft_fn = daft if "daft" in ctx.names() else None
    return _with(ctx, _sql(ctx, duck, batcher=True), polars=polars, daft=daft_fn)


# --------------------------------------------------------------------------- #
# json-groupby-sql — the group-by-count through Batcher's SQL surface
# --------------------------------------------------------------------------- #
@js.case("json-groupby-sql")
def json_groupby_sql(ctx: Context) -> EngineQueries:
    """GROUP BY user.tier, COUNT(*) — the group-by expressed as SQL on every SQL engine."""
    duck = (
        f"SELECT {_JSON_STR}(payload,'$.user.tier') AS tier, COUNT(*) AS n FROM events GROUP BY 1"
    )

    def polars() -> pa.Table:
        import polars as pl

        df = _pl(ctx)
        return (
            df.select(pl.col("payload").str.json_path_match("$.user.tier").alias("tier"))
            .group_by("tier")
            .agg(pl.len().alias("n"))
            .to_arrow()
        )

    return _with(ctx, _sql(ctx, duck, batcher=True), polars=polars)
