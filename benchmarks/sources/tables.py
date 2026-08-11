"""Established public benchmark tables — the only tabular data the suite reads.

The benchmarks **never generate data**. Every table is read from a canonical public
parquet location and normalized to a stable cross-engine schema here, in one place,
so all engines see byte-identical inputs (the parity discipline the harness's
correctness gate relies on).

Three table-based datasets are wired up:

- **TPC-H** — the Ray public benchmark bucket
  (``s3://ray-benchmark-data/tpch/parquet/sf{scale}/{table}/``), whose files carry
  positional ``column0..N`` names; we rename them to the canonical ``l_``/``o_``...
  columns the TPC-H queries use and normalize decimal/date types.
- **ClickBench** — the anonymous ClickHouse ``hits`` dataset
  (``https://datasets.clickhouse.com/hits_compatible/...``); already named.
- **TPC-DS** — a configurable parquet base (no single canonical public mirror); the
  default is overridable via ``--source`` / ``BENCH_TPCDS_BASE``.

The file-corpus benchmarks (scan layouts, images), which measure the read path rather
than a query over loaded tables, live in :mod:`sources.corpora`.

Sources, scale, and the ClickBench partition count are overridable via environment
variables (``BENCH_TPCH_BASE``, ``BENCH_CLICKBENCH_BASE``, ``BENCH_TPCDS_BASE``,
``BENCH_CLICKBENCH_PARTS``) or the ``--source`` CLI flag, so a private mirror or a
different scale factor needs no code change.

Tables are materialized to in-memory Arrow once and shared across engines. That keeps
small/medium scale (the dev and CI path) exact and simple; reading parquet natively
per engine for PB-scale multi-node runs is the documented follow-up (each adapter
already has ``read_parquet``) — see :func:`table_uris`.
"""

from __future__ import annotations

import os

import duckdb
import pyarrow as pa
import pyarrow.compute as pc

# --------------------------------------------------------------------------- #
# Default public sources (override via env or --source)
# --------------------------------------------------------------------------- #
TPCH_BASE = os.environ.get("BENCH_TPCH_BASE", "s3://ray-benchmark-data/tpch/parquet")
CLICKBENCH_BASE = os.environ.get(
    "BENCH_CLICKBENCH_BASE",
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned",
)
CLICKBENCH_PARTS = int(os.environ.get("BENCH_CLICKBENCH_PARTS", "1"))
# TPC-DS has no published parquet mirror the way TPC-H does — `ray-benchmark-data` holds no
# `tpcds` prefix at all, so the registered suite failed every table read and could never run.
# It is therefore materialized locally, once, from DuckDB's `tpcds` extension: that extension
# implements the spec's own `dsdgen`, so this is the official dataset rather than a synthetic
# stand-in (the "no data generation" rule is about not inventing a substrate to benchmark on,
# and every published TPC-DS result generates its data the same way). `ensure_tpcds_data`
# below writes it on the first run and is a no-op once the directory is there.
TPCDS_LOCAL = os.path.expanduser(os.environ.get("BENCH_TPCDS_LOCAL", "~/bench-data/tpcds"))
TPCDS_BASE = os.environ.get("BENCH_TPCDS_BASE", TPCDS_LOCAL)

# --------------------------------------------------------------------------- #
# TPC-H — canonical column order per table (the Ray files are positional)
# --------------------------------------------------------------------------- #
TPCH_COLUMNS: dict[str, tuple[str, ...]] = {
    "region": ("r_regionkey", "r_name", "r_comment"),
    "nation": ("n_nationkey", "n_name", "n_regionkey", "n_comment"),
    "supplier": (
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ),
    "customer": (
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ),
    "part": (
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ),
    "partsupp": ("ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"),
    "orders": (
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ),
    "lineitem": (
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    ),
}
TPCH_TABLES = tuple(TPCH_COLUMNS)

# The full TPC-DS schema — all 24 tables `dsdgen` produces. The registered suite is the
# whole 99-query benchmark, which reaches every one of them (all three sales channels,
# their returns, and every dimension), so there is no subset left to fetch.
TPCDS_TABLES = (
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
)


def _normalize_types(table: pa.Table) -> pa.Table:
    """Cast decimals to float64 for cross-engine parity (no float128/Decimal skew).

    Date and timestamp columns are left in their source type: the TPC-H / TPC-DS
    queries compare them against ``date '...'`` literals, so casting dates to
    timestamps would break those comparisons on engines that don't implicitly
    coerce ``timestamp`` vs ``date``.
    """
    arrays, fields = [], []
    for fld in table.schema:
        arr = table.column(fld.name)
        if pa.types.is_decimal(fld.type):
            arr = pc.cast(arr, pa.float64())
            fld = fld.with_type(pa.float64())
        arrays.append(arr)
        fields.append(fld)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _reader() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection set up to read parquet from local / ``s3://`` / ``https://``.

    DuckDB's ``httpfs`` reads anonymous public buckets and HTTPS directly, which the
    plain PyArrow filesystem cannot — and DuckDB is already a core dependency, so the
    loader needs no extra cloud client. The data it returns is shared across engines;
    this connection is only the fetch path, never a benchmarked query.
    """
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql("SET enable_progress_bar=false")
    region = os.environ.get("BENCH_S3_REGION")
    if region:
        con.sql(f"SET s3_region='{region}'")
    return con


def _read(uri: str) -> pa.Table:
    """Read a parquet file or glob (local, ``s3://``, or ``https://``) into Arrow."""
    return _reader().sql(f"SELECT * FROM read_parquet('{uri}')").to_arrow_table()


def _rename_positional(table: pa.Table, names: tuple[str, ...]) -> pa.Table:
    """Rename columns positionally to the canonical TPC-H names (drop any extras)."""
    keep = min(len(names), table.num_columns)
    table = table.select(list(range(keep)))
    return table.rename_columns(list(names[:keep]))


def _tpch_tables(scale: float, base: str) -> dict[str, pa.Table]:
    sf = int(scale) if float(scale).is_integer() else scale
    out: dict[str, pa.Table] = {}
    for name, cols in TPCH_COLUMNS.items():
        raw = _read(f"{base}/sf{sf}/{name}/*.parquet")
        out[name] = _normalize_types(_rename_positional(raw, cols))
    return out


# ClickBench's `hits_compatible` parquet stores its temporal columns as raw integers:
# `EventDate` as days since the epoch (`uint16`), the three `*EventTime`s as seconds
# (`int64`). The benchmark's queries treat them as a DATE and a TIMESTAMP — comparing
# `EventDate >= '2013-07-01'` and calling `extract(minute FROM EventTime)` — so the
# reference loaders reconstruct the types on ingest (the official DuckDB one does
# `DATE '1970-01-01' + EventDate` and `epoch_ms(EventTime * 1000)`). Without it, every
# engine fails those queries identically (`Could not convert string '2013-07-01' to
# UINT16`), which is a broken benchmark rather than an engine result.
_CLICKBENCH_DATE_COLUMNS = ("EventDate",)
_CLICKBENCH_TIME_COLUMNS = ("EventTime", "ClientEventTime", "LocalEventTime")


def _reconstruct_clickbench_temporals(table: pa.Table) -> pa.Table:
    """Rebuild ClickBench's DATE / TIMESTAMP columns from their integer storage.

    Idempotent, because there are two supported layouts and only one of them stores these
    as integers. The public ``hits_compatible`` parquet does; a **normalized local mirror**
    (`tools/mirror_bench_data.py --dataset clickbench`, which scan mode requires) has
    already converted them. Re-converting the second reads a timestamp as a *second* count
    and overflows — `1373809127000000` out of bounds — which aborts the whole suite before
    a single query runs, and reads as "the mirror is corrupt" rather than "it was already
    right". A column that is already temporal is therefore left exactly as it is.
    """
    for name in _CLICKBENCH_DATE_COLUMNS:
        if name in table.column_names:
            column = table.column(name)
            if pa.types.is_date(column.type):
                continue
            days = pc.cast(column, pa.int32())
            table = table.set_column(
                table.schema.get_field_index(name), name, pc.cast(days, pa.date32())
            )
    for name in _CLICKBENCH_TIME_COLUMNS:
        if name in table.column_names:
            column = table.column(name)
            if pa.types.is_timestamp(column.type):
                continue
            secs = pc.cast(column, pa.int64())
            stamps = pc.cast(secs, pa.timestamp("s"))
            table = table.set_column(
                table.schema.get_field_index(name), name, pc.cast(stamps, pa.timestamp("us"))
            )
    return table


def _binary_to_utf8(table: pa.Table) -> pa.Table:
    """Cast ClickBench's string columns from their parquet ``binary`` storage to ``utf8``.

    The ``hits_compatible`` parquet stores every text column (URL, Title, SearchPhrase,
    Referer, ...) as ``binary``, but every published ClickBench loader — including DuckDB's —
    treats them as ``VARCHAR``. Without the cast, string queries run on ``binary``: ``LIKE``
    and ``MIN``/``MAX`` are undefined for a blob (DuckDB errors, so the query becomes an
    un-timeable ``n/a``), and the benchmark measures the wrong thing. Casting here makes every
    engine see the same ``utf8`` columns the standard benchmark specifies.
    """
    arrays, names = [], []
    for field in table.schema:
        arr = table.column(field.name)
        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            arr = pc.cast(arr, pa.string())
        arrays.append(arr)
        names.append(field.name)
    return pa.table(arrays, names=names)


def _clickbench_tables(base: str, parts: int) -> dict[str, pa.Table]:
    uris = [f"{base}/hits_{i}.parquet" for i in range(parts)]
    hits = pa.concat_tables([_read(u) for u in uris]) if len(uris) > 1 else _read(uris[0])
    return {"hits": _binary_to_utf8(_reconstruct_clickbench_temporals(_normalize_types(hits)))}


def ensure_tpcds_data(scale: float, base: str) -> None:
    """Materialize the TPC-DS tables at `base/sf{scale}` if they are not there yet.

    Only fires for the local mirror (`TPCDS_LOCAL`); an explicitly configured remote `base` is
    read as given, so this can never mask a misconfigured `--source` by silently generating
    data underneath it. Generation is DuckDB's `dsdgen` — the spec's own generator.
    """
    if base != TPCDS_LOCAL:
        return
    sf = _sf(scale)
    target = os.path.join(base, f"sf{sf}")
    if all(os.path.isdir(os.path.join(target, name)) for name in TPCDS_TABLES):
        return
    con = duckdb.connect()
    con.sql("INSTALL tpcds; LOAD tpcds;")
    con.sql("SET enable_progress_bar=false")
    con.sql(f"CALL dsdgen(sf={sf})")
    for name in TPCDS_TABLES:
        out_dir = os.path.join(target, name)
        os.makedirs(out_dir, exist_ok=True)
        con.sql(f"COPY {name} TO '{os.path.join(out_dir, 'part0.parquet')}' (FORMAT PARQUET)")


def _tpcds_tables(scale: float, base: str) -> dict[str, pa.Table]:
    ensure_tpcds_data(scale, base)
    sf = _sf(scale)
    out: dict[str, pa.Table] = {}
    for name in TPCDS_TABLES:
        out[name] = _normalize_types(_read(f"{base}/sf{sf}/{name}/*.parquet"))
    return out


def _sf(scale: float) -> str | int:
    return int(scale) if float(scale).is_integer() else scale


def table_uris(benchmark: str, scale: float, source: str | None = None) -> dict[str, str]:
    """Per-table parquet globs for the *scan* path — each engine reads these natively.

    The large-scale counterpart to :func:`load_tables`: rather than materialize every
    table into shared Arrow, return ``{table -> glob}`` so each engine binds a lazy
    native scan (this module no longer touches the bulk data). ``source`` must point
    at a base holding **canonical-named** parquet (``{base}/{table}/*.parquet``) — at
    sf100 that is the normalized local mirror, since the raw Ray files are positional
    and every engine's scan needs the ``l_``/``o_``... names.
    """
    if benchmark == "tpch":
        base = source or TPCH_BASE
        tables = TPCH_TABLES
    elif benchmark == "tpcds":
        base = source or TPCDS_BASE
        tables = TPCDS_TABLES
    elif benchmark == "clickbench":
        # ClickBench has one table and no scale factor. Scan mode is the *only* way to run
        # it at its full 100 M rows: `load_tables` concatenates every part into one shared
        # Arrow table for all engines, which exceeds memory. The scan base must hold parquet
        # whose `EventDate`/`*EventTime` columns are already DATE/TIMESTAMP — the raw files
        # store them as integers and the four queries that call `toMonth`/`toHour` on them
        # fail on every engine. `tools/mirror_bench_data.py --dataset clickbench` writes that.
        return {"hits": f"{source or CLICKBENCH_BASE}/hits/*.parquet"}
    else:
        raise ValueError(f"scan mode unsupported for benchmark {benchmark!r}")
    # A base already pointing at a specific scale dir (contains a table subdir) is used
    # verbatim; otherwise the canonical ``{base}/sf{N}`` layout is assumed.
    root = base if os.path.isdir(os.path.join(base, tables[0])) else f"{base}/sf{_sf(scale)}"
    return {name: f"{root}/{name}/*.parquet" for name in tables}


def scan_rename(benchmark: str, uris: dict[str, str]) -> dict[str, dict[str, str]]:
    """Per-table ``{positional -> canonical}`` column renames for the *scan* path.

    The public Ray TPC-H parquet names its columns positionally (``column00``...), while
    every TPC-H query names them (``l_orderkey``...). :func:`load_tables` renames after
    materializing into Arrow; the scan path never materializes, so instead each engine
    binds its lazy scan with this rename applied — schema-on-read, the same pure-metadata
    projection for every engine, and no 40 GB mirror to normalize the names.

    Returns ``{}`` for a source whose columns are already canonical (a normalized mirror,
    and TPC-DS / ClickBench, which ship real names), so those pay nothing.
    """
    if benchmark != "tpch":
        return {}  # TPC-DS and ClickBench parquet already carry real column names
    con = _reader()
    out: dict[str, dict[str, str]] = {}
    for name, glob in uris.items():
        want = TPCH_COLUMNS.get(name)
        if not want:
            continue
        found = tuple(
            row[0] for row in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()
        )
        if not found or found[0] == want[0]:
            continue  # already canonically named
        out[name] = dict(zip(found, want, strict=False))
    return out


def load_tables(benchmark: str, scale: float, source: str | None = None) -> dict[str, pa.Table]:
    """Load the named tables for ``benchmark`` from its public parquet source.

    ``benchmark`` is one of ``"tpch"``, ``"clickbench"``, ``"tpcds"``. ``source``
    overrides the default base URI for that benchmark; ``scale`` selects the TPC-H /
    TPC-DS scale factor (ignored by ClickBench, which is a fixed single table).
    """
    if benchmark == "tpch":
        return _tpch_tables(scale, source or TPCH_BASE)
    if benchmark == "clickbench":
        return _clickbench_tables(source or CLICKBENCH_BASE, CLICKBENCH_PARTS)
    if benchmark == "tpcds":
        return _tpcds_tables(scale, source or TPCDS_BASE)
    if benchmark == "json":
        # The one generated dataset: no public nested-JSON parquet corpus exists, so a
        # fixed-seed generator builds byte-identical documents shared across every engine
        # (the same parity the parquet loaders provide). See ``json_source``.
        from datagen import build_events

        return build_events(scale)
    if benchmark == "job":
        # The Join Order Benchmark's IMDb database — a real one, fetched and converted
        # once. Lives in its own module because the fetch/extract/convert path is nothing
        # like the others. See ``sources.job``.
        from sources.job import job_tables

        return job_tables(source)
    if benchmark in ("h2o-groupby", "h2o-join"):
        # The H2O.ai db-benchmark publishes generators rather than data — every leaderboard
        # entry runs its own `groupby-datagen.R` / `join-datagen.R`. `datagen.h2o_tables`
        # follows that spec, the same way TPC-DS above runs the spec's own `dsdgen`.
        from datagen import build_groupby, build_join

        return build_groupby(scale) if benchmark == "h2o-groupby" else build_join(scale)
    raise ValueError(f"unknown benchmark dataset: {benchmark!r}")
