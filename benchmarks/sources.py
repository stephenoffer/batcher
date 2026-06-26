"""Established public benchmark datasets — the only data the suite reads.

The benchmarks **never generate data**. Every table is read from a canonical public
parquet location and normalized to a stable cross-engine schema here, in one place,
so all engines see byte-identical inputs (the parity discipline the harness's
correctness gate relies on).

Three datasets are wired up:

- **TPC-H** — the Ray public benchmark bucket
  (``s3://ray-benchmark-data/tpch/parquet/sf{scale}/{table}/``), whose files carry
  positional ``column0..N`` names; we rename them to the canonical ``l_``/``o_``...
  columns the TPC-H queries use and normalize decimal/date types.
- **ClickBench** — the anonymous ClickHouse ``hits`` dataset
  (``https://datasets.clickhouse.com/hits_compatible/...``); already named.
- **TPC-DS** — a configurable parquet base (no single canonical public mirror); the
  default is overridable via ``--source`` / ``BENCH_TPCDS_BASE``.

Sources, scale, and the ClickBench partition count are overridable via environment
variables (``BENCH_TPCH_BASE``, ``BENCH_CLICKBENCH_BASE``, ``BENCH_TPCDS_BASE``,
``BENCH_CLICKBENCH_PARTS``) or the ``--source`` CLI flag, so a private mirror or a
different scale factor needs no code change.

Tables are materialized to in-memory Arrow once and shared across engines. That keeps
small/medium scale (the dev and CI path) exact and simple; reading parquet natively
per engine for PB-scale multi-node runs is the documented follow-up (each adapter
already has ``read_parquet``).
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
TPCDS_BASE = os.environ.get("BENCH_TPCDS_BASE", "s3://ray-benchmark-data/tpcds/parquet")

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

# The tables the wired-up TPC-DS subset actually touches (so we only fetch those).
TPCDS_TABLES = (
    "store_sales",
    "store_returns",
    "date_dim",
    "item",
    "customer",
    "customer_address",
    "store",
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


def _clickbench_tables(base: str, parts: int) -> dict[str, pa.Table]:
    uris = [f"{base}/hits_{i}.parquet" for i in range(parts)]
    hits = pa.concat_tables([_read(u) for u in uris]) if len(uris) > 1 else _read(uris[0])
    return {"hits": _normalize_types(hits)}


def _tpcds_tables(scale: float, base: str) -> dict[str, pa.Table]:
    sf = int(scale) if float(scale).is_integer() else scale
    out: dict[str, pa.Table] = {}
    for name in TPCDS_TABLES:
        out[name] = _normalize_types(_read(f"{base}/sf{sf}/{name}/*.parquet"))
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
    raise ValueError(f"unknown benchmark dataset: {benchmark!r}")
