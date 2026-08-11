"""Real datasets for the example suite, sourced from S3 and cached locally.

Every example that needs more than a handful of literal rows reads a *real* dataset:
the TPC-H mirror in the public ``s3://ray-benchmark-data`` bucket, plus a corpus of
small JPEGs for the multimodal examples. Nothing here is synthetic while the network
is up.

Three things this module does that each example should not repeat:

- **Restores the column names.** The mirror stores TPC-H positionally
  (``column0``, ``column1``, ...) and carries one trailing all-null column per table,
  an artifact of the ``|``-terminated ``.tbl`` source. Reading it raw would make every
  example unreadable, so the canonical names (``l_orderkey``, ``o_totalprice``, ...)
  are restored on the way into the cache.
- **Caches to local Parquet.** 500 scripts each re-reading S3 would take longer than
  the release check is worth and would fail whenever the network hiccups. The first
  call downloads a bounded slice, writes it once, and every later call — in this
  process or a later one — opens the local file.
- **Degrades loudly, never silently.** With no network the helper synthesizes a
  schema-identical stand-in so the suite still runs, and prints a one-line notice to
  stderr saying so. A quiet fallback would let the corpus rot invisibly, which is the
  exact failure this suite exists to catch.

Point the cache somewhere else with ``BATCHER_EXAMPLES_CACHE``. Ask for more rows
than the defaults below with ``BATCHER_EXAMPLES_ROWS`` (or ``full`` for whole tables).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import batcher as bt

__all__ = [
    "TPCH_COLUMNS",
    "TPCH_ROWS",
    "images",
    "is_offline",
    "tpch",
    "tpch_csv_uri",
    "tpch_path",
    "tpch_uri",
]

S3_BUCKET = "ray-benchmark-data"
TPCH_S3_BASE = f"s3://{S3_BUCKET}/tpch/parquet/sf1"
TPCH_CSV_S3_BASE = f"s3://{S3_BUCKET}/tpch/csv/sf1"
IMAGES_S3_BASE = f"s3://{S3_BUCKET}/profile-pictures/1GiB"

# The canonical TPC-H column names, in the order the mirror stores them. The mirror has
# one extra trailing column per table (always null) which is dropped, so each tuple is
# exactly one shorter than the file's field count.
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

# How much of each table the cache holds. Small enough that the whole suite stays a
# release check rather than a benchmark, large enough that group-bys, joins and window
# frames have real skew and real cardinality to work on. The three dimension tables are
# whole; the fact tables are a prefix.
TPCH_ROWS: dict[str, int] = {
    "region": 5,
    "nation": 25,
    "supplier": 10_000,
    "customer": 30_000,
    "part": 40_000,
    "partsupp": 60_000,
    "orders": 60_000,
    "lineitem": 200_000,
}

_OFFLINE = False


def is_offline() -> bool:
    """Report whether a fetch has already fallen back to synthesized data."""
    return _OFFLINE


def _cache_root() -> Path:
    override = os.environ.get("BATCHER_EXAMPLES_CACHE")
    root = Path(override) if override else Path.home() / ".cache" / "batcher-examples"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _row_budget(table: str) -> int | None:
    """Rows to cache for ``table``; ``None`` means the whole table."""
    request = os.environ.get("BATCHER_EXAMPLES_ROWS", "").strip().lower()
    if request == "full":
        return None
    if request.isdigit():
        return int(request)
    return TPCH_ROWS[table]


def _s3() -> tuple[object, str]:
    """An anonymous S3 filesystem and the bucket root, or raise if unreachable."""
    import pyarrow.fs as pafs

    try:
        region = pafs.resolve_s3_region(S3_BUCKET)
    except Exception:  # no network, or no metadata endpoint
        region = "us-west-2"
    return pafs.S3FileSystem(anonymous=True, region=region), S3_BUCKET


def _download_tpch(table: str, rows: int | None) -> pa.Table:
    """Read ``rows`` rows of ``table`` from S3 and apply the canonical column names."""
    filesystem, bucket = _s3()
    import pyarrow.fs as pafs

    directory = f"{bucket}/tpch/parquet/sf1/{table}"
    listing = filesystem.get_file_info(pafs.FileSelector(directory, recursive=False))
    members = sorted(
        entry.path for entry in listing if entry.path.endswith(".parquet") and entry.size
    )
    if not members:
        raise FileNotFoundError(directory)

    names = TPCH_COLUMNS[table]
    collected: list[pa.RecordBatch] = []
    seen = 0
    for member in members:
        reader = pq.ParquetFile(filesystem.open_input_file(member))
        # `iter_batches` stops after the row groups it needs, so a 200k-row slice of the
        # 265 MB lineitem file transfers a few row groups rather than the whole object.
        for batch in reader.iter_batches(batch_size=64 * 1024):
            wanted = batch if rows is None else batch.slice(0, rows - seen)
            collected.append(wanted)
            seen += wanted.num_rows
            if rows is not None and seen >= rows:
                break
        if rows is not None and seen >= rows:
            break

    table_data = pa.Table.from_batches(collected)
    # Drop the trailing all-null artifact column, then name the rest.
    table_data = table_data.select(list(range(len(names))))
    return table_data.rename_columns(list(names))


def _synthesize(table: str, rows: int | None) -> pa.Table:
    """A deterministic, schema-identical stand-in used only when S3 is unreachable."""
    import datetime as dt

    count = min(rows or 5_000, 5_000)
    names = TPCH_COLUMNS[table]
    epoch = dt.date(1994, 1, 1)
    columns: list[pa.Array] = []
    for position, name in enumerate(names):
        if name.endswith(("date",)):
            values = [epoch + dt.timedelta(days=(i * 7 + position) % 2000) for i in range(count)]
            columns.append(pa.array(values, type=pa.date32()))
        elif name.endswith(("price", "cost", "acctbal", "quantity", "discount", "tax")):
            columns.append(pa.array([round((i % 97) * 1.07 + position, 2) for i in range(count)]))
        elif name.endswith(("key", "size", "availqty", "linenumber", "shippriority")):
            columns.append(pa.array([(i % max(count // 4, 1)) + position for i in range(count)]))
        else:
            alphabet = ("alpha", "bravo", "charlie", "delta", "echo")
            columns.append(pa.array([f"{alphabet[i % 5]}-{name}-{i % 23}" for i in range(count)]))
    return pa.table(dict(zip(names, columns, strict=True)))


def tpch_uri(table: str) -> str:
    """Return the S3 glob for one raw TPC-H table.

    Use this in examples that are *about* reading from object storage. The columns are
    positional there; :func:`tpch` is the named, cached form everything else wants.
    """
    return f"{TPCH_S3_BASE}/{table}/*.parquet"


def tpch_csv_uri(table: str) -> str:
    """Return the S3 glob for one raw TPC-H table in the delimited-text mirror.

    The files are the original `dbgen` output: ``.tbl``, pipe-delimited, no header row,
    and one trailing delimiter per line. Read them with ``delimiter="|"`` and
    ``has_header=False``.
    """
    return f"{TPCH_CSV_S3_BASE}/{table}/*.tbl"


def tpch_path(table: str, *, rows: int | None = -1) -> str:
    """Return a local Parquet path holding ``table``, fetching it from S3 once.

    The file has canonical TPC-H column names. Pass ``rows`` to override the default
    slice for this table; pass ``None`` for the whole table.
    """
    if table not in TPCH_COLUMNS:
        raise KeyError(f"unknown TPC-H table {table!r}; have {sorted(TPCH_COLUMNS)}")
    budget = _row_budget(table) if rows == -1 else rows
    target = _cache_root() / f"tpch_sf1_{table}_{budget if budget is not None else 'full'}.parquet"
    if target.exists():
        return str(target)

    global _OFFLINE
    try:
        data = _download_tpch(table, budget)
    except Exception as exc:  # any transport failure degrades the same way
        _OFFLINE = True
        print(
            f"[examples] S3 unreachable ({type(exc).__name__}); "
            f"synthesizing a stand-in for TPC-H {table}.",
            file=sys.stderr,
        )
        data = _synthesize(table, budget)

    # Write via a unique temp file in the cache directory, then rename. Several example
    # scripts can run at once (pytest -n), and a half-written Parquet file read by a
    # sibling process is a confusing failure a long way from its cause.
    handle, staging = tempfile.mkstemp(dir=str(target.parent), suffix=".parquet")
    os.close(handle)
    pq.write_table(data, staging)
    os.replace(staging, target)
    return str(target)


def tpch(table: str, *, rows: int | None = -1) -> bt.Dataset:
    """Open one TPC-H table as a Dataset with canonical column names.

    Examples:
        >>> orders = tpch("orders")
        >>> "o_totalprice" in orders.schema.names
        True
    """
    return bt.read.parquet(tpch_path(table, rows=rows))


def images(count: int = 10) -> str:
    """Return an S3 glob matching roughly ``count`` small JPEGs.

    The corpus holds 211,742 zero-padded ``NNNNNN.jpg`` files of about 5 KiB each, so a
    multimodal example decodes real image bytes without turning the release check into a
    download. A glob keeps the selection bounded *and* keeps the listing cheap: matching
    a name prefix lists only those keys rather than paging the whole directory.

    ``count`` is rounded up to the next power of ten it can express as a prefix, so 10,
    100 and 1000 are exact and anything between rounds up.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    stars = max(1, len(str(max(count - 1, 1))))
    return f"{IMAGES_S3_BASE}/{'0' * (6 - stars)}{'*' * stars}.jpg"
