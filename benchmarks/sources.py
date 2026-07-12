"""Established public benchmark datasets — the only data the suite reads.

The benchmarks **never generate data**. Every table is read from a canonical public
parquet location and normalized to a stable cross-engine schema here, in one place,
so all engines see byte-identical inputs (the parity discipline the harness's
correctness gate relies on).

Four datasets are wired up:

- **TPC-H** — the Ray public benchmark bucket
  (``s3://ray-benchmark-data/tpch/parquet/sf{scale}/{table}/``), whose files carry
  positional ``column0..N`` names; we rename them to the canonical ``l_``/``o_``...
  columns the TPC-H queries use and normalize decimal/date types.
- **ClickBench** — the anonymous ClickHouse ``hits`` dataset
  (``https://datasets.clickhouse.com/hits_compatible/...``); already named.
- **TPC-DS** — a configurable parquet base (no single canonical public mirror); the
  default is overridable via ``--source`` / ``BENCH_TPCDS_BASE``.
- **Scan (file layout)** — the same 16-column ``int64`` data laid out three ways in
  the Ray bucket, for the scan-planning benchmark. See :class:`ScanCorpus`.

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
from dataclasses import dataclass

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs

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

# --------------------------------------------------------------------------- #
# Scan corpus — one logical table, three physical file layouts
# --------------------------------------------------------------------------- #
SCAN_BASE = os.environ.get("BENCH_SCAN_BASE", "s3://ray-benchmark-data")

# Benchmark ``--scale`` -> the bucket's size directory, mirroring the TPC-H convention
# that scale 1 is ~1 GiB of data.
SCAN_SIZES: dict[int, str] = {
    1: "1GiB",
    10: "10GiB",
    100: "100GiB",
    1000: "1TiB",
    10000: "10TiB",
}

# The three layouts, in increasing file count. Every layout holds the *same* schema
# (``column0..column15``, all ``int64``, uniformly random over ``[0, 2^63)``), so the
# only variable across them is how the bytes are split into files.
SCAN_LAYOUTS = ("one_big", "ideal", "many_small")

_SCAN_DIRS = {
    "one_big": "{base}/parquet/{size}",  # ~1 GiB files (8.4M rows, 8 row groups each)
    "ideal": "{base}/parquet/128MiB-file/{size}",  # ~132 MiB files — the recommended size
    "many_small": "{base}/small-parquet/{size}",  # ~1.2 MiB files (8,192 rows each)
}

# `small-parquet/{1GiB,100GiB}` mixes a few ~133 MiB files (named `75_*`) in among the
# 1.2 MiB ones (`80_*`). Reading them would make the "many small files" layout not
# actually many-small, and at 1GiB it also breaks row-count parity with the other two
# layouts (9,445,376 rows rather than 8,388,608). Restrict those two dirs to `80_*`.
# The other sizes are homogeneous and need no filter.
_SCAN_FILE_PREFIX = {("many_small", "1GiB"): "80_", ("many_small", "100GiB"): "80_"}


def _list_corpus(
    directory: str, suffix: str, file_prefix: str
) -> tuple[pafs.FileSystem, list[str]]:
    """List ``directory`` now, returning its filesystem and sorted member paths.

    Filters to files ending in ``suffix`` whose basename starts with ``file_prefix``. The
    listing is on demand (not at corpus construction) so the corpus benchmarks time it as
    part of the workload.

    When a ``file_prefix`` selects a subset of a large directory, the listing is
    **prefix-scoped** through fsspec (one object-store LIST of the matching keys) rather
    than paging the whole directory — so the non-Batcher engines, which read this file
    list, are timed against the same efficient listing Batcher's own reader now does
    (a fair comparison, not a handicap from the harness's listing code). Falls back to the
    full pyarrow directory listing when fsspec / its backend is absent.
    """
    filesystem, base = pafs.FileSystem.from_uri(directory)
    scoped = _prefix_scoped_paths(directory, base, suffix, file_prefix)
    if scoped is not None:
        return filesystem, scoped
    entries = filesystem.get_file_info(pafs.FileSelector(base, recursive=False))
    paths = [
        entry.path
        for entry in entries
        if entry.path.endswith(suffix) and entry.base_name.startswith(file_prefix)
    ]
    return filesystem, sorted(paths)


def _prefix_scoped_paths(
    directory: str, base: str, suffix: str, file_prefix: str
) -> list[str] | None:
    """Filesystem-native paths under ``base`` matching ``file_prefix``, via a prefix LIST.

    Uses fsspec's prefix-scoped glob (so a subset of a huge bucket lists in one call) and
    maps the results back to the filesystem-native form pyarrow readers expect. Returns
    ``None`` (fall back to a full listing) for a local path, a missing fsspec backend, or
    any listing error — the fast path is purely an optimization, never the correctness path.
    """
    if "://" not in directory or file_prefix == "":
        return None
    try:
        import fsspec

        backend = fsspec.filesystem(directory.split("://", 1)[0])
        matches = backend.glob(f"{base}/{file_prefix}*{suffix}")
    except Exception:
        return None
    paths = sorted(m for m in matches if m.endswith(suffix))
    return paths or None


@dataclass(frozen=True)
class ScanCorpus:
    """One (layout, size) corpus of parquet files, resolved lazily.

    Exposes the two forms the engines need. ``glob`` is for the engines that expand a
    glob themselves (every SQL engine); ``open()`` performs the listing explicitly, for
    PyArrow and Ray Data, whose readers take a file list rather than a pattern.
    """

    layout: str
    size: str
    directory: str
    file_prefix: str = ""

    @property
    def glob(self) -> str:
        """The ``*.parquet`` pattern covering this corpus's files."""
        return f"{self.directory}/{self.file_prefix}*.parquet"

    def open(self) -> tuple[pafs.FileSystem, list[str]]:
        """List the corpus now, returning its filesystem and sorted member paths."""
        return _list_corpus(self.directory, ".parquet", self.file_prefix)


def scan_corpora(scale: float, source: str | None = None) -> dict[str, ScanCorpus]:
    """The three file layouts at ``scale``, keyed by layout name.

    ``scale`` selects the size directory via :data:`SCAN_SIZES` (1 -> ``1GiB``,
    10 -> ``10GiB``, ...); ``source`` overrides the bucket base.
    """
    size = SCAN_SIZES.get(int(scale))
    if size is None:
        valid = ", ".join(str(s) for s in SCAN_SIZES)
        raise ValueError(f"scan scale must be one of {valid} (got {scale})")
    base = source or SCAN_BASE
    return {
        layout: ScanCorpus(
            layout=layout,
            size=size,
            directory=_SCAN_DIRS[layout].format(base=base, size=size),
            file_prefix=_SCAN_FILE_PREFIX.get((layout, size), ""),
        )
        for layout in SCAN_LAYOUTS
    }


# --------------------------------------------------------------------------- #
# Image corpus — an unstructured (JPEG) multimodal read/decode workload
# --------------------------------------------------------------------------- #
IMAGES_BASE = os.environ.get("BENCH_IMAGES_BASE", "s3://ray-benchmark-data/profile-pictures/1GiB")

# Benchmark ``--scale`` -> a filename prefix selecting how many images to read. The
# corpus is 211,742 JPEGs named ``000000.jpg``..``211741.jpg`` (all 110x110 RGB, ~5 KiB),
# so a zero-padded prefix bounds the count by a power of ten. The default is deliberately
# small: image reads over S3 are per-file (one object open each), so thousands of tiny
# files is minutes of wall-clock even before any engine's overhead (see the README note).
IMAGE_COUNTS: dict[int, tuple[str, int]] = {
    1: ("00000", 10),
    10: ("0000", 100),
    100: ("000", 1_000),
    1000: ("00", 10_000),
    10000: ("0", 100_000),
}
IMAGE_SUFFIX = ".jpg"
# The corpus's native pixel dimensions (H, W). The decode benchmark targets these so all
# engines produce identically-sized tensors — and because Batcher's ``read.images`` reader
# requires an explicit decode size (it has no native-resolution decode mode), a fact this
# makes the decode shape express uniformly rather than special-case.
IMAGE_NATIVE_SIZE = (110, 110)


@dataclass(frozen=True)
class ImageCorpus:
    """A bounded set of JPEG files for the multimodal read/decode benchmark.

    Like :class:`ScanCorpus`, exposes ``glob`` (for readers that expand a pattern, e.g.
    Batcher's ``read.images``) and ``open()`` (an explicit path list, for Ray Data / Daft
    / PyArrow, whose image/binary readers take a list). Resolution is lazy so that file
    listing is timed as part of the workload.
    """

    directory: str
    file_prefix: str
    count: int
    native_size: tuple[int, int] = IMAGE_NATIVE_SIZE
    suffix: str = IMAGE_SUFFIX

    @property
    def glob(self) -> str:
        """The pattern covering this corpus's image files."""
        return f"{self.directory}/{self.file_prefix}*{self.suffix}"

    def open(self) -> tuple[pafs.FileSystem, list[str]]:
        """List the corpus now, returning its filesystem and sorted member paths.

        The paths are filesystem-native (bucket-relative for S3, absolute for local) —
        the form Ray Data and PyArrow want alongside the filesystem object.
        """
        return _list_corpus(self.directory, self.suffix, self.file_prefix)

    def uris(self) -> list[str]:
        """Scheme-qualified URIs (``s3://...`` / ``file://...``) — the form Daft downloads.

        Derived from the corpus directory's scheme so a Daft ``url.download`` gets a URI
        it can resolve, whether the base is an S3 bucket or a local directory.
        """
        _, paths = self.open()
        if "://" in self.directory:
            scheme = self.directory.split("://", 1)[0]
            return [f"{scheme}://{p}" for p in paths]
        return [f"file://{p}" for p in paths]


def image_corpus(scale: float, source: str | None = None) -> ImageCorpus:
    """The image corpus at ``scale`` (1 -> 10 files, 10 -> 100, ... via :data:`IMAGE_COUNTS`)."""
    entry = IMAGE_COUNTS.get(int(scale))
    if entry is None:
        valid = ", ".join(str(s) for s in IMAGE_COUNTS)
        raise ValueError(f"images scale must be one of {valid} (got {scale})")
    prefix, count = entry
    return ImageCorpus(directory=source or IMAGES_BASE, file_prefix=prefix, count=count)


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
    """Rebuild ClickBench's DATE / TIMESTAMP columns from their integer storage."""
    for name in _CLICKBENCH_DATE_COLUMNS:
        if name in table.column_names:
            days = pc.cast(table.column(name), pa.int32())
            table = table.set_column(
                table.schema.get_field_index(name), name, pc.cast(days, pa.date32())
            )
    for name in _CLICKBENCH_TIME_COLUMNS:
        if name in table.column_names:
            secs = pc.cast(table.column(name), pa.int64())
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


def _tpcds_tables(scale: float, base: str) -> dict[str, pa.Table]:
    sf = int(scale) if float(scale).is_integer() else scale
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
    native scan (``sources.py`` no longer touches the bulk data). ``source`` must point
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
    raise ValueError(f"unknown benchmark dataset: {benchmark!r}")
