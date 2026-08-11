"""The Join Order Benchmark's IMDb database — fetched, typed, and converted to parquet once.

JOB is the odd one out among the suite's sources, and deliberately so. Every other dataset is
either public parquet or a generator; JOB is a **real** database — a 2014 IMDb snapshot,
3.6 GB of CSV across 21 tables — and that is the entire point of it. Its predicates are
correlated the way real data is (a film's country correlates with its language, its company
with its year), which is exactly what defeats the independence assumptions a textbook cost
model makes. Leis et al. built it to show that cardinality estimation, not the join-order
search, is where optimizers actually lose. That makes it the sharpest available test of a
plan chosen from *measured* cardinalities rather than estimated ones.

Nothing here is invented. The archive is the one the paper's reference implementation
distributes, and the column names and types come from the `schematext.sql` **shipped inside
that archive** rather than from anything typed here — the same discipline
`sources.tables` applies to TPC-DS, and for the same reason: a hand-transcribed schema
produces a benchmark that measures the transcription.

The first run downloads (~1.2 GiB), extracts, and converts to parquet; every later run reads
the parquet. Set ``BENCH_JOB_LOCAL`` to relocate it, or ``BENCH_JOB_BASE`` to point at a
mirror that already holds ``{base}/{table}/*.parquet`` and skip the fetch entirely.
"""

from __future__ import annotations

import os
import re
import tarfile
import urllib.request

import duckdb
import pyarrow as pa

__all__ = ["JOB_BASE", "JOB_LOCAL", "JOB_TABLES", "ensure_job_data", "job_tables"]

# The archive the reference implementation distributes. The canonical `homepages.cwi.nl`
# path is dead; this is the live CWI mirror.
JOB_ARCHIVE_URL = os.environ.get("BENCH_JOB_ARCHIVE", "https://event.cwi.nl/da/job/imdb.tgz")
JOB_LOCAL = os.path.expanduser(os.environ.get("BENCH_JOB_LOCAL", "~/bench-data/job"))
JOB_BASE = os.environ.get("BENCH_JOB_BASE", os.path.join(JOB_LOCAL, "parquet"))

# The 21 tables, in the order `schematext.sql` declares them. Named here only so a caller can
# ask "which tables does this dataset have" without a download; the *columns* are never
# hard-coded — see `_schema`.
JOB_TABLES = (
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
)

_CREATE_TABLE = re.compile(r"CREATE TABLE (\w+)\s*\((.*?)\n\);", re.S)


def _schema(schema_sql: str) -> dict[str, list[tuple[str, str]]]:
    """Parse the archive's own ``schematext.sql`` into ``{table -> [(column, type)]}``.

    Only two types occur across the 21 tables: PostgreSQL ``integer`` (every key and count)
    and ``character varying`` (everything else). They map to DuckDB ``BIGINT`` and
    ``VARCHAR``. Widening the integers to 64 bits matters: the FFI boundary normalizes narrow
    integers anyway, so reading them as ``BIGINT`` up front keeps every engine on the same
    type rather than leaving each to pick its own.

    Args:
        schema_sql: The contents of the archive's ``schematext.sql``.

    Returns:
        Each table's ordered ``(column name, DuckDB type)`` pairs.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for match in _CREATE_TABLE.finditer(schema_sql):
        columns: list[tuple[str, str]] = []
        for line in match.group(2).strip().splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            name, _, rest = line.partition(" ")
            columns.append((name, "BIGINT" if rest.startswith("integer") else "VARCHAR"))
        out[match.group(1)] = columns
    return out


def _read_csv_sql(path: str, columns: list[tuple[str, str]]) -> str:
    """A ``read_csv`` call typed by ``columns``, matching the archive's CSV dialect.

    The files are a PostgreSQL ``COPY ... CSV`` dump: no header row, double-quoted strings,
    backslash escapes, and an empty field meaning NULL rather than the empty string. Getting
    `nullstr` wrong is the subtle one — it turns every absent `note` into `''`, and the many
    `IS NOT NULL` predicates in the 113 queries then select rows the benchmark excludes.
    """
    spec = "{" + ", ".join(f"'{name}': '{typ}'" for name, typ in columns) + "}"
    return (
        f"SELECT * FROM read_csv('{path}', header=false, columns={spec}, "
        f"quote='\"', escape='\\', nullstr='')"
    )


def _connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("SET enable_progress_bar=false")
    con.sql("SET preserve_insertion_order=false")
    return con


def ensure_job_data(base: str = JOB_BASE, *, local: str = JOB_LOCAL) -> None:
    """Materialize the IMDb tables as parquet under ``base`` if they are not there yet.

    Downloads and extracts the archive on the first call only. A ``base`` pointing somewhere
    other than the default local mirror is read as given and never written to, so an
    explicitly configured ``BENCH_JOB_BASE`` can't be silently filled in underneath.

    Args:
        base: Destination for ``{base}/{table}/part0.parquet``.
        local: Directory holding (or to hold) the extracted CSVs and the archive.
    """
    if base != os.path.join(local, "parquet"):
        return
    if all(os.path.isdir(os.path.join(base, name)) for name in JOB_TABLES):
        return

    os.makedirs(local, exist_ok=True)
    archive = os.path.join(local, "imdb.tgz")
    if not os.path.exists(os.path.join(local, "schematext.sql")):
        if not os.path.exists(archive):
            print(f"downloading the JOB IMDb archive (~1.2 GiB) to {archive} ...", flush=True)
            urllib.request.urlretrieve(JOB_ARCHIVE_URL, archive)
        print(f"extracting {archive} ...", flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(local)

    with open(os.path.join(local, "schematext.sql")) as fh:
        schema = _schema(fh.read())
    missing = sorted(set(JOB_TABLES) - set(schema))
    if missing:
        raise RuntimeError(f"the archive's schematext.sql is missing tables: {missing}")

    con = _connection()
    for name in JOB_TABLES:
        out_dir = os.path.join(base, name)
        if os.path.isdir(out_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)
        csv = os.path.join(local, f"{name}.csv")
        target = os.path.join(out_dir, "part0.parquet")
        print(f"converting {name} ...", flush=True)
        con.sql(f"COPY ({_read_csv_sql(csv, schema[name])}) TO '{target}' (FORMAT PARQUET)")


def job_tables(base: str | None = None) -> dict[str, pa.Table]:
    """Load the 21 IMDb tables into Arrow, converting from CSV on the first call.

    Args:
        base: Parquet base directory; defaults to the local mirror.

    Returns:
        Table name mapped to its Arrow table.
    """
    base = base or JOB_BASE
    ensure_job_data(base)
    con = _connection()
    return {
        name: con.sql(
            f"SELECT * FROM read_parquet('{os.path.join(base, name, '*.parquet')}')"
        ).to_arrow_table()
        for name in JOB_TABLES
    }
