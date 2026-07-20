"""Mirror the public benchmark parquet locally, normalized to the cross-engine schema.

The benchmark suite never generates data: it reads TPC-H from the Ray public bucket and
ClickBench from the ClickHouse mirror. Those raw files are unusable for the ``--scan``
path at scale, though — the TPC-H files carry positional ``column0..N`` names and
decimal types, and every engine's *native* scan needs the canonical ``l_``/``o_``...
names and the float64 normalization that ``benchmarks/sources/`` otherwise applies
after materializing into Arrow (which does not fit at sf100).

This tool performs that normalization once, out-of-core, into a local mirror laid out as
``{out}/tpch/sf{N}/{table}/part-0.parquet`` — exactly the ``{base}/{table}/*.parquet``
shape ``sources.table_uris`` expects. Point the suite at it with ``--source``.

Usage:
    python tools/mirror_bench_data.py --dataset tpch --scale 1 --scale 10 --scale 100
    python tools/mirror_bench_data.py --dataset clickbench --parts 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
from sources import CLICKBENCH_BASE, TPCH_BASE, TPCH_COLUMNS

DEFAULT_OUT = os.environ.get("BENCH_DATA_DIR", os.path.expanduser("~/bench-data"))


def _connect(threads: int) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql("SET enable_progress_bar=false")
    con.sql(f"SET threads={threads}")
    con.sql("SET preserve_insertion_order=false")
    region = os.environ.get("BENCH_S3_REGION")
    if region:
        con.sql(f"SET s3_region='{region}'")
    return con


def _select_list(con: duckdb.DuckDBPyConnection, uri: str, names: tuple[str, ...]) -> str:
    """Positional rename + decimal->double cast, as a SQL select list."""
    desc = con.sql(f"SELECT * FROM read_parquet('{uri}') LIMIT 0").description
    src = [d[0] for d in desc]
    types = {
        r[0]: r[1]
        for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{uri}') LIMIT 0").fetchall()
    }
    parts = []
    for i, target in enumerate(names):
        if i >= len(src):
            break
        col, typ = src[i], types[src[i]]
        expr = f'CAST("{col}" AS DOUBLE)' if typ.startswith("DECIMAL") else f'"{col}"'
        parts.append(f"{expr} AS {target}")
    return ", ".join(parts)


def _copy(con: duckdb.DuckDBPyConnection, query: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    con.sql(f"COPY ({query}) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")


def mirror_tpch(con: duckdb.DuckDBPyConnection, scale: int, base: str, out: str) -> None:
    root = os.path.join(out, "tpch", f"sf{scale}")
    for table, names in TPCH_COLUMNS.items():
        dest = os.path.join(root, table, "part-0.parquet")
        if os.path.exists(dest):
            print(f"  sf{scale}/{table}: exists, skipping")
            continue
        uri = f"{base}/sf{scale}/{table}/*.parquet"
        t0 = time.perf_counter()
        sel = _select_list(con, uri, names)
        _copy(con, f"SELECT {sel} FROM read_parquet('{uri}')", dest)
        size = os.path.getsize(dest) / 1e9
        print(f"  sf{scale}/{table}: {size:.2f} GB in {time.perf_counter() - t0:.1f}s", flush=True)


# ClickBench stores its temporal columns as integers: `EventDate` is days since the epoch
# (uint16), the `*EventTime` columns are unix seconds (int64). The *loaded* path rebuilds them
# in memory (`benchmarks/sources/tables.py::_reconstruct_clickbench_temporals`), but a **scan**
# hands each engine the raw file — so the four queries that call `toMonth`/`toHour` on them (q18,
# q36, q37, q41) fail on every engine, ours included. Normalizing them once, on disk, is what
# makes ClickBench runnable in scan mode at its full 100 M rows.
_CLICKBENCH_DATE_COLUMNS = ("EventDate",)
_CLICKBENCH_TIME_COLUMNS = ("EventTime", "ClientEventTime", "LocalEventTime")


def _clickbench_select(con: duckdb.DuckDBPyConnection, uri: str) -> str:
    """`SELECT` list that rebuilds ClickBench's DATE/TIMESTAMP columns, passing the rest."""
    names = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{uri}')").fetchall()]
    parts = []
    for name in names:
        if name in _CLICKBENCH_DATE_COLUMNS:
            days = f'CAST("{name}" AS INTEGER)'
            parts.append(f"CAST(DATE '1970-01-01' + INTERVAL ({days}) DAY AS DATE) AS \"{name}\"")
        elif name in _CLICKBENCH_TIME_COLUMNS:
            secs = f'CAST("{name}" AS BIGINT)'
            parts.append(f'CAST(TO_TIMESTAMP({secs}) AS TIMESTAMP) AS "{name}"')
        else:
            parts.append(f'"{name}"')
    return ", ".join(parts)


def mirror_clickbench(con: duckdb.DuckDBPyConnection, parts: int, base: str, out: str) -> None:
    dest = os.path.join(out, "clickbench", "hits", "part-0.parquet")
    if os.path.exists(dest):
        print("  clickbench/hits: exists, skipping")
        return
    uris = ", ".join(f"'{base}/hits_{i}.parquet'" for i in range(parts))
    t0 = time.perf_counter()
    sel = _clickbench_select(con, f"{base}/hits_0.parquet")
    _copy(con, f"SELECT {sel} FROM read_parquet([{uris}])", dest)
    size = os.path.getsize(dest) / 1e9
    print(f"  clickbench/hits: {size:.2f} GB in {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("tpch", "clickbench"), required=True)
    p.add_argument("--scale", type=int, action="append", default=None)
    p.add_argument("--parts", type=int, default=100, help="ClickBench partitions to mirror")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = p.parse_args()

    con = _connect(args.threads)
    print(f"mirroring {args.dataset} -> {args.out}")
    if args.dataset == "tpch":
        for scale in args.scale or [1]:
            mirror_tpch(con, scale, TPCH_BASE, args.out)
    else:
        mirror_clickbench(con, args.parts, CLICKBENCH_BASE, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
