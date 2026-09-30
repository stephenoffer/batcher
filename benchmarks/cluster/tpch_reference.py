"""DuckDB's answers to the standard TPC-H suite, over the parquet the cluster runs read.

`tpch_vs_raydata.py --suite standard` has no oracle of its own: Ray's scripts are the
reference for the Ray-variant suite, and nothing was for the standard one, so a cluster
run could only say a query did not raise. This writes one ``tpch_<q>.parquet`` per query,
computed by DuckDB over the same files and the same column renaming, into a directory
(local or ``s3://``) that ``--reference`` then reads.

Run it where the scale fits in DuckDB's reach -- the head node for SF1/SF10, a large
single node for SF100 and up (DuckDB spills to ``--temp-dir``):

    python benchmarks/cluster/tpch_reference.py --sf 10 --base-uri s3://.../tpch_zstd \\
        --out s3://.../tpch_reference/sf10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.fs as pafs
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tpch_raydata import TABLE_COLUMNS

from suites.standard.tpch import QUERIES


def connect(base_uri: str, sf: int, temp_dir: str | None, memory_limit: str | None = None):
    """A DuckDB connection with one view per TPC-H table, named and typed as Batcher reads it."""
    import duckdb

    con = duckdb.connect()
    if base_uri.startswith("s3://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain, REGION 'us-west-2')")
    if temp_dir:
        con.execute(f"SET temp_directory = '{temp_dir}'")
    if memory_limit:
        # DuckDB's default is 80% of the host's RAM, which a container's own limit can sit
        # below: SF1000 q9 was OOM-killed that way.
        con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    for table, mapping in TABLE_COLUMNS.items():
        cols = ", ".join(f'"{src}" AS {dst}' for src, dst in mapping.items())
        path = f"{base_uri}/sf{sf}/{table}/*.parquet"
        con.execute(f"CREATE VIEW {table} AS SELECT {cols} FROM read_parquet('{path}')")
    return con


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sf", type=int, required=True)
    p.add_argument("--base-uri", required=True)
    p.add_argument("--out", required=True, help="directory (local or s3://) for tpch_<q>.parquet")
    p.add_argument("--queries", default="all")
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--memory-limit", default=None, help="DuckDB memory_limit, e.g. 150GB")
    args = p.parse_args()

    con = connect(args.base_uri, args.sf, args.temp_dir, args.memory_limit)
    fs, root = pafs.FileSystem.from_uri(args.out)
    fs.create_dir(root, recursive=True)
    names = [n.removeprefix("tpch-") for n in QUERIES]
    if args.queries != "all":
        names = args.queries.split(",")
    for name in names:
        t = time.perf_counter()
        table = con.execute(QUERIES[f"tpch-{name}"]).to_arrow_table()
        pq.write_table(table, f"{root}/tpch_{name}.parquet", filesystem=fs)
        print(f"{name} {time.perf_counter() - t:.1f}s rows={table.num_rows}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
