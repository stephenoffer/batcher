"""Format-read benchmark: Batcher vs DuckDB / Polars / PyArrow across file formats.

The industry suites all read Parquet; this covers the *other* formats a real pipeline
ingests — Avro, CSV, ORC, Arrow IPC — reading the identical data (written once, shared) with
every engine that supports each format, and verifying row counts agree before timing.

The headline is **Avro**: PyArrow has no Avro reader, so the ecosystem default is the
row-by-row `fastavro` path (a Python dict per row). Batcher decodes Avro natively with
``arrow-avro`` in the Rust data plane — the same columnar reader Polars uses — so it reads a
format that is otherwise a Python-speed cliff at Arrow speed.

Run:
    python benchmarks/scenarios/formats/read.py                 # 3M rows
    python benchmarks/scenarios/formats/read.py --rows 10000000
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.orc as paorc
import pyarrow.parquet as pq

import batcher as bt


def _table(rows: int) -> pa.Table:
    """A mixed-type table (int / float / low-card int / short string), shared by all writers."""
    rng = np.random.default_rng(0)
    words = np.array(["abc", "defg", "hijkl", "mn", "opqrs"])
    return pa.table(
        {
            "id": pa.array(np.arange(rows), type=pa.int64()),
            "x": pa.array(rng.random(rows)),
            "k": pa.array(rng.integers(0, 1000, rows), type=pa.int64()),
            "s": pa.array(words[rng.integers(0, len(words), rows)]),
        }
    )


def _best_ms(fn, runs: int) -> tuple[float, int]:
    """Best-of-``runs`` ms plus the engine's row count (for the correctness gate)."""
    best, rowcount = float("inf"), -1
    for _ in range(runs):
        t0 = time.perf_counter()
        rowcount = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000, rowcount


def _engines_for(fmt: str, path: str) -> dict:
    """Row-count-returning readers per engine that supports ``fmt`` (else absent → n/a)."""
    out: dict = {"batcher": lambda: getattr(bt.read, fmt)(path).collect().num_rows}

    if fmt in ("csv", "parquet", "orc"):

        def _duck() -> int:
            import duckdb

            reader = {"csv": "read_csv", "parquet": "read_parquet", "orc": "read_orc"}[fmt]
            # Materialize to Arrow (not count(*), which DuckDB answers from metadata for
            # Parquet/ORC without reading a column) so every engine does a real full read.
            return duckdb.sql(f"SELECT * FROM {reader}('{path}')").to_arrow_table().num_rows

        out["duckdb"] = _duck

    if fmt in ("csv", "parquet", "arrow", "avro"):

        def _polars() -> int:
            import polars as pl

            reader = {
                "csv": pl.read_csv,
                "parquet": pl.read_parquet,
                "arrow": pl.read_ipc,
                "avro": pl.read_avro,
            }[fmt]
            return reader(path).height

        out["polars"] = _polars

    if fmt == "avro":

        def _fastavro() -> int:
            import fastavro

            with open(path, "rb") as fh:
                return sum(1 for _ in fastavro.reader(fh))

        out["fastavro"] = _fastavro

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Format-read benchmark")
    parser.add_argument("--rows", type=int, default=3_000_000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    table = _table(args.rows)
    tmp = tempfile.mkdtemp()
    paths = {
        "parquet": f"{tmp}/t.parquet",
        "orc": f"{tmp}/t.orc",
        "csv": f"{tmp}/t.csv",
        "arrow": f"{tmp}/t.arrow",
        "avro": f"{tmp}/t.avro",
    }
    pq.write_table(table, paths["parquet"])
    paorc.write_table(table, paths["orc"])
    import pyarrow.csv as pacsv

    pacsv.write_csv(table, paths["csv"])
    feather.write_feather(table, paths["arrow"])
    try:
        bt.from_arrow(table).write.avro(paths["avro"])
    except Exception as exc:  # missing fastavro extra → skip avro
        print(f"(avro write skipped: {exc})")
        paths.pop("avro")

    print(f"\n{args.rows:,} rows, best-of-{args.runs}\n")
    header = f"  {'format':<9} {'size':>8}   " + "  ".join(
        f"{e:>10}" for e in ("batcher", "duckdb", "polars", "fastavro")
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fmt, path in paths.items():
        readers = _engines_for(fmt, path)
        times: dict[str, float] = {}
        counts = set()
        for name, fn in readers.items():
            try:
                ms, n = _best_ms(fn, args.runs)
                times[name] = ms
                counts.add(n)
            except Exception as exc:
                times[name] = -1.0
                print(f"  ({fmt}/{name} error: {str(exc)[:60]})")
        gate = "OK" if len(counts) == 1 else f"MISMATCH {counts}"
        size = os.path.getsize(path) / (1 << 20)
        cells = []
        for e in ("batcher", "duckdb", "polars", "fastavro"):
            v = times.get(e)
            cells.append(f"{v:>10.1f}" if v and v > 0 else f"{'n/a':>10}")
        print(f"  {fmt:<9} {size:>7.0f}M   " + "  ".join(cells) + f"   {gate}")
    print("\n(ms per read; lower is better. avro: batcher native arrow-avro vs Python fastavro.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
