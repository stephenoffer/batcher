"""Format-write benchmark: Batcher vs DuckDB / Polars / PyArrow, across output *shapes*.

`read.py` (beside this file) covers the read side; nothing covered the write side, and that is where
the regressions were. A write's cost is not a property of the format alone — it is the
format crossed with the **shape** of the output, and the shapes diverge by more than the
formats do:

* **one file** — a single encoder, so it is the shape with no parallelism to find;
* **a directory of N files** — N independent encodes, which is what every engine fans out;
* **Hive-partitioned** — a fan-out by key, plus the run detection that finds the keys.

Timing only the first shape hides the other two entirely, which is how a directory write
came to run its parts one after another and a partitioned write came to spend most of its
time in a per-row Python loop, both while the one-file number looked fine.

Correctness is checked before anything is timed: every output is read back and its row
count compared against the source, so a fast writer that drops rows fails rather than wins.

Run:
    python benchmarks/scenarios/formats/write.py                  # 4M rows
    python benchmarks/scenarios/formats/write.py --rows 16000000
    python benchmarks/scenarios/formats/write.py --formats parquet,json
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import statistics
import tempfile
import time

import numpy as np
import pyarrow as pa

import batcher as bt

#: Output shapes, as ``(label, ds.write kwargs)``. `files` is filled in per run from the
#: row count, since a fixed row cap would mean a different file count at each scale.
SHAPES = ("1file", "8files", "hive")

#: Distinct values in the partition key. Low enough that a partition is a sensible file
#: size at every scale here, high enough that the fan-out is real work.
PARTITIONS = 97


def _table(rows: int) -> pa.Table:
    """A mixed-type table (int / float / low-card key / short string) shared by all engines."""
    rng = np.random.default_rng(0)
    words = np.array(["abc", "defg", "hijkl", "mn", "opqrs"])
    return pa.table(
        {
            "id": pa.array(np.arange(rows), type=pa.int64()),
            "x": pa.array(rng.random(rows)),
            "k": pa.array(rng.integers(0, PARTITIONS, rows), type=pa.int64()),
            "s": pa.array(words[rng.integers(0, len(words), rows)]),
        }
    )


def _write_kwargs(shape: str, rows: int) -> dict:
    if shape == "1file":
        return {"single_file": True}
    if shape == "8files":
        return {"max_rows_per_file": max(1, rows // 8)}
    return {"partition_by": ["k"]}


def _out_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def _median_ms(fn, runs: int) -> float:
    """Median-of-``runs`` ms. Median, not best-of: a write is I/O bound enough that the
    fastest run is the one where the page cache happened to absorb it, and the shared
    machines these run on make the spread wide."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _batcher_write(src: str, dest: str, fmt: str, shape: str, rows: int) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    if os.path.exists(dest):
        os.remove(dest)
    bt.read(src).write(dest, fmt, **_write_kwargs(shape, rows))


def _duckdb_write(src: str, dest: str, fmt: str, shape: str, rows: int):
    """DuckDB's ``COPY``, or None where it cannot express the shape/format."""
    try:
        import duckdb
    except ImportError:
        return None
    if fmt not in ("parquet", "csv"):
        return None  # DuckDB has no NDJSON/Arrow COPY target here

    def run() -> None:
        shutil.rmtree(dest, ignore_errors=True)
        con = duckdb.connect()
        opts = [f"FORMAT {fmt.upper()}"]
        if shape == "hive":
            opts += ["PARTITION_BY (k)", "OVERWRITE_OR_IGNORE"]
        elif shape == "8files":
            opts += [f"PER_THREAD_OUTPUT TRUE, FILE_SIZE_BYTES {max(1, rows // 8) * 32}"]
        con.execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{dest}' ({', '.join(opts)})")
        con.close()

    return run


def _polars_write(src: str, dest: str, fmt: str, shape: str, rows: int):  # noqa: ARG001
    """Polars' writer, or None where it cannot express the shape/format."""
    try:
        import polars as pl
    except ImportError:
        return None
    if shape != "1file" or fmt not in ("parquet", "csv", "json"):
        return None  # partitioned / multi-file writing is not a like-for-like here

    def run() -> None:
        if os.path.exists(dest):
            os.remove(dest)
        df = pl.read_parquet(src)
        if fmt == "parquet":
            df.write_parquet(dest)
        elif fmt == "csv":
            df.write_csv(dest)
        else:
            df.write_ndjson(dest)

    return run


def _rows_written(dest: str, fmt: str) -> int:
    """Read the output back through Batcher, so a writer that lost rows cannot be timed."""
    return bt.read(dest, format=fmt).count()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4_000_000)
    parser.add_argument("--formats", default="parquet,csv,json,arrow")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    work = tempfile.mkdtemp(prefix="fmtwrite-")
    src = os.path.join(work, "src.parquet")
    import pyarrow.parquet as pq

    pq.write_table(_table(args.rows), src)
    print(f"batcher {bt.versions().get('engine_profile')}  rows={args.rows:,}\n")
    header = f"{'format':8s} {'shape':7s} {'batcher':>10s} {'duckdb':>10s} {'polars':>10s}"
    print(f"{header}  {'Mrow/s':>8s} {'out MB':>8s}")

    for fmt in args.formats.split(","):
        for shape in SHAPES:
            dest = os.path.join(work, f"out-{fmt}-{shape}")
            if shape == "1file":
                dest += "." + fmt
            try:
                write = functools.partial(_batcher_write, src, dest, fmt, shape, args.rows)
                ms = _median_ms(write, args.runs)
            except Exception as exc:
                print(f"{fmt:8s} {shape:7s} {'n/a':>10s}  ({type(exc).__name__})")
                continue
            got = _rows_written(dest, fmt)
            if got != args.rows:
                raise SystemExit(
                    f"{fmt}/{shape}: wrote {got:,} rows, expected {args.rows:,} — "
                    "refusing to report a timing for a write that lost rows"
                )
            size = _out_bytes(dest)
            others = []
            for build in (_duckdb_write, _polars_write):
                other_dest = dest + "-" + build.__name__.split("_")[1]
                run = build(src, other_dest, fmt, shape, args.rows)
                if run is None:
                    others.append("n/a")
                    continue
                try:
                    others.append(f"{_median_ms(run, args.runs):.0f}")
                except Exception:
                    others.append("err")
                finally:
                    shutil.rmtree(other_dest, ignore_errors=True)
            print(
                f"{fmt:8s} {shape:7s} {ms:10.0f} {others[0]:>10s} {others[1]:>10s}"
                f"  {args.rows / ms * 1000 / 1e6:8.2f} {size / 1e6:8.1f}"
            )
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
