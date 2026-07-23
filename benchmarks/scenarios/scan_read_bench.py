"""Scan benchmark — what it costs to read a Parquet table and hand back Arrow.

`read_parquet(path).collect()` is the most common thing anyone asks a data engine to do, and
it is the one where an engine's *fixed overhead* has nowhere to hide: there is no join to
dominate, no aggregation to amortize against. Whatever the engine does besides decoding the
file shows up directly in the wall clock.

Three physical layouts of one logical table, because the layout is what exposes that
overhead: a single large file (can the engine parallelize *inside* a file?), a handful of
mid-size files, and many small ones (per-file planning cost — where Ray Data and Spark are
known to struggle).

## Each measurement runs in its own process

Not fastidiousness — necessity. Batcher learns from execution and caches source statistics,
so a second read in the same process is not the same query as the first. And a full-table
read allocates gigabytes, so back-to-back reps measure the allocator and the GC as much as
the reader. One process, one measurement, `min` over processes.

Run:

    python benchmarks/scenarios/scan_read_bench.py --build   # write the corpus once
    python benchmarks/scenarios/scan_read_bench.py           # batcher vs pyarrow
    python benchmarks/scenarios/scan_read_bench.py --ray     # + Ray Data (needs a cluster)
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time

ROWS = 20_000_000
COLUMNS = 16
# Where the corpus lives. Ray Data's workers are on other nodes, so it has to be somewhere
# every node can see — a local /tmp would have the workers reading an empty directory.
BASE = os.environ.get("BENCH_SCAN_DIR", "/mnt/cluster_storage/bcscan")
LAYOUTS = {"one": ROWS, "mid": 2_000_000, "many": 100_000}


def build() -> None:
    """Write one logical table three ways: 1 file, 10 files, 200 files."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    table = pa.table(
        {f"c{i}": rng.integers(0, 1_000_000, ROWS, dtype=np.int64) for i in range(COLUMNS)}
    )
    for name, rows_per_file in LAYOUTS.items():
        out = f"{BASE}/{name}"
        os.makedirs(out, exist_ok=True)
        if os.listdir(out):
            print(f"{name}: already present")
            continue
        for i, start in enumerate(range(0, ROWS, rows_per_file)):
            pq.write_table(
                table.slice(start, rows_per_file),
                f"{out}/part-{i:05d}.parquet",
                compression="snappy",
            )
        print(f"{name}: {len(os.listdir(out))} files")


# ======================================================================================
# The child: one engine, one layout, one process.
# ======================================================================================


def _child(engine: str, layout: str) -> dict:
    """One engine, one layout. Warm the engine and the page cache once, then time the read.

    The warm-up is what makes this a measurement of *reading* rather than of starting: a
    cold process pays to connect to Ray, JIT-warm the interpreter, and fault the file into
    the page cache, and none of that is what anyone means by "how fast is read_parquet".
    Every engine gets exactly the same courtesy.
    """
    path = f"{BASE}/{layout}"
    read = _reader(engine, path)
    warm = read()  # engine start-up + page cache, for every engine alike
    del warm
    gc.collect()

    start = time.perf_counter()
    rows = read()
    elapsed = time.perf_counter() - start
    return {"ms": elapsed * 1000, "rows": rows}


def _reader(engine: str, path: str):
    """A zero-arg callable that reads `path` and returns its row count."""
    if engine in ("batcher", "batcher-local"):
        import batcher as bt

        distributed = engine == "batcher"
        return lambda: bt.read.parquet(path).collect(distributed=distributed or False).num_rows

    if engine == "pyarrow":
        import pyarrow.dataset as pds

        return lambda: pds.dataset(path, format="parquet").to_table().num_rows

    if engine == "ray":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from engines.ray import _ensure_ray

        _ensure_ray()
        import ray.data as rd

        return lambda: rd.read_parquet(path).materialize().count()

    raise SystemExit(f"unknown engine {engine!r}")


# ======================================================================================
# The parent.
# ======================================================================================


def _measure(engine: str, layout: str, reps: int) -> dict:
    best: dict | None = None
    for _ in range(reps):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", engine, layout],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
        if not lines:
            raise SystemExit(f"{engine}/{layout} failed:\n{proc.stderr[-1500:]}")
        got = json.loads(lines[-1])
        if best is None or got["ms"] < best["ms"]:
            best = got
    assert best is not None
    if best["rows"] != ROWS:
        raise SystemExit(f"{engine}/{layout}: read {best['rows']} rows, expected {ROWS}")
    return best


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--child":
        print(json.dumps(_child(args[1], args[2])))
        return
    if args and args[0] == "--build":
        build()
        return

    engines = ["batcher", "pyarrow"]
    if "--ray" in args:
        engines.append("ray")
    reps = 3

    print(f"\nread_parquet(...) → Arrow, {ROWS:,} rows x {COLUMNS} int64 columns")
    print(f"one measurement per process, min of {reps}; row count verified every run\n")
    header = f"{'layout':<6} {'files':>6}  " + "  ".join(f"{e:>10}" for e in engines)
    if "--ray" in args:
        header += f"  {'vs ray':>8}"
    print(header)
    print("-" * len(header))

    for layout in LAYOUTS:
        files = len(os.listdir(f"{BASE}/{layout}"))
        results = {e: _measure(e, layout, reps) for e in engines}
        row = f"{layout:<6} {files:>6}  " + "  ".join(f"{results[e]['ms']:9.0f}m" for e in engines)
        if "ray" in results:
            row += f"  {results['ray']['ms'] / results['batcher']['ms']:7.1f}x"
        print(row)


if __name__ == "__main__":
    main()
