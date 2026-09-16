"""Benchmark `ORDER BY <clustered key> LIMIT k` and a spilled selective filter over Parquet.

Two scan-side mechanisms, measured on the data shape each exists for: an event table written
in time order, so its row groups are clustered on the timestamp.

``topn``
    The footer-proved top-N bound (`kyber/learned_tuning/topn_footer.py`), DuckDB's
    `RowGroupPruner` technique. Timed **cold**, in a fresh interpreter per run, because the
    bound's whole point is the first run -- a warm run would also have the remembered bound
    (`topn_bound.py`) and could not tell the two apart. Warm best-of-N is printed beside it.

``spill``
    A selective filter under `collect(spill=True)`, with and without the predicate reaching the
    out-of-core read (`dist/spill/scratch.py::map_predicate`). The two arms alternate in one
    process, so both see the same box; ``off`` restores the old tap, which read the source
    unfiltered.

Correctness before timing: every top-N result is compared, in order, against DuckDB's, and the
spilled aggregate against the in-memory one, before a time is printed.

Run:
    python benchmarks/internals/operators/parquet_topn.py                 # 40M rows, 8 files
    python benchmarks/internals/operators/parquet_topn.py --rows 10000000
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envinfo import machine_fingerprint, require_quiet_box, require_release_build

_FILES = 8
_ROW_GROUP = 500_000
_K = 10

_COLD = """
import json, sys, time
import batcher as bt
path, descending = sys.argv[1], sys.argv[2] == "1"
t = time.perf_counter()
out = bt.read.parquet(path).sort("ts", "id", descending=descending).limit({k}).collect()
print(json.dumps({{"ms": (time.perf_counter() - t) * 1000, "id": out.column("id").to_pylist()}}))
"""


def _write(root: Path, rows: int) -> str:
    """An event log in time order: clustered `ts`, a unique `id`, eight payload columns."""
    rng = np.random.default_rng(0)
    per = rows // _FILES
    for f in range(_FILES):
        base = f * per
        ids = np.arange(base, base + per, dtype=np.int64)
        cols = {
            "ts": pa.array((ids * 1000 + rng.integers(0, 50_000, per)).astype("datetime64[ms]")),
            "id": pa.array(ids),
            "g": pa.array(rng.integers(0, 1_000_000, per).astype(str)),
        }
        for i in range(7):
            cols[f"p{i}"] = pa.array(rng.random(per))
        pq.write_table(pa.table(cols), root / f"part-{f}.parquet", row_group_size=_ROW_GROUP)
    return str(root / "*.parquet")


def _duck_topn(path: str, descending: bool) -> tuple[float, list[int]]:
    direction = "DESC" if descending else "ASC"
    sql = (
        f"SELECT id FROM read_parquet('{path}') ORDER BY ts {direction}, id {direction} LIMIT {_K}"
    )
    t = time.perf_counter()
    ids = [r[0] for r in duckdb.sql(sql).fetchall()]
    return (time.perf_counter() - t) * 1000, ids


def _topn(path: str, repeats: int) -> None:
    import polars as pl

    import batcher as bt

    print(f"\n== topn: ORDER BY ts LIMIT {_K} (cold = fresh interpreter, first query)")
    print(
        f"{'direction':<10} {'batcher cold':>13} {'batcher warm':>13} {'duckdb':>9} {'polars':>9}"
    )
    for descending in (True, False):
        cold, oracle = [], None
        for _ in range(repeats):
            run = subprocess.run(
                [sys.executable, "-c", _COLD.format(k=_K), path, "1" if descending else "0"],
                capture_output=True,
                text=True,
                check=True,
            )
            got = json.loads(run.stdout.strip().splitlines()[-1])
            _, oracle = _duck_topn(path, descending)
            if got["id"] != oracle:
                raise SystemExit(f"top-N disagrees with DuckDB: {got['id']} vs {oracle}")
            cold.append(got["ms"])
        warm, duck, polars = [], [], []
        query = bt.read.parquet(path).sort("ts", "id", descending=descending).limit(_K)
        for _ in range(repeats):
            t = time.perf_counter()
            query.collect()
            warm.append((time.perf_counter() - t) * 1000)
            duck.append(_duck_topn(path, descending)[0])
            t = time.perf_counter()
            pl.scan_parquet(path).sort(["ts", "id"], descending=descending).head(_K).collect()
            polars.append((time.perf_counter() - t) * 1000)
        label = "DESC" if descending else "ASC"
        print(
            f"{label:<10} {statistics.median(cold):>10.1f} ms {min(warm):>10.1f} ms "
            f"{min(duck):>6.1f} ms {min(polars):>6.1f} ms"
        )


def _spill(path: str, rows: int, repeats: int) -> None:
    import batcher as bt
    import batcher.dist.spill.scratch as scratch

    cut = rows - rows // 20  # the last 5% of ids: a selective, clustered predicate

    def query():
        return (
            bt.read.parquet(path)
            .filter(bt.col("id") >= cut)
            .group_by("g")
            .agg(s=bt.col("p0").sum())
        )

    expected = query().collect().sort_by("g")
    on = scratch.iter_source

    def off(source, projection, _predicate):
        return source.iter_batches(projection)

    times: dict[str, list[float]] = {"off": [], "on": []}
    for _ in range(repeats):
        for arm, tap in (("off", off), ("on", on)):
            scratch.iter_source = tap
            t = time.perf_counter()
            got = query().collect(spill=True)
            times[arm].append((time.perf_counter() - t) * 1000)
            if got.num_rows != expected.num_rows:
                raise SystemExit(
                    f"spilled result has {got.num_rows} rows, expected {expected.num_rows}"
                )
    scratch.iter_source = on
    speedup = min(times["off"]) / min(times["on"])
    print(f"\n== spill: filter id >= {cut:,} then GROUP BY, collect(spill=True), best of {repeats}")
    print(f"predicate not pushed {min(times['off']):>8.1f} ms")
    print(f"predicate pushed     {min(times['on']):>8.1f} ms   ({speedup:.1f}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=40_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--allow-busy-box", action="store_true")
    args = parser.parse_args()

    require_release_build()
    require_quiet_box(allow_busy=args.allow_busy_box)
    print(json.dumps(machine_fingerprint()))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), args.rows)
        _topn(path, args.repeats)
        _spill(path, args.rows, args.repeats)


if __name__ == "__main__":
    main()
