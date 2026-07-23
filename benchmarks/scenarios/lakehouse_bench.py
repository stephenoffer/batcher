"""Lakehouse benchmark: transaction-log file skipping (read) and metadata-only commit (write).

Two numbers this measures, both of which the table-shaped suites cannot see because they
scan whole tables and write once:

**Read — data skipping.** A lakehouse table's log records each data file's partition
values and column bounds. A selective predicate should therefore be answered by opening
*one* file, not all of them. The gap this exposes is between an engine that consults the
log at plan time and one that opens every footer and filters afterwards. Batcher is
compared against DuckDB's ``delta_scan`` on the same table.

**Write — commit cost.** A distributed write's driver should register the files its
workers wrote, not re-encode them. This times the driver's commit phase against the
"stream every shard back through the writer" design it replaces, on identical worker
output. That cost is ``O(rows)`` in the old shape and ``O(files)`` in the new one, so the
gap grows with the data — which is the point, and why a single ratio understates it.

Correctness is checked before anything is timed (the harness rule: never report a speed
on a path whose answer has not been verified). Run it directly::

    python benchmarks/scenarios/lakehouse_bench.py            # default 10M rows / 200 files
    python benchmarks/scenarios/lakehouse_bench.py 40000000   # a bigger table
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Files the read benchmark spreads its rows across. Each holds one distinct `day`, so a
# `day = k` predicate is answerable from the log alone: exactly one file can match, and a
# reader that consults the log opens 1 of `FILES` instead of all of them.
FILES = 200
SHARDS = 16  # "workers" whose output the write benchmark commits


def _build_table(root: str, rows: int) -> None:
    """A naturally-clustered Delta table: `FILES` data files, one `day` each."""
    from deltalake import write_deltalake

    rng = np.random.default_rng(0)
    per_file = max(1, rows // FILES)
    for day in range(FILES):
        chunk = pa.table(
            {
                "day": pa.array(np.full(per_file, day, dtype=np.int64)),
                "id": pa.array(rng.integers(0, 1_000_000, per_file, dtype=np.int64)),
                "val": pa.array(rng.random(per_file)),
            }
        )
        write_deltalake(root, chunk, mode="overwrite" if day == 0 else "append")


def _time(fn: Any, repeat: int = 5) -> float:
    """Best-of-`repeat` milliseconds, after one warm-up."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def bench_read(root: str) -> None:
    """Selective read: how much work does a `day = k` predicate actually cost?"""
    import duckdb

    import batcher as bt
    from batcher.io.formats.lakehouse import DeltaSource

    predicate = {
        "e": "binary",
        "op": "eq",
        "left": {"e": "col", "name": "day"},
        "right": {"e": "lit", "value": {"int": 42}},
    }

    def batcher_selective() -> int:
        return bt.read.delta(root).filter(bt.col("day") == 42).count()

    def duckdb_selective() -> int:
        return duckdb.sql(f"select count(*) from delta_scan('{root}') where day = 42").fetchone()[0]

    # Correctness first — never time a path whose answer is not verified.
    got, expected = batcher_selective(), duckdb_selective()
    if got != expected:
        raise SystemExit(f"CORRECTNESS FAILURE: batcher={got} duckdb={expected}")

    source = DeltaSource(root)
    all_splits = len(source.splits())
    pruned = len(source.splits(predicate=predicate))

    batcher_ms = _time(batcher_selective)
    duckdb_ms = _time(duckdb_selective)

    print("\n--- read: selective predicate (day = 42) ---")
    print(f"  rows matched          {got:,} (verified against DuckDB)")
    print(f"  files in table        {all_splits}")
    print(f"  files planned         {pruned}   <- opened; the rest are skipped from the log")
    print(f"  batcher               {batcher_ms:8.1f} ms")
    print(f"  duckdb (delta_scan)   {duckdb_ms:8.1f} ms")
    print(f"  ratio                 {duckdb_ms / batcher_ms:8.2f}x  (>1 = batcher faster)")


def bench_write(rows: int) -> None:
    """Driver commit cost: re-encode every shard vs. register the files they wrote."""
    from deltalake import DeltaTable, write_deltalake

    from batcher.io.formats.lakehouse.delta._commit import collect_file_stats, commit_add_actions
    from batcher.io.manifest import WriteManifest, WrittenFile

    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "day": pa.array(rng.integers(0, 100, rows, dtype=np.int64)),
            "id": pa.array(np.arange(rows, dtype=np.int64)),
            "val": pa.array(rng.random(rows)),
        }
    )
    per = rows // SHARDS
    shards = [table.slice(i * per, per) for i in range(SHARDS)]

    # --- OLD: workers stage files; the driver reads every one back and re-encodes it.
    old_root = tempfile.mkdtemp(prefix="lakehouse_old_")
    stage = os.path.join(old_root, "_stage")
    os.makedirs(stage)
    staged = []
    for i, shard in enumerate(shards):
        path = os.path.join(stage, f"part-{i:05d}.parquet")
        pq.write_table(shard, path, compression="zstd")
        staged.append(path)
    schema = pq.ParquetFile(staged[0]).schema_arrow

    def _stream() -> Any:
        for path in staged:
            yield from pq.ParquetFile(path).iter_batches()

    old_target = os.path.join(old_root, "t")
    start = time.perf_counter()
    write_deltalake(old_target, pa.RecordBatchReader.from_batches(schema, _stream()), mode="append")
    old_ms = (time.perf_counter() - start) * 1000.0

    # --- NEW: workers write FINAL files (+ stats); the driver commits only AddActions.
    new_root = tempfile.mkdtemp(prefix="lakehouse_new_")
    new_target = os.path.join(new_root, "t")
    os.makedirs(new_target)
    written = []
    for i, shard in enumerate(shards):  # this loop is the *worker's* work, not the driver's
        name = os.path.join(new_target, f"part-{i:05d}-tok.parquet")
        pq.write_table(shard, name, compression="zstd")
        written.append(
            WrittenFile(
                path=name,
                rows=shard.num_rows,
                bytes=os.path.getsize(name),
                stats=collect_file_stats(shard),
            )
        )
    start = time.perf_counter()
    commit_add_actions(
        WriteManifest(tuple(written), schema=table.schema), new_target, mode="append"
    )
    new_ms = (time.perf_counter() - start) * 1000.0

    old_rows = DeltaTable(old_target).to_pyarrow_table().num_rows
    new_rows = DeltaTable(new_target).to_pyarrow_table().num_rows
    if not old_rows == new_rows == rows:
        raise SystemExit(f"CORRECTNESS FAILURE: old={old_rows} new={new_rows} expected={rows}")
    stats_written = "minValues" in _first_commit(new_target)

    megabytes = table.nbytes / 1e6
    print(f"\n--- write: driver commit of {SHARDS} worker shards ({megabytes:.0f} MB) ---")
    print(f"  rows committed        {new_rows:,} (both designs agree)")
    print(f"  OLD  re-encode        {old_ms:8.1f} ms   (driver rewrites every shard: O(rows))")
    print(f"  NEW  metadata commit  {new_ms:8.1f} ms   (driver registers files: O(files))")
    print(f"  speedup               {old_ms / new_ms:8.1f}x")
    print(f"  bytes through driver  OLD ~{megabytes:.0f} MB    NEW 0 MB")
    print(f"  skipping stats in log {stats_written}  <- what makes the next read prunable")

    shutil.rmtree(old_root, ignore_errors=True)
    shutil.rmtree(new_root, ignore_errors=True)


def _first_commit(root: str) -> str:
    """The table's first commit entry, to confirm the write recorded skipping statistics."""
    log = os.path.join(root, "_delta_log")
    entry = os.path.join(log, sorted(f for f in os.listdir(log) if f.endswith(".json"))[0])
    with open(entry) as fh:
        return fh.read()


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    try:
        import deltalake  # noqa: F401
        import duckdb  # noqa: F401
    except ImportError:
        raise SystemExit(
            "the lakehouse benchmark needs the delta extra and duckdb:\n"
            "  pip install 'batcher-engine[delta,duckdb]'"
        ) from None

    root = tempfile.mkdtemp(prefix="lakehouse_bench_")
    table_root = os.path.join(root, "t")
    print(f"building a {rows:,}-row Delta table across {FILES} files ...")
    start = time.perf_counter()
    _build_table(table_root, rows)
    print(f"built in {time.perf_counter() - start:.1f}s")

    bench_read(table_root)
    bench_write(rows)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
