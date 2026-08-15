"""Benchmark the shuffle a partitioned table's own layout makes unnecessary.

A Hive-partitioned table read one directory per split has already co-located equal partition
values on one worker, so a ``GROUP BY`` over the partition column needs no exchange:
`dist/executor.py::_partition_aligned_aggregate` runs the aggregate per partition and
concatenates. This script measures what that exchange cost, by running the *same* query both
ways -- once with the elimination on, once with it forced off -- and checking they agree.

Correctness first, as the harness insists: a fast wrong answer is a bug, and here the wrong
answer has a specific shape (a group reported twice as two partial finals), so the two results
are compared row for row before any timing is reported.

`--sweep` answers the second question the scheduler has to: not whether the elimination is
faster on a well-shaped table, but *where it stops being*. It varies the partition count
against a fixed fleet, forcing the aligned path at every point, and is what
`dist/executors/map.py::_MIN_PARALLELISM_RETENTION` is set from.

Both reader families are covered, because they split differently and the guarantee is reached
differently. A Hive Parquet tree splits one *directory* per partition, so co-location is
immediate; Delta splits one *data file*, so a partition is many splits and co-location comes
from grouping them before assignment. The second is the shape every real lakehouse table has.

Run:
    source .venv/bin/activate
    python3 benchmarks/internals/partition_aligned.py                     # parquet, 16 x 500k
    python3 benchmarks/internals/partition_aligned.py --format delta      # 4 files/partition
    python3 benchmarks/internals/partition_aligned.py --partitions 32 --workers 16
    python3 benchmarks/internals/partition_aligned.py --sweep                 # both readers

Ray must be able to package the working directory, so run it from a small directory (or let
this script's own default local cluster start it).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARTITIONS = 16
ROWS_PER_PARTITION = 500_000
WORKERS = 8
REPEATS = 3
# Delta writes one data file per partition per append, so this is how many splits a partition
# is cut into -- the whole reason the Delta case is measured separately.
DELTA_APPENDS = 4


def _write_parquet(root: Path, partitions: int, rows_each: int) -> None:
    """One ``day=<n>`` directory per partition, each a single Parquet file."""
    rng = np.random.default_rng(0)
    for day in range(partitions):
        (root / f"day={day}").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "v": rng.integers(0, 1000, rows_each).astype("int64"),
                    "g": rng.integers(0, 50, rows_each).astype("int64"),
                }
            ),
            root / f"day={day}" / "part.parquet",
        )


def _write_delta(root: Path, partitions: int, rows_each: int) -> None:
    """The same data as a partitioned Delta table, in `DELTA_APPENDS` commits.

    Each append writes one data file per partition, so a partition ends up as several splits
    and the read cannot rely on the split set being distinct.
    """
    import batcher as bt

    rng = np.random.default_rng(0)
    per_append = max(1, rows_each // DELTA_APPENDS)
    for append in range(DELTA_APPENDS):
        table = pa.table(
            {
                "day": pa.array(np.repeat(np.arange(partitions), per_append), pa.int64()),
                "v": pa.array(rng.integers(0, 1000, partitions * per_append), pa.int64()),
                "g": pa.array(rng.integers(0, 50, partitions * per_append), pa.int64()),
            }
        )
        bt.from_arrow(table).write.delta(
            str(root), partition_by=["day"], mode="overwrite" if append == 0 else "append"
        )


def _never_aligned(*_args, **_kwargs) -> tuple[str, ...]:
    """Force the shuffle path, so the same query is measured both ways."""
    return ()


def _query(root: Path, fmt: str):
    import batcher as bt
    from batcher import col, count

    read = bt.read.delta if fmt == "delta" else bt.read.parquet
    return read(str(root)).group_by("day").agg(s=col("v").sum(), n=count())


def _time(root: Path, fmt: str, workers: int, repeats: int) -> tuple[float, list[dict]]:
    """Best-of-`repeats` wall time, and the rows, for the query as currently routed."""
    times = []
    rows: list[dict] = []
    for _ in range(repeats):
        start = time.perf_counter()
        table = _query(root, fmt).collect(distributed=True, num_workers=workers)
        times.append(time.perf_counter() - start)
        rows = sorted(table.to_pylist(), key=lambda r: r["day"])
    return min(times), rows


def main(
    partitions: int = PARTITIONS,
    rows_each: int = ROWS_PER_PARTITION,
    workers: int = WORKERS,
    fmt: str = "parquet",
) -> int:
    import ray

    import batcher.dist.executor as dist_executor

    root = Path(tempfile.mkdtemp(prefix="batcher_partition_aligned_"))
    try:
        if fmt == "delta":
            shutil.rmtree(root, ignore_errors=True)  # the writer creates the table root
            _write_delta(root, partitions, rows_each)
        else:
            _write_parquet(root, partitions, rows_each)
        ray.init(
            num_cpus=workers,
            include_dashboard=False,
            logging_level="ERROR",
            ignore_reinit_error=True,
        )
        try:
            # A first run pays cluster warm-up and the Parquet footer reads; it is timed
            # with the rest and discarded by the best-of, but it must not be the *only* run.
            aligned_t, aligned_rows = _time(root, fmt, workers, REPEATS)
            keep = dist_executor._partition_aligned_aggregate
            dist_executor._partition_aligned_aggregate = _never_aligned
            try:
                shuffle_t, shuffle_rows = _time(root, fmt, workers, REPEATS)
            finally:
                dist_executor._partition_aligned_aggregate = keep
        finally:
            ray.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if aligned_rows != shuffle_rows:
        print("MISMATCH: the two paths disagree — not reporting a timing on an unverified path")
        print(f"  aligned: {aligned_rows[:3]}\n  shuffle: {shuffle_rows[:3]}")
        return 1

    total = partitions * rows_each
    layout = f"{partitions} day partitions"
    if fmt == "delta":
        layout += f" x {DELTA_APPENDS} data files"
    print(f"{total:,} rows across {layout} ({fmt}), {workers} workers, GROUP BY day")
    print(f"  results agree: {len(aligned_rows)} groups\n")
    print(f"  partition-aligned (no exchange)  {aligned_t * 1e3:8.1f} ms")
    print(f"  hash shuffle                     {shuffle_t * 1e3:8.1f} ms")
    print(f"\n  aligned vs shuffle: {shuffle_t / aligned_t:.2f}x")
    print(
        "\nWhat this removes is the shuffle's map barrier, its network transfer and its "
        "reduce barrier, so the ratio grows with the data the shuffle no longer moves. It "
        "is a property of this query shape: a GROUP BY on a column the table is NOT "
        "partitioned by keeps its shuffle and is unaffected."
    )
    return 0


def _sweep(workers: int, rows: int) -> int:
    """Aligned vs shuffled across partition counts, for both readers.

    The aligned path is forced at every point (past the retention floor it would normally
    refuse), because the question is what the floor *should* be, and a run that declines to
    measure the losing cases cannot answer it.
    """
    import ray

    import batcher.dist.executor as dist_executor
    import batcher.dist.executors.map as map_mod

    ray.init(
        num_cpus=workers, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True
    )
    counts = (1, 2, 4, 8, 16)
    try:
        print(f"{rows:,} rows, {workers} workers, GROUP BY the partition column")
        header = f"{'reader':>9} {'parts':>6} {'splits':>7} {'aligned':>10}"
        print(f"{header} {'shuffled':>10} {'ratio':>7}")
        for fmt in ("parquet", "delta"):
            for parts in counts:
                root = Path(tempfile.mkdtemp(prefix="batcher_sweep_"))
                try:
                    per = max(1, rows // parts)
                    if fmt == "delta":
                        shutil.rmtree(root, ignore_errors=True)
                        _write_delta(root, parts, per)
                    else:
                        _write_parquet(root, parts, per)
                    splits = _split_count(root, fmt)
                    floor = map_mod._MIN_PARALLELISM_RETENTION
                    map_mod._MIN_PARALLELISM_RETENTION = 10**9  # force it, to measure the losses
                    try:
                        aligned, _ = _time(root, fmt, workers, REPEATS)
                    finally:
                        map_mod._MIN_PARALLELISM_RETENTION = floor
                    keep = dist_executor._partition_aligned_aggregate
                    dist_executor._partition_aligned_aggregate = _never_aligned
                    try:
                        shuffled, _ = _time(root, fmt, workers, REPEATS)
                    finally:
                        dist_executor._partition_aligned_aggregate = keep
                    print(
                        f"{fmt:>9} {parts:>6} {splits:>7} {aligned * 1e3:>9.1f}ms "
                        f"{shuffled * 1e3:>9.1f}ms {shuffled / aligned:>6.2f}x",
                        flush=True,
                    )
                finally:
                    shutil.rmtree(root, ignore_errors=True)
    finally:
        ray.shutdown()
    return 0


def _split_count(root: Path, fmt: str) -> int:
    """How many splits the read plans — the denominator of the retention the sweep varies."""
    import batcher as bt

    read = bt.read.delta if fmt == "delta" else bt.read.parquet
    return len(read(str(root))._sources[0].splits())


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partitions", type=int, default=PARTITIONS)
    parser.add_argument("--rows-per-partition", type=int, default=ROWS_PER_PARTITION)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--format", choices=("parquet", "delta"), default="parquet")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="vary the partition count for both readers, to locate the retention floor",
    )
    ns = parser.parse_args()
    if ns.sweep:
        return _sweep(ns.workers, ns.partitions * ns.rows_per_partition)
    return main(ns.partitions, ns.rows_per_partition, ns.workers, ns.format)


if __name__ == "__main__":
    raise SystemExit(_cli())
