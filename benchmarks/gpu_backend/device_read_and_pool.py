"""What the device Parquet read and the RMM pool are actually worth, on real hardware.

Two changes to the GPU backend are argued for mechanically and have never been measured, which
is why no number for either appears anywhere in the tree:

* **Reading on the device.** A worker that reads its own Parquet skips a CPU decode and a trip
  across PCIe. The argument is that on a scan-heavy query the decode is most of the wall clock;
  the counter-argument is that the compressed bytes still cross and a fast host may decode
  faster than the device does. Only a measurement settles it, and it will settle it differently
  per file layout — row-group size and compression codec both move it.
* **The allocator.** Unconfigured, RAPIDS asks the CUDA driver for every intermediate column
  and a driver allocation synchronizes the device. The argument is that a chain of many
  operators makes thousands of them; the size of that constant is unknown.

What this script measures is the **allocator**, against the CPU engine as a baseline. The
device read is deliberately *not* a second axis: it is chosen per shard by the worker that
holds it, from whether the shard is device-readable and whether a predicate was pushed into
its scan, so there is no configuration switch to flip and a cell that pretended otherwise
would be measuring something else. To isolate it, compare a run over this file against one
over the same data written in a layout the device reader declines — a decimal column will
do it.

Correctness is checked before anything is timed: every cell must return the same rows as
Batcher's CPU engine, and a cell that does not is reported and not timed. A fast wrong answer
is a bug.

Run:
    python benchmarks/gpu_backend/device_read_and_pool.py            # 50M rows
    BENCH_ROWS=200000000 python benchmarks/gpu_backend/device_read_and_pool.py
    BENCH_ROW_GROUP=1000000 python benchmarks/gpu_backend/device_read_and_pool.py

Results belong in `benchmarks/BENCHMARK_RESULTS.md` with the device model, the driver version,
the row-group size and the codec named. Without those four the ratio is not reproducible.

Nothing here has been run on a GPU. It is written to produce the numbers, not to assert them.
"""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

print = functools.partial(print, flush=True)

_SEED = 20240730
ROWS = int(os.environ.get("BENCH_ROWS", 50_000_000))
ROW_GROUP = int(os.environ.get("BENCH_ROW_GROUP", 512 * 1024))
CODEC = os.environ.get("BENCH_CODEC", "snappy")


def _write_source(path: str) -> None:
    """A wide-ish scan-heavy table: a group key, two measures, and a string nobody selects.

    The unselected string column is the point of the projection: a device read that honors it
    moves a fraction of the file, and a benchmark whose every column is read cannot see that.
    """
    rng = np.random.default_rng(_SEED)
    chunk = 4_000_000
    writer = None
    try:
        for start in range(0, ROWS, chunk):
            n = min(chunk, ROWS - start)
            batch = pa.table(
                {
                    "k": rng.integers(0, 5000, n, dtype=np.int64),
                    "v": rng.random(n),
                    "w": rng.random(n),
                    "label": pa.array([f"row-{i}" for i in range(start, start + n)]),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(path, batch.schema, compression=CODEC)
            writer.write_table(batch, row_group_size=ROW_GROUP)
    finally:
        if writer is not None:
            writer.close()


def _query(path: str, backend: str):
    """The scan-heavy shape a GPU is worth using for: read, derive, reduce."""
    import batcher as bt
    from batcher import col

    ds = bt.read.parquet(path)
    plan = (
        ds.select("k", "v", "w")
        .group_by("k")
        .agg(total=(col("v") * col("w")).sum(), n=col("v").count())
    )
    return plan.collect(backend=backend)


def _canonical(table) -> list[tuple]:
    rows = table.to_pydict()
    out = [
        (int(k), round(float(t), 6), int(n))
        for k, t, n in zip(rows["k"], rows["total"], rows["n"], strict=True)
    ]
    return sorted(out)


def _time(path: str, label: str, expected) -> None:
    """Run one cell, refuse to time it unless it agrees with the CPU engine, then report."""
    start = time.perf_counter()
    try:
        got = _query(path, "gpu")
    except Exception as exc:  # a cell the fleet cannot run is reported, not hidden
        print(f"{label:34s} FAILED: {type(exc).__name__}: {exc}")
        return
    elapsed = (time.perf_counter() - start) * 1000.0
    if _canonical(got) != expected:
        print(f"{label:34s} WRONG RESULT — not timed")
        return
    print(f"{label:34s} {elapsed:9.1f} ms")


def main() -> None:
    import batcher as bt
    from batcher import Config
    from batcher.config import AcceleratorConfig, DeviceMemoryConfig
    from batcher.core.gpu_transform import gpu_available

    if not gpu_available():
        # `collect(backend="gpu")` on a host with no device silently uses the CPU engine, which
        # is what makes the flag safe and what would make this script lie: the rows below would
        # be the CPU engine twice, the second one over a warm page cache, and they would read
        # as a speedup. Refusing is the only honest option.
        print("no accelerator on this host: the GPU rows would be the CPU engine. Not running.")
        return

    tmp = tempfile.mkdtemp(prefix="bench-device-read-")
    path = os.path.join(tmp, "source.parquet")
    try:
        print(f"writing {ROWS:,} rows (row group {ROW_GROUP:,}, codec {CODEC})")
        _write_source(path)
        print(f"file is {os.path.getsize(path) / (1 << 30):.2f} GiB")

        start = time.perf_counter()
        expected = _canonical(_query(path, "cpu"))
        print(f"{'cpu engine (the oracle)':34s} {(time.perf_counter() - start) * 1000.0:9.1f} ms")

        for allocator in ("default", "pool"):
            memory = DeviceMemoryConfig(allocator=allocator)
            cfg = Config().replace(accelerator=AcceleratorConfig(memory=memory))
            with bt.config_context(cfg):
                _time(path, f"gpu, {allocator} allocator", expected)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
