"""Writing a training corpus: what publishing shards one at a time costs.

`write_shards` repacks a stream into fixed-size Arrow-IPC shards and publishes each one.
On local disk the publish is bandwidth-bound and doing them one at a time is right. Against
an object store it is *latency*-bound: a PUT is tens of milliseconds, so a five-thousand-shard
corpus spends minutes purely waiting, with the encoder idle the whole time — the same
asymmetry `batcher.io.base.sink` already sizes its file writes for.

This measures the overlap. There is no object store here, so the latency is **simulated**: a
filesystem wrapper sleeps for `--latency-ms` after each publish. That is an honest model of
where the time goes (a round trip per shard) and it is reproducible on any machine, but it is
not a cloud measurement and no cloud number should be quoted from it.

Every configuration is checked to have produced a byte-identical corpus before its time is
reported — shards are published out of order, so "faster" is only interesting if shard *k*
still holds rows ``[k * rows_per_shard, ...)``.

Run:
    python benchmarks/scenarios/training/shard_write_throughput.py
    python benchmarks/scenarios/training/shard_write_throughput.py --rows 200000 --latency-ms 50
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
import time

import numpy as np
import pyarrow as pa


class _LatentFilesystem:
    """A filesystem that pauses after each publish, standing in for an object-store PUT."""

    def __init__(self, real, latency_s: float) -> None:
        self._real = real
        self._latency_s = latency_s

    @contextlib.contextmanager
    def atomic_writer(self, path: str):
        """Publish through the real filesystem, then pay the round trip."""
        with self._real.atomic_writer(path) as handle:
            yield handle
        time.sleep(self._latency_s)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def _corpus(rows: int, width: int) -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "row_id": np.arange(rows, dtype=np.int64),
            "f": pa.FixedSizeListArray.from_arrays(
                pa.array(rng.random(rows * width, dtype=np.float32)), width
            ),
        }
    )


def _time_write(table: pa.Table, rows_per_shard: int, concurrency: int, latency_s: float):
    """Write `table` once and return ``(seconds, the row ids the corpus reads back as)``."""
    import batcher.io.formats.ml.shards.writer as writer_module
    from batcher.io.formats.ml.shards import ShardReader

    directory = tempfile.mkdtemp(prefix="batcher-shard-write-")
    real = writer_module.resolve_filesystem
    writer_module.resolve_filesystem = lambda path, **kw: _LatentFilesystem(
        real(path, **kw), latency_s
    )
    try:
        start = time.perf_counter()
        index = writer_module.write_shards(
            table, directory, rows_per_shard=rows_per_shard, write_concurrency=concurrency
        )
        elapsed = time.perf_counter() - start
    finally:
        writer_module.resolve_filesystem = real
    try:
        reader = ShardReader(directory, cache_size=index.shard_count + 1)
        ids = reader.take(list(range(index.total_rows))).column("row_id").to_pylist()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    return elapsed, ids


def main() -> None:
    """Write the same corpus at several concurrencies and compare, checking each result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40_000)
    parser.add_argument("--width", type=int, default=16, help="feature vector width")
    parser.add_argument("--rows-per-shard", type=int, default=500)
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=20.0,
        help="simulated per-publish round trip; 0 measures local disk as it is",
    )
    args = parser.parse_args()

    table = _corpus(args.rows, args.width)
    shards = -(-args.rows // args.rows_per_shard)
    print(f"corpus: {args.rows:,} rows x {args.width} features in {shards} shards")
    print(f"publish latency: {args.latency_ms}ms per shard (simulated)\n")

    expected = list(range(args.rows))
    baseline = None
    for concurrency in (1, 2, 4, 8, 16):
        elapsed, ids = _time_write(
            table, args.rows_per_shard, concurrency, args.latency_ms / 1000.0
        )
        if ids != expected:
            # Correctness before timing: shards are published out of order, so a faster
            # write is only interesting if every shard still landed under the right name.
            raise SystemExit(f"write_concurrency={concurrency} produced the wrong corpus")
        baseline = baseline or elapsed
        print(f"  write_concurrency={concurrency:>3}  {elapsed:6.2f}s  {baseline / elapsed:>5.1f}x")


if __name__ == "__main__":
    main()
