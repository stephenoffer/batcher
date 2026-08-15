"""Sharded training-corpus loading: what the shuffle costs the shard cache.

`batcher.ml.shard_stream_loader` is the larger-than-RAM training path. It reads a corpus of
Arrow-IPC shards by *sample index*, through a bounded LRU shard cache, so the resident set is
a few shards no matter how large the corpus.

That only works if the sample order has a working set. A **global** shuffle does not: every
sample is uniform over the whole corpus, so a batch of `batch_size` samples lands in up to
`batch_size` different shards, the cache misses on nearly all of them, and each miss reads a
whole shard to use one row of it. The epoch then reads the corpus many times over, and the
cache size is not the problem — the order is.

The default order is therefore *blocked*: the shards are shuffled, and the rows inside each
shard are shuffled (MosaicML Streaming's ``py1s``; the reason WebDataset pairs a shard
shuffle with a sample buffer). A batch stays inside one shard, so each shard is read once per
epoch, and the order is still seed-keyed and different every epoch.

This measures that difference, and the prefetch that overlaps the shard read with the
consumer's step. Every configuration must yield the *same* multiset of rows — the script
checks that before reporting a rate, per the harness discipline.

Run:
    python benchmarks/scenarios/training/shard_loader_throughput.py
    python benchmarks/scenarios/training/shard_loader_throughput.py --rows 800000 --width 128
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import warnings

import numpy as np
import pyarrow as pa


def _write_corpus(directory: str, rows: int, width: int, rows_per_shard: int) -> None:
    """Write a `rows`-row corpus of `width`-wide float32 feature vectors plus a label."""
    from batcher.io.formats.ml.shards import write_shards

    rng = np.random.default_rng(0)
    table = pa.table(
        {
            # A unique id, so the correctness check can compare the exact rows an epoch
            # read rather than a histogram of labels that different rows could produce.
            "row_id": np.arange(rows, dtype=np.int64),
            "f": pa.FixedSizeListArray.from_arrays(
                pa.array(rng.random(rows * width, dtype=np.float32)), width
            ),
        }
    )
    del rng
    write_shards(table, directory, rows_per_shard=rows_per_shard)


def _drain(
    directory: str, *, block: int | None, cache: int, prefetch: int, batch: int, step_ms: float
):
    """Read one whole epoch and return ``(rows_per_second, the row ids seen)``.

    A *whole* epoch, not a fixed number of steps: two shuffle orders agree on the rows an
    epoch contains and disagree completely on any prefix of it, so timing a prefix compares
    two different samples of the corpus. And ``drop_last=False``, because `drop_last`
    discards the epoch's ragged *tail* — which is a different set of rows under a different
    order. Both of those were caught by this script's own correctness check rather than
    reasoned about in advance, which is the argument for having one.
    """
    from batcher.ml.loader import shard_stream_loader

    with warnings.catch_warnings():
        # A global shuffle over a small cache warns that it will thrash; that is the very
        # thing being measured here, so it is expected rather than a problem to report.
        warnings.simplefilter("ignore")
        loader = shard_stream_loader(
            directory,
            batch_size=batch,
            seed=1,
            cache_size=cache,
            shuffle_block_size=block,
            prefetch_batches=prefetch,
            drop_last=False,
        )
        stream = iter(loader)
        seen: list[int] = []
        first = next(stream)  # excluded from the timing: it pays the first shard read
        seen.extend(int(v) for v in first["row_id"].tolist())
        start = time.perf_counter()
        rows = len(first["row_id"])
        for item in stream:
            seen.extend(int(v) for v in item["row_id"].tolist())
            rows += len(item["row_id"])
            if step_ms:
                time.sleep(step_ms / 1000.0)
        elapsed = time.perf_counter() - start
    return rows / elapsed, seen


def main() -> None:
    """Write a corpus, then time the loader under each shuffle/prefetch configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=400_000)
    parser.add_argument("--width", type=int, default=64, help="feature vector width")
    parser.add_argument("--rows-per-shard", type=int, default=16_384)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--cache", type=int, default=4, help="shards held resident")
    parser.add_argument(
        "--step-ms",
        type=float,
        default=1.0,
        help="simulated training-step time per batch; 0 measures the loader in isolation",
    )
    args = parser.parse_args()

    directory = tempfile.mkdtemp(prefix="batcher-shard-bench-")
    try:
        _write_corpus(directory, args.rows, args.width, args.rows_per_shard)
        shards = -(-args.rows // args.rows_per_shard)
        print(f"corpus: {args.rows:,} rows x {args.width} features in {shards} shards")
        print(
            f"loader: batch={args.batch} cache_size={args.cache} "
            f"step={args.step_ms}ms/batch, one full epoch timed\n"
        )

        common = {"cache": args.cache, "batch": args.batch, "step_ms": args.step_ms}
        configurations = [
            ("global shuffle,  no prefetch", {"block": 0, "prefetch": 0}),
            ("blocked shuffle, no prefetch", {"block": None, "prefetch": 0}),
            ("blocked shuffle, prefetch=2 ", {"block": None, "prefetch": 2}),
        ]
        baseline = None
        reference = None
        for name, options in configurations:
            rate, seen = _drain(directory, **common, **options)
            if reference is None:
                reference = sorted(seen)
            elif sorted(seen) != reference:
                # Correctness before timing: a configuration that reads different rows is
                # not a faster loader, it is a broken one.
                raise SystemExit(f"{name}: read a different multiset of rows than the first run")
            baseline = baseline or rate
            print(f"  {name}  {rate:>12,.0f} rows/s   {rate / baseline:>5.1f}x")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
