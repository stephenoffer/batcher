"""Checkpoint cost of a stateful streaming query: whole snapshots vs a changelog.

A running aggregate with no watermark never evicts, so its state only grows. Snapshotting
the whole state every micro-batch therefore costs more on every epoch than on the one
before: the total written over a run is **quadratic** in the number of epochs. Recording
the partial each micro-batch folded in — a changelog — costs the *batch's* distinct group
count instead, which is flat, so the total is linear.

This measures both, at several run lengths, so the shape of the two curves is visible rather
than a single ratio that would look like a fixed constant. What it reports is bytes written
to the checkpoint's state directory, counted at the store's own write boundary, plus the
wall time of the run — because the flush sits on the critical path of every epoch, so the
bytes are latency, not just disk.

There is no cross-engine bar here on purpose. Spark's equivalent is RocksDB changelog
checkpointing, which is a different storage engine rather than a different policy over the
same one, so a number against it would compare two stacks and attribute the difference to
this change.

Run:
    python benchmarks/scenarios/streaming/state_checkpoint.py
    python benchmarks/scenarios/streaming/state_checkpoint.py --epochs 100 200 400 800
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import tempfile
import time

import pyarrow as _pa

import batcher as bt
from batcher.config import option_context
from batcher.io.formats.streaming.checkpoint import state_store

#: Rows per micro-batch. One group per row, so the state grows by this much every epoch —
#: the high-cardinality shape a changelog exists for.
ROWS_PER_BATCH = 8


class _ByteCounter:
    """Counts what the state store writes, at the store's own boundary.

    Wrapping `_write` rather than stat-ing the directory afterwards is deliberate: the store
    prunes superseded files as it goes, so a directory listing at the end measures what
    *survived*, not what was written — and what was written is the cost.
    """

    def __init__(self) -> None:
        self.written = 0
        self._original = state_store.StateStore._write

    def __enter__(self) -> _ByteCounter:
        counter = self

        def counting(store, name, state):
            counter.written += len(state_store._serialize(state))
            return counter._original(store, name, state)

        state_store.StateStore._write = counting
        return self

    def __exit__(self, *_exc: object) -> None:
        state_store.StateStore._write = self._original


def _run(checkpoint: str, epochs: int, interval: int) -> tuple[int, float]:
    """Run a high-cardinality running aggregate; return bytes written and wall seconds."""
    rows = epochs * ROWS_PER_BATCH
    with _ByteCounter() as counter, option_context("streaming.checkpoint_delta_interval", interval):
        started = time.perf_counter()
        query = (
            bt.read.rate(ROWS_PER_BATCH, num_rows=rows, pace=False)
            .with_columns(bucket=bt.col("value"))
            .group_by("bucket")
            .agg(total=bt.col("value").sum())
            .write.for_each_batch(
                lambda _table, _batch_id: None,
                trigger=bt.Trigger.available_now(),
                checkpoint=checkpoint,
                output_mode="update",
            )
        )
        query.await_termination()
        elapsed = time.perf_counter() - started
    return counter.written, elapsed


def _windowed(checkpoint: str, epochs: int, interval: int, keys: int = 200) -> int:
    """A watermarked windowed aggregate under a wide lateness; return bytes written.

    The other operator that can use a changelog, and for a different reason. This one
    *removes* state — but eviction drops a prefix of the window axis, so the whole tombstone
    is one bound in the entry's metadata. A generous `allowed_lateness` is what makes the
    open set big enough for the difference to show.
    """
    base = _dt.datetime(2024, 1, 1)
    schema = _pa.schema([("ts", _pa.timestamp("us")), ("k", _pa.string()), ("v", _pa.int64())])

    def batches():
        for step in range(epochs):
            yield _pa.record_batch(
                {
                    "ts": _pa.array(
                        [base + _dt.timedelta(seconds=step * 2)] * keys,
                        type=_pa.timestamp("us"),
                    ),
                    "k": _pa.array([f"key-{i:05d}" for i in range(keys)]),
                    "v": _pa.array([1] * keys, type=_pa.int64()),
                },
                schema=schema,
            )

    with _ByteCounter() as counter, option_context("streaming.checkpoint_delta_interval", interval):
        query = (
            bt.from_batches(batches, schema, bounded=False)
            .with_watermark("ts", "300 seconds")
            .group_by(w=bt.window(bt.col("ts"), "10 seconds"), k=bt.col("k"))
            .agg(total=bt.col("v").sum())
            .write.for_each_batch(
                lambda _table, _batch_id: None,
                trigger=bt.Trigger.available_now(),
                checkpoint=checkpoint,
            )
        )
        query.await_termination()
    return counter.written


def main() -> None:
    """Measure both checkpoint policies across a range of run lengths and print a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400],
        help="micro-batch counts to measure (state grows by 8 groups per epoch)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="streaming.checkpoint_delta_interval for the incremental run",
    )
    args = parser.parse_args()

    work = tempfile.mkdtemp(prefix="bt-ckpt-bench-")
    try:
        print("running aggregate (no watermark — state only grows, nothing evicts)")
        print(
            f"{'epochs':>7} {'whole-snapshot':>16} {'changelog':>14} {'reduction':>10} {'wall':>14}"
        )
        for epochs in args.epochs:
            whole, whole_s = _run(os.path.join(work, f"whole-{epochs}"), epochs, 0)
            delta, delta_s = _run(os.path.join(work, f"delta-{epochs}"), epochs, args.interval)
            print(
                f"{epochs:>7} {whole:>15,}B {delta:>13,}B "
                f"{whole / max(delta, 1):>9.1f}x {whole_s:>6.2f}s→{delta_s:>5.2f}s"
            )
        print()
        print("windowed aggregate (300s lateness — evicts, and the bound is one integer)")
        print(f"{'epochs':>7} {'whole-snapshot':>16} {'changelog':>14} {'reduction':>10}")
        for epochs in args.epochs:
            whole = _windowed(os.path.join(work, f"w-whole-{epochs}"), epochs, 0)
            delta = _windowed(os.path.join(work, f"w-delta-{epochs}"), epochs, args.interval)
            print(f"{epochs:>7} {whole:>15,}B {delta:>13,}B {whole / max(delta, 1):>9.1f}x")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
