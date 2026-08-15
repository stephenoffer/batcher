"""Writing a training corpus: streaming, crash-safe, and resumable.

Three properties, each of which a long write needs and none of which it had.

**Streaming** — the input is consumed a batch at a time and at most one shard's rows are
held, so writing a corpus larger than memory does not first need a `pa.Table` larger than
memory. This used to build one table of the *entire* corpus and slice it, which made the
sharded format require exactly the condition it exists to remove.

**Crash-safe** — the manifest is published as the write proceeds, not once at the end. A
write that died after four thousand of five thousand shards left four thousand perfectly good
shards on disk and no ``index.json``, so nothing could read them and the only move was to
start over. Now the corpus on disk is readable at every moment.

**Resumable** — `resume=True` picks up from the manifest instead of rewriting what is
already there. The rows already written are skipped from the input rather than re-encoded,
which is what makes restarting a ten-hour write cost minutes.

Shards are published **concurrently**, which matters for the destination this format is for.
Writing one shard at a time is right on local disk, where a write is bandwidth-bound; against
an object store each shard is a round trip of tens of milliseconds, so a five-thousand-shard
corpus spent minutes purely waiting, with the encoder idle throughout. The number in flight is
sized by `batcher.io.base.sink.stream_part_concurrency` — the same decision `FileSink` makes,
not a second copy of it — and bounds resident memory to that many shards.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import pyarrow as pa

from batcher.io.base._transient import with_retry
from batcher.io.base.sink import stream_part_concurrency
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.ml.shards.index import (
    INDEX_NAME,
    ShardIndex,
    read_shard_index,
    shard_name,
    write_index,
)
from batcher.plan.types import retained_bytes

__all__ = ["write_shards"]

#: Shards written between manifest publications. The manifest of a uniform corpus is a few
#: hundred bytes whatever the corpus size, so this is a bound on *recovery loss*, not on
#: bytes: a crash costs at most this many shards of re-done work. It is not 1 only because
#: on an object store each publication is a round trip, and paying one per shard doubles the
#: request count of a write whose shards are small.
_INDEX_FLUSH_SHARDS = 16

#: Retry budget for publishing one shard. A multi-hour write against object storage will meet
#: a throttle; failing the whole corpus over one is the difference between a job that
#: finishes and a job that is restarted by hand.
_WRITE_ATTEMPTS = 4
_WRITE_BACKOFF_S = 0.5


def _iter_input_batches(batches: Iterable[pa.RecordBatch] | pa.Table) -> Iterator[pa.RecordBatch]:
    """Normalize the writer's input to a stream of `RecordBatch`.

    A `pa.Table` is walked batch by batch rather than kept whole, so the two input spellings
    take the same bounded-memory path through the writer.
    """
    if isinstance(batches, pa.Table):
        yield from batches.to_batches()
        return
    for item in batches:
        if isinstance(item, pa.Table):  # tolerate a stream of tables
            yield from item.to_batches()
        else:
            yield item


def _skip_rows(batches: Iterator[pa.RecordBatch], count: int) -> Iterator[pa.RecordBatch]:
    """Drop the first `count` rows of a batch stream, slicing the batch they end inside.

    The resume path. The rows are skipped rather than re-encoded, so restarting a write that
    died near the end costs a pass over the source rather than a rewrite of the corpus.
    """
    remaining = count
    for batch in batches:
        if remaining <= 0:
            yield batch
        elif batch.num_rows <= remaining:
            remaining -= batch.num_rows
        else:
            yield batch.slice(remaining)
            remaining = 0


class _ShardPacker:
    """Repack an arbitrary batch stream into exactly `rows_per_shard`-row shards.

    Holds at most one shard's rows, which is what makes `write_shards` usable on a corpus
    larger than memory.
    """

    __slots__ = ("_buffer", "_rows", "rows_per_shard")

    def __init__(self, rows_per_shard: int) -> None:
        self.rows_per_shard = rows_per_shard
        self._buffer: list[pa.RecordBatch] = []
        self._rows = 0

    def push(self, batch: pa.RecordBatch) -> Iterator[pa.Table]:
        """Absorb one input batch, yielding every full shard it completes."""
        if batch.num_rows == 0:
            return
        self._buffer.append(batch)
        self._rows += batch.num_rows
        while self._rows >= self.rows_per_shard:
            table = pa.Table.from_batches(self._buffer)
            yield table.slice(0, self.rows_per_shard).combine_chunks()
            remainder = table.slice(self.rows_per_shard)
            self._buffer = remainder.to_batches()
            self._rows = remainder.num_rows

    def flush(self) -> Iterator[pa.Table]:
        """Yield the trailing partial shard, if any."""
        if self._rows:
            yield pa.Table.from_batches(self._buffer).combine_chunks()
            self._buffer, self._rows = [], 0


def _resume_point(directory: str, rows_per_shard: int, resume: bool) -> tuple[int, int]:
    """``(complete shards, rows already written)`` to continue from, or ``(0, 0)``.

    Only **whole** shards count as done. A partial final shard from the previous attempt is
    rewritten, because the manifest is published every `_INDEX_FLUSH_SHARDS` shards and the
    shards written since the last publication are not described by it — trusting a file the
    manifest does not describe is how a corpus ends up with a shard of the wrong width in the
    middle, which breaks global indexing silently.
    """
    if not resume:
        return 0, 0
    from batcher._internal.errors import FormatError

    fs = resolve_filesystem(directory)
    if not fs.exists(f"{directory}/{INDEX_NAME}"):
        return 0, 0  # nothing to resume from; a fresh write
    existing = read_shard_index(directory)
    if existing.rows_per_shard != rows_per_shard:
        raise FormatError(
            f"cannot resume {directory!r}: it was written with "
            f"rows_per_shard={existing.rows_per_shard}, not {rows_per_shard}. Shard width "
            "has to match, or a global row index means two different things in one corpus."
        )
    complete = existing.total_rows // rows_per_shard
    return complete, complete * rows_per_shard


def write_shards(
    batches: Iterable[pa.RecordBatch] | pa.Table,
    directory: str,
    *,
    rows_per_shard: int = 65_536,
    resume: bool = False,
    write_concurrency: int | None = None,
) -> ShardIndex:
    """Write `batches` as equal-size Arrow-IPC shards + an index under `directory`.

    Rows are repacked to exactly `rows_per_shard` per shard (the last is the remainder), so
    shard boundaries are independent of the input batching — a prerequisite for deterministic
    global indexing. The input is consumed as a **stream**: at most one shard's rows are
    resident, so a corpus larger than memory writes in bounded memory.

    The manifest is republished as the write proceeds, so the corpus on disk is readable at
    every moment rather than only once the last shard lands, and a crashed write can be
    continued with `resume`.

    Args:
        batches: The rows to write, as a `pyarrow.Table` or an iterable of
            `pyarrow.RecordBatch` (for example ``ds.iter_batches()``).
        directory: Where to write the shards and their ``index.json``.
        rows_per_shard: Rows in every shard but the last.
        resume: Continue a previous write instead of starting over. The already-written rows
            are skipped from the input, so this requires `batches` to reproduce the same rows
            in the same order — the same determinism a shuffled global order already assumes.
        write_concurrency: Shards to publish at once. Defaults to a size derived from the
            destination and the measured shard size: cores for local disk, requests in flight
            for an object store, narrowed so the shards held in memory stay bounded.

    Returns:
        The `ShardIndex` describing the whole corpus.

    Raises:
        ValueError: If `rows_per_shard` is below 1, or the input batches disagree on their
            schema (which would make the shards unreadable as one dataset).
        FormatError: If `resume` is set and the existing corpus used a different shard width.
    """
    if rows_per_shard < 1:
        raise ValueError(f"rows_per_shard must be >= 1, got {rows_per_shard}")
    import pyarrow.ipc as ipc

    fs = resolve_filesystem(directory)
    fs.mkdirs(directory, exist_ok=True)
    written_shards, written_rows = _resume_point(directory, rows_per_shard, resume)

    schema: pa.Schema | None = None
    state = {"shards": written_shards, "rows": written_rows, "published": written_shards}

    def _publish() -> ShardIndex:
        index = ShardIndex(
            rows_per_shard=rows_per_shard,
            total_rows=state["rows"],
            shard_count=state["shards"],
            directory=directory,
            schema=schema,
        )
        write_index(directory, index, filesystem=fs)
        state["published"] = state["shards"]
        return index

    packer = _ShardPacker(rows_per_shard)
    # (shard index, future, rows) in submission order. Retired from the front, so the count
    # the manifest publishes is always a *contiguous* prefix of the corpus — a manifest that
    # skipped a shard still in flight would describe rows that are not there yet.
    pending: list[tuple[int, Future, int]] = []
    limit = 1

    def _submit(pool: ThreadPoolExecutor, chunk: pa.Table, shard: int) -> None:
        path = f"{directory}/{shard_name(shard)}"

        def _put() -> None:
            with fs.atomic_writer(path) as fh:
                writer = ipc.new_file(fh, chunk.schema)
                writer.write_table(chunk)
                writer.close()

        # Safe to repeat: the write is atomic, so a failed attempt published nothing and the
        # chunk is still in memory. A throttle mid-corpus is a blip, not a lost corpus.
        pending.append(
            (
                shard,
                pool.submit(
                    with_retry, _put, attempts=_WRITE_ATTEMPTS, backoff_base_s=_WRITE_BACKOFF_S
                ),
                chunk.num_rows,
            )
        )

    def _retire(*, block: bool) -> None:
        """Account for finished shards from the front, optionally waiting for the first."""
        while pending and (block or pending[0][1].done()):
            shard, future, rows = pending[0]
            future.result()  # re-raises whatever the write failed with
            pending.pop(0)
            state["shards"] = shard + 1
            state["rows"] += rows
            block = False
            if state["shards"] - state["published"] >= _INDEX_FLUSH_SHARDS:
                _publish()

    next_shard = written_shards
    pool: ThreadPoolExecutor | None = None

    def _ensure_pool(chunk: pa.Table) -> ThreadPoolExecutor:
        """The write pool, sized on first use from the first shard's real cost.

        Sized from the shard rather than from a constant because a shard can be a few KB or
        several GB depending on `rows_per_shard` and the row width, and the shards in flight
        are held live — a fixed count would make resident memory a function of the row width.
        """
        nonlocal pool, limit
        if pool is None:
            limit = write_concurrency or stream_part_concurrency(retained_bytes(chunk), directory)
            pool = ThreadPoolExecutor(max_workers=limit)
        return pool

    try:
        for batch in _skip_rows(_iter_input_batches(batches), written_rows):
            if schema is None:
                schema = batch.schema
            elif not batch.schema.equals(schema):
                # Shards written from disagreeing schemas cannot be read back as one
                # dataset; the failure would otherwise surface as an unrelated concat error
                # at train time.
                raise ValueError(
                    "write_shards: input batches have differing schemas; "
                    f"expected {schema} but got {batch.schema}"
                )
            for chunk in packer.push(batch):
                _submit(_ensure_pool(chunk), chunk, next_shard)
                next_shard += 1
                # Backpressure. Without it the packer runs ahead of the writers and the whole
                # corpus queues up in memory — the regression the streaming rewrite removed.
                _retire(block=len(pending) > limit)
        for chunk in packer.flush():
            _submit(_ensure_pool(chunk), chunk, next_shard)
            next_shard += 1
        _retire(block=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        # Account for whatever landed even when the write is failing, so a crash still
        # leaves a readable corpus. Errors here are suppressed: one shard's failure is
        # already being raised, and a second must not replace it.
        with contextlib.suppress(Exception):
            _retire(block=False)
        if schema is None and written_shards:
            # A resume that had nothing left to add: keep the schema the corpus records.
            schema = read_shard_index(directory).schema
        published = _publish()
    return published
