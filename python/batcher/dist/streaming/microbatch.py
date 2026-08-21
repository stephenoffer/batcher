"""A streaming micro-batch, run across the cluster — one epoch, one transaction.

This is the distributed half of `core.streaming_runner.MicroBatchRunner`. The engine's
loop, triggers, offset log and recovery are unchanged and stay on the driver; only *where
the epoch's rows are touched* changes. The driver never sees one.

**Stage.** The epoch's work units come from the source's own `splits()` — the new files an
Auto Loader pass discovered, the partitions of a broker. Each goes to a worker, which reads
it, runs the (already Kyber-optimized) plan in Rust, writes its output as **data files it
does not commit**, and returns locators. Metadata comes back; data does not.

**Publish.** The driver takes every worker's locators and makes **one** commit for the
epoch, carrying the Delta ``txn`` action ``(app_id, batch_id)``. This is the difference
between a stream whose log reads as one transaction per micro-batch and one that records
a commit per *worker* — a log that no longer describes the stream that produced it. It is
also what makes a replay free: a re-run epoch finds its own transaction already in the log,
writes nothing, and commits nothing, so the rows land exactly once no matter how many times
a lost worker forced the epoch to be retried.

A stateful epoch (a streaming aggregation) fans out the same way, but each worker returns a
*partial aggregate* instead of files — bounded by the group count, not the input size — and
the driver `combine`s them into the running state. That is the mergeable algebra the engine
is built on (`partial → combine → finalize`), so a distributed streaming aggregate is the
same operator as the single-node one, not a second implementation of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.native import engine
from batcher.core.mergeable import RunningAggregate
from batcher.io.manifest import WriteManifest, WrittenFile

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = ["DistributedRunner"]

#: How long a source's work units stay cached before they are enumerated again.
#:
#: `splits()` costs a round trip for most unbounded sources — a Kafka `list_topics`, a Kinesis
#: `list_shards`, an Event Hubs management call — and the idle loop below asks five times a
#: second, so a stream with nothing to read spent its whole trigger budget re-asking the
#: broker how many partitions it has. Partitions change when a topic is rescaled, which is
#: rare enough to notice within this window and not rare enough to cache forever.
_SPLIT_REFRESH_SECONDS = 30.0


@dataclass(slots=True)
class _StagedEpoch:
    """One epoch's worker output — locators and counts, never rows (except an aggregate's
    partial state, which is bounded by the group count)."""

    files: list[WrittenFile] = field(default_factory=list)
    partials: list[pa.RecordBatch] = field(default_factory=list)
    schema: pa.Schema | None = None
    input_rows: int = 0
    output_rows: int = 0
    consumed: list[str] = field(default_factory=list)
    offsets: dict[str, Any] = field(default_factory=dict)


class DistributedRunner:
    """Run each micro-batch as a cluster-wide epoch: fan out to stage, commit once."""

    __slots__ = (
        "_agg",
        "_cfg",
        "_drain",
        "_fmt",
        "_fold",
        "_offsets",
        "_partition_by",
        "_path",
        "_plan_ir",
        "_projection",
        "_query_name",
        "_should_stop",
        "_sink_kwargs",
        "_sink_writer",
        "_source",
        "_spent",
        "_split_cache",
        "_workers",
    )

    def __init__(
        self,
        *,
        plan_ir: str,
        projection: list[str] | None,
        source: Source,
        path: str,
        fmt: str,
        sink_kwargs: dict[str, Any] | None,
        query_name: str,
        num_workers: int,
        drain: bool,
        should_stop: Callable[[], bool],
        agg: Any | None = None,
    ) -> None:
        from batcher.dist.executors.ray_runtime import engine_config_json

        self._source = source
        self._path = path.rstrip("/")
        self._fmt = fmt
        self._sink_kwargs = dict(sink_kwargs or {})
        self._partition_by = self._sink_kwargs.pop("partition_by", None)
        self._query_name = query_name
        self._workers = max(1, num_workers)
        self._drain = drain
        self._should_stop = should_stop
        self._cfg = engine_config_json()  # the driver's config, shipped to every worker

        # The conductor already chose what the workers run (the optimized plan, or an
        # aggregate's input pipeline) — `dist` schedules it, it does not re-decide it.
        self._agg = agg
        self._fold = RunningAggregate(agg) if agg is not None else None
        self._plan_ir = plan_ir
        self._projection = projection

        self._offsets: dict[str, Any] = {}
        self._sink_writer = _sink_writer(fmt)
        self._spent = False
        # `(monotonic_deadline, splits)` for a source whose `splits()` is a pure question.
        # See `_splits`.
        self._split_cache: tuple[float, list[Any]] | None = None

    # --- MicroBatchRunner -------------------------------------------------
    def stage(self, batch_id: int) -> _StagedEpoch | None:
        """Fan this epoch across the cluster; None when the source is spent (or stopped).

        An *idle* pass of an unbounded source is not a finished one: a continuous stream
        waits and looks again rather than ending the query, and only a drain trigger
        (`available_now`/`once`) reads the same empty pass as the end — which is what makes
        it a drain. Waiting here, instead of returning an empty epoch, is also what keeps
        an idle stream from inflating the batch counter and the offset log with epochs that
        carry no rows.
        """
        import time

        from batcher.config import active_config
        from batcher.io.source import is_bounded

        # One knob, both runners: an idle stream must wait the same amount on one machine
        # and on a cluster, or the two disagree about how a quiet source behaves.
        idle_poll_seconds = active_config().streaming.idle_poll_seconds
        while not self._spent:
            splits = self._splits()
            epoch = self._fan_out(batch_id, splits) if splits else _StagedEpoch()
            if epoch.input_rows:
                # A bounded source hands back the *same* splits on every pass, so its
                # entire content is this one epoch; asking again would re-read it.
                self._spent = is_bounded(self._source)
                return epoch
            if self._drain or self._should_stop() or is_bounded(self._source):
                return None
            time.sleep(idle_poll_seconds)
        return None

    def _splits(self) -> list[Any]:
        """This epoch's work units, re-enumerated only when that is not merely a question.

        A source whose `splits()` *claims* the work it returns — the incremental file source,
        whose files become pending the moment they are handed out — must be asked every pass,
        because asking **is** the discovery. Everything else is answering a question about
        shape, and for an unbounded source that question costs a round trip to the broker.
        The idle loop asks five times a second, so a stream with nothing to read spent its
        whole trigger budget re-asking Kafka how many partitions its topic has.
        """
        import time

        if getattr(self._source, "confirm", None) is not None:
            return list(self._source.splits())
        now = time.monotonic()
        if self._split_cache is not None and now < self._split_cache[0]:
            return self._split_cache[1]
        splits = list(self._source.splits())
        self._split_cache = (now + _SPLIT_REFRESH_SECONDS, splits)
        return splits

    def positions(self) -> dict[int, dict]:
        from batcher.io.source import is_checkpointable

        if self._offsets:
            return {0: {"offsets": dict(self._offsets)}}
        if is_checkpointable(self._source):
            return {0: self._source.snapshot_position()}
        return {}

    def publish(self, batch_id: int, staged: _StagedEpoch) -> tuple[int, int]:
        """Make the epoch visible: one commit for the whole cluster's output."""
        if self._fold is not None:
            emitted = self._publish_aggregate(batch_id, staged)
        else:
            emitted = self._publish_files(batch_id, staged)
        confirm = getattr(self._source, "confirm", None)
        if confirm is not None:
            confirm()  # the epoch is durable — its files may now be forgotten
        return staged.input_rows, emitted

    def seek(self, position: dict) -> None:
        from batcher.io.source import is_checkpointable

        self._offsets = dict(position.get("offsets", {}))
        if not self._offsets and is_checkpointable(self._source):
            self._source.seek(position)

    def snapshot_state(self) -> pa.RecordBatch | None:
        return self._fold.state() if self._fold is not None else None

    def restore_state(self, state: pa.RecordBatch) -> None:
        if self._fold is not None:
            self._fold.restore(state)

    def restore_state_chain(self, states: list[pa.RecordBatch]) -> None:
        """Rebuild the running state from a base snapshot plus the changelog after it.

        This runner does not *write* deltas — a state backend is not something to change on
        a path with no automated coverage — but it must be able to *read* one, because a
        checkpoint is portable across runners and a single-node query restarted with
        `distributed=True` resumes from exactly the chain the single-node run left behind.
        Without this the engine would fall back to restoring the base alone and the query
        would resume with a fraction of its aggregate: silently short totals, no error, and
        only on the restart that changed configuration.

        `combine_all` is the same algebra this runner already applies to worker partials on
        every epoch, so replaying a chain is not a new code path — it is `_publish_aggregate`
        with the partials coming off disk instead of off the wire.

        Args:
            states: The base snapshot followed by each recorded delta, oldest first.
        """
        if self._fold is not None:
            self._fold.combine_all(list(states))

    def has_state(self) -> bool:
        return self._fold is not None

    def finalize(self) -> list[pa.RecordBatch]:
        return []  # the running result is emitted every epoch; nothing is left open

    def emit_final(
        self,
        batch_id: int,  # noqa: ARG002 (protocol signature; this path never runs)
        rows: pa.RecordBatch,  # noqa: ARG002 (same)
    ) -> None:  # pragma: no cover
        raise AssertionError("distributed streaming emits its result each epoch")

    # --- staging ----------------------------------------------------------
    def _fan_out(self, batch_id: int, splits: list[Any]) -> _StagedEpoch:
        from batcher.dist.executors.ray_runtime import _ensure_ray, gather_map_results

        workers = min(self._workers, len(splits))
        _ensure_ray(workers)
        assignments = _balance(splits, workers)

        results = gather_map_results(
            lambda idx: _stage_shard.remote(
                assignments[idx],
                self._plan_ir,
                self._projection,
                self._offsets,
                batch_id,
                idx,
                self._fmt,
                self._sink_kwargs,
                self._path,
                self._partition_by,
                self._cfg,
                self._agg_spec(),
                self._sink_writer,
            ),
            len(assignments),
        )
        return self._merge(splits, [r for r in results if r is not None])

    def _agg_spec(self) -> tuple[str, str] | None:
        """The aggregate's ``(group_keys, aggregates)`` IR, or None for a stateless epoch."""
        return self._fold.spec if self._fold is not None else None

    def _merge(self, splits: list[Any], results: list[dict]) -> _StagedEpoch:
        """Combine the workers' metadata into one epoch (a commutative merge)."""
        epoch = _StagedEpoch()
        for r in results:
            epoch.files.extend(r["files"])
            if r["partial"] is not None:
                epoch.partials.append(r["partial"])
            epoch.input_rows += r["input_rows"]
            epoch.output_rows += r["output_rows"]
            epoch.offsets.update(r["offsets"])
            if epoch.schema is None and r["schema"] is not None:
                epoch.schema = r["schema"]
        if epoch.offsets:
            self._offsets.update(epoch.offsets)
        # File-locator sources (an Auto Loader pass) carry their position as the set of
        # files the epoch read; the driver knows them because it listed them.
        epoch.consumed = [getattr(s, "path", "") for s in splits if getattr(s, "path", None)]
        complete = getattr(self._source, "complete", None)
        if complete is not None and epoch.consumed:
            complete(epoch.consumed)
        return epoch

    # --- publishing -------------------------------------------------------
    def _publish_files(self, batch_id: int, staged: _StagedEpoch) -> int:
        """One transaction for the epoch's data files (or nothing, if it had no rows)."""
        if not staged.files:
            return 0
        if self._sink_writer != "transactional":
            return staged.output_rows  # workers wrote final files; there is nothing to commit
        from batcher.io.formats import SINKS

        sink = SINKS.get(self._fmt)(
            app_id=self._query_name,
            txn_version=batch_id,
            partition_by=self._partition_by,
            **self._sink_kwargs,
        )
        if sink.is_committed(self._path):
            return 0  # a replayed epoch: its rows are already in the table
        sink.commit(WriteManifest(tuple(staged.files), schema=staged.schema), self._path)
        return staged.output_rows

    def _publish_aggregate(self, batch_id: int, staged: _StagedEpoch) -> int:
        """Combine the workers' partial states and emit the running result."""
        for partial in staged.partials:
            self._fold.absorb(partial)
        result = self._fold.finalize()
        if result is None or not result.num_rows:
            return 0
        from batcher.io.formats.streaming.sinks import FileStreamSink, TransactionalStreamSink

        sink: Any = (
            TransactionalStreamSink(
                self._path, self._fmt, query_name=self._query_name, **self._sink_kwargs
            )
            if self._sink_writer == "transactional"
            else FileStreamSink(self._path, self._fmt, **self._sink_kwargs)
        )
        sink.write_batch(batch_id, pa.Table.from_batches([result]))
        return result.num_rows


def _sink_writer(fmt: str) -> str:
    """Whether `fmt` commits its files in a transaction, or the files *are* the write."""
    from batcher.io.formats import SINKS

    cls = SINKS.get(fmt)
    return "transactional" if hasattr(cls, "is_committed") else "files"


def _balance(splits: list[Any], workers: int) -> list[list[Any]]:
    """Deal the epoch's splits round-robin so every worker gets a contiguous share."""
    buckets: list[list[Any]] = [[] for _ in range(workers)]
    for i, split in enumerate(splits):
        buckets[i % workers].append(split)
    return [b for b in buckets if b]


def _stage_shard(
    splits: list[Any],
    plan_ir: str,
    projection: list[str] | None,
    offsets: dict[str, Any],
    batch_id: int,
    idx: int,
    fmt: str,
    sink_kwargs: dict[str, Any],
    path: str,
    partition_by: list[str] | None,
    engine_config: str,
    agg_spec: tuple[str, str] | None,
    sink_writer: str,
) -> dict:
    """One worker's share of an epoch: read it, run the plan, write files — never commit.

    Returns only metadata (file locators, row counts, the new source offsets), so the
    epoch's rows never travel to the driver. A stateful epoch returns its partial aggregate
    instead of files, which is bounded by the group count.
    """
    from batcher.io.formats import SINKS

    nat = engine()
    batches, new_offsets = _read_epoch(splits, projection, offsets)
    input_rows = sum(b.num_rows for b in batches)
    empty: dict[str, Any] = {
        "files": [],
        "partial": None,
        "schema": None,
        "input_rows": input_rows,
        "output_rows": 0,
        "offsets": new_offsets,
    }
    if not batches:
        return empty

    out = nat.execute_plan(plan_ir, [batches], engine_config)
    if not out or sum(b.num_rows for b in out) == 0:
        return empty

    if agg_spec is not None:
        keys, aggs = agg_spec
        partial = nat.partial_aggregate(keys, aggs, out)
        return {**empty, "partial": partial, "output_rows": partial.num_rows}

    table = pa.Table.from_batches(out)
    sink = SINKS.get(fmt)(**sink_kwargs)
    if sink_writer == "transactional":
        # A fresh sink per shard carries its own file token, so two workers — and two
        # epochs — never write the same data-file name. The commit happens on the driver.
        files = sink.write_partitioned(table, path, partition_by=partition_by, file_index=idx)
    else:
        # A plain file sink has no commit, so the file name *is* the idempotency: naming it
        # for the epoch and the shard means a replayed epoch overwrites its own output
        # instead of appending a second copy of it.
        name = f"{path}/part-batch{batch_id:05d}-{idx:05d}{sink.suffix}"
        files = [sink.write(table, name, resume=False)]
    return {
        "files": files,
        "partial": None,
        "schema": table.schema,
        "input_rows": input_rows,
        "output_rows": table.num_rows,
        "offsets": new_offsets,
    }


def _read_epoch(
    splits: list[Any], projection: list[str] | None, offsets: dict[str, Any]
) -> tuple[list[pa.RecordBatch], dict[str, Any]]:
    """Read this shard's splits, returning its batches and the positions it advanced to.

    A split that carries its own bounds (a file) simply reads; one that is a cursor into an
    unbounded partition (a broker) is resumed from the driver-supplied offset and read for
    exactly one micro-batch, reporting where it stopped.
    """
    batches: list[pa.RecordBatch] = []
    new_offsets: dict[str, Any] = {}
    for split in splits:
        read_epoch = getattr(split, "read_epoch", None)
        if read_epoch is None:
            batches.extend(split.read(projection))
            continue
        key = str(getattr(split, "partition", ""))
        rows, position = read_epoch(offsets.get(key), projection)
        batches.extend(rows)
        if position is not None:
            new_offsets[key] = position
    return batches, new_offsets
