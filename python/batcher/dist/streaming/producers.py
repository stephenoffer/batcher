"""The CPU producer stage of the streaming CPU-to-GPU pipeline.

The twin of `consumers`: `pipeline` owns the overlap loop, this module owns the actor that
streams one input partition through the stateless-CPU sub-plan and publishes its output on
a node-local Flight server, one morsel at a time, for a GPU consumer to fetch in place.

Split out of `pipeline` for size; the overlap loop, the credit windows, and the recovery
policy stay there.
"""

from __future__ import annotations

from collections import deque

from batcher.plan.logical import LogicalPlan
from batcher.plan.types import one_batch

__all__ = ["ProducerActor", "coalesce", "consumer_batch_rows"]


def coalesce(batches: list, target_rows: int) -> list:
    """Regroup `batches` to about `target_rows` rows each — merging short ones, splitting long.

    The eager counterpart of `ProducerActor._take_batch`, for a stage whose whole output is
    already in hand. Both exist for one reason: a published morsel is one model call on the
    stage above, so a morsel below that stage's declared batch size is a small forward pass no
    downstream re-batching can undo. The producer has to gather *lazily*, pulling more input
    only when it is short, or it would defeat the streaming it exists to do; a relay has run
    its whole morsel already and has nothing left to pull.

    **Splitting matters as much as merging, and this did only the latter**: it regrouped to
    "at least `target_rows`", so a batch already over the target was published whole. A scan
    morsel is sized in *rows*, so on the workload this pipeline exists for — decode a batch of
    images, score them on a device — one upstream morsel of 16,384 rows decodes to about
    **2.4 GB**, and the relay published it as a single morsel. That is one 16,384-image
    forward pass, a fetch the consumer cannot break up, and a resident block far past the
    256 MiB per-channel credit budget. Measured on 400,000 384x384 JPEGs over 8 T4s: the
    staged pipeline died with `3 worker(s) were killed due to the node running low on memory`
    at 28.53 GB of 30 GB. Splitting to the size the consumer declared is what makes the credit
    window mean anything for wide rows.

    Slices are zero-copy and the remainder carries forward, so no row is copied, dropped or
    reordered, and the concatenation of the result is the concatenation of the input.

    Args:
        batches: The stage's output batches, in order.
        target_rows: Rows per morsel; `0` leaves the batches as they are.

    Returns:
        The regrouped batches, each `target_rows` rows except the last. A short final morsel
        is correct — there are no more rows.
    """
    import pyarrow as pa

    if target_rows <= 0:
        return list(batches)
    out: list = []
    held: list = []
    rows = 0
    for batch in batches:
        if not held and batch.num_rows >= target_rows:
            # The common case on a wide stage, and the one worth not concatenating for: a
            # single oversized batch is sliced where it lies.
            offset = 0
            while batch.num_rows - offset >= target_rows:
                out.append(batch.slice(offset, target_rows))
                offset += target_rows
            rest = batch.slice(offset)
            if rest.num_rows:
                held, rows = [rest], rest.num_rows
            continue
        held.append(batch)
        rows += batch.num_rows
        if rows >= target_rows:
            # `concat_batches` rather than `combine_chunks`: the latter splits at the 32-bit
            # offset limit, so a morsel holding more than 2 GiB of string or binary data comes
            # back as several batches — see `_take_batch`, which learned this the hard way.
            merged = held[0] if len(held) == 1 else pa.concat_batches(held)
            offset = 0
            while merged.num_rows - offset >= target_rows:
                out.append(merged.slice(offset, target_rows))
                offset += target_rows
            rest = merged.slice(offset)
            held, rows = ([rest], rest.num_rows) if rest.num_rows else ([], 0)
    if held:
        out.append(held[0] if len(held) == 1 else pa.concat_batches(held))
    return out


def consumer_batch_rows(sub_plan: LogicalPlan) -> int:
    """Rows the consumer stage wants per model call, or 0 when it declares none.

    The first `map_batches` in the stage owns the hand-off size — it is the one the produced
    morsel feeds directly. A stage that declared no `batch_size` gets 0, which leaves the
    producer publishing at the engine's own morsel granularity, exactly as before.
    """
    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import walk

    for node in walk(sub_plan):
        if isinstance(node, MapBatches) and getattr(node, "batch_size", None):
            return int(node.batch_size)
    return 0


try:
    import ray

    @ray.remote
    class ProducerActor:
        """A CPU producer stage: streams a partition through its sub-plan and publishes
        each output morsel on its node-local Flight server for the consumer to fetch.

        The model/decoder (a class UDF) builds once here (`_prebuild_factories`), so a
        load-once preprocess stage reuses it across partitions. The partition is
        consumed one input batch at a time (`iter_partition_descriptor`) and each input
        batch's mapped output is buffered and published morsel by morsel, so the
        producer never materializes the whole partition — only one input chunk's output
        plus the published-but-unreleased window. Only `(addr, ticket)` ever crosses
        Ray; the batches move over credit-bounded Flight.
        """

        def __init__(
            self,
            plan0: LogicalPlan,
            credits: int,
            target_rows: int = 0,
            cpu_workers: int | None = None,
        ) -> None:
            from batcher.carbonite.transfer import ShuffleSession
            from batcher.dist.executors.map import _prebuild_factories, _with_inference_workers

            # Threads this actor runs its sub-plan with. Left at the plan's own width, a
            # producer ran the stage on **one** thread: `execute_with_udfs` is called once per
            # published morsel, so a pool of N producers was an N-thread decode however many
            # cores the fleet had. Measured on 400,000 images over a 208-core / 17-node
            # cluster, 32 producers: 11-17% cluster CPU, and the staged form lost to the fused
            # one it exists to beat. The caller sizes this from the cores the pool actually
            # holds (`_producer_cpu_width`), which is the same quantity `_agg_actor_width`
            # gives a CPU map/aggregate pool and for the same reason.
            self._plan = _with_inference_workers(_prebuild_factories(plan0), cpu_workers)
            # Rows to gather into one published morsel. A morsel is one *model call* on the
            # consumer, so publishing at the engine's own morsel granularity hands the GPU
            # whatever the scan happened to emit — for a wide row (a 150 KB image) the
            # byte-sized morsel is a handful of rows, and a device that wants a batch of 128
            # gets forward passes of six. The consumer stage already declares the batch it
            # wants; this is where it has to be honored, because past this point the rows are
            # a Flight ticket the consumer cannot re-group across.
            self._target_rows = max(0, int(target_rows))
            # Advertise the node's routable IP so a consumer on another host can dial
            # this server (loopback would be unreachable cross-node).
            host = ray.util.get_node_ip_address()
            self.session = ShuffleSession(credits, advertise_host=host)
            self._it = None  # iterator over the current partition's input batches
            self._pending: deque = deque()  # mapped output morsels awaiting publish
            self._inp_rest = None  # unconsumed tail of an oversized input batch
            self._peak = 0  # peak published-but-unreleased morsels (memory-bound probe)

        def addr(self) -> str:
            return self.session.addr

        def open(self, partition: dict) -> str:
            """Begin streaming `partition`: reset the per-partition input iterator and
            output buffer. Returns this server's address."""
            from batcher.dist.executors.partition_io import iter_partition_descriptor

            self._it = iter_partition_descriptor(partition)
            self._pending = deque()
            self._inp_rest = None
            return self.session.addr

        def publish_next(self, ticket) -> bool:
            """Publish the next output morsel under `ticket`; `False` when the partition
            is exhausted. Holds only one input chunk's output at a time, so producer
            memory is bounded by the chunk plus the unreleased window."""
            batch = self._next_output()
            if batch is None:
                return False
            self.session.publish(ticket, [batch])
            self._peak = max(self._peak, self.session.partition_count)
            return True

        def _next_input(self):
            """The next input chunk, bounded to `_target_rows` rows. `None` when exhausted.

            The bound is what keeps the producer's resident output small, and it has to be on
            the **input** because that is what decides the output's size. A scan morsel is
            sized in rows (16,384 by default), so on a decode stage one chunk's *output* is
            16,384 decoded images — about 2.4 GB — and it all lands in `_pending` before a
            single morsel is published. Capping the published morsel does not help: the slices
            are views on that same 2.4 GB parent, which stays resident until the last of them
            drains. Measured on 400,000 384x384 JPEGs, 32 producers over 8 CPU nodes: a worker
            node died at 28.0 GB of 30. Feeding the sub-plan the consumer's batch size instead
            holds ~37 MB.
            """
            if self._inp_rest is not None:
                inp, self._inp_rest = self._inp_rest, None
            else:
                try:
                    inp = next(self._it)
                except StopIteration:
                    return None
            if self._target_rows and inp.num_rows > self._target_rows:
                self._inp_rest = inp.slice(self._target_rows)
                inp = inp.slice(0, self._target_rows)
            return inp

        def _next_output(self):
            """The next mapped output morsel, advancing the input stream as needed.

            Running the CPU sub-plan over one input batch at a time yields exactly the
            concatenation of the whole-partition result, because the stage is
            breaker-free (only per-batch Filter/Project/MapBatches) — so this streams
            without changing the result."""
            from batcher import core
            from batcher.io.source import InMemorySource

            while not self._pending:
                inp = self._next_input()
                if inp is None:
                    return None
                if inp.num_rows == 0:
                    continue
                self._pending.extend(core.execute_with_udfs(self._plan, [InMemorySource([inp])]))
            return self._take_batch()

        def _take_batch(self):
            """Pop about `_target_rows` rows as one morsel, pulling or splitting as needed.

            Concatenation is half the point: the consumer runs one model call per published
            morsel, so a morsel below the stage's batch size is a small forward pass that no
            downstream re-batching can undo. Short at end-of-partition is correct — there are
            no more rows to wait for — and the result is unchanged either way, because the
            stage is breaker-free and its output is the concatenation of its inputs'.

            **Splitting is the other half, and it was missing.** `_target_rows` was a floor
            with no ceiling, so an input chunk whose mapped output already exceeded it was
            published whole. That is harmless for narrow rows and ruinous for the rows this
            pipeline exists to carry: a scan morsel is sized in *rows*, so one row-group of
            decoded 384x384 images is ~460 MB of output published as a single morsel, against
            a per-channel credit budget of 256 MiB. Measured: the staged route OOM-killed two
            16-core nodes out of the fleet, twice, and narrowing the decode from `float32` to
            `uint8` did not stop it because the morsel was oversized by row count either way.
            Splitting to the size the consumer actually asked for bounds what is resident and
            hands the device the forward pass it declared.

            The slice is zero-copy and the remainder goes back on the queue, so no row is
            copied, dropped or reordered.
            """

            from batcher import core
            from batcher.io.source import InMemorySource

            first = self._pending.popleft()
            if self._target_rows <= 0:
                return first
            if first.num_rows > self._target_rows:
                self._pending.appendleft(first.slice(self._target_rows))
                return first.slice(0, self._target_rows)
            if first.num_rows == self._target_rows:
                return first
            held = [first]
            rows = first.num_rows
            while rows < self._target_rows:
                if not self._pending:
                    inp = self._next_input()
                    if inp is None:
                        break
                    if inp.num_rows == 0:
                        continue
                    self._pending.extend(
                        core.execute_with_udfs(self._plan, [InMemorySource([inp])])
                    )
                    continue
                nxt = self._pending.popleft()
                held.append(nxt)
                rows += nxt.num_rows
            # `one_batch` is the shared compaction: it keeps every row and raises rather
            # than returning a prefix when a published morsel genuinely exceeds the 32-bit
            # offset limit, which more than 2 GiB of string or binary data does.
            return one_batch(held)

        def release(self, ticket) -> None:
            """Evict a published morsel once its consumer has fetched it — frees one
            production credit and bounds the producer's resident output."""
            self.session.release(ticket)

        def peak_retained(self) -> int:
            """Peak number of published-but-unreleased morsels this producer ever held
            (a test probe for the production-window memory bound)."""
            return self._peak

except ImportError:  # pragma: no cover - ray optional
    ProducerActor = None  # type: ignore
