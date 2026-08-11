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

__all__ = ["ProducerActor", "coalesce", "consumer_batch_rows"]


def coalesce(batches: list, target_rows: int) -> list:
    """Regroup `batches` so each holds at least `target_rows` rows, where the data allows.

    The eager counterpart of `ProducerActor._take_batch`, for a stage whose whole output is
    already in hand. Both exist for one reason: a published morsel is one model call on the
    stage above, so a morsel below that stage's declared batch size is a small forward pass no
    downstream re-batching can undo. The producer has to gather *lazily*, pulling more input
    only when it is short, or it would defeat the streaming it exists to do; a relay has run
    its whole morsel already and has nothing left to pull.

    Args:
        batches: The stage's output batches, in order.
        target_rows: Rows to gather per morsel; `0` leaves the batches as they are.

    Returns:
        The regrouped batches. A short final morsel is correct — there are no more rows.
    """
    import pyarrow as pa

    if target_rows <= 0 or len(batches) <= 1:
        return list(batches)
    out: list = []
    held: list = []
    rows = 0
    for batch in batches:
        held.append(batch)
        rows += batch.num_rows
        if rows >= target_rows:
            # `concat_batches` rather than `combine_chunks`: the latter splits at the 32-bit
            # offset limit, so a morsel holding more than 2 GiB of string or binary data comes
            # back as several batches — see `_take_batch`, which learned this the hard way.
            out.append(held[0] if len(held) == 1 else pa.concat_batches(held))
            held, rows = [], 0
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

        def __init__(self, plan0: LogicalPlan, credits: int, target_rows: int = 0) -> None:
            from batcher.carbonite.transfer import ShuffleSession
            from batcher.dist.executors.map import _prebuild_factories

            self._plan = _prebuild_factories(plan0)
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
            self._peak = 0  # peak published-but-unreleased morsels (memory-bound probe)

        def addr(self) -> str:
            return self.session.addr

        def open(self, partition: dict) -> str:
            """Begin streaming `partition`: reset the per-partition input iterator and
            output buffer. Returns this server's address."""
            from batcher.dist.executors.partition_io import iter_partition_descriptor

            self._it = iter_partition_descriptor(partition)
            self._pending = deque()
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

        def _next_output(self):
            """The next mapped output morsel, advancing the input stream as needed.

            Running the CPU sub-plan over one input batch at a time yields exactly the
            concatenation of the whole-partition result, because the stage is
            breaker-free (only per-batch Filter/Project/MapBatches) — so this streams
            without changing the result."""
            from batcher import core
            from batcher.io.source import InMemorySource

            while not self._pending:
                try:
                    inp = next(self._it)
                except StopIteration:
                    return None
                if inp.num_rows == 0:
                    continue
                self._pending.extend(core.execute_with_udfs(self._plan, [InMemorySource([inp])]))
            return self._take_batch()

        def _take_batch(self):
            """Pop at least `_target_rows` rows as one morsel, pulling more input as needed.

            Concatenation is the whole point: the consumer runs one model call per published
            morsel, so a morsel below the stage's batch size is a small forward pass that no
            downstream re-batching can undo. Short at end-of-partition is correct — there are
            no more rows to wait for — and the result is unchanged either way, because the
            stage is breaker-free and its output is the concatenation of its inputs'.
            """
            import pyarrow as pa

            from batcher import core
            from batcher.io.source import InMemorySource

            first = self._pending.popleft()
            if self._target_rows <= 0 or first.num_rows >= self._target_rows:
                return first
            held = [first]
            rows = first.num_rows
            while rows < self._target_rows:
                if not self._pending:
                    try:
                        inp = next(self._it)
                    except StopIteration:
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
            if len(held) == 1:
                return held[0]
            # `concat_batches` rather than `combine_chunks().to_batches()[0]`: the latter
            # splits at the 32-bit offset limit, so a published morsel holding more than
            # 2 GiB of string or binary data came back as several batches and taking the
            # first silently dropped every row after it.
            return pa.concat_batches(held)

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
