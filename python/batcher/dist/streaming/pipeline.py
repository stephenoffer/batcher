"""Distributed streaming heterogeneous inference pipeline (the GPU-feeding moat).

`dist/executors/map.py` distributes a linear `map_batches` chain *embarrassingly* —
one actor runs the whole CPU→GPU chain per partition, so the GPU sits idle while its
actor reads and preprocesses. This module is the distributed image of the single-node
`ml/pipeline.py::run_pipeline`: it splits the chain into stages **by resource class**
(a CPU preprocess stage, a GPU/load-once inference stage), gives each its own actor
pool, and streams partitions stage→stage so CPU and GPU **overlap** — while the GPU
runs morsel *k*, the CPU producers prepare *k+1*.

The hand-off is Carbonite Arrow Flight, not the Ray object store: each CPU producer
PUBLISHES its output **one morsel at a time** on its node-local `ShuffleSession` and
returns only a small `(addr, ticket)`; the GPU consumer FETCHES it in place
(`_MapActor.run_split`) over credit-bounded Flight. The result equals running the
stages in sequence — the single-node `run_pipeline` it mirrors — because each stage
runs the identical sub-plan through `core.execute_with_udfs`; only the *scheduling*
overlaps.

Two nested credit windows bound memory. The **transfer** window (the Rust Flight
credit window) bounds a morsel's wire buffer. The **production** window bounds how far
a producer may run ahead of its consumer: the driver issues `publish_next` only while
the producer holds fewer than `credits` published-but-unreleased morsels, and frees
one with `release` after each consumer returns — so a producer's resident output is
bounded by `credits` morsels regardless of partition size (the distributed image of
`run_pipeline`'s `Queue(maxsize=credits)`). PR3 generalizes to N stages, per-stage
autoscaling, and AIMD windows; PR4 (optional) moves the production window into the
data plane so the driver round-trip per morsel disappears.
"""

from __future__ import annotations

import contextlib
from collections import deque

import pyarrow as pa

from batcher._internal.mathx import clamp
from batcher.config import active_config
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["stream_distributed_pipeline"]

# Ticket stage id for the CPU→GPU hand-off (stage 0 is the source; 1 is this channel).
_STAGE_ID = 1


try:
    import ray

    @ray.remote
    class _ProducerActor:
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

        def __init__(self, plan0: LogicalPlan, credits: int) -> None:
            from batcher.carbonite.transfer import ShuffleSession
            from batcher.dist.executors.map import _prebuild_factories

            self._plan = _prebuild_factories(plan0)
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
            return self._pending.popleft()

        def release(self, ticket) -> None:
            """Evict a published morsel once its consumer has fetched it — frees one
            production credit and bounds the producer's resident output."""
            self.session.release(ticket)

        def peak_retained(self) -> int:
            """Peak number of published-but-unreleased morsels this producer ever held
            (a test probe for the production-window memory bound)."""
            return self._peak

except ImportError:  # pragma: no cover - ray optional
    _ProducerActor = None  # type: ignore


def _consumer_pool_size(gpu_stage, workers: int, num_partitions: int) -> int:
    """Actor count for the GPU consumer stage: its explicit `concurrency`, else a
    GPU-aware default (one actor per GPU), clamped to the partition count."""
    from batcher.dist.executors.map import _resolve_pool_size
    from batcher.ml.gpu import gpu_aware_pool_default

    default = gpu_aware_pool_default(
        gpu_stage.num_gpus,
        workers,
        num_partitions,
        getattr(gpu_stage, "accelerator_type", None),
        resources=dict(getattr(gpu_stage, "resources", ()) or ()),
    )
    size = _resolve_pool_size(gpu_stage.concurrency, num_partitions, default)
    return clamp(num_partitions, 1, size)


def stream_distributed_pipeline(
    plan: LogicalPlan, sources: list[Source], workers: int, hub=None
) -> pa.Table:
    """Run a two-stage (CPU→GPU) linear map pipeline with overlapped, credit-bounded
    stages.

    The CPU producers stream each partition's preprocessed output morsel by morsel over
    Flight while the GPU consumers run inference on already-produced morsels — so the
    GPU stays fed instead of waiting on the CPU stage, and producer memory stays bounded
    by the production credit window regardless of partition size. The result is
    identical to the single-node sequential composition (and to the non-overlapped
    distributed map); only the scheduling overlaps. The caller guarantees a 2-stage
    split (the dispatch hook checks `split_into_resource_stages`); other shapes use
    `_distributed_map`.
    """
    from batcher.dist.executors.map import _gpu_options
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.plan_analysis import split_at_first_pool_boundary
    from batcher.dist.executors.ray_runtime import _ensure_ray
    from batcher.dist.flight_worker import new_plan_id
    from batcher.plan.visitor import scanned_source_ids

    _ensure_ray(workers)
    # Imported AFTER `_ensure_ray` so it is the Ray-remote-wrapped class (the rebind in
    # `ray_runtime._wrap_tasks` applies the ambient scheduling grant to the base actor).
    from batcher.dist.executors.map import _MapActor

    cpu_stage, gpu_stage = split_at_first_pool_boundary(plan)
    sid = next(iter(scanned_source_ids(plan)))
    partitions = partition_descriptors(sources[sid], workers)
    n = len(partitions)
    if n == 0:
        return pa.table({})

    credits = max(1, active_config().flow_control.default_credits)
    n_producers = clamp(n, 1, workers)
    n_consumers = _consumer_pool_size(gpu_stage, workers, n)
    gpu_opts = _gpu_options(gpu_stage.num_gpus, gpu_stage.accelerator_type)

    consumer_cls = _MapActor.options(**gpu_opts) if gpu_opts else _MapActor

    def _spawn_producer():
        return _ProducerActor.remote(cpu_stage.sub_plan, credits)

    def _spawn_consumer():
        return consumer_cls.remote(gpu_stage.sub_plan)

    producers = [_spawn_producer() for _ in range(n_producers)]
    consumers = [_spawn_consumer() for _ in range(n_consumers)]
    # Every actor ever spawned (including preemption replacements) is killed at the end;
    # the recovery loop mutates `producers`/`consumers` in place to hold the live pools.
    alive: set = {*producers, *consumers}
    try:
        results = _run_streamed(
            producers,
            consumers,
            partitions,
            new_plan_id(),
            credits,
            spawn_producer=_spawn_producer,
            spawn_consumer=_spawn_consumer,
            alive=alive,
        )
        # The GPU consumers measured their utilization; record it so the next run's
        # `num_gpus` request adapts (the feedback half of GPU scheduling).
        _record_consumer_feedback(consumers, plan, hub)
    finally:
        for actor in alive:
            with contextlib.suppress(Exception):
                ray.kill(actor)

    # Concatenate morsels in (partition, seq) order — a valid grouping of the input
    # multiset (the result is an unordered relation; callers that need order sort).
    batches: list[pa.RecordBatch] = []
    for _key, out in sorted(results.items()):
        if out:
            batches.extend(out)
    return pa.Table.from_batches(batches) if batches else pa.table({})


def _record_consumer_feedback(consumers, plan: LogicalPlan, hub) -> None:
    """Persist the GPU consumers' peak utilization for next-run `num_gpus` adaptation
    (a no-op when `hub` is None or no GPU was observed)."""
    from batcher.dist.executors.map import _record_gpu_feedback

    samples = [s for s in ray.get([c.gpu_stats.remote() for c in consumers]) if s is not None]
    _record_gpu_feedback(hub, plan, max(samples) if samples else None)


def _worker_loss_errors() -> tuple[type[BaseException], ...]:
    """Ray exception types that signal a lost actor/task (safe to recover), built lazily
    so `ray` stays an optional import."""
    try:
        import ray as _ray

        return (_ray.exceptions.RayActorError, _ray.exceptions.RayTaskError)
    except Exception:  # pragma: no cover - ray optional
        return ()


def _run_streamed(
    producers,
    consumers,
    partitions,
    plan_id,
    credits,
    *,
    spawn_producer=None,
    spawn_consumer=None,
    alive: set | None = None,
):
    """Stream partitions morsel by morsel through the producer and consumer pools.

    Each producer streams one partition at a time, publishing a morsel only while it
    holds fewer than `credits` unreleased morsels (the production window). As morsels
    become ready they are handed to free consumers; a consumer's completion releases
    the morsel (freeing a credit) and lets the producer publish the next. So CPU and
    GPU overlap, and a producer's resident output never exceeds `credits` morsels
    regardless of partition size. Returns `{(partition_idx, seq): output_batches}`.

    Fault tolerance (spot preemption). Actors are stateful, so a lost actor is
    recovered by lineage recompute, not lost:

    - A **consumer** that dies mid-morsel had not released that morsel (release runs only
      after a successful fetch), so the morsel is still on its producer's Flight server —
      re-dispatch it to a fresh consumer. A deterministic `RayTaskError` (a UDF bug, not a
      preemption) is bounded by `recovery_max_attempts` so it surfaces instead of looping.
    - A **producer** that dies takes its in-flight partition (open iterator + published
      morsels) with it, so re-run that whole partition on a fresh producer from its durable
      descriptor. Output morsels are keyed by `(pidx, seq)` and the sub-plan is
      deterministic, so the re-run reproduces the same keys/values — `results` overwrites
      idempotently and stays complete. In-flight fetches from the dead producer fail and
      are discarded (their partition is being re-run).

    `spawn_producer`/`spawn_consumer` mint replacement actors (registered in `alive` for
    teardown); with neither provided (single-actor tests) a loss re-raises.
    """
    from batcher.carbonite.transfer import ShuffleTicket
    from batcher.config import active_config

    loss_errors = _worker_loss_errors()
    max_attempts = max(1, active_config().distributed.recovery_max_attempts)

    free_producers = deque(producers)
    pending_parts = deque(enumerate(partitions))  # (partition_idx, descriptor)
    state: dict = {}  # producer -> per-partition streaming state
    addr_of: dict = {}  # producer -> its (stable) Flight address
    dead_producers: set = set()  # producers lost to preemption (their morsels are void)
    open_inflight: dict = {}  # ref -> producer
    publish_inflight: dict = {}  # ref -> (producer, partition_idx, seq, ticket)
    consume_inflight: dict = {}  # ref -> (consumer, producer, key, addr, ticket)
    free_consumers = deque(consumers)
    ready: deque = deque()  # (addr, ticket, producer, key, attempts) awaiting a consumer
    results: dict = {}
    part_attempts: dict = {}  # pidx -> re-run count (bounds a deterministic producer crash)

    def start_part(prod) -> None:
        pidx, desc = pending_parts.popleft()
        state[prod] = {
            "pidx": pidx,
            "desc": desc,
            "seq": 0,
            "outstanding": 0,
            "done": False,
            "open": False,
        }
        open_inflight[prod.open.remote(desc)] = prod

    def recycle(prod) -> None:
        del state[prod]
        if pending_parts:
            start_part(prod)
        else:
            free_producers.append(prod)

    def replace_consumer(dead) -> None:
        """Drop a lost consumer and, if we can, add a fresh one so pool size holds."""
        with contextlib.suppress(ValueError):
            free_consumers.remove(dead)
        with contextlib.suppress(ValueError):
            consumers.remove(dead)
        if spawn_consumer is not None:
            fresh = spawn_consumer()
            if alive is not None:
                alive.add(fresh)
            consumers.append(fresh)
            free_consumers.append(fresh)

    def lose_producer(dead, *, exc) -> None:
        """Re-queue a lost producer's partition for a full deterministic re-run."""
        if dead in dead_producers:
            return
        dead_producers.add(dead)
        st = state.pop(dead, None)
        with contextlib.suppress(ValueError):
            free_producers.remove(dead)
        with contextlib.suppress(ValueError):
            producers.remove(dead)
        # Drop this producer's ready-but-unconsumed morsels — its Flight server is gone.
        for item in [i for i in ready if i[2] is dead]:
            ready.remove(item)
        if st is not None:
            pidx = st["pidx"]
            part_attempts[pidx] = part_attempts.get(pidx, 0) + 1
            if part_attempts[pidx] > max_attempts:
                raise exc  # a partition that keeps killing its producer is not recoverable
            pending_parts.append((pidx, st["desc"]))
        if spawn_producer is not None:
            fresh = spawn_producer()
            if alive is not None:
                alive.add(fresh)
            producers.append(fresh)
            free_producers.append(fresh)

    while True:
        # Assign any pending partitions (initial fan-out and recovery re-runs) to free
        # producers at the top of every iteration, so a re-queued partition restarts.
        while free_producers and pending_parts:
            start_part(free_producers.popleft())

        # Issue a publish for every opened producer that has production-window headroom
        # and no publish already in flight (one outstanding publish per producer).
        publishing = {p for p, _pi, _s, _t in publish_inflight.values()}
        for prod, st in state.items():
            has_headroom = st["open"] and not st["done"] and st["outstanding"] < credits
            if has_headroom and prod not in publishing:
                seq = st["seq"]
                st["seq"] += 1
                ticket = ShuffleTicket(plan_id, _STAGE_ID, st["pidx"], seq)
                ref = prod.publish_next.remote(ticket)
                publish_inflight[ref] = (prod, st["pidx"], seq, ticket)

        # Assign ready morsels to free consumers.
        while ready and free_consumers:
            addr, ticket, prod, key, attempts = ready.popleft()
            consumer = free_consumers.popleft()
            ref = consumer.run_split.remote(addr, ticket)
            consume_inflight[ref] = (consumer, prod, key, addr, ticket, attempts)

        waitset = list(open_inflight) + list(publish_inflight) + list(consume_inflight)
        if not waitset:
            break  # nothing in flight and no headroom to issue ⇒ all partitions drained
        done, _ = ray.wait(waitset, num_returns=1)
        ref = done[0]
        if ref in open_inflight:
            prod = open_inflight.pop(ref)
            try:
                addr_of[prod] = ray.get(ref)
            except loss_errors as exc:
                lose_producer(prod, exc=exc)
                continue
            if prod in state:
                state[prod]["open"] = True
        elif ref in publish_inflight:
            prod, pidx, seq, ticket = publish_inflight.pop(ref)
            try:
                more = ray.get(ref)
            except loss_errors as exc:
                lose_producer(prod, exc=exc)
                continue
            st = state.get(prod)
            if st is None:  # producer was lost between issue and completion
                continue
            if more:
                st["outstanding"] += 1
                ready.append((addr_of[prod], ticket, prod, (pidx, seq), 0))
            else:
                st["done"] = True
                if st["outstanding"] == 0:
                    recycle(prod)
        else:
            consumer, prod, key, addr, ticket, attempts = consume_inflight.pop(ref)
            try:
                out = ray.get(ref)
            except loss_errors as exc:
                # The consumer (or its fetch from a now-dead producer) failed. Replace the
                # consumer; if the producer is still alive the morsel is still published,
                # so re-dispatch it (bounded); if the producer died, its partition is being
                # re-run, so drop this morsel.
                replace_consumer(consumer)
                if prod not in dead_producers and prod in state:
                    if attempts + 1 > max_attempts:
                        raise exc
                    ready.append((addr, ticket, prod, key, attempts + 1))
                continue
            results[key] = out
            free_consumers.append(consumer)
            st = state.get(prod)
            if prod not in dead_producers:
                prod.release.remote(ticket)  # free one production credit
            if st is not None:
                st["outstanding"] -= 1
                if st["done"] and st["outstanding"] == 0:
                    recycle(prod)
    return results
