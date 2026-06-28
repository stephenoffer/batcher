"""The shared Arrow Flight shuffle worker actor.

`_FlightWorker` is one Ray actor per worker slot, hosting a Carbonite
`ShuffleSession` (its node-local Flight server). Every Flight-shuffle operator —
aggregate, join, window, sort — drives this *same* actor: mappers PUBLISH their
hash- or range-partitioned output on their own server and advertise only their
`addr`; reducers FETCH their bucket from every mapper over credit-bounded Flight.
Only `(addr, ticket)` strings (and the small finalized results) transit Ray — no
`RecordBatch` becomes a Ray object, and the heavy shuffle never touches the object
store.

The actor is operator-agnostic: each method supplies the opaque IR / partition
function for its operator, and the session moves bytes under the Carbonite-granted
credit window (reading co-located buckets straight from the local store, no
loopback). Keeping it in one module lets every `flight_*` operator share the actor
and its lineage-recovery contract without a circular import.
"""

from __future__ import annotations

from batcher.carbonite.transfer import ShuffleTicket

__all__ = [
    "_FlightWorker",
    "_combine_sources",
    "_ticket",
    "new_plan_id",
    "set_current_plan_id",
    "spawn_flight_workers",
]

# The shuffle plan id for the query in flight on THIS process (driver or worker). One
# in-flight plan per process, so a module-level value is correct; it is set once per
# query (`set_current_plan_id`) on the driver and on every worker so all tickets carry
# the same id. Fences a query's published partitions from another query's — and from a
# crashed prior query's leftovers when a persistent fleet actor is reused.
_DEFAULT_PLAN_ID = 1
_current_plan_id = _DEFAULT_PLAN_ID
_RESULT_STAGE = 100  # ticket stage for a stage's *finalized* result (kept on the actor)


def new_plan_id() -> int:
    """A fresh, process-unique-enough shuffle plan id (63-bit, fits the ticket field).

    Generated once per query at fleet spawn. Two queries — or a crashed query and its
    replacement reusing the same fleet actor — get different ids, so a stale partition
    left at the same stage/src/dst/epoch under the old id can never be fetched (the
    cross-query / cross-restart analogue of the per-recompute `epoch` fence)."""
    import uuid

    return uuid.uuid4().int & ((1 << 63) - 1)


def set_current_plan_id(plan_id: int) -> None:
    """Set this process's current shuffle plan id so `_ticket` fences this query.

    Called once per query: on the driver (which builds the tree-combine tickets) and
    inside every `_FlightWorker` (which builds publish/fetch tickets), with the same
    id, so the whole shuffle agrees."""
    global _current_plan_id
    _current_plan_id = plan_id


def _ticket(stage: int, src: int, dst: int, epoch: int = 0) -> ShuffleTicket:
    """A shuffle ticket for this query: `plan/stage/src(mapper)/dst(reducer)/epoch`.

    `plan` is the per-query id (`set_current_plan_id`) fencing this query from another.
    `epoch` (default 0) fences a recomputed partition from the stale one a lost worker
    published: a fresh recompute bumps the source's epoch, so the partition is
    published *and* fetched under a new ticket and a zombie worker's old-epoch partial
    can never be read — defense in depth atop the address-redirect the recovery loop
    already does.
    """
    return ShuffleTicket(_current_plan_id, stage, src, dst, epoch)


def _combine_sources(session, gk, aj, sources):
    """Fetch each `(addr, ticket)` concurrently and merge into one running partial.

    The bounded-memory merge: hold one combined partial, never the whole source
    list. `sources` is at most `fan_in` long in the tree shuffle, so a combiner
    node's fan-in (and memory) is bounded regardless of cluster size. A lost source
    surfaces as a `RetryableShuffleError` (the tree node's task fails → driver
    recompute), preserving the propagate-on-fault contract of the serial path.
    """
    import batcher._native as nat

    running, unreachable = session.gather_combine(gk, aj, list(sources), finalize=False)
    if unreachable:
        raise nat.RetryableShuffleError(f"combiner lost sources {unreachable}")
    return running


try:
    import ray

    @ray.remote
    class _FlightWorker:
        """A Ray actor hosting a Carbonite `ShuffleSession` for one worker slot.

        The session owns this worker's Flight server, moves buckets under the
        Carbonite-granted credit window, and reads co-located buckets straight from
        the local store (no loopback). Map/reduce supply opaque partials/partition
        functions; the session is operator-agnostic.
        """

        def __init__(
            self,
            worker_id: int,
            credits: int,
            engine_config: str = "",
            adaptive: bool = False,
            token: str = "",
            idle_timeout_ms: int = 0,
            keepalive_ms: int = 0,
            plan_id: int = _DEFAULT_PLAN_ID,
            shm: bool = False,
            preemption: bool = False,
        ) -> None:
            import batcher._native as nat
            from batcher.carbonite.transfer import ShuffleSession

            # Fence this worker's tickets to the query it was spawned for, so a
            # reused fleet actor cannot serve a prior (crashed) query's stale buckets.
            set_current_plan_id(plan_id)

            # Under the spot profile, watch for a preemption notice (SIGTERM / cloud
            # metadata) so the driver can migrate this worker's shuffle output to a
            # survivor *before* it is reclaimed — turning a reactive recompute into a
            # zero-loss proactive migration. Started only here (one poller per worker
            # process); a stable on-demand cluster never starts it and pays nothing.
            if preemption:
                from batcher.carbonite.resilience import preemption_monitor

                preemption_monitor().start()

            # Apply the driver's Flight transport timeouts in this worker process
            # (it can't see the driver's config_context): bound the fetch idle gap so
            # a dead peer is detected, and set keepalive to catch a dropped connection
            # promptly. A long GC pause under a generous idle window is not misread as
            # death. 0 keeps the process default.
            if idle_timeout_ms or keepalive_ms:
                nat.set_flight_transport_config(idle_timeout_ms, keepalive_ms)

            self.id = worker_id
            # The node's routable IP, so this worker's Flight server advertises a
            # cross-node-reachable address instead of loopback (which a reducer on
            # another host could never dial). On a single-host cluster this is the
            # local IP and behaves like before.
            advertise_host = ray.util.get_node_ip_address()
            shuffle_token = token or None
            # Opt-in AIMD adaptive credits: the window adjusts to this worker's memory
            # pressure per fetch. Decided on the driver (the worker can't see the
            # driver's config_context) and passed in, so it reaches every worker.
            if adaptive:
                from batcher.carbonite.memory.pressure import PressureMonitor
                from batcher.carbonite.policies import AIMDFlowControl

                self.session = ShuffleSession(
                    credits,
                    flow_control=AIMDFlowControl(),
                    pressure=PressureMonitor(),
                    advertise_host=advertise_host,
                    token=shuffle_token,
                    shm=shm,
                )
            else:
                self.session = ShuffleSession(
                    credits, advertise_host=advertise_host, token=shuffle_token, shm=shm
                )
            # The driver's EngineConfig (this worker process can't see the driver's
            # config_context), used for every local execute_plan on this actor.
            self._engine_config = engine_config
            # Bytes this mapper published per reducer bucket on its last `map_publish`,
            # so the driver can place each reducer where its bucket is concentrated
            # (locality-aware scheduling). Overwritten each map; read after the barrier.
            self._bucket_bytes: dict[int, int] = {}

        def addr(self) -> str:
            return self.session.addr

        def is_draining(self) -> bool:
            """Whether this worker has seen a spot-preemption notice (reclamation
            imminent). The driver consults this at a stage boundary to migrate the
            worker's shuffle output before it dies. Always `False` when the monitor
            was not started (the non-spot path), so the query is safe to call anywhere."""
            from batcher.carbonite.resilience import preemption_monitor

            return preemption_monitor().is_draining()

        def published_bucket_bytes(self) -> dict[int, int]:
            """Bytes published per reducer bucket on this mapper's last `map_publish`
            (for locality-aware reducer placement)."""
            return dict(self._bucket_bytes)

        def node_id(self) -> str:
            """The Ray node this worker's actor landed on — for locality routing and
            observing how well the placement group spread the fleet."""
            import ray

            return ray.get_runtime_context().get_node_id()

        def map_publish(
            self, map_ir, gk, aj, partition, n_keys, n_reducers, src=None, epoch=0
        ) -> str:
            import batcher._native as nat
            from batcher.dist.executors.partition_io import (
                iter_partition_descriptor,
                streaming_partial_aggregate,
            )

            # `src` overrides the mapper id on recompute: a surviving worker
            # regenerates a lost worker's output, so it publishes under the
            # *original* src and the reducers' tickets still resolve. `epoch` rises on
            # each recompute so the fresh partition can't be confused with the stale
            # one a lost worker left under the previous epoch.
            src = self.id if src is None else src
            # Stream the partition through the map prefix + partial-aggregate one chunk at
            # a time, so the map side never materializes the whole partition or its whole
            # mapped output — the #1 distributed memory peak. Mergeable: the folded
            # per-chunk partials equal one partial over the whole partition.
            partial = streaming_partial_aggregate(
                nat, map_ir, gk, aj, iter_partition_descriptor(partition), self._engine_config
            )
            if n_keys == 0:
                buckets = [[partial]]
            else:
                buckets = nat.partition_batches([partial], list(range(n_keys)), n_reducers)
            # Publish EVERY bucket, empty included: then a reducer's failed fetch can
            # only mean a lost worker, never a legitimately empty bucket — the clean
            # signal the recompute loop keys on. Record each bucket's bytes for
            # locality-aware reducer placement.
            self._bucket_bytes = {}
            for r in range(n_reducers):
                bucket = buckets[r] if r < len(buckets) else []
                self.session.publish(_ticket(0, src, r, epoch), bucket)
                self._bucket_bytes[r] = sum(b.nbytes for b in bucket)
            return self.session.addr

        def reduce_fetch(self, gk, aj, mapper_addrs, reducer_id, epochs=None):
            # Fetch every mapper's partial *concurrently* and fold them into one running
            # merged state in Rust (bounded by the session's fan-in), instead of one
            # blocking round-trip per mapper. The reducer holds one merged partial (sized
            # by the group count) plus at most `fan_in` in-flight fetches — memory
            # independent of the mapper count, so the shuffle scales to a wide cluster.
            # `combine` is associative, so the concurrent fold equals a serial one.
            # `epochs` maps a recomputed source to its current epoch (default 0) so the
            # fetch resolves the fresh partition, never a lost worker's stale one. A
            # retryable fault (unreachable/idle peer == worker loss, since every bucket
            # is published) is reported so the driver recomputes + retries; a fatal fault
            # propagates and fails the query fast.
            epochs = epochs or {}
            sources = [
                (addr, _ticket(0, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(mapper_addrs)
            ]
            payload, unreachable = self.session.gather_combine(gk, aj, sources, finalize=True)
            if unreachable:
                return ("retry", unreachable)
            return ("ok", payload)

        def reduce_fetch_publish(self, gk, aj, mapper_addrs, reducer_id, epochs=None):
            """Like `reduce_fetch`, but PUBLISH the finalized result on this worker's own
            Flight server and return only a `(addr, ticket, rows, schema)` handle.

            This keeps the stage's output partitioned on the workers — the adaptive
            executor scans it in place for the next stage instead of pulling every
            reducer's result back to the driver. The status protocol is unchanged
            (`"retry"` on a lost mapper), so it composes with the recovery loop.
            """
            status, payload = self.reduce_fetch(gk, aj, mapper_addrs, reducer_id, epochs)
            if status != "ok" or payload is None:
                return (status, payload)  # retry, or an empty bucket (no handle)
            ticket = _ticket(_RESULT_STAGE, self.id, reducer_id)
            self.session.publish(ticket, [payload])
            return ("ok", (self.session.addr, ticket, payload.num_rows, payload.schema))

        def combine_publish(self, gk, aj, sources, out_ticket):
            # One interior node of the combiner tree: merge <= fan_in upstream
            # partials and republish the result for the next level to fetch.
            running = _combine_sources(self.session, gk, aj, sources)
            self.session.publish(out_ticket, [running] if running is not None else [])
            return self.session.addr

        def combine_finalize_fetch(self, gk, aj, sources):
            # The tree root for one bucket: merge the last <= fan_in partials and
            # finalize to output rows.
            import batcher._native as nat

            running = _combine_sources(self.session, gk, aj, sources)
            return None if running is None else nat.combine_finalize(gk, aj, [running])

        def map_publish_raw(self, sub_ir, key_names, partition, n_buckets, stage, src=None) -> str:
            import batcher._native as nat
            from batcher.dist.executors.partition_io import read_partition_descriptor

            # `src` overrides the mapper id on recompute (a survivor regenerates a
            # lost worker's side). Publish EVERY bucket, empty included, so a
            # reducer's failed fetch means a lost worker, not an empty bucket.
            src = self.id if src is None else src
            rows = nat.execute_plan(
                sub_ir, [read_partition_descriptor(partition)], self._engine_config
            )
            if not rows:
                buckets = []
            else:
                key_idx = [rows[0].schema.get_field_index(k) for k in key_names]
                buckets = (
                    [rows] if n_buckets == 1 else nat.partition_batches(rows, key_idx, n_buckets)
                )
            for r in range(n_buckets):
                self.session.publish(_ticket(stage, src, r), buckets[r] if r < len(buckets) else [])
            return self.session.addr

        def reduce_window(self, win_ir, addrs, reducer_id):
            import batcher._native as nat

            # A window partition is computed whole, so this reducer holds all of its
            # bucket's raw rows (memory = the bucket, which shrinks as workers grow).
            # Fetch every mapper concurrently, tracking lost workers so the driver
            # recomputes + retries.
            sources = [(addr, _ticket(0, src, reducer_id)) for src, addr in enumerate(addrs)]
            rows, unreachable = self.session.gather_concat(sources)
            if unreachable:
                return ("retry", unreachable)
            if not rows:
                return ("ok", None)
            return ("ok", nat.execute_plan(win_ir, [rows], self._engine_config))

        def reduce_join(
            self, join_ir, addrs, reducer_id, left_schema, right_schema, gk=None, aj=None
        ):
            import batcher._native as nat

            # A join needs its bucket's whole left and right side, so it holds them
            # both (memory = the bucket's data, which shrinks as workers grow). Fetch
            # every mapper's left (stage 0) and right (stage 1) side concurrently,
            # tracking lost workers so the driver can recompute and retry.
            left_sources = [(addr, _ticket(0, src, reducer_id)) for src, addr in enumerate(addrs)]
            right_sources = [(addr, _ticket(1, src, reducer_id)) for src, addr in enumerate(addrs)]
            left, lost_left = self.session.gather_concat(left_sources)
            right, lost_right = self.session.gather_concat(right_sources)
            unreachable = sorted(set(lost_left) | set(lost_right))
            if unreachable:
                return ("retry", unreachable)
            if not left and not right:
                return ("ok", None)
            # Schema-bearing empties so an outer join can null-extend the missing side.
            joined = nat.execute_plan(
                join_ir, [left or [left_schema], right or [right_schema]], self._engine_config
            )
            if gk is not None:
                # Fused post-join aggregate: the group keys ⊇ the join key, so every group's
                # rows share one join key and land in THIS bucket — a per-bucket
                # partial+finalize is therefore complete (no cross-bucket combine). Only the
                # small aggregated bucket leaves the worker; the full join never reaches the
                # driver (exchange elimination). `combine_finalize` returns one batch (or
                # None); reduce_join's contract is a list of batches, so wrap it.
                out = nat.combine_finalize(gk, aj, [nat.partial_aggregate(gk, aj, joined)])
                joined = [out] if out is not None else []
            return ("ok", joined)

        def sample_quantiles(self, map_ir, key_name, probs, partition):
            """Sample this split's leading-key distribution as a small quantile grid.

            Each worker samples its *own* split — the input is never read on the
            driver. The grid (a few floats) plus the row count go back; the driver
            merges them into range boundaries. Stateless w.r.t. the shuffle session.
            """
            import batcher._native as nat
            from batcher.dist.executors.partition_io import read_partition_descriptor

            rows = nat.execute_plan(
                map_ir, [read_partition_descriptor(partition)], self._engine_config
            )
            n = sum(b.num_rows for b in rows)
            if n == 0:
                return ([], 0)
            grid = nat.column_quantiles([key_name], rows, list(probs)).get(key_name, [])
            return (grid, n)

        def range_publish(
            self, map_ir, key_name, boundaries, n_buckets, nulls_first, desc, partition, src=None
        ) -> str:
            """Range-partition this split's rows by `boundaries` and publish each bucket.

            Bucket b holds keys in `(boundaries[b-1], boundaries[b]]` so the buckets
            are globally ordered; equal keys never span a boundary (boundaries are
            deduplicated on the driver and `searchsorted(side="right")` is used), so
            the per-bucket sorts concatenate to a globally sorted result. Nulls go to
            the bucket that lands at the correct end of the *final* (post-`desc`)
            concatenation, so they sort first/last exactly as single-node would.
            """
            import batcher._native as nat
            from batcher.dist.executors.partition_io import bucketize, read_partition_descriptor

            src = self.id if src is None else src
            rows = nat.execute_plan(
                map_ir, [read_partition_descriptor(partition)], self._engine_config
            )
            buckets = bucketize(rows, key_name, boundaries, n_buckets, nulls_first, desc)
            # Publish EVERY bucket (empty included) so a reducer's failed fetch means
            # a lost worker, not an empty bucket — the recompute loop's clean signal.
            for r in range(n_buckets):
                self.session.publish(_ticket(0, src, r), buckets[r])
            return self.session.addr

        def sort_reduce(self, sort_ir, addrs, reducer_id):
            import batcher._native as nat

            # This reducer owns one contiguous key range; fetch its bucket from every
            # mapper concurrently, concatenate, and sort by all keys — the bucket is
            # globally ordered relative to the others, so a final concat needs no merge.
            sources = [(addr, _ticket(0, src, reducer_id)) for src, addr in enumerate(addrs)]
            rows, unreachable = self.session.gather_concat(sources)
            if unreachable:
                return ("retry", unreachable)
            if not rows:
                return ("ok", None)
            return ("ok", nat.execute_plan(sort_ir, [rows], self._engine_config))

except ImportError:  # pragma: no cover - ray optional
    _FlightWorker = None  # type: ignore


def spawn_flight_workers(workers: int, credits: int, cfg_json: str, plan_id: int | None = None):
    """Gang-schedule `workers` `_FlightWorker` actors in one SPREAD placement group.

    Returns `(actors, placement_group)`. The whole fleet is reserved before the
    shuffle starts (no partial-fleet deadlock) and spread across nodes for even data
    distribution and locality; pass the PG to `release_placement` when done. The PG
    is `None` when placement is unavailable or over-subscribed, and the actors then
    fall back to default scheduling — the result is identical either way.

    `plan_id` fences this query's shuffle from another's; a fresh one is minted when
    omitted. It is set on the driver here (so the driver's tree-combine tickets agree)
    and passed to every worker (so its publish/fetch tickets agree).
    """
    if plan_id is None:
        plan_id = new_plan_id()
    set_current_plan_id(plan_id)
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import (
        create_worker_placement,
        current_envelope,
        placement_actor_options,
    )

    dc = active_config().distributed
    adaptive = dc.adaptive_credits
    # The shuffle auth token is decided on the driver (the worker can't see the
    # driver's config_context) and shipped to every actor, so all servers expect and
    # all clients present the same secret. Env var overrides config (N5).
    import os

    token = os.environ.get("BATCHER_SHUFFLE_TOKEN") or dc.shuffle_token or ""
    # Flight transport timeouts decided on the driver and shipped to every worker
    # (which can't see the driver's config_context), in milliseconds for the native
    # setter. 0 keepalive = off.
    idle_ms = int(dc.flight_idle_timeout_s * 1000)
    keepalive_ms = int((dc.flight_keepalive_s or 0) * 1000)
    # Same-node shared-memory transfer, decided on the driver and shipped to every
    # worker (which can't see the driver's config_context). Gated on the native probe so
    # it is never enabled where no shared directory exists (it would just churn fallbacks).
    import batcher._native as nat

    shm = bool(dc.shared_memory_transfer) and nat.shm_available()
    # Each worker watches for a preemption notice so the driver can migrate its output
    # before reclamation (proactive, not reactive). Engaged under the spot profile —
    # which a spot deployment gets automatically: `config.profiles.detect_spot_environment`
    # auto-upgrades `resilience` to "spot" on a detected spot node, so a fresh user is
    # protected without setting it by hand.
    preemption = dc.resilience == "spot"
    pg = create_worker_placement(workers, current_envelope())
    actors = [
        _FlightWorker.options(**placement_actor_options(pg, i)).remote(
            i, credits, cfg_json, adaptive, token, idle_ms, keepalive_ms, plan_id, shm, preemption
        )
        for i in range(workers)
    ]
    return actors, pg
