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

from concurrent import futures
from typing import TYPE_CHECKING

from batcher._internal.native import engine
from batcher.carbonite.transfer import ShuffleTicket

if TYPE_CHECKING:
    from batcher.config.config import ShuffleTlsConfig

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
    nat = engine()
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
            connections_per_peer: int = 0,
            compression: int = 1,
            plan_id: int = _DEFAULT_PLAN_ID,
            shm: bool = False,
            preemption: bool = False,
            tls_config: ShuffleTlsConfig | None = None,
        ) -> None:
            nat = engine()
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
            nat.set_flight_transport_config(
                idle_timeout_ms, keepalive_ms, connections_per_peer, compression
            )

            # Shuffle TLS (off unless the operator mounted certs and enabled it). Read
            # this node's mounted PEM files, install the process-wide *client* TLS for
            # outbound fetches, and keep the *server* material to hand to the session's
            # Flight server below. A misconfigured deployment raises here, at worker
            # startup, rather than at the first cross-node fetch.
            shuffle_tls = None
            if tls_config is not None and tls_config.enabled:
                from batcher.carbonite.transfer.tls import load_shuffle_tls

                shuffle_tls = load_shuffle_tls(tls_config)
                nat.set_flight_client_tls(
                    shuffle_tls.ca_pem,
                    shuffle_tls.server_name,
                    shuffle_tls.client_cert_pem,
                    shuffle_tls.client_key_pem,
                )

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
                    tls=shuffle_tls,
                )
            else:
                self.session = ShuffleSession(
                    credits,
                    advertise_host=advertise_host,
                    token=shuffle_token,
                    shm=shm,
                    tls=shuffle_tls,
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

        def set_grant(self, credits: int, engine_config: str) -> None:
            """Re-grant this worker for the query about to borrow it.

            A worker is built from the grant of whichever query *spawned* it: its credit
            window (1 credit = 1 in-flight batch) and the `EngineConfig` every local
            `execute_plan` runs under (memory budget, morsel size, parallelism). A reused
            session fleet therefore ran every later query under the first query's grant —
            so a global `COUNT(*)` (granted 1 credit and a 1 MB budget, which is all it
            needs) left the 8-node join after it shuffling one batch at a time against a
            1 MB budget: measured 0.6 s -> 3.2 s on TPC-H sf10, same plan, same data.

            Re-granting in place is what makes the warm fleet *correct* to reuse. It is two
            attribute writes on the worker — no respawn, so it costs neither the placement
            group nor the Flight server bind, and it cannot fail to place (a fleet asks for
            the cluster's whole CPU capacity, so a respawn is exactly what one cannot do
            reliably while the fleet it replaces is still being reaped).
            """
            self.session.set_credits(credits)
            self._engine_config = engine_config

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
            nat = engine()
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

        def replicate_buckets(self, primary_addr, src, n_buckets, stage=0, epoch=0) -> str:
            """Pull every bucket source `src` published on `primary_addr` and re-publish it
            here, under the *same* ticket — a second copy of that mapper's shuffle output.

            This is the core of recompute-free recovery. A reducer whose mapper is gone
            fetches the byte-identical copy from this worker instead of forcing the driver
            to re-read the source partition and re-run the map. It is also the spot-drain
            hand-off: a survivor pulls a doomed-but-still-alive worker's buckets, so a
            preemption notice costs one copy of (small, pre-aggregated) partial state
            rather than a full recompute of it.

            Returns this worker's address once **every** bucket is registered, and raises if
            any fetch failed. That all-or-nothing ack is load-bearing: an unregistered ticket
            reads back as an *empty* bucket rather than an error, so a reducer allowed to
            fall back to a half-filled replica would silently drop that mapper's rows. The
            driver therefore only advertises a replica whose ack it has in hand.
            """
            for r in range(n_buckets):
                ticket = _ticket(stage, src, r, epoch)
                self.session.publish(ticket, self.session.fetch(primary_addr, ticket))
            return self.session.addr

        def reduce_fetch(self, gk, aj, mapper_addrs, reducer_id, epochs=None, replicas=None):
            # Fetch every mapper's partial *concurrently* and fold them into one running
            # merged state in Rust (bounded by the session's fan-in), instead of one
            # blocking round-trip per mapper. The reducer holds one merged partial (sized
            # by the group count) plus at most `fan_in` in-flight fetches — memory
            # independent of the mapper count, so the shuffle scales to a wide cluster.
            # `combine` is associative, so the concurrent fold equals a serial one.
            # `epochs` maps a recomputed source to its current epoch (default 0) so the
            # fetch resolves the fresh partition, never a lost worker's stale one.
            # `replicas[src]` are the peers holding a copy of that mapper's bucket: the
            # gather falls over to one when the primary is unreachable, so a lost worker
            # costs a re-fetch and no recovery round at all. Only when *every* copy is gone
            # is the source reported retryable, and the driver recomputes + retries; a fatal
            # fault propagates and fails the query fast.
            epochs = epochs or {}
            sources = [
                (addr, _ticket(0, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(mapper_addrs)
            ]
            payload, unreachable = self.session.gather_combine(
                gk, aj, sources, finalize=True, replicas=replicas
            )
            if unreachable:
                return ("retry", unreachable)
            return ("ok", payload)

        def reduce_fetch_publish(
            self, gk, aj, mapper_addrs, reducer_id, epochs=None, replicas=None
        ):
            """Like `reduce_fetch`, but PUBLISH the finalized result on this worker's own
            Flight server and return only a `(addr, ticket, rows, schema)` handle.

            This keeps the stage's output partitioned on the workers — the adaptive
            executor scans it in place for the next stage instead of pulling every
            reducer's result back to the driver. The status protocol is unchanged
            (`"retry"` on a lost mapper), so it composes with the recovery loop.
            """
            status, payload = self.reduce_fetch(gk, aj, mapper_addrs, reducer_id, epochs, replicas)
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
            nat = engine()
            running = _combine_sources(self.session, gk, aj, sources)
            return None if running is None else nat.combine_finalize(gk, aj, [running])

        def map_publish_raw(
            self, sub_ir, key_names, partition, n_buckets, stage, src=None, epoch=0
        ) -> str:
            nat = engine()
            from batcher.dist.executors.partition_io import read_partition_descriptor

            # `src` overrides the mapper id on recompute (a survivor regenerates a
            # lost worker's side). `epoch` rises on each recompute so the fresh partition
            # is published under a new ticket and a zombie worker's stale one can never be
            # read — the same fence the aggregate's `map_publish` carries. Publish EVERY
            # bucket, empty included, so a reducer's failed fetch means a lost worker, not
            # an empty bucket.
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
                self.session.publish(
                    _ticket(stage, src, r, epoch), buckets[r] if r < len(buckets) else []
                )
            return self.session.addr

        def map_publish_join(self, left, right, n_buckets, src=None, epoch=0) -> str:
            """Publish BOTH sides of one join source partition in a single actor call.

            `left`/`right` are each `(sub_ir, key_names, partition)`.

            A join mapper must land its left (stage 0) *and* right (stage 1) buckets before
            any reducer fetches them. Issuing the two `map_publish_raw` calls separately and
            awaiting only the second one silently swallowed a left-side failure: the barrier
            saw the right side's address come back, declared the mapper healthy, and then every
            reducer failed to fetch its stage-0 ticket — so a deterministic application error
            surfaced, three retries later, as a phantom "unreachable worker". One call means
            one `ObjectRef`, so either side's exception propagates to the barrier as the error
            it actually is.
            """
            self.map_publish_raw(left[0], left[1], left[2], n_buckets, 0, src, epoch)
            return self.map_publish_raw(right[0], right[1], right[2], n_buckets, 1, src, epoch)

        def reduce_window(self, win_ir, addrs, reducer_id, epochs=None, replicas=None):
            nat = engine()
            # A window partition is computed whole, so this reducer holds all of its
            # bucket's raw rows (memory = the bucket, which shrinks as workers grow).
            # Fetch every mapper concurrently, falling over to a replica of any mapper that
            # is gone, and tracking the sources whose every copy is lost so the driver
            # recomputes + retries.
            epochs = epochs or {}
            sources = [
                (addr, _ticket(0, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            rows, unreachable = self.session.gather_concat(sources, replicas=replicas)
            if unreachable:
                return ("retry", unreachable)
            if not rows:
                return ("ok", None)
            return ("ok", nat.execute_plan(win_ir, [rows], self._engine_config))

        def reduce_join(
            self,
            join_ir,
            addrs,
            reducer_id,
            left_schema,
            right_schema,
            gk=None,
            aj=None,
            finalize=True,
            epochs=None,
            replicas=None,
        ):
            import functools

            nat = engine()
            # A join needs its bucket's whole left and right side, so it holds them
            # both (memory = the bucket's data, which shrinks as workers grow). Fetch
            # every mapper's left (stage 0) and right (stage 1) side concurrently, falling
            # over to a replica of any mapper that is gone and tracking the sources whose
            # every copy is lost, so the driver can recompute and retry. Both sides of a
            # source live on the same worker, so one `replicas` list covers them both.
            epochs = epochs or {}
            left_sources = [
                (addr, _ticket(0, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            right_sources = [
                (addr, _ticket(1, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            # The two sides are independent streams from the same peers, so fetch them at
            # the same time rather than draining the left before dialing the right. Each
            # `gather_concat` drops the GIL and blocks on the shared Rust runtime, so two
            # Python threads genuinely overlap the transfers; serially, the right side's
            # round-trips were dead time on a link the left side had already finished with.
            gather = functools.partial(self.session.gather_concat, replicas=replicas)
            with futures.ThreadPoolExecutor(max_workers=2) as pool:
                l_fut = pool.submit(gather, left_sources)
                r_fut = pool.submit(gather, right_sources)
                left, lost_left = l_fut.result()
                right, lost_right = r_fut.result()
            unreachable = sorted(set(lost_left) | set(lost_right))
            if unreachable:
                return ("retry", unreachable)
            if not left and not right:
                return ("ok", None)
            # Schema-bearing empties so an outer join can null-extend the missing side.
            relations = [left or [left_schema], right or [right_schema]]
            if gk is not None:
                # Fused post-join aggregate (only a small bucket leaves the worker — the
                # full join never reaches the driver). `finalize=True` when group keys ⊇
                # join key (each group is whole in this bucket, so finalize here; driver
                # concatenates disjoint groups); `finalize=False` otherwise (a group spans
                # buckets, so emit PARTIAL state and let the driver `combine_finalize`).
                #
                # Run the join and the fold in ONE native call: the join's output is the
                # reducer's largest object by far (TPC-H sf10: 3.75M rows / ~106 MB) and is
                # consumed immediately by the aggregate, so handing it back to Python only
                # to pass it straight into the next FFI call built a Python mirror of it for
                # nothing. Same plan, same fold — a bit-identical partial.
                out = nat.execute_plan_aggregated(
                    join_ir, relations, gk, aj, self._engine_config, finalize
                )
                return ("ok", [out] if out is not None else [])
            joined = nat.execute_plan(join_ir, relations, self._engine_config)
            return ("ok", joined)

        def reduce_join_publish(
            self,
            join_ir,
            addrs,
            reducer_id,
            left_schema,
            right_schema,
            epochs=None,
            replicas=None,
        ):
            """Like `reduce_join`, but PUBLISH this bucket's joined rows on the worker's own
            Flight server and hand back only an `(addr, ticket, rows, schema)` handle.

            The join analogue of `reduce_fetch_publish`, and what lets a *multi-join* query
            run without a driver round-trip per join. Previously every intermediate join
            collected its whole output to the driver, which then re-partitioned it back out
            to the workers for the next join — so a 6-table query moved every intermediate
            twice through one process (TPC-H q5 sf1 pushed 12M then 18M rows through the
            driver for a 5-row answer, and at sf10 it OOM-killed the node). Keeping the
            bucket where it was computed means the next stage's mappers fetch it
            shared-nothing, straight from the holding actor.

            The status protocol is `reduce_join`'s, so it composes with the same recovery
            loop: `("retry", unreachable)` on a lost mapper, `("ok", None)` for an empty
            bucket (nothing to publish).
            """
            status, batches = self.reduce_join(
                join_ir,
                addrs,
                reducer_id,
                left_schema,
                right_schema,
                None,
                None,
                True,
                epochs,
                replicas,
            )
            if status != "ok":
                return (status, batches)  # retry: `batches` carries the unreachable mappers
            # An EMPTY bucket publishes nothing and yields no handle. It contributes no rows
            # to the relation, and a zero-row partition is not merely pointless to serve: the
            # Flight encoder emits no data message for it, so a reader blocks until the fetch
            # idle timeout (60s) and then reports the perfectly healthy worker holding it as
            # unreachable. (`reduce_fetch_publish` returns `None` for an empty bucket for the
            # same reason.) The join's probe schema — not an empty bucket — carries the schema.
            batches = [b for b in (batches or []) if b.num_rows > 0]
            if not batches:
                return ("ok", None)
            ticket = _ticket(_RESULT_STAGE, self.id, reducer_id)
            self.session.publish(ticket, batches)
            rows = sum(b.num_rows for b in batches)
            return ("ok", (self.session.addr, ticket, rows, batches[0].schema))

        def local_topn(self, plan_ir, partition):
            """Run `plan_ir` (the map prefix + sort + limit) on this worker's own split and
            return its local top-N rows — no shuffle. For a top-N (`ORDER BY ... LIMIT k`)
            the global answer is the top-N of the union of per-worker top-Ns, so each
            worker reads its split, applies the single-node top-N heap, and ships only k
            rows; the driver merges. Reads the split directly (never on the driver)."""
            nat = engine()
            from batcher.dist.executors.partition_io import read_partition_descriptor

            return nat.execute_plan(
                plan_ir, [read_partition_descriptor(partition)], self._engine_config
            )

        def sample_quantiles(self, map_ir, key_name, probs, partition):
            """Sample this split's leading-key distribution as a small quantile grid.

            Each worker samples its *own* split — the input is never read on the
            driver. The grid (a few floats) plus the row count go back; the driver
            merges them into range boundaries. Stateless w.r.t. the shuffle session.
            """
            nat = engine()
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
            self,
            map_ir,
            key_name,
            boundaries,
            n_buckets,
            nulls_first,
            desc,
            partition,
            src=None,
            epoch=0,
        ) -> str:
            """Range-partition this split's rows by `boundaries` and publish each bucket.

            Bucket b holds keys in `(boundaries[b-1], boundaries[b]]` so the buckets
            are globally ordered; equal keys never span a boundary (boundaries are
            deduplicated on the driver and `searchsorted(side="right")` is used), so
            the per-bucket sorts concatenate to a globally sorted result. Nulls go to
            the bucket that lands at the correct end of the *final* (post-`desc`)
            concatenation, so they sort first/last exactly as single-node would.

            `epoch` rises on each recompute, so a regenerated bucket is published under a
            fresh ticket and a zombie worker's stale one can never be read.
            """
            nat = engine()
            from batcher.dist.executors.partition_io import bucketize, read_partition_descriptor

            src = self.id if src is None else src
            rows = nat.execute_plan(
                map_ir, [read_partition_descriptor(partition)], self._engine_config
            )
            buckets = bucketize(rows, key_name, boundaries, n_buckets, nulls_first, desc)
            # Publish EVERY bucket (empty included) so a reducer's failed fetch means
            # a lost worker, not an empty bucket — the recompute loop's clean signal.
            for r in range(n_buckets):
                self.session.publish(_ticket(0, src, r, epoch), buckets[r])
            return self.session.addr

        def sort_reduce(self, sort_ir, addrs, reducer_id, epochs=None, replicas=None):
            nat = engine()
            # This reducer owns one contiguous key range; fetch its bucket from every
            # mapper concurrently, concatenate, and sort by all keys — the bucket is
            # globally ordered relative to the others, so a final concat needs no merge.
            # A mapper that is gone is served from a replica; only a source whose every
            # copy is lost is reported retryable for the driver to recompute.
            epochs = epochs or {}
            sources = [
                (addr, _ticket(0, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            rows, unreachable = self.session.gather_concat(sources, replicas=replicas)
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
        fleet_actor_options,
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
    connections_per_peer = int(dc.flight_connections_per_peer or 0)
    compression = {"none": 0, "lz4": 1, "zstd": 2}.get(dc.flight_compression, 1)
    # Same-node shared-memory transfer, decided on the driver and shipped to every
    # worker (which can't see the driver's config_context). Gated on the native probe so
    # it is never enabled where no shared directory exists (it would just churn fallbacks).
    nat = engine()
    shm = bool(dc.shared_memory_transfer) and nat.shm_available()
    # Each worker watches for a preemption notice so the driver can migrate its output
    # before reclamation (proactive, not reactive). Engaged under the spot profile —
    # which a spot deployment gets automatically: `config.profiles.detect_spot_environment`
    # auto-upgrades `resilience` to "spot" on a detected spot node, so a fresh user is
    # protected without setting it by hand.
    preemption = dc.resilience == "spot"
    pg = create_worker_placement(workers, current_envelope())
    # Resolve the fleet-uniform actor options once (they read the live topology), then vary
    # only the per-bundle index — so spawning W workers is O(W), not O(W x nodes).
    opts = fleet_actor_options(pg, workers)
    actors = [
        _FlightWorker.options(**opts[i]).remote(
            i,
            credits,
            cfg_json,
            adaptive,
            token,
            idle_ms,
            keepalive_ms,
            connections_per_peer,
            compression,
            plan_id,
            shm,
            preemption,
            dc.tls,
        )
        for i in range(workers)
    ]
    return actors, pg
