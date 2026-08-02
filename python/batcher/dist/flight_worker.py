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

import contextvars
import logging
from concurrent import futures
from typing import TYPE_CHECKING

from batcher._internal.errors import ConfigError
from batcher._internal.hardware.cpu import available_cpu_count
from batcher._internal.logging import get_logger, log_kv, note_suppressed
from batcher._internal.native import engine
from batcher.carbonite.transfer import ShuffleTicket
from batcher.carbonite.transfer.codec import resolve_codec
from batcher.kyber.cost.fabric import measured_fabric_gbps

if TYPE_CHECKING:
    from batcher.config.config import ShuffleTlsConfig

__all__ = [
    "_FlightWorker",
    "_combine_sources",
    "_ticket",
    "current_plan_id",
    "new_plan_id",
    "set_current_plan_id",
    "spawn_flight_workers",
]

# The shuffle plan id for the query in flight in THIS context (driver thread or worker
# task). Set once per query on the driver and re-asserted per call on every worker, so
# all of a query's tickets carry the same id. Fences a query's published partitions from
# another query's — and from a crashed prior query's leftovers when a fleet actor is
# reused.
#
# A `ContextVar`, not a module global: several pipelines share one warm session fleet
# (`dist.fleet`) precisely so they need not each reserve the cluster's whole CPU
# capacity, and they run concurrently in separate driver threads. Under a plain global
# the later query's `set_current_plan_id` retroactively changed the tickets the earlier
# query's driver was still building. A fresh thread starts from the default rather than
# inheriting, which is the isolation we want here.
_DEFAULT_PLAN_ID = 1
_current_plan_id: contextvars.ContextVar[int] = contextvars.ContextVar(
    "batcher_shuffle_plan_id", default=_DEFAULT_PLAN_ID
)
# Default ticket stage for a stage's *finalized* result (kept on the actor). Every publish
# should pass its own `result_stage` from `fleet.plan_id.next_result_stage` — two results
# sharing one stage are byte-identical tickets on the same worker, and the second overwrites
# the first. This constant remains only so an older caller keeps working.
_RESULT_STAGE = 100
# Sub-buckets a memory-bounded join reduce re-partitions each staged side into on disk, so
# it joins one sub-bucket pair at a time rather than the whole bucket. Matches the
# single-node spilling join's default fan-out (`execute_spilling_join`).
_JOIN_REDUCE_SUBBUCKETS = 16


def _reduce_spill_opts(engine_config: str) -> tuple[int, str | None, str | None]:
    """`(memory_budget_bytes, spill_dir, spill_compression)` from a worker's engine config.

    A positive budget routes the flight aggregate reduce through its memory-bounded path
    (`_bounded_reduce`): stage each mapper's partial to disk, then merge in memory if the
    bucket fits the budget or grace-partition out of core if not — so a high-cardinality
    bucket never assembles whole in RAM. `(0, ...)` means unbounded (the in-memory fold).
    """
    if not engine_config:
        return (0, None, None)
    import json

    cfg = json.loads(engine_config)
    return (
        int(cfg.get("memory_budget_bytes", 0) or 0),
        cfg.get("spill_dir"),
        cfg.get("spill_compression"),
    )


def _reduce_work_dir(prefix: str, spill_dir: str | None) -> str:
    """A scratch directory for a bounded reduce, on the fastest disk this worker actually has.

    The configured `spill_dir` wins, because an operator who named one has already decided.
    With none, this prefers the node's measured local volume over a bare tempdir: on a GPU
    node the tempdir is an overlay on the container root — commonly under 100 GB and shared
    with the image — while the terabytes of NVMe the node ships with sit under a
    provider-specific mount. A reduce that spills to the overlay fails with `ENOSPC` beside
    unused storage, and the failure reads as an undersized query rather than a misplaced
    directory.

    Args:
        prefix: The temporary directory's name prefix.
        spill_dir: The configured scratch root, or `None`.

    Returns:
        A fresh directory the caller owns and removes.
    """
    import tempfile

    from batcher._internal.site import local_scratch_root

    return tempfile.mkdtemp(prefix=prefix, dir=spill_dir or local_scratch_root() or None)


def new_plan_id() -> int:
    """A fresh, process-unique-enough shuffle plan id (63-bit, fits the ticket field).

    Generated once per query at fleet spawn. Two queries — or a crashed query and its
    replacement reusing the same fleet actor — get different ids, so a stale partition
    left at the same stage/src/dst/epoch under the old id can never be fetched (the
    cross-query / cross-restart analogue of the per-recompute `epoch` fence)."""
    import uuid

    return uuid.uuid4().int & ((1 << 63) - 1)


def set_current_plan_id(plan_id: int) -> None:
    """Set this context's shuffle plan id so `_ticket` fences this query.

    Called once per query on the driver (which builds the tree-combine tickets), and
    re-asserted per call inside every `_FlightWorker` (which builds publish/fetch
    tickets) from the id the driver sent, so the whole shuffle agrees. Scoped to the
    calling context, so a second pipeline sharing the same fleet cannot retarget the
    tickets this one is still building."""
    _current_plan_id.set(plan_id)


def current_plan_id() -> int:
    """The shuffle plan id fencing the query in flight in this context."""
    return _current_plan_id.get()


def _use_plan(plan_id: int | None) -> None:
    """Adopt the driver-sent plan id for the duration of this actor call.

    A fleet actor is shared by every query in the session — including, once several
    pipelines run at once, by two queries interleaving calls on it. So the fence cannot
    live in the actor's own state the way it did when a worker served one query at a
    time: each call carries the id of the query making it. Ray runs an actor's tasks one
    at a time, so setting it at method entry is sufficient. `None` means a caller that
    predates the plumbing — keep the actor's spawn-time id.
    """
    if plan_id is not None:
        set_current_plan_id(plan_id)


_worker_log = get_logger("dist.shuffle")


def _ticket(stage: int, src: int, dst: int, epoch: int = 0) -> ShuffleTicket:
    """A shuffle ticket for this query: `plan/stage/src(mapper)/dst(reducer)/epoch`.

    `plan` is the per-query id (`set_current_plan_id`) fencing this query from another.
    `epoch` (default 0) fences a recomputed partition from the stale one a lost worker
    published: a fresh recompute bumps the source's epoch, so the partition is
    published *and* fetched under a new ticket and a zombie worker's old-epoch partial
    can never be read — defense in depth atop the address-redirect the recovery loop
    already does.
    """
    return ShuffleTicket(_current_plan_id.get(), stage, src, dst, epoch)


def _lost(unreachable: list[tuple[int, str]]) -> list[int]:
    """The source indices from a gather's `(index, fault)` pairs, with the faults logged.

    The driver's recovery loop speaks in indices — it recomputes a source and retries — so
    that is what a reducer returns. But an index on its own is what let a deterministic bug
    masquerade as worker loss: a ticket collision here surfaced as
    `shuffle did not recover after 3 attempts`, three frames and one wrong noun from its
    cause. The transport's own words for *why* a source was unreachable go to the log, on
    the worker that saw them, before the index is all that is left.
    """
    for src, why in unreachable:
        log_kv(_worker_log, logging.WARNING, "shuffle source unreachable", source=src, fault=why)
    return sorted({src for src, _ in unreachable})


def _combine_sources(session, gk, aj, sources, replicas=None):
    """Fetch each `(addr, ticket)` concurrently and merge into one running partial.

    The bounded-memory merge: hold one combined partial, never the whole source
    list. `sources` is at most `fan_in` long in the tree shuffle, so a combiner
    node's fan-in (and memory) is bounded regardless of cluster size. A lost source
    surfaces as a `RetryableShuffleError` (the tree node's task fails → driver
    recompute), preserving the propagate-on-fault contract of the serial path.

    `replicas[i]` are fallback addresses holding a copy of `sources[i]` — positional,
    so it must be built alongside `sources` rather than indexed by worker id. A lost
    source served from a replica costs a re-fetch instead of a recompute round; only
    when every copy is gone does it raise.
    """
    nat = engine()
    running, unreachable = session.gather_combine(
        gk, aj, list(sources), finalize=False, replicas=replicas
    )
    if unreachable:
        raise nat.RetryableShuffleError(f"combiner lost sources {_lost(unreachable)}")
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
            port_range: tuple[int, int] | None = None,
            credit_ceiling: int = 0,
            prefer_fabric: bool = False,
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
            # The cap on this worker's *published* shuffle output — the one large
            # footprint Carbonite's buffer pool cannot see, since a bucket is held for a
            # reducer rather than reserved by anyone. Above it the store spills its
            # largest buckets to local disk and reads them back on fetch, which is
            # result-preserving. Set before the server is created: each store captures the
            # cap at construction so its bound cannot shift mid-query.
            from batcher.carbonite.policies import shuffle_store_cap
            from batcher.config import active_config

            nat.set_flight_transport_config(
                idle_timeout_ms,
                keepalive_ms,
                connections_per_peer,
                compression,
                shuffle_store_cap(active_config()),
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
            #
            # `BATCHER_ADVERTISE_HOST` overrides it for a topology where Ray's view of
            # the node is not the address peers must dial: a multi-homed host whose
            # shuffle traffic belongs on a second NIC, or a NAT'd / VPC-peered network
            # where the reachable address is not the local one. It is read *here*, in the
            # worker, rather than shipped from the driver, because the right value differs
            # per node — a single driver-side setting would hand every worker one host and
            # be wrong everywhere but one. Set it per node (pod spec, node env) and each
            # worker advertises its own address.
            #
            # `prefer_fabric` is the same fix expressed once for the cluster instead of once
            # per node: each worker resolves *its own* fabric interface address, so one
            # setting covers a fleet whose nodes each need a different value. It is shipped
            # from the driver rather than read from config here, because a Ray actor cannot
            # see the driver's `config_context`. A node with no IPoIB address configured
            # finds nothing and keeps its Ray address, so a partially-configured fleet
            # degrades one node at a time instead of advertising an address nobody can dial.
            import os

            from batcher._internal.hardware.fabric import fabric_interface_address

            advertise_host = os.environ.get("BATCHER_ADVERTISE_HOST") or ""
            if not advertise_host and prefer_fabric:
                advertise_host = fabric_interface_address()
            advertise_host = advertise_host or ray.util.get_node_ip_address()
            shuffle_token = token or None
            # Opt-in AIMD adaptive credits: the window adjusts to this worker's memory
            # pressure per fetch. Decided on the driver (the worker can't see the
            # driver's config_context) and passed in, so it reaches every worker.
            if adaptive:
                from batcher.carbonite.memory.pressure import PressureMonitor
                from batcher.carbonite.policies import AIMDFlowControl

                self.session = ShuffleSession(
                    credits,
                    # Warm-start AIMD at the driver's grant. `credits` is what Carbonite's
                    # `grant_credits(signature=)` just computed from Kyber's per-operator
                    # estimate *and* this shuffle's learned converged window — and under
                    # adaptive credits `_window()` reads the controller, never the session's
                    # static `credits`, so a bare controller silently discarded all of it and
                    # every channel re-climbed from `default_credits` (4) on every query.
                    # The ceiling comes from the driver, not from this process. A Ray
                    # actor sees neither the driver's `config_context` nor the metadata hub
                    # the learned row width is fit from, so every input `credit_ceiling`
                    # needs is wrong or missing here — and AIMD *grows toward* its ceiling,
                    # so re-deriving a wrong one is not an approximation, it is the window
                    # the controller settles at and the memory it buffers there. A
                    # wide-row shuffle (embeddings, blobs) was the case that mattered: its
                    # learned per-batch width is what holds the window under
                    # `credit_byte_budget`, and it is invisible from inside the worker.
                    flow_control=AIMDFlowControl(
                        initial_window=credits, ceiling=credit_ceiling or None
                    ),
                    pressure=PressureMonitor(),
                    advertise_host=advertise_host,
                    token=shuffle_token,
                    shm=shm,
                    tls=shuffle_tls,
                    port_range=port_range,
                )
            else:
                self.session = ShuffleSession(
                    credits,
                    advertise_host=advertise_host,
                    token=shuffle_token,
                    shm=shm,
                    tls=shuffle_tls,
                    port_range=port_range,
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

        def clear_plan(self, plan_id: int) -> None:
            """Evict every bucket this worker published for `plan_id`.

            The shuffle store is append-only until something evicts it, and until this was
            wired the *batch* path evicted nothing: `ShuffleSession.clear_plan` existed and
            was bound all the way through to Rust, with exactly one caller — a test. So a
            session-scoped fleet (`reuse_session_fleet`, on by default) accumulated every
            bucket of every stage of every query it ever served, until the node ran out of
            memory. Only the streaming pipeline released anything.
            """
            self.session.clear_plan(plan_id)

        def release_ticket(self, ticket: str) -> None:
            """Evict one published bucket, once the stage that reads it has finished.

            The finer-grained half of `clear_plan`: an adaptive query's stage `k` holds its
            buckets resident through stages `k+1..n` otherwise, which is the leak that
            exhausts memory *inside a single query* rather than across a session.
            """
            from batcher.carbonite.transfer.server import ShuffleTicket

            parts = [int(p) for p in str(ticket).split("/")]
            self.session.release(ShuffleTicket(*parts))

        def partition_count(self) -> int:
            """How many buckets this worker still holds.

            The leak oracle, and the reason this is a permanent method rather than a test
            helper: a bucket leak has no symptom until the node dies, so the only way to
            test for it is to ask a live worker what it is still holding.
            """
            return self.session.partition_count

        def map_publish(
            self,
            map_ir,
            gk,
            aj,
            partition,
            n_keys,
            n_reducers,
            src=None,
            epoch=0,
            plan_id=None,
            stage_base=0,
        ) -> str:
            _use_plan(plan_id)
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
                self.session.publish(_ticket(stage_base, src, r, epoch), bucket)
                # `nbytes`, deliberately, where the memory guards nearby use
                # `plan.types.retained_bytes`: this figure predicts what a reducer will
                # *pull over the wire*, and Arrow IPC writes only the rows a batch
                # addresses. A window's pinned parent costs this worker memory (which the
                # store's own cap governs) but costs the transfer nothing.
                self._bucket_bytes[r] = sum(b.nbytes for b in bucket)
            return self.session.addr

        def replicate_buckets(
            self, primary_addr, src, n_buckets, stage=0, epoch=0, plan_id=None
        ) -> str:
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
            _use_plan(plan_id)
            for r in range(n_buckets):
                ticket = _ticket(stage, src, r, epoch)
                self.session.publish(ticket, self.session.fetch(primary_addr, ticket))
            return self.session.addr

        def reduce_fetch(
            self,
            gk,
            aj,
            mapper_addrs,
            reducer_id,
            epochs=None,
            replicas=None,
            plan_id=None,
            stage_base=0,
        ):
            _use_plan(plan_id)
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
                (addr, _ticket(stage_base, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(mapper_addrs)
            ]
            budget, _sdir, _codec = _reduce_spill_opts(self._engine_config)
            if budget > 0:
                # Bounded reduce: never assemble the whole bucket in RAM (a high-cardinality
                # bucket would OOM the in-memory fold). Stage each mapper's partial to disk
                # (fan-in-bounded) and merge in memory when it fits the envelope, else
                # grace-partition out of core — the flight arm of "spill, never crash".
                return self._bounded_reduce(gk, aj, sources, replicas)
            payload, unreachable = self.session.gather_combine(
                gk, aj, sources, finalize=True, replicas=replicas
            )
            if unreachable:
                return ("retry", _lost(unreachable))
            return ("ok", payload)

        def _bounded_reduce(self, gk, aj, sources, replicas):
            """Memory-bounded aggregate reduce: spill each mapper's partial to disk (never
            holding the whole assembled bucket), then merge in memory if it fits the budget
            or grace-partition out of core if not. Result-identical to `gather_combine`."""
            import os
            import shutil

            from batcher.dist.shuffle_io import read_ipc

            nat = engine()
            budget, sdir, codec = _reduce_spill_opts(self._engine_config)
            work = _reduce_work_dir("bc_flight_reduce_", sdir)
            try:
                paths, unreachable = self.session.gather_to_files(sources, work, replicas=replicas)
                if unreachable:
                    return ("retry", _lost(unreachable))
                if not paths:
                    return ("ok", None)
                # On-disk IPC is uncompressed here, so its size ≈ the in-memory partials: when
                # the whole bucket fits the envelope, fold it in memory (bounded, one file at a
                # time); otherwise grace-partition the merge out of core.
                on_disk = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
                if on_disk <= budget:
                    running = None
                    for p in paths:
                        batch = read_ipc(p)
                        if not batch:
                            continue
                        merged = batch if running is None else [running, *batch]
                        running = nat.combine(gk, aj, merged)
                    result = (
                        nat.combine_finalize(gk, aj, [running]) if running is not None else None
                    )
                else:
                    result = nat.combine_finalize_spilling(gk, aj, paths, budget, work, codec)
                return ("ok", result if (result is not None and result.num_rows) else None)
            finally:
                shutil.rmtree(work, ignore_errors=True)

        def reduce_fetch_publish(
            self,
            gk,
            aj,
            mapper_addrs,
            reducer_id,
            epochs=None,
            replicas=None,
            plan_id=None,
            result_stage=_RESULT_STAGE,
            stage_base=0,
        ):
            """Like `reduce_fetch`, but PUBLISH the finalized result on this worker's own
            Flight server and return only a `(addr, ticket, rows, schema)` handle.

            This keeps the stage's output partitioned on the workers — the adaptive
            executor scans it in place for the next stage instead of pulling every
            reducer's result back to the driver. The status protocol is unchanged
            (`"retry"` on a lost mapper), so it composes with the recovery loop.
            """
            _use_plan(plan_id)
            status, payload = self.reduce_fetch(
                gk, aj, mapper_addrs, reducer_id, epochs, replicas, None, stage_base
            )
            if status != "ok" or payload is None:
                return (status, payload)  # retry, or an empty bucket (no handle)
            ticket = _ticket(result_stage, self.id, reducer_id)
            self.session.publish(ticket, [payload])
            return ("ok", (self.session.addr, ticket, payload.num_rows, payload.schema))

        def combine_publish(self, gk, aj, sources, out_ticket, replicas=None):
            # One interior node of the combiner tree: merge <= fan_in upstream
            # partials and republish the result for the next level to fetch.
            # `replicas` is positional over `sources` (see `_combine_sources`); it is
            # populated only at the leaf level, since a combiner's own output is
            # published on one node and never replicated.
            running = _combine_sources(self.session, gk, aj, sources, replicas)
            self.session.publish(out_ticket, [running] if running is not None else [])
            return self.session.addr

        def combine_finalize_fetch(self, gk, aj, sources, replicas=None):
            # The tree root for one bucket: merge the last <= fan_in partials and
            # finalize to output rows.
            nat = engine()
            running = _combine_sources(self.session, gk, aj, sources, replicas)
            return None if running is None else nat.combine_finalize(gk, aj, [running])

        def map_publish_raw(
            self, sub_ir, key_names, partition, n_buckets, stage, src=None, epoch=0, plan_id=None
        ) -> str:
            _use_plan(plan_id)
            nat = engine()
            from batcher.dist.executors.partition_io import (
                iter_partition_descriptor,
                streaming_map_buckets,
            )

            # `src` overrides the mapper id on recompute (a survivor regenerates a
            # lost worker's side). `epoch` rises on each recompute so the fresh partition
            # is published under a new ticket and a zombie worker's stale one can never be
            # read — the same fence the aggregate's `map_publish` carries. Publish EVERY
            # bucket, empty included, so a reducer's failed fetch means a lost worker, not
            # an empty bucket.
            src = self.id if src is None else src
            # Stream the partition through the map prefix and the hash a chunk at a time, as
            # the aggregate's `map_publish` does. Reading it whole held three things at once —
            # the partition, its entire mapped output, and the second full copy
            # `partition_batches` gathers into — and this is the path the memory envelope does
            # not cover: `memory_budget_bytes` bounds allocations inside `execute_plan`, not
            # what the worker keeps afterwards. That gap is what OOM-kills a shuffle worker
            # (BENCHMARK_RESULTS.md, sf10 q5), and at sf100 it killed two of them on TPC-H q9.
            buckets = streaming_map_buckets(
                nat,
                sub_ir,
                key_names,
                iter_partition_descriptor(partition),
                n_buckets,
                self._engine_config,
            )
            for r in range(n_buckets):
                self.session.publish(
                    _ticket(stage, src, r, epoch), buckets[r] if r < len(buckets) else []
                )
                # `publish` is synchronous and moves the batches into the store's own
                # `Vec<RecordBatch>`, so the mapper's reference is redundant the moment it
                # returns. Dropping it here means the peak is one bucket past what the
                # store already holds, instead of every bucket until the last is sent.
                if r < len(buckets):
                    buckets[r] = []
            return self.session.addr

        def map_publish_join(
            self, left, right, n_buckets, src=None, epoch=0, plan_id=None, stage_base=0
        ) -> str:
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
            _use_plan(plan_id)
            self.map_publish_raw(
                left[0], left[1], left[2], n_buckets, stage_base, src, epoch, plan_id
            )
            return self.map_publish_raw(
                right[0], right[1], right[2], n_buckets, stage_base + 1, src, epoch, plan_id
            )

        def reduce_window(
            self, win_ir, addrs, reducer_id, epochs=None, replicas=None, plan_id=None, stage=0
        ):
            _use_plan(plan_id)
            nat = engine()
            # A window partition is computed whole, so this reducer holds all of its
            # bucket's raw rows (memory = the bucket, which shrinks as workers grow).
            # Fetch every mapper concurrently, falling over to a replica of any mapper that
            # is gone, and tracking the sources whose every copy is lost so the driver
            # recomputes + retries.
            epochs = epochs or {}
            sources = [
                (addr, _ticket(stage, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            rows, unreachable = self.session.gather_concat(sources, replicas=replicas)
            if unreachable:
                return ("retry", _lost(unreachable))
            if not rows:
                return ("ok", None)
            return ("ok", nat.execute_plan(win_ir, [rows], self._engine_config))

        def reduce_join(
            self,
            join_ir,
            addrs,
            reducer_id,
            left_empty,
            right_empty,
            gk=None,
            aj=None,
            finalize=True,
            epochs=None,
            replicas=None,
            plan_id=None,
            stage_base=0,
        ):
            import functools

            _use_plan(plan_id)
            nat = engine()
            # `left_empty`/`right_empty` are 0-row RecordBatches, not schemas — the driver's
            # `probe()` runs each side's sub-plan over an empty input and sends the batch it
            # gets back, so a bucket missing one side still has something schema-bearing to
            # null-extend from. They were once called `*_schema`, and the spilling branch
            # below duly handed them to a parameter typed `pa.Schema`: every multi-way join
            # whose reduce spilled died on `Schema must be an instance of pyarrow.Schema`,
            # and the recovery loop reported it as four unreachable workers.
            # A join needs its bucket's whole left and right side, so it holds them
            # both (memory = the bucket's data, which shrinks as workers grow). Fetch
            # every mapper's left (stage 0) and right (stage 1) side concurrently, falling
            # over to a replica of any mapper that is gone and tracking the sources whose
            # every copy is lost, so the driver can recompute and retry. Both sides of a
            # source live on the same worker, so one `replicas` list covers them both.
            epochs = epochs or {}
            left_sources = [
                (addr, _ticket(stage_base, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            right_sources = [
                (addr, _ticket(stage_base + 1, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            budget, _sdir, _codec = _reduce_spill_opts(self._engine_config)
            if budget > 0:
                # Bounded join reduce: never assemble both whole sides in RAM. A skewed or
                # high-cardinality bucket would OOM the in-memory gather + build (the flight
                # arm of the sf10 q5 worker death). Stage each side to disk (fan-in bounded)
                # and join co-partitioned sub-bucket pairs one at a time — symmetric with the
                # aggregate reduce's `_bounded_reduce`, the flight arm of "spill, never crash".
                return self._bounded_reduce_join(
                    join_ir,
                    left_sources,
                    right_sources,
                    left_empty,
                    right_empty,
                    gk,
                    aj,
                    finalize,
                    replicas,
                )
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
                return ("retry", _lost(unreachable))
            if not left and not right:
                return ("ok", None)
            # Schema-bearing empties so an outer join can null-extend the missing side.
            relations = [left or [left_empty], right or [right_empty]]
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

        def _bounded_reduce_join(
            self,
            join_ir,
            left_sources,
            right_sources,
            left_empty,
            right_empty,
            gk,
            aj,
            finalize,
            replicas,
        ):
            """Memory-bounded join reduce: stage each mapper's left and right bucket to disk
            (never holding the whole assembled bucket), then join co-partitioned sub-bucket
            pairs one pair at a time. Peak memory is one sub-bucket pair, not the whole
            (possibly skewed) bucket — result-identical to `gather_concat` + `execute_plan`.

            The join analogue of `_bounded_reduce`: it wires the two out-of-core primitives
            that already exist — `session.gather_to_files` (never assembles the bucket in RAM)
            and `reduce_join_paths_spilling` (re-partitions on disk, joins one pair at a
            time) — which the flight join path had left unconnected, so its reducer built the
            whole bucket in RAM regardless of the memory envelope."""
            import functools
            import json
            import os
            import shutil

            from batcher.dist.spill_breakers.join import reduce_join_paths_spilling

            nat = engine()
            budget, sdir, _codec = _reduce_spill_opts(self._engine_config)
            spec = json.loads(join_ir)
            left_keys = list(spec.get("left_keys", []))
            right_keys = list(spec.get("right_keys", []))
            work = _reduce_work_dir("bc_flight_joinreduce_", sdir)
            try:
                # Stage the two sides concurrently, each into its own subdir so the two
                # `gather_to_files` waves cannot collide on a ticket-named file, and each
                # dropping the GIL on the shared Rust runtime so the transfers overlap.
                left_dir = os.path.join(work, "L")
                right_dir = os.path.join(work, "R")
                os.mkdir(left_dir)
                os.mkdir(right_dir)
                l_gather = functools.partial(
                    self.session.gather_to_files, spill_dir=left_dir, replicas=replicas
                )
                r_gather = functools.partial(
                    self.session.gather_to_files, spill_dir=right_dir, replicas=replicas
                )
                with futures.ThreadPoolExecutor(max_workers=2) as pool:
                    l_fut = pool.submit(l_gather, left_sources)
                    r_fut = pool.submit(r_gather, right_sources)
                    left_paths, lost_left = l_fut.result()
                    right_paths, lost_right = r_fut.result()
                unreachable = sorted(set(lost_left) | set(lost_right))
                if unreachable:
                    return ("retry", _lost(unreachable))
                if not left_paths and not right_paths:
                    return ("ok", None)
                # Join in memory when the staged bucket fits the envelope, exactly as the
                # aggregate's `_bounded_reduce` folds in memory when its partials do. Without
                # this the bounded path grace-partitioned *every* bucket: fetch to disk, read
                # it back, re-partition it to disk, read that back, then join — three disk
                # passes for a bucket that fitted memory all along. On TPC-H sf100 q3 that is
                # a ten-second plateau in the middle of a twenty-three-second query, sitting at
                # one core per node with the network carrying 2 MB/s. Neither CPU-bound nor
                # network-bound: waiting on a disk round trip it did not need.
                #
                # Staging first and deciding after is deliberate, and is what the aggregate
                # does too — the size is not known until the fetch has happened, and it is the
                # fetch that has to stay bounded.
                from batcher.dist.shuffle_io import read_ipc

                on_disk = sum(
                    os.path.getsize(p) for p in (*left_paths, *right_paths) if os.path.exists(p)
                )
                if on_disk <= budget:
                    left = [b for p in left_paths for b in read_ipc(p)]
                    right = [b for p in right_paths for b in read_ipc(p)]
                    # Schema-bearing empties so an outer join still null-extends, as the
                    # unbounded path's `relations` does.
                    joined = nat.execute_plan(
                        join_ir,
                        [left or [left_empty], right or [right_empty]],
                        self._engine_config,
                    )
                else:
                    joined = reduce_join_paths_spilling(
                        join_ir,
                        left_keys,
                        right_keys,
                        left_paths,
                        right_paths,
                        work,
                        _JOIN_REDUCE_SUBBUCKETS,
                        self._engine_config,
                        left_empty.schema,
                        right_empty.schema,
                    )
                if gk is not None:
                    # Fused post-join aggregate: fold the bounded join output into partial
                    # state (finalize only when each group is whole in this bucket), so only
                    # the small aggregate — never the join output — leaves the worker.
                    # Bit-identical to `execute_plan_aggregated` (join → partial → finalize?).
                    if not joined:
                        return ("ok", [])
                    partial = nat.partial_aggregate(gk, aj, joined)
                    out = nat.combine_finalize(gk, aj, [partial]) if finalize else partial
                    return ("ok", [out] if out is not None else [])
                return ("ok", joined)
            finally:
                shutil.rmtree(work, ignore_errors=True)

        def reduce_join_publish(
            self,
            join_ir,
            addrs,
            reducer_id,
            left_empty,
            right_empty,
            epochs=None,
            replicas=None,
            plan_id=None,
            result_stage=_RESULT_STAGE,
            stage_base=0,
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
            _use_plan(plan_id)
            status, batches = self.reduce_join(
                join_ir,
                addrs,
                reducer_id,
                left_empty,
                right_empty,
                None,
                None,
                True,
                epochs,
                replicas,
                plan_id,
                stage_base,
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
            ticket = _ticket(result_stage, self.id, reducer_id)
            self.session.publish(ticket, batches)
            rows = sum(b.num_rows for b in batches)
            return ("ok", (self.session.addr, ticket, rows, batches[0].schema))

        def local_topn(self, plan_ir, partition, merge_ir=None):
            """Run `plan_ir` (the map prefix + sort + limit) on this worker's own split and
            return its local top-N rows — no shuffle. For a top-N (`ORDER BY ... LIMIT k`)
            the global answer is the top-N of the union of per-worker top-Ns, so each
            worker reads its split, applies the single-node top-N heap, and ships only k
            rows; the driver merges. Reads the split directly (never on the driver).

            With `merge_ir` (the same sort+limit over the projected output schema the driver
            merges with), the split is folded through the heap a **chunk at a time**, so peak
            memory is one chunk plus `k` rows instead of the whole split — the same fold the
            aggregate map side already uses. Reading the split whole made a worker hold ~125M
            rows at 1B scale just to pick 100: ~25 GB, an OOM-killed actor, and 130 s for a
            100-row answer.
            """
            nat = engine()
            from batcher.dist.executors.partition_io import (
                iter_partition_descriptor,
                read_partition_descriptor,
                streaming_topn,
            )

            if merge_ir is None:  # legacy call: materialize (kept so an older driver still works)
                return nat.execute_plan(
                    plan_ir, [read_partition_descriptor(partition)], self._engine_config
                )
            return streaming_topn(
                nat, plan_ir, merge_ir, iter_partition_descriptor(partition), self._engine_config
            )

        def sample_quantiles(self, map_ir, key_name, probs, partition):
            """Sample this split's leading-key distribution as a small quantile grid.

            Each worker samples its *own* split — the input is never read on the
            driver. The grid (a few floats) plus the row count go back; the driver
            merges them into range boundaries. Stateless w.r.t. the shuffle session.
            """
            nat = engine()
            from batcher.dist.executors.partition_io import (
                read_partition_descriptor,
                sample_key_grid,
            )

            rows = nat.execute_plan(
                map_ir, [read_partition_descriptor(partition)], self._engine_config
            )
            n = sum(b.num_rows for b in rows)
            if n == 0:
                return ([], 0)
            return (sample_key_grid(rows, key_name, list(probs)), n)

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
            plan_id=None,
            stage_base=0,
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
            _use_plan(plan_id)
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
                self.session.publish(_ticket(stage_base, src, r, epoch), buckets[r])
            return self.session.addr

        def sort_reduce(
            self,
            sort_ir,
            addrs,
            reducer_id,
            epochs=None,
            replicas=None,
            plan_id=None,
            stage_base=0,
        ):
            _use_plan(plan_id)
            nat = engine()
            # This reducer owns one contiguous key range; fetch its bucket from every
            # mapper concurrently, concatenate, and sort by all keys — the bucket is
            # globally ordered relative to the others, so a final concat needs no merge.
            # A mapper that is gone is served from a replica; only a source whose every
            # copy is lost is reported retryable for the driver to recompute.
            epochs = epochs or {}
            sources = [
                (addr, _ticket(stage_base, src, reducer_id, epochs.get(src, 0)))
                for src, addr in enumerate(addrs)
            ]
            rows, unreachable = self.session.gather_concat(sources, replicas=replicas)
            if unreachable:
                return ("retry", _lost(unreachable))
            if not rows:
                return ("ok", None)
            return ("ok", nat.execute_plan(sort_ir, [rows], self._engine_config))

except ImportError:  # pragma: no cover - ray optional
    _FlightWorker = None  # type: ignore


def _connections_per_peer(dc) -> int:
    """How many TCP flows a consumer opens to one peer, floored at the node's rail count.

    A striped fetch can only use as many paths as it has flows: four connections on an
    eight-rail node leave half the fabric unused however the routing hashes them, because
    there are not enough flows to hash. The configured value is a floor rather than a
    ceiling here — it was chosen against a NIC count, not against this node's rails — and a
    node with no readable rail map keeps exactly the configured number.
    """
    from batcher._internal.hardware.fabric.rails import rail_summary

    configured = int(dc.flight_connections_per_peer or 0)
    try:
        rails = int(rail_summary().get("loaded_rails", 0))
    except Exception as exc:  # a transport hint must never fail a worker's startup
        note_suppressed("dist", "read the node's rail count", exc)
        return configured
    return max(configured, rails)


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
    # all clients present the same secret. Env var overrides config.
    import os

    # An `env:`/`file:` reference is resolved here so an operator can mount the token as a
    # secret file instead of writing it into a config. It resolves on the *driver*, not per
    # worker, because every peer must present the SAME shared secret — shipping the
    # reference and resolving per node would silently split the fleet if one node's mount
    # differed. The token then travels to actors exactly as it does today.
    from batcher.io.credentials import resolve_secret

    token = (
        resolve_secret(
            os.environ.get("BATCHER_SHUFFLE_TOKEN") or dc.shuffle_token or "",
            what="shuffle_token",
        )
        or ""
    )
    # Flight transport timeouts decided on the driver and shipped to every worker
    # (which can't see the driver's config_context), in milliseconds for the native
    # setter. 0 keepalive = off.
    idle_ms = int(dc.flight_idle_timeout_s * 1000)
    keepalive_ms = int((dc.flight_keepalive_s or 0) * 1000)
    connections_per_peer = _connections_per_peer(dc)
    # `auto` decides the codec against the fabric this node actually has. A compressor that
    # cannot keep up with the wire is a ceiling below it, so the right answer on a 400 Gb/s
    # port is usually no compression at all and on a 10 Gb/s VM is the highest ratio available.
    # An explicitly named codec is never overruled by a measurement.
    compression = resolve_codec(
        dc.flight_compression, measured_fabric_gbps(), available_cpu_count()
    )
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
    # Shuffle listener port range, decided on the driver (the worker can't see the driver's
    # config_context) and shipped to every actor so the whole fleet binds inside the range
    # the operator opened in their firewall. Env var overrides config, matching the token.
    port_range = _shuffle_port_range(os.environ.get("BATCHER_SHUFFLE_PORT_RANGE")) or (
        tuple(dc.shuffle_port_range) if dc.shuffle_port_range else None
    )
    # The AIMD ceiling, computed once here where the driver's config and the metadata hub
    # are both visible, and shipped to every actor as a plain int. `workers` is the channel
    # width: in a hash shuffle every reducer fetches from every mapper. Only the adaptive
    # path reads it, so a static-credit fleet pays one cheap call and ignores the result.
    ceiling = 0
    if adaptive:
        from batcher.carbonite import ResourceManager

        ceiling = ResourceManager().credit_window_ceiling(channels=max(1, workers))

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
            port_range,
            ceiling,
            dc.prefer_fabric_interface,
        )
        for i in range(workers)
    ]
    return actors, pg


def _shuffle_port_range(raw: str | None) -> tuple[int, int] | None:
    """Parse a ``"40000-40100"`` port range, or None when unset.

    A malformed value is a deployment mistake that would otherwise degrade silently to an
    ephemeral port outside the operator's firewall rule — where the shuffle then hangs
    unreachable rather than failing. Raise instead."""
    if not raw or not raw.strip():
        return None
    lo, _, hi = raw.strip().partition("-")
    try:
        port_range = (int(lo), int(hi))
    except ValueError as e:
        raise ConfigError(
            f"BATCHER_SHUFFLE_PORT_RANGE must look like '40000-40100', got {raw!r}"
        ) from e
    if not (0 < port_range[0] <= port_range[1] <= 65535):
        raise ConfigError(
            f"BATCHER_SHUFFLE_PORT_RANGE must be an ascending range within 1-65535, got {raw!r}"
        )
    return port_range
