"""The shared bucket-reduce driver for every Flight shuffle (join, sort, window).

The three shuffles differ only in *what* a reducer computes and *how* a lost worker's
mapped output is republished. Everything else — hosting a reducer on a live worker,
speculating on stragglers, caching completed buckets across recovery rounds,
classifying a dead host, and driving the policy-bounded recovery loop — is identical,
and lived as three near-copies before it moved here.

The caller supplies two closures:

- ``remote_reduce(host, bucket)`` launches the bucket's reduce on `actors[host]` and
  returns its ``ObjectRef``. The task must be deterministic: a straggler backup on
  another worker has to produce a byte-identical result.
- ``republish(target, src)`` regenerates worker `src`'s mapped output onto the live
  worker `target`, from the on-disk source partition, and updates the address table
  the reducers read.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from batcher._internal import events

__all__ = ["gather_in_windows", "run_bucket_reduce"]


def gather_in_windows(launch: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """`ray.get` over `items`, with at most a window of them launched at once.

    The plain `ray.get([launch(i) for i in items])` shape is correct and unbounded: it
    materializes one `ObjectRef` per item on the driver and hands the scheduler every task
    at once. That is fine while the list is the worker count and is not while it is a
    product of two fan-outs — an aggregate's combiner tree launches
    `n_reducers x ceil(sources / shuffle_fan_in)` tasks per level, which at the reducer
    ceiling and a fan-in of 8 is six figures of simultaneously-pending tasks for one level.

    Results come back in `items` order, so a caller that zips them against its own task list
    is unaffected. `len(items) <= window` launches everything before the first wait, which is
    byte-identical to the un-windowed call.

    Args:
        launch: Starts one item's remote task and returns its `ObjectRef`.
        items: The work, in the order the results are wanted.
        workers: The stage's worker fan-out, which sizes the window.

    Returns:
        One result per item, in `items` order.
    """
    import ray

    window = _reduce_window(workers)
    if len(items) <= window:
        return list(ray.get([launch(item) for item in items]))
    out: list[Any] = []
    for start in range(0, len(items), window):
        chunk = items[start : start + window]
        out.extend(ray.get([launch(item) for item in chunk]))
    return out


def _reduce_window(workers: int) -> int:
    """How many bucket reduces may be outstanding at once.

    The reduce side's answer to the same question `map_barrier._pending_window` answers for
    the map side, and it is deliberately a different one. A map task is stateless and Ray may
    place it anywhere, so its window is sized by *cores*. A bucket reduce runs on the actor
    the bucket hashes to (``bucket % workers``), so no more than `workers` of them can run at
    all — everything past that is a task sitting in the scheduler's queue, which is exactly
    the flood the map barrier's window exists to prevent.

    `distributed.max_pending_tasks` pins it when the user set one; otherwise it is
    `pending_window_factor` (default 4) times the worker count, so a slot frees as soon as a
    reducer finishes and the pipeline never drains. Reusing those two settings keeps one
    vocabulary for "how deep may a stage queue" across both barriers.

    Args:
        workers: The shuffle's worker fan-out.

    Returns:
        The maximum outstanding reduces, at least 1.
    """
    from batcher.config import active_config

    cfg = active_config().distributed
    if cfg.max_pending_tasks > 0:
        return max(1, cfg.max_pending_tasks)
    return max(1, max(1, workers) * max(1, cfg.pending_window_factor))


def run_bucket_reduce(
    *,
    kind: str,
    n_buckets: int,
    workers: int,
    actors: list[Any],
    remote_reduce: Callable[[int, int], Any],
    republish: Callable[[int, int], None],
    dead: set[int] | None = None,
    mapper_addrs: list[str] | None = None,
    replicas: list[list[str]] | None = None,
    placement: Any = None,
) -> dict[int, Any]:
    """Reduce every shuffle bucket under recompute-on-worker-loss recovery.

    Returns `{bucket_index: payload}` — keyed by bucket so a caller that needs the
    ranges in key order (sort) gets it regardless of completion order. `dead` seeds
    the workers the map barrier already lost, so no reducer is hosted on an actor that
    is gone. `kind` names the shuffle in the exhaustion error.

    Args:
        kind: The shuffle's name ("join", "sort", "window"), used in error messages.
        n_buckets: How many shuffle buckets to reduce.
        workers: The worker count; worker indices are `range(workers)`.
        actors: The worker actor handles, indexed by worker.
        remote_reduce: Launches one bucket's reduce on a host, returning an `ObjectRef`.
        republish: Regenerates a lost worker's mapped output onto a live worker.
        dead: Workers already known to be gone.
        mapper_addrs: `mapper_addrs[src]` is the address source `src` published on, used
            with `replicas` to tell a lost *copy* from a lost *source*.
        replicas: `replicas[src]` are the peers holding a copy of that source's buckets.
            Supplying both this and `mapper_addrs` is what lets a dead worker that also
            hosted a reducer avoid an unnecessary recompute — see `_survives_on_a_peer`.
        placement: Where the map barrier put each source. Required once a shuffle maps more
            partitions than it has workers, because then "the source with the dead worker's
            id" names one arbitrary partition of the several that worker was holding, and
            the rest stay unreachable — which reads back as empty buckets, not as an error.
            Defaults to the one-source-per-worker identity.

    Returns:
        One entry per bucket, mapping the bucket index to the reducer's payload.
    """
    from batcher._internal.errors import ResourceError
    from batcher.carbonite.resilience import (
        ShuffleRecovery,
        SourcePlacement,
        gather_with_backups,
    )
    from batcher.dist.executors.ray_runtime import (
        blame_host_for_reduce_failure,
        draining_workers,
        recovery_policy,
        speculation_policy,
    )

    lost: set[int] = set(dead or ())
    # Where each source's latest mapped output lives. Identity until a recompute relocates a
    # source, after which the source id and its host are different numbers — and it is the
    # HOST that dies. The flat aggregate reduce keeps the same mapping for the same reason;
    # see `SourcePlacement` for what conflating the two costs.
    placement = SourcePlacement(workers) if placement is None else placement

    def _pick_live(avoid: set[int]) -> int:
        for i in range(workers):
            if i not in lost and i not in avoid:
                return i
        raise ResourceError(f"no surviving worker to recover the {kind} shuffle on")

    def _host_for(bucket: int, avoid: set[int]) -> int:
        # `bucket b → actor b % workers`, unless dead/avoided; `avoid` lets a straggler's
        # backup land on a different live worker than the slow original. The modulo is what
        # permits more buckets than workers (see `shuffle_partitions`); indexing `actors[b]`
        # directly made the bucket count the worker count by construction.
        host = bucket % workers
        return host if host not in lost and host not in avoid else _pick_live(avoid)

    def _survives_on_a_peer(src: int) -> bool:
        """Whether `src`'s published buckets still exist somewhere that is not lost.

        A dead worker is two losses at once: the *reducer* it was hosting, and the
        *mapped output* it was serving. Those are independent, and treating them as one
        is what made replication look like it did nothing on a small cluster. When
        `workers == n_buckets` the bucket-`b`-on-actor-`b` rule puts a reducer on every
        worker, so any kill also kills a reducer host — and the host branch below
        scheduled a recompute of that worker's sources unconditionally, even with a
        byte-identical copy sitting on a survivor. The reducer only has to be re-launched
        elsewhere; its gather then falls over to the replica by itself.

        Returns False when replication is off, so the unreplicated path is unchanged.
        """
        if not replicas or mapper_addrs is None or src >= len(replicas):
            return False
        # A replica hosted on a worker that is itself lost is no help. Workers are known
        # here by index and replicas by address, so map back through the addresses the
        # lost workers published on.
        gone = {
            mapper_addrs[s]
            for host in lost
            for s in placement.sources_on(host)
            if s < len(mapper_addrs)
        }
        return any(addr not in gone for addr in replicas[src])

    # A bucket reduce that returns "ok" consumed its complete input and is
    # deterministic, so cache it across recovery rounds (keyed by bucket index) and
    # never re-run it. Only pending buckets re-launch, so one lost mapper doesn't
    # re-reduce every surviving bucket — the amplification that hurt most on a
    # churning spot/autoscaling cluster.
    done: dict[int, Any] = {}

    def attempt() -> tuple[dict[int, Any], set[int]]:
        failed: set[int] = set()
        pending = [b for b in range(n_buckets) if b not in done]
        # Submit-ahead window, the reduce-side twin of `map_barrier`'s. A bucket is reduced
        # by exactly one worker, so launching every bucket at once queues
        # `n_buckets - workers` tasks in Ray's scheduler that cannot start — the "too many
        # pending tasks" anti-pattern the map stage already guards against, and one the
        # reduce could reach on its own: `max_shuffle_partitions` allows 2,048 buckets, and
        # sizing a big shuffle by its volume makes counts near that ceiling ordinary rather
        # than exotic. The window is a multiple of *workers* rather than of cores, because
        # workers is what bounds how many reduces can actually run.
        #
        # `len(pending) <= window` submits everything before the first wait, so an ordinary
        # query is byte-identical to the un-windowed behaviour — the same guarantee the map
        # barrier gives.
        window = _reduce_window(workers)
        for start in range(0, len(pending), window):
            _reduce_chunk(pending[start : start + window], failed)
        return done, failed

    def _reduce_chunk(chunk: list[int], failed: set[int]) -> None:
        """Launch and collect one window's worth of buckets, folding results into `done`.

        Collected via `gather_with_backups`: a degraded-but-alive bucket gets a backup on
        another live worker (deterministic => byte-identical), and a dead host is classified
        for recompute — so one slow node cannot stall the barrier. A chunk that discovers a
        dead host updates `lost` before the next chunk launches, so later windows are placed
        on survivors rather than re-discovering the loss one bucket at a time.
        """
        ref_host: dict[Any, int] = {}

        def _launch(bucket: int, avoid: set[int]) -> Any:
            host = _host_for(bucket, avoid)
            ref = remote_reduce(host, bucket)
            ref_host[ref] = host
            return ref

        refs = [_launch(b, set()) for b in chunk]

        def _relaunch(idx: int) -> Any:
            try:
                return _launch(chunk[idx], {ref_host[refs[idx]]})
            except ResourceError:
                return _launch(chunk[idx], set())

        def _on_failure(_idx: int, ref: Any, exc: Exception) -> tuple[str, int | None]:
            # Only a failure that means *lost data* is charged to a host. A deterministic one
            # re-raises here, because blaming a healthy worker for it marks the whole fleet
            # dead one recompute at a time — see `blame_host_for_reduce_failure`.
            return ("__dead__", blame_host_for_reduce_failure(exc, ref_host.get(ref)))

        def _doomed() -> frozenset[int]:
            """Slots whose host is being reclaimed, so a backup is certainly needed.

            A drain notice arrives *during* the barrier, which is precisely when it is
            worth acting on: waiting for the doomed reducer to also look slow spends the
            notice period, and then the loss costs a full recovery round instead of a
            duplicate that was already running elsewhere. `draining_workers` short-circuits
            to an empty set when the cluster reports nothing draining, so a healthy fleet
            pays one GCS read per poll rather than a round trip per worker.
            """
            hosts = draining_workers(actors, workers)
            if not hosts:
                return frozenset()
            return frozenset(
                slot
                for slot, ref in enumerate(refs)
                if ref_host.get(ref) in hosts  # the host it is running on, not the bucket
            )

        gathered = gather_with_backups(
            refs,
            _relaunch,
            speculation_policy(),
            on_failure=_on_failure,
            doomed_slots=_doomed,
            stage="shuffle.reduce",
        )
        for bucket, (status, payload) in zip(chunk, gathered, strict=True):
            if status == "ok":
                done[bucket] = payload  # complete + deterministic -> cache, never re-run
            elif status == "__dead__":
                if payload is not None:
                    if payload not in lost:
                        events.publish(
                            events.RECOVERY,
                            name=kind,
                            event="worker_lost",
                            shuffle=kind,
                            worker=payload,
                            dead_total=len(lost) + 1,
                        )
                    lost.add(payload)  # host died — the reducer it hosted with it
                    # `payload` is a HOST id; `failed` carries SOURCE ids (the other branch
                    # reports unreachable sources). Translate through the current placement so
                    # a relocated source is recomputed and an unrelated one is not — on a
                    # clean run this is exactly `{payload}`.
                    #
                    # Everything the host held goes in, *including* sources a replica makes
                    # cheap to recover. Whether to regenerate is `recompute`'s decision, not
                    # this branch's. Filtering here instead looks equivalent and is not:
                    # `failed` is also the signal that another round is needed at all, so
                    # emptying it returns `done` **without the bucket this dead host never
                    # finished** — a silently short answer rather than an error. That is what
                    # the correctness assertions in `test_shuffle_replication.py` caught.
                    failed.update(placement.sources_on(payload))
            else:
                failed.update(payload)  # the reducer named the mappers it could not reach

    # Sources whose regeneration was skipped once because a replica still held them. The
    # skip is deliberately at most once per source: if the copy really does serve the next
    # attempt, this source never comes back and the recompute was pure waste avoided. If it
    # does not — a replica that is unreachable for its own reason — the second visit falls
    # through to the ordinary recompute instead of spending the whole attempt budget
    # re-trying a fetch that cannot work.
    replica_served: set[int] = set()

    def recompute(failed_srcs: set[int]) -> None:
        for src in failed_srcs:
            # The HOST holding `src` is what died, and that is `src` itself only until this
            # source has been relocated once. Marking `src` unconditionally would re-mark an
            # already-dead worker and leave the real one live for `_pick_live`/`_host_for` to
            # hand out again, spending the recovery budget on a host that cannot answer.
            host = placement.host_of(src)
            lost.add(host)
            if src not in replica_served and _survives_on_a_peer(src):
                # The worker serving this output died; the output did not. Re-running the
                # map to regenerate a partition a peer already holds byte-for-byte is the
                # expensive half of recovery, and it buys nothing — the next attempt
                # re-launches the bucket on a live host and its gather falls over to the
                # copy. Placement is left alone on purpose: the source has not moved.
                replica_served.add(src)
                continue
            target = _pick_live({host})
            placement.relocate(src, target)  # it lives here now, not on `src`
            republish(target, src)

    # Proactive spot-preemption migration: move a draining worker's mapped output to a
    # survivor before reclamation (no recovery round, no idle-timeout stall).
    # Best-effort — a failure falls through to the reactive recompute below.
    # `draining_workers` reports HOST ids, so translate to the sources they hold.
    proactive = draining_workers(actors, workers)
    if proactive:
        # Announced before it is attempted, and deliberately outside the `suppress`. This is
        # the engine's best fault-tolerance behaviour — it moves work off a spot node before
        # the node dies, so the query never pays a recovery round — and it was previously
        # invisible in both outcomes: silent on success, and silent on failure too, because
        # the bare `suppress` swallowed the reason. An operator comparing spot against
        # on-demand had no way to see it working.
        events.publish(
            events.RECOVERY,
            name=kind,
            event="preempt_migrate",
            shuffle=kind,
            draining=sorted(proactive),
        )
        with contextlib.suppress(Exception):
            recompute({s for host in proactive for s in placement.sources_on(host)})

    return ShuffleRecovery(recovery_policy(), label=kind).run(attempt, recompute)
