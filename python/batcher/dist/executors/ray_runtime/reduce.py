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

__all__ = ["run_bucket_reduce"]


def run_bucket_reduce(
    *,
    kind: str,
    n_buckets: int,
    workers: int,
    actors: list[Any],
    remote_reduce: Callable[[int, int], Any],
    republish: Callable[[int, int], None],
    dead: set[int] | None = None,
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
        draining_workers,
        recovery_policy,
        speculation_policy,
    )

    lost: set[int] = set(dead or ())
    # Where each source's latest mapped output lives. Identity until a recompute relocates a
    # source, after which the source id and its host are different numbers — and it is the
    # HOST that dies. The flat aggregate reduce keeps the same mapping for the same reason;
    # see `SourcePlacement` for what conflating the two costs.
    placement = SourcePlacement(workers)

    def _pick_live(avoid: set[int]) -> int:
        for i in range(workers):
            if i not in lost and i not in avoid:
                return i
        raise ResourceError(f"no surviving worker to recover the {kind} shuffle on")

    def _host_for(bucket: int, avoid: set[int]) -> int:
        # `bucket b → actor b`, unless dead/avoided; `avoid` lets a straggler's backup
        # land on a different live worker than the slow original.
        return bucket if bucket not in lost and bucket not in avoid else _pick_live(avoid)

    # A bucket reduce that returns "ok" consumed its complete input and is
    # deterministic, so cache it across recovery rounds (keyed by bucket index) and
    # never re-run it. Only pending buckets re-launch, so one lost mapper doesn't
    # re-reduce every surviving bucket — the amplification that hurt most on a
    # churning spot/autoscaling cluster.
    done: dict[int, Any] = {}

    def attempt() -> tuple[dict[int, Any], set[int]]:
        failed: set[int] = set()
        # Launch every *pending* bucket concurrently, then collect via
        # `gather_with_backups`: a degraded-but-alive bucket gets a backup on another
        # live worker (deterministic => byte-identical), and a dead host is classified
        # for recompute — so one slow node cannot stall the barrier.
        ref_host: dict[Any, int] = {}

        def _launch(bucket: int, avoid: set[int]) -> Any:
            host = _host_for(bucket, avoid)
            ref = remote_reduce(host, bucket)
            ref_host[ref] = host
            return ref

        pending = [b for b in range(n_buckets) if b not in done]
        refs = [_launch(b, set()) for b in pending]

        def _relaunch(idx: int) -> Any:
            try:
                return _launch(pending[idx], {ref_host[refs[idx]]})
            except ResourceError:
                return _launch(pending[idx], set())

        def _on_failure(_idx: int, ref: Any, _exc: Exception) -> tuple[str, int | None]:
            return ("__dead__", ref_host.get(ref))

        gathered = gather_with_backups(
            refs, _relaunch, speculation_policy(), on_failure=_on_failure
        )
        for bucket, (status, payload) in zip(pending, gathered, strict=True):
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
                    lost.add(payload)  # host died — its mapped output is lost too
                    # `payload` is a HOST id; `failed` carries SOURCE ids (the other branch
                    # reports unreachable sources). Translate through the current placement so
                    # a relocated source is recomputed and an unrelated one is not — on a
                    # clean run this is exactly `{payload}`.
                    failed.update(placement.sources_on(payload))
            else:
                failed.update(payload)  # the reducer named the mappers it could not reach
        return done, failed

    def recompute(failed_srcs: set[int]) -> None:
        for src in failed_srcs:
            # The HOST holding `src` is what died, and that is `src` itself only until this
            # source has been relocated once. Marking `src` unconditionally would re-mark an
            # already-dead worker and leave the real one live for `_pick_live`/`_host_for` to
            # hand out again, spending the recovery budget on a host that cannot answer.
            host = placement.host_of(src)
            lost.add(host)
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
