"""Shuffle lineage — how to recompute an output a lost worker produced.

Carbonite's fault tolerance is Spark-style *recompute from lineage*: a shuffle
output isn't replicated, it's regenerated from the deterministic map task that
produced it. `ShuffleLineage` records the coordinate of that work (which source
partition of which stage) and the epoch that distinguishes a fresh recompute from
the stale output a dead worker left behind. The recompute *action* is a caller
thunk — the distributed layer owns the map IR and the workers — so this carries
only what Carbonite needs to coordinate: the identity and the epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ShuffleLineage", "SourcePlacement"]


@dataclass(frozen=True, slots=True)
class ShuffleLineage:
    """The coordinate + epoch of one recomputable map output.

    `stage`/`src_partition` identify the map task; `epoch` increments each time the
    output is regenerated, so a reducer never confuses a recomputed partition with
    the stale one a lost worker had published under the previous epoch.
    """

    stage: int
    src_partition: int
    epoch: int = 0

    def reincarnate(self) -> ShuffleLineage:
        """The lineage for a fresh recompute attempt (the next epoch)."""
        return ShuffleLineage(self.stage, self.src_partition, self.epoch + 1)


class SourcePlacement:
    """Which worker currently holds each shuffle source's latest map output.

    The other half of the recovery coordinate `ShuffleLineage` carries: lineage says
    *what* to regenerate and at which epoch, this says *where* the current copy lives.

    It exists because the two are trivially equal until they aren't. On a clean run
    source `s` is published by worker `s`, so a recovery loop can use one number for
    both — and every such loop did. After a single recompute relocates `s` to a
    survivor, the source id and its host diverge permanently, and code that still
    conflates them marks the *original* (already-dead) worker on the next failure while
    the genuinely dead host keeps looking alive. The recovery budget then drains
    re-picking a host that cannot answer, and the stage fails with `ResourceError`
    despite survivors being available. Keeping the mapping explicit is what makes the
    second failure of a relocated source recoverable.
    """

    __slots__ = ("_hosts", "_on_host", "_workers")

    def __init__(self, workers: int, hosts: list[int] | None = None) -> None:
        self._workers = workers
        # Only relocated sources are stored; an absent entry means "still on its own
        # worker", so a clean run allocates nothing and behaves exactly as before.
        self._hosts: dict[int, int] = {}
        # The reverse index: host -> the relocated sources it now holds. Maintained rather
        # than derived because `sources_on` is asked once per worker death, and deriving it
        # meant walking every source in the fleet to ask where each one lives — an O(W)
        # scan per failure, on a path that only runs when the cluster is already losing
        # workers and a correlated preemption wave can be losing many at once.
        self._on_host: dict[int, set[int]] = {}
        # `hosts[src]` is where source `src` landed on its first attempt. Needed as soon as
        # a shuffle has more sources than workers, because then "source `s` is on worker
        # `s`" is not merely unrelocated-yet, it is out of range — the identity the sparse
        # form assumes never held. Recording every source up front keeps `host_of` and
        # `sources_on` exact without giving either one a second code path: the reverse
        # index is simply complete from the start rather than filled in by relocations.
        if hosts is not None:
            for src, host in enumerate(hosts):
                self._hosts[src] = host
                self._on_host.setdefault(host, set()).add(src)

    def host_of(self, src: int) -> int:
        """The worker holding `src`'s latest output — where it was first placed until it moves."""
        return self._hosts.get(src, src)

    def sources_on(self, host: int) -> set[int]:
        """The sources whose latest output lives on `host` — what its death loses.

        A dead worker loses the map output it was holding, but which source that is
        depends on the current placement rather than on the host id.

        Args:
            host: The worker index.

        Returns:
            The source ids whose latest output is on `host`.
        """
        out = set(self._on_host.get(host, ()))
        # `host` still holds its own source unless that one was itself relocated away.
        if 0 <= host < self._workers and host not in self._hosts:
            out.add(host)
        return out

    def relocate(self, src: int, host: int) -> None:
        """Record that `src` was recomputed onto `host`.

        Args:
            src: The source whose output was regenerated.
            host: The worker it now lives on.
        """
        previous = self._hosts.get(src, src)
        if previous in self._on_host:
            self._on_host[previous].discard(src)
            if not self._on_host[previous]:
                del self._on_host[previous]
        self._hosts[src] = host
        self._on_host.setdefault(host, set()).add(src)
