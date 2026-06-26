"""Shuffle recovery — the recompute-on-failure coordination loop.

When a worker is lost mid-shuffle, the partitions it held vanish and the reducers
fetching them fail. Carbonite recovers by *recomputing* those partitions on a
surviving worker (from lineage) and retrying the fetch — bounded by a policy so a
permanently broken shuffle fails loudly instead of looping forever.

Carbonite owns only the *coordination* (try, see what failed, recompute, retry,
give up); the distributed layer supplies the closures that know how to attempt the
shuffle round and how to regenerate a lost partition on a live worker. Keeping the
loop here means the policy and the give-up semantics are one tested thing, reused
by every shuffle shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from batcher._internal.errors import ResourceError

__all__ = ["RecoveryPolicy", "ShuffleRecovery"]

_Result = TypeVar("_Result")
_Failed = TypeVar("_Failed")


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """How hard Carbonite tries to recover a shuffle from worker loss.

    `max_attempts` bounds the recompute→retry cycles; a shuffle that still has
    unreachable inputs after that many rounds is treated as unrecoverable.
    `backoff_base_s` is the base of an exponential backoff slept between rounds, so
    a flaky cluster is not retried in a tight loop (0 disables the sleep). The backoff
    carries *equal jitter*, so a correlated preemption wave (many reducers losing
    workers at once) does not retry in lockstep and stampede the survivors.
    """

    max_attempts: int = 3
    backoff_base_s: float = 0.0


class ShuffleRecovery:
    """Runs a shuffle round under recompute-on-failure, policy-bounded.

    Construct one per shuffle; `run` drives the caller's `attempt`/`recompute`
    closures. `recomputes` counts how many recovery rounds were needed (0 on a
    clean run) — useful telemetry and a test hook.
    """

    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        self._policy = policy or RecoveryPolicy()
        self.recomputes = 0

    def run(
        self,
        attempt: Callable[[], tuple[_Result, _Failed]],
        recompute: Callable[[_Failed], None],
    ) -> _Result:
        """Drive `attempt` until it reports no failures, recomputing in between.

        `attempt()` runs one shuffle round and returns `(result, failed)`, where a
        falsy `failed` means every input was reached (done — `result` is returned).
        `recompute(failed)` regenerates the unreachable inputs on a live worker.
        After `max_attempts` exhausted rounds, raises `ResourceError`.
        """
        failed: _Failed | None = None
        for round_idx in range(self._policy.max_attempts):
            result, failed = attempt()
            if not failed:
                return result
            recompute(failed)
            self.recomputes += 1
            # Exponential backoff before the next round so a flaky network/cluster
            # is not hammered in a tight recompute→retry loop. Equal jitter
            # (`half + U[0, half]`) keeps at least half the backoff while decorrelating
            # a correlated preemption wave's retries, so survivors aren't stampeded by
            # every lost reducer retrying at the same instant.
            if self._policy.backoff_base_s > 0 and round_idx + 1 < self._policy.max_attempts:
                import random
                import time

                ceiling = self._policy.backoff_base_s * (2**round_idx)
                half = ceiling / 2.0
                time.sleep(half + random.uniform(0.0, half))
        raise ResourceError(
            f"shuffle did not recover after {self._policy.max_attempts} attempts "
            f"(still unreachable: {failed})"
        )
