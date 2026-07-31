"""A ceiling on how much of a job may be spent retrying.

Per-task retry limits do not bound a job. `max_retries=3` over a hundred thousand tasks
authorizes three hundred thousand retries, and a fleet that is broken in some way no probe
catches — a bad image, an expired credential, a driver that stopped matching its runtime — will
use every one of them. What the operator sees is a job that runs for hours at a fraction of its
rate and then fails with whatever error happened to be last, long after the first one said
exactly what was wrong.

A budget converts that into a bounded loss. Retries are drawn from a pool sized *relative to
the work*, so a large job tolerates proportionally more transient failure than a small one
without either of them being able to spend the whole run on retries. When the pool is empty the
next failure is raised instead of retried — with its own traceback, at the point it happened.

Two things this deliberately does not do. It does not distinguish *why* a retry was taken;
that is `classify`'s job, and a caller consults it before reaching for the budget. And it never
blocks — an exhausted budget is an immediate `False`, because a resilience mechanism that waits
is a resilience mechanism that turns a broken fleet into a hung job.

Carbonite owns it (a protection concern). Thread-safe: the distributed executor draws from one
budget across every concurrently retrying task.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from batcher._internal import events

__all__ = ["BudgetState", "RetryBudget"]


@dataclass(frozen=True, slots=True)
class BudgetState:
    """A budget's accounting, for a report or a test.

    Attributes:
        attempts: Units of work attempted so far, which is what the pool is sized against.
        spent: Retries taken.
        allowance: Retries authorized at the current attempt count.
        exhausted: Whether the next retry would be refused.
    """

    attempts: int
    spent: int
    allowance: int
    exhausted: bool

    @property
    def remaining(self) -> int:
        """Retries still available, never negative."""
        return max(0, self.allowance - self.spent)


class RetryBudget:
    """A job-wide pool of retries, sized as a fraction of the work attempted.

    The allowance is `max(floor, fraction * attempts)`, which gives the two behaviors a fixed
    number cannot give at once. The floor lets a *small* job retry at all — ten tasks at a 5%
    fraction would otherwise authorize nothing, so one flaky node fails the whole thing. The
    fraction keeps a *large* job from spending itself on retries, because a hundred thousand
    tasks at a fixed budget of twenty would exhaust it in the first second of a bad fleet and
    fail a job that was merely unlucky.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.resilience import RetryBudget
            >>> budget = RetryBudget(fraction=0.5, floor=1)
            >>> for _ in range(4):
            ...     budget.record_attempt()
            >>> [budget.try_consume() for _ in range(3)]
            [True, True, False]
    """

    def __init__(self, fraction: float = 0.1, floor: int = 16, *, label: str = "") -> None:
        """Build a budget.

        Args:
            fraction: Share of attempted work that may be retried. `0.1` means a job may spend
                a tenth of its task count on retries before failing on the next error.
            floor: Retries authorized regardless of size, so a short job is not left with a
                budget of zero.
            label: What this budget covers (a stage or shuffle name), carried on the event
                published when it runs out. Without it an exhausted budget is observable but
                not attributable, which on a plan with several stages is barely a signal.
        """
        self._fraction = max(0.0, float(fraction))
        self._floor = max(0, int(floor))
        self._label = label
        self._lock = threading.Lock()
        self._attempts = 0
        self._spent = 0
        self._announced = False

    def record_attempt(self, count: int = 1) -> None:
        """Count units of work attempted, which is what the allowance is sized against.

        Args:
            count: How many attempts to add.
        """
        with self._lock:
            self._attempts += max(0, int(count))

    def _allowance(self) -> int:
        return max(self._floor, int(self._fraction * self._attempts))

    def try_consume(self, count: int = 1) -> bool:
        """Draw retries from the pool.

        All-or-nothing: a partial draw would let a caller retry some of a batch and abandon
        the rest, which is neither of the two outcomes it knows how to handle.

        Args:
            count: How many retries to take.

        Returns:
            True when the pool covered them, False when it did not — in which case the caller
            must raise the failure it was about to retry rather than waiting for capacity that
            does not arrive.
        """
        want = max(1, int(count))
        with self._lock:
            if self._spent + want > self._allowance():
                announce = not self._announced
                self._announced = True
                state = self._snapshot()
            else:
                self._spent += want
                return True
        if announce:
            # Published once, on the transition. A budget that has run out will be asked again
            # by every task still in flight, and an event per refusal would bury the one that
            # matters under thousands of copies.
            events.publish(
                events.RECOVERY,
                name=self._label,
                event="budget_exhausted",
                attempts=state.attempts,
                spent=state.spent,
                allowance=state.allowance,
            )
        return False

    def _snapshot(self) -> BudgetState:
        """The current accounting. The lock must already be held."""
        allowance = self._allowance()
        return BudgetState(
            attempts=self._attempts,
            spent=self._spent,
            allowance=allowance,
            exhausted=self._spent >= allowance,
        )

    def state(self) -> BudgetState:
        """The budget's accounting right now.

        Returns:
            A `BudgetState` snapshot.
        """
        with self._lock:
            return self._snapshot()

    def reset(self) -> None:
        """Return the budget to empty, for a new job on a reused process."""
        with self._lock:
            self._attempts = 0
            self._spent = 0
            self._announced = False
