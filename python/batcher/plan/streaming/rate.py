"""The contract between a streaming query's rate controller and the loop it paces.

A micro-batch stream has one backpressure question the shuffle's credit window cannot answer:
*how much should the next trigger read?* Read too much and the batch overruns its interval,
the next one starts late carrying a larger backlog, and the query falls further behind on
every pass until it OOMs on the epoch it can no longer hold. Read too little and the stream
is idle while its source accumulates.

The static answer is a per-trigger cap — ``max_offsets_per_trigger``, ``max_files_per_trigger``
— and it is a real bound but a hand-tuned one. It has to be set for the worst trigger the
query will ever see, so it throttles every other one, and it goes stale the moment the cluster,
the data, or the plan changes.

The adaptive answer measures. Every micro-batch already reports what it consumed and how long
it took, which is a processing *rate*; comparing that against the rate the source is being
admitted at says whether the query is keeping up. That is the loop this module's types carry.

**Why the contract lives here.** `core` owns the micro-batch loop and `carbonite` owns the
policy that decides a resource bound, and the two subsystems must not import each other. So
the vocabulary they exchange lives in `plan`, which is neutral, exactly as the progress record
they are both written against already does. `api` wires the implementation into the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from batcher.plan.streaming.progress import StreamingQueryProgress

__all__ = ["RateController", "RateLimit"]


@dataclass(frozen=True, slots=True)
class RateLimit:
    """How much the next micro-batch may admit.

    Carries the derived rate as well as the row cap because the two answer different
    questions: the cap is what the source is told, and the rate is what a progress reader
    needs in order to see *why* it was told that.

    Examples:
        .. doctest::

            >>> from batcher.plan.streaming import RateLimit
            >>> RateLimit(max_rows=5000, rows_per_second=1000.0).max_rows
            5000
    """

    #: Rows the next trigger may read at most. Always at least 1: a limit of zero is not
    #: backpressure, it is a stalled query that never recovers, because a stream that reads
    #: nothing produces no progress record and so never revises its own limit.
    max_rows: int
    #: The sustainable ingestion rate the cap was derived from, in rows per second.
    rows_per_second: float


@runtime_checkable
class RateController(Protocol):
    """Derives the next micro-batch's admission cap from what the last one measured.

    One method, called once per completed micro-batch. Returning `None` leaves the source's
    configured limit alone, which is what an unconvinced controller must do: a rate estimate
    built from too little evidence is worse than no estimate, because it is acted on.
    """

    def next_limit(self, progress: StreamingQueryProgress) -> RateLimit | None:
        """Fold one micro-batch's outcome in and return the next cap, or `None` to abstain."""
        ...
