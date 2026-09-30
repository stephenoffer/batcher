"""What the Python out-of-core executors wrote to disk, measured where they write it.

The engine reports its own spills per operator (`ExecMetrics.ops[].spilled`/`spill_bytes`),
and `QueryProfile.spilled` is read from those. The Python spill path (`dist.spill`,
`dist.spill_breakers`, `dist.global_window`) never produced such a reading: it runs thousands
of unmetered engine dispatches with its own partition and reduce phases around them, and its
buckets go to disk through Carbonite's tiered store rather than through an engine operator.
A query that went out of core that way reported `spilled: False`, which is the one reading a
reader reaches for while diagnosing an OOM, and it was backwards.

Every one of those executors opens its scratch through one function
(`dist.spill.buckets.spill_scratch`), so that is where the reading is taken, from the store's
own accounting, and folded into the innermost active `SpillMeter`. The conductor opens a meter
around the out-of-core phase and records the total into the profile. The meter is carried by a
`ContextVar`, so a run with no meter open (every path that does not profile) pays one lookup
per spilled breaker and records nothing.

Neutral: it imports nothing, so `dist` (which writes) and `api` (which reads) can share it.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

__all__ = ["SpillMeter", "note_spill", "spill_meter"]


class SpillMeter:
    """The running total of what the Python spill executors wrote while this meter was open."""

    __slots__ = ("buckets_written", "bytes_written")

    def __init__(self) -> None:
        self.bytes_written = 0
        self.buckets_written = 0

    @property
    def spilled(self) -> bool:
        """Whether anything reached disk while the meter was open.

        Returns:
            True when at least one bucket or byte was written.
        """
        return self.buckets_written > 0 or self.bytes_written > 0


_active: contextvars.ContextVar[SpillMeter | None] = contextvars.ContextVar(
    "batcher_spill_meter", default=None
)


@contextlib.contextmanager
def spill_meter() -> Iterator[SpillMeter]:
    """Open a meter that accumulates every spill recorded inside the `with` block.

    Examples:
        .. doctest::

            >>> from batcher.plan.profile.spill import note_spill, spill_meter
            >>> with spill_meter() as meter:
            ...     note_spill(bytes_written=4096, buckets_written=2)
            >>> meter.spilled, meter.bytes_written
            (True, 4096)

    Yields:
        The meter, whose totals remain readable after the block exits.
    """
    meter = SpillMeter()
    token = _active.set(meter)
    try:
        yield meter
    finally:
        _active.reset(token)


def note_spill(*, bytes_written: int, buckets_written: int) -> None:
    """Add one spill store's lifetime volume to the innermost open meter, if there is one.

    Args:
        bytes_written: Bytes the store wrote to disk over its lifetime.
        buckets_written: Buckets the store wrote over its lifetime.
    """
    meter = _active.get()
    if meter is not None:
        meter.bytes_written += max(0, int(bytes_written))
        meter.buckets_written += max(0, int(buckets_written))
