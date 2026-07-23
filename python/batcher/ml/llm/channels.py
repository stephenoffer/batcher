"""Per-call side channels an engine uses to report token usage and finish reasons.

An engine's contract is ``list[str] -> list[str]``: one string per request and nothing
else. Everything *about* a generation — how many tokens it cost, whether it stopped
because the model was done or because it hit ``max_tokens`` — has to travel beside that
return value.

The original channel was a mutable ``engine.last_usage`` attribute. It works only while
one engine object is touched by one thread at a time, which is true of `InferencePool`
today (a worker is checked out of a queue for the duration of a batch) but is not part
of any contract, and is not true of an engine a user shares across their own threads. A
stale attribute read is silent: the token counts land on the wrong batch and look
entirely plausible.

These sinks make the channel **per call** instead. A reader opens a `capture` scope, the
engine calls `report` inside it, and the reader takes the values back out of its own
thread's slot. Nothing is shared between threads, so nothing can be misattributed.
`last_usage` stays supported for user-written engines that already set it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Generic, TypeVar

__all__ = ["finish_reason_sink", "logprob_sink", "usage_sink"]

T = TypeVar("T")


class _Sink(Generic[T]):
    """A thread-local, per-call slot an engine reports into and its caller reads back.

    Outside a `capture` scope `report` is a no-op, so an engine can report
    unconditionally without knowing whether anyone asked for the values.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Open a scope in which `report` records, restoring any outer scope on exit."""
        previous = getattr(self._local, "slot", None)
        self._local.slot = []
        try:
            yield
        finally:
            self._local.slot = previous

    def report(self, values: Sequence[T]) -> None:
        """Record one value per request, in request order. A no-op outside a scope."""
        slot = getattr(self._local, "slot", None)
        if slot is not None:
            slot[:] = list(values)

    def collected(self) -> list[T] | None:
        """The values reported in this thread's scope, or `None` if nothing reported."""
        slot = getattr(self._local, "slot", None)
        return list(slot) if slot else None


_USAGE: _Sink[tuple[int | None, int | None] | None] = _Sink()
_FINISH_REASON: _Sink[str | None] = _Sink()
_LOGPROB: _Sink[float | None] = _Sink()


def usage_sink() -> _Sink[tuple[int | None, int | None] | None]:
    """The process-wide sink carrying ``(prompt_tokens, completion_tokens)`` per request."""
    return _USAGE


def finish_reason_sink() -> _Sink[str | None]:
    """The process-wide sink carrying each request's finish reason (``"stop"``/``"length"``)."""
    return _FINISH_REASON


def logprob_sink() -> _Sink[float | None]:
    """The process-wide sink carrying each generation's cumulative log-probability.

    The model's own confidence in what it produced, summed over the generated tokens, so
    it is negative and more negative the less certain the model was. It travels the same
    per-call route as usage and the finish reason because it is the same kind of fact:
    something *about* a generation that the ``list -> list[str]`` return value has no
    room for.
    """
    return _LOGPROB
