"""Why this query got the fan-out it got.

The worker count a distributed query runs at is the end of a chain of narrowings, each in a
different module and each with its own reason: Carbonite's data-driven `n_tasks`, the
cluster-fill rewrite, the even-CPU-share raise, the schedulable clamp, and the drain and
node-class exclusions inside it. Every step is well documented and none of it is
*observable* — the query runs at 6 workers on a 40-node cluster and nothing anywhere says
which step took it there.

That gap is expensive in exactly the environments this engine targets. "The job used a
tenth of the cluster" is the single most common distributed complaint, and answering it
currently means reading five modules and guessing which branch ran. The steps are already
computed; this records them.

Reported as a structured log record on the `dist` logger, which is how this module already
reports its other scheduling observation (`scheduling._report_collective_fabric`) and which
reaches the human formatter, the JSON formatter, and the web UI's typed log view from one
call.

It is **also** published as a `plan.profile.Decision` on the event bus, which took a second
attempt to make work. The bus attributes every event to a `query_id`, and `observe.store`
drops any event whose id does not match a live record — silently, by design, so a late event
cannot resurrect an aged-out query as a ghost. The id is minted in `api` and `dist` must not
import `api` to ask for it, so a `Decision` published from here reached the bus and was
discarded by the one consumer that matters. `events.query_scope`, set by the conductor around
execution, closes that: the id is ambient in layer 0, where both `api` and `dist` can reach
it, and `publish` falls back to it. Outside a scope the id is empty and the event is
engine-level, exactly as before — so nothing here depends on a conductor being present.

Purely observational: nothing here decides anything, and a failure to record must never
disturb the schedule it is describing.
"""

from __future__ import annotations

import logging

from batcher._internal import events
from batcher._internal.logging import get_logger, log_kv, note_suppressed
from batcher.plan.profile import Decision

__all__ = ["FanoutTrace"]


class FanoutTrace:
    """Accumulates the fan-out narrowing chain for one distributed execution.

    Steps are appended in the order they are applied, each naming the worker count *after*
    it and why it changed. Publishing is a single event at the end rather than one per step,
    because the useful artifact is the whole chain — a reader wants "8 wanted, 40 available,
    6 placed, because the memory grant only fits 6" in one place, not five events to join.

    Examples:
        .. doctest::

            >>> trace = FanoutTrace(8)
            >>> trace.step("cluster_fill", 40, "one worker per 16-core slice")
            >>> trace.step("clamp", 6, "memory grant fits 6 per node")
            >>> trace.summary()
            'fan-out 8 -> 6: cluster_fill 40, clamp 6'
    """

    __slots__ = ("_start", "_steps")

    def __init__(self, requested: int) -> None:
        self._start = int(requested)
        self._steps: list[tuple[str, int, str]] = []

    def step(self, name: str, workers: int, why: str) -> None:
        """Record that `name` moved the fan-out to `workers`, because `why`.

        A step that changes nothing is still recorded. "The clamp did not reduce it" is an
        answer to the question this exists to answer, and dropping no-ops would leave a
        reader unable to tell a step that ran and agreed from one that never ran at all.
        """
        self._steps.append((str(name), int(workers), str(why)))

    @property
    def final(self) -> int:
        """The fan-out after the last recorded step, or the requested count if none ran."""
        return self._steps[-1][1] if self._steps else self._start

    def summary(self) -> str:
        """A one-line rendering of the chain, for a log or an `EXPLAIN` row."""
        chain = ", ".join(f"{name} {workers}" for name, workers, _ in self._steps)
        return f"fan-out {self._start} -> {self.final}" + (f": {chain}" if chain else "")

    def to_decision(self) -> Decision:
        """The chain as a `Decision`, the neutral record every hand-off is explained with."""
        return Decision(
            subsystem="core",
            category="scheduling",
            summary=self.summary(),
            detail={
                "requested": self._start,
                "final": self.final,
                "steps": [
                    {"step": name, "workers": workers, "why": why}
                    for name, workers, why in self._steps
                ],
            },
        )

    def report(self) -> None:
        """Record the chain on the `dist` logger and the event bus. Never raises.

        An observation about a schedule that has already been decided must not be able to
        disturb it, so every failure here is swallowed the way the rest of the distributed
        layer swallows its diagnostics.

        Logged at INFO: a reader asking why a job used a tenth of the cluster should not
        have to have known to turn on debug logging *before* the run they are asking about.
        One record per distributed execution, so the cost is negligible.

        The bus copy carries the ambient query id (`events.query_scope`), which is what puts
        the chain in the dashboard's decision list beside Kyber's and Carbonite's instead of
        only in a log file. It is a no-op when nothing is subscribed.
        """
        try:
            steps = [
                {"step": name, "workers": workers, "why": why} for name, workers, why in self._steps
            ]
            log_kv(
                get_logger("dist"),
                logging.INFO,
                "fan-out decided",
                requested=self._start,
                final=self.final,
                steps=steps,
            )
            events.publish(events.DECISION, **self.to_decision().to_dict())
        except Exception as exc:  # pragma: no cover - observation must never fail a query
            note_suppressed("dist", "report the fan-out trace", exc)
