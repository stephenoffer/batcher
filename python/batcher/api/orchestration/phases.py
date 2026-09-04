"""The control plane's phase vocabulary: what a query is doing, while it is doing it.

A terminal op spends its time in a handful of named phases -- reading source statistics,
optimizing, admission, executing, and on the distributed path sizing the fan-out and
gathering worker metadata. Two things consume that: the DEBUG log records what each phase
*cost* once it is over, and the live progress line needs to know which one is running *now*.

The second is the reason this module exists. The engine runs inside Rust and reports
nothing until it returns, and every operator stage the profile carries is replayed onto the
bus *after* the query ends -- so between `QUERY_START` and `QUERY_END` the bus was silent
for the whole of the slow part. Measured on a six-operator join over three million rows,
that was 122 ms of a 142 ms query, and the progress line read ``running`` from the first
millisecond to the last. On a distributed run it was worse: seven seconds showing the phase
that had ended in its first millisecond.

Two vocabularies, deliberately. The log's phase name is a field that gets joined and
grepped and must stay stable; the label is prose for someone watching a line, and
``kyber.optimize_full`` is not it.

Lives here rather than in `run` because `stages` reports the distributed phases and `run`
imports `stages` -- so the shared vocabulary has to sit below both.
"""

from __future__ import annotations

import logging

from batcher._internal import events
from batcher._internal.logging import get_logger, log_kv

__all__ = ["MAX_LABEL", "PHASE_LABELS", "begin", "record"]

_log = get_logger("api.run")

#: What each phase is *doing*, in the words a person watching the line would use. Keys are
#: the stable machine names the log records. A phase with no entry shows its machine name,
#: which is worse but never wrong.
PHASE_LABELS = {
    "collect_source_stats": "reading stats",
    "kyber.optimize_full": "optimizing",
    "carbonite.validate": "admission",
    "core.execute": "executing",
    "core.execute.spilled": "spilling",
    "distributed_grant": "sizing fan-out",
    "execute_distributed": "on cluster",
    "collect_source_metadata": "learning stats",
}

#: The widest a label may be, so it fits `observe.console.paint.STAGE_W` without truncating.
#: A phase label is read at a glance and re-read on every frame; ``executing (cl…`` costs
#: the reader more than the two words it was trying to say.
MAX_LABEL = 14


def begin(name: str) -> None:
    """Announce that the query has entered phase `name`, for anything watching live.

    Costs a tuple-truthiness check when nothing is attached -- `events.publish` returns
    before building an `Event` -- which is the common case.

    Args:
        name: The stable phase name, a key of `PHASE_LABELS`.

    Returns:
        None.
    """
    events.publish(events.PHASE, name=PHASE_LABELS.get(name, name), phase=name)


def record(name: str, seconds: float, **fields: object) -> None:
    """Record what phase `name` cost, at DEBUG, once it is over.

    Reported in **milliseconds with microsecond resolution**, not in rounded seconds. These
    phases are routinely tens of microseconds -- `carbonite.validate` on a small plan is one
    -- and rounding to three decimal places of a second printed every one of them as `0.0`,
    which is indistinguishable from a phase that was never measured. The `_ms` suffix is the
    OpenTelemetry unit convention `_internal.logging` names as its own example, so the
    number's unit does not depend on the reader knowing which helper wrote it.

    Args:
        name: The stable phase name.
        seconds: How long the phase took.
        **fields: Extra structured detail for this phase.

    Returns:
        None.
    """
    log_kv(
        _log,
        logging.DEBUG,
        "run phase",
        phase=name,
        duration_ms=round(seconds * 1000, 3),
        **fields,
    )
