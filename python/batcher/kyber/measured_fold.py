"""The incremental per-signature fold the measured-quantity readers share.

`measured_selectivity` and `measured_width` ask one question of one history — "what did Core
actually measure for this plan shape, and is it consistent enough to believe?" — and differ
only in *which field of a feedback row is the sample*. Everything around that field is the
same: the cursor arithmetic that keeps the fold incremental, the cap on tracked signatures,
the sample window, and the two confidence gates.

It was written twice, which is what this module exists to undo. Both copies live in `kyber`,
so the pasted code was not a layering violation — it was the ordinary kind of duplication,
where the second copy silently stops matching the first. The two had already begun to: only
one of them documented why the fold is incremental at all, and a fix to the cursor rebuild
would have had to be found twice.

## The fold

Core records one row per operator per execution, under the stable signature `annotate_ops`
stamped on that operator. `MetadataHub.op_stats_with_signature` returns them oldest first and
`MetadataHub.signed_appends` counts how many have ever been appended, so the difference
against a retained cursor is exactly how many rows at the tail are new.

Absorbing only those is load-bearing rather than tidy. Re-folding the retained history puts a
cost proportional to the session's cumulative query count on the critical path of *every*
`optimize`, and `optimize` runs several times per query: the first version of this fold
re-folded, and cost 3.3% on an eight-query loop and 9.1% over twenty-two — worse the longer
the session ran, because the history it walked kept growing.

## The two gates

A signature needs `cardinality_correction_min_samples` observations before it reports, and
those observations must be *concentrated*. The second gate is the one that is easy to skip
and expensive to omit. A signature is structural, so two different relations can share one
entry — a scan with no data-stable identity renders as the bare token `["scan"]` — and a mean
over two populations describes neither. Measured on a selectivity: `x < 40` keeping 40 of
20,000 rows in one table and every row in another averaged to a **500x** under-estimate on
the permissive table, against a structural estimate that was off by 3x. Refusing a wide
spread does not fix the key; it bounds the damage, and falls back to the structural estimate,
which is the honest answer when the evidence is two things wearing one name.
"""

from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from batcher._internal.logging import note_suppressed
from batcher._internal.mathx import is_concentrated
from batcher.config import active_config
from batcher.metadata import MetadataHub

__all__ = ["fold_measured"]


@dataclass(slots=True)
class _State:
    """One (hub, quantity) fold: how far it has read, and the windows it holds."""

    consumed: int  # `MetadataHub.signed_appends` as of the last fold
    window: int  # the per-signature sample window this state was built for
    min_samples: int  # the confidence gate this state's `result` was computed under
    samples: dict[str, deque[float]] = field(default_factory=dict)
    result: dict[str, float] = field(default_factory=dict)


# Per-hub fold state, keyed weakly so a dropped hub evicts its own, then by quantity: the
# readers share a hub and a cursor position but not a sample, so one state per hub would have
# each of them absorbing the other's rows and reporting nothing.
_STATE: weakref.WeakKeyDictionary[MetadataHub, dict[str, _State]] = weakref.WeakKeyDictionary()

# Cap on distinct signatures tracked in one fold, mirroring `learning`'s: a session issuing
# endlessly many shapes must not grow this without bound.
_MAX_TRACKED_SIGNATURES = 4096

# How far apart a signature's observations may be before their mean is refused, as a multiple
# of the median — the gate the module docstring justifies. `cpu_shares` applies the same test
# to its utilization medians, which is why the predicate itself lives in `_internal.mathx`.
_MAX_REL_SPREAD = 1.0


def fold_measured(
    hub: MetadataHub,
    sample_of: Callable[[dict], float | None],
    *,
    what: str,
) -> dict[str, float]:
    """`{signature: mean of the recent samples}` for shapes with enough consistent evidence.

    The mean of the most recent `cardinality_correction_window` samples, gated on that same
    loop's `cardinality_correction_min_samples` and on the samples being concentrated. An
    arithmetic mean, because the quantities folded here are magnitudes and probabilities
    rather than the multiplicative errors a geometric mean is right for.

    Best-effort throughout: a row `sample_of` rejects is skipped, and any failure yields
    nothing rather than raising into planning.

    Args:
        hub: The metadata hub holding the measured operator history.
        sample_of: Reads one feedback row and returns the sample it contributes, or `None`
            to skip it — the only thing that varies between the quantities folded here.
        what: Names the quantity, both to key its fold state apart from the others sharing
            this hub and to name it in a suppressed-error note.

    Returns:
        The measured quantity per plan signature.
    """
    cfg = active_config().optimizer
    window = cfg.cardinality_correction_window
    min_samples = cfg.cardinality_correction_min_samples
    if window <= 0 or min_samples <= 0:
        return {}
    try:
        # The view before the cursor: the first read is what materializes the Hub's view from
        # the backend, and that load is what gives `signed_appends` its value. Reading the
        # cursor first would see 0 and absorb nothing.
        rows = hub.op_stats_with_signature()  # oldest first
        appends = hub.signed_appends
        folds = _STATE.setdefault(hub, {})
        state = folds.get(what)
        if (
            state is None
            or state.window != window
            or state.min_samples != min_samples
            or state.consumed > appends
        ):
            # First fold, a reconfigured window or gate, or a counter that moved backwards (a
            # fresh backend behind the hub): rebuild from whatever history is retained.
            state = _State(consumed=0, window=window, min_samples=min_samples)
            folds[what] = state
        fresh = appends - state.consumed
        if fresh <= 0:
            return state.result
        # The Hub's view is bounded, so a cursor left far enough behind can name more rows
        # than remain; absorb whatever is still there.
        touched: set[str] = set()
        for row in rows[-fresh:] if fresh < len(rows) else rows:
            _absorb(state, row, sample_of, touched)
        state.consumed = appends
        # Recompute **only the signatures this round changed**. The absorption above was
        # already incremental; the summary was not, and rebuilding it walked every signature
        # the session had ever seen — computing a median per signature, per query, forever.
        # That put a cost proportional to the session's cumulative history back on the
        # critical path, which is the exact thing the incremental cursor exists to remove.
        # Measured on TPC-DS at scale 1, replaying the suite in one session: a probe query
        # cost 9.4 ms with nothing else run and 21.5 ms after 100 other queries — and this
        # rebuild, across the two quantities folded here and the correction factors next
        # door, was most of the difference.
        #
        # A signature's entry is *removed* when it no longer qualifies, not left behind: the
        # window is bounded, so new observations evict old ones and a shape whose samples
        # have spread out must stop reporting a mean nobody should trust.
        for sig in touched:
            values = state.samples[sig]
            if len(values) >= min_samples and is_concentrated(values, _MAX_REL_SPREAD):
                state.result[sig] = sum(values) / len(values)
            else:
                state.result.pop(sig, None)
        return state.result
    except Exception as exc:  # pragma: no cover - learning must never break planning
        note_suppressed("kyber", f"read measured {what}", exc)
        return {}


def _absorb(
    state: _State,
    row: dict,
    sample_of: Callable[[dict], float | None],
    touched: set[str],
) -> None:
    """Fold one feedback row into `state`, recording whose window it changed.

    A row carrying no signature has nothing to be attributed to, and a signature past the cap
    is dropped rather than growing the fold without bound. What makes a *sample* admissible is
    the caller's judgement, not this function's.

    `touched` collects the signatures whose sample window this call altered, so the caller
    re-summarizes those and leaves the rest of the fold alone.
    """
    sig = row.get("signature")
    if not sig:
        return
    sample = sample_of(row)
    if sample is None:
        return
    if sig not in state.samples and len(state.samples) >= _MAX_TRACKED_SIGNATURES:
        return
    state.samples.setdefault(sig, deque(maxlen=state.window)).append(sample)
    touched.add(sig)
