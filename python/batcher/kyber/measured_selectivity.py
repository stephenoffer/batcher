"""Filter selectivity derived from what Core measured, per plan signature.

`learning` records learned values *into* the keyed-parameter store; this module derives one
*out of* the per-operator feedback history instead, the way `learning._cardinality_corrections`
derives its q-error factors. Kyber cannot hook the recording — `core` builds the
`OperatorFeedback` rows and the two subsystems are independent — so reading the hub's history
is the only correct layering for it.

## The loop this closes

Core already records, for every filter it runs, the measured `rows_out / rows_in` under the
stable signature `annotate_ops` stamped on that operator. That happens on *every* execution,
profiled or not, because both executor paths pass `feedback=hub`.

Nothing consumed it. The estimator reads a `selectivity` key per signature
(`StatsEstimator._selectivity`, where a measured value always beats the structural guess), and
the only writer of that key was `learning.record_selectivity` — which is handed the **query's**
final row count and therefore guards on `_filter_over_scan`: the whole plan must be a filter
over a single scan, modulo row-preserving projections. Every filter underneath a join,
aggregate, sort or limit — 21 of the 22 TPC-H queries, and essentially every real analytical
query — re-derived a structural guess the engine had already measured, on every run forever.

Measured on TPC-H sf1: q12's `lineitem` filter measures 0.0869 selectivity (521,289 of
6,001,215 rows) and was estimated at 0.327 — the flat range constant — identically on ten
consecutive runs, with all six of its signed feedback rows present in the hub and none of
their signatures carrying a `selectivity` entry.

## What this does not fix

A signature is structural, so this learns "how selective is *this predicate shape* here", not
a correlation model. It also cannot help a shape's first execution, by construction: one run
must be measured before there is anything to consume.
"""

from __future__ import annotations

import weakref
from collections import deque

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.metadata import MetadataHub

__all__ = ["measured_selectivities"]

# Same contract as `learning._CORRECTION_CACHE`: `(hub.version, fingerprint, result)`, valid
# while the hub has absorbed no new feedback and the relevant config is unchanged. Without it
# every `optimize` re-folds the history, which is the fixed overhead a sub-millisecond query
# cannot afford.
_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, tuple, dict[str, float]]] = (
    weakref.WeakKeyDictionary()
)

# Cap on distinct signatures tracked in one fold, mirroring `learning`'s: a session issuing
# endlessly many shapes must not grow this without bound.
_MAX_TRACKED_SIGNATURES = 4096


def measured_selectivities(hub: MetadataHub) -> dict[str, float]:
    """`{signature: measured selectivity}` for filters with enough recent observations.

    The mean of the most recent `cardinality_correction_window` samples, gated on that same
    loop's `cardinality_correction_min_samples`, because the question is the same one: how
    many recent observations of a shape are enough to trust. An arithmetic mean is right here
    where the correction factor needs a geometric one — a selectivity is a probability, not a
    multiplicative error.

    Best-effort throughout: a malformed row is skipped and any failure yields no
    selectivities rather than raising into planning.
    """
    cfg = active_config().optimizer
    window = cfg.cardinality_correction_window
    min_samples = cfg.cardinality_correction_min_samples
    if window <= 0 or min_samples <= 0:
        return {}
    fingerprint = (window, min_samples)
    cached = _CACHE.get(hub)
    if cached is not None and cached[0] == hub.version and cached[1] == fingerprint:
        return cached[2]
    samples: dict[str, deque[float]] = {}
    try:
        for row in hub.op_stats_with_signature():  # oldest first, already bounded
            sig, sel = row.get("signature"), row.get("selectivity")
            if row.get("kind") != "filter" or not sig or not isinstance(sel, (int, float)):
                continue
            # A ratio outside [0, 1] is not a selectivity; a new signature past the cap is
            # dropped rather than growing the fold without bound.
            if not 0.0 <= float(sel) <= 1.0:
                continue
            if sig not in samples and len(samples) >= _MAX_TRACKED_SIGNATURES:
                continue
            samples.setdefault(sig, deque(maxlen=window)).append(float(sel))
    except Exception as exc:  # pragma: no cover - learning must never break planning
        note_suppressed("kyber", "read measured filter selectivities", exc)
        return {}
    out = {s: sum(v) / len(v) for s, v in samples.items() if len(v) >= min_samples}
    _CACHE[hub] = (hub.version, fingerprint, out)
    return out
