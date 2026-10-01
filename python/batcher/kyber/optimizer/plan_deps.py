"""Re-validate a memoized plan against the measurements its own planning read.

The plan cache (`kyber.plan_cache`) keys a plan on the learning generation, which moves when a
learned value is *written*. Three values the estimator reads are not written that way: a
filter's measured selectivity, a shape's cardinality-correction factor and its measured row
width are all folded from operator feedback as it arrives. A plan memoized before any of them
existed was therefore served unchanged ever after. TPC-H q18's `HAVING sum(l_quantity) > 300`
subquery was measured keeping 57 of 1.5M orders on every run, and planned at a third of them
on every run.

Bumping the global generation when a measurement appears was tried and is recorded against
`learning._cardinality_corrections`: it drops *every* memoized plan whenever any query learns
anything, and a mixed workload learns continuously (TPC-DS q34 17 -> 85-210 ms). So the check
here is per plan instead. The estimator records which signatures it looked a measurement up
for (`StatsEstimator.consulted`, found or not); the cache entry stores each one's measured
values as half-octave buckets; and a lookup re-plans only when one of *its own* dependencies
appeared, disappeared, or left its bucket. One query learning something invalidates no other
query's plan.

The one-bucket deadband is `plan_cache`'s treatment of the cost coefficients, for the same
reason: a mean that wanders across a bucket edge must not flip the entry run after run.

A measurement that *appears* re-plans only for the first `LEARNING_ROUNDS` rounds of a key.
Planning consults far more signatures than the plan it picks, and only the picked plan's
operators are measured, so each re-plan can choose an order whose operators nobody measured
yet — which appear on the next run and re-plan it again. On a 17-way JOB join that cascade
re-planned for 100-200 ms on six runs out of seven, against 20-50 ms for DuckDB's whole
query. Past the rounds, only a measurement that moves by a full bucket or disappears still
invalidates, which keeps what the check exists for: the first run's own measurements (q18's
57-of-1.5M `HAVING`) are always spent.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["FINGERPRINT_ROUNDS", "LEARNING_ROUNDS", "dependencies_hold", "dependency_snapshot"]

#: Re-plans a key may take for measurements that newly appeared (see the module note), per
#: quantity. A measured *selectivity* gets two, because a plan's own first measurements do not
#: all arrive on its first run: a filter's selectivity is folded from operator feedback after
#: a row width or correction already re-planned it once, and it is the one the q18 case needs.
#: A row width or cardinality correction gets one: on TPC-H q8 at sf10 a second round let a
#: width that appeared and disappeared between runs move the plan, from 69 ms to 104 ms.
LEARNING_ROUNDS = 2
_WIDTH_CORRECTION_ROUNDS = 1

#: Re-plans a key may take for its learned fingerprint moving (`kyber.plan_cache.cache_key`),
#: which another query's learning moves as readily as this one's.
FINGERPRINT_ROUNDS = 1

Snapshot = tuple[tuple[str, int | None, int | None, int | None], ...]


def _bucket(value: float | None) -> int | None:
    if value is None:
        return None
    return round(math.log2(max(abs(float(value)), 1e-12)) * 2)


def _measured(hub: Any) -> tuple[dict, dict, dict]:
    """The three feedback-folded quantities, each an incremental fold (cheap when idle)."""
    from batcher.kyber.learning import measured_corrections
    from batcher.kyber.measured_selectivity import measured_selectivities
    from batcher.kyber.measured_width import measured_widths

    return measured_selectivities(hub), measured_corrections(hub), measured_widths(hub)


def dependency_snapshot(hub: Any, signatures: set[str]) -> Snapshot:
    """The measured values `signatures` have now, as buckets (None where unmeasured).

    Args:
        hub: The metadata hub the measurements are folded from.
        signatures: The signatures the planning consulted.

    Returns:
        A snapshot `dependencies_hold` can compare a later state against.
    """
    if hub is None or not signatures:
        return ()
    sel, corr, width = _measured(hub)
    return tuple(
        (s, _bucket(sel.get(s)), _bucket(corr.get(s)), _bucket(width.get(s)))
        for s in sorted(signatures)
    )


def dependencies_hold(hub: Any, snapshot: Snapshot, rounds: int = 0) -> bool:
    """Whether every dependency in `snapshot` still measures what it did, within a bucket.

    Args:
        hub: The metadata hub the measurements are folded from.
        snapshot: What `dependency_snapshot` recorded when the plan was built.
        rounds: How many times this plan's key was already re-planned for its dependencies.
            Past its quantity's rounds (`LEARNING_ROUNDS` for a selectivity, one for a width
            or correction), a newly appeared measurement no longer invalidates.

    Returns:
        False when any dependency lost or materially moved a measurement, or gained one
        while the key is still within its learning rounds.
    """
    if hub is None or not snapshot:
        return True
    admits = (rounds < LEARNING_ROUNDS,) + (rounds < _WIDTH_CORRECTION_ROUNDS,) * 2
    sel, corr, width = _measured(hub)
    for sig, s_b, c_b, w_b in snapshot:
        pairs = ((s_b, sel.get(sig)), (c_b, corr.get(sig)), (w_b, width.get(sig)))
        for (then, now), admit_new in zip(pairs, admits, strict=True):
            current = _bucket(now)
            if then is None and current is not None and not admit_new:
                continue
            if (then is None) != (current is None):
                return False
            if then is not None and current is not None and abs(current - then) >= 2:
                return False
    return True
