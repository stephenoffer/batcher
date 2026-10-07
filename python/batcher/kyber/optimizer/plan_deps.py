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

#: One entry per consulted signature, all as buckets (None where absent): its measured
#: selectivity, correction factor and width; the selectivity, row count and width the plan was
#: built on (`StatsEstimator.used`); and its measured row count.
Snapshot = tuple[tuple[str, *tuple[int | None, ...]], ...]


def _bucket(value: float | None) -> int | None:
    if value is None:
        return None
    return round(math.log2(max(abs(float(value)), 1e-12)) * 2)


def _measured(hub: Any) -> tuple[dict, dict, dict, dict]:
    """The feedback-folded quantities, each an incremental fold (cheap when idle)."""
    from batcher.kyber.learning import measured_corrections, measured_rows
    from batcher.kyber.measured_selectivity import measured_selectivities
    from batcher.kyber.measured_width import measured_widths

    return (
        measured_selectivities(hub),
        measured_corrections(hub),
        measured_widths(hub),
        measured_rows(hub),
    )


def dependency_snapshot(
    hub: Any, signatures: set[str], used: dict[str, list[float | None]] | None = None
) -> Snapshot:
    """The measured values `signatures` have now, as buckets (None where unmeasured).

    Args:
        hub: The metadata hub the measurements are folded from.
        signatures: The signatures the planning consulted.
        used: Per signature, the `[selectivity, rows, width]` the plan was built on
            (`StatsEstimator.used`), measured or estimated.

    Returns:
        A snapshot `dependencies_hold` can compare a later state against.
    """
    if hub is None or not signatures:
        return ()
    sel, corr, width, rows = _measured(hub)
    used = used or {}
    out = []
    for s in sorted(signatures):
        planned = used.get(s) or (None, None, None)
        out.append(
            (
                s,
                _bucket(sel.get(s)),
                _bucket(corr.get(s)),
                _bucket(width.get(s)),
                *(_bucket(v) for v in planned),
                _bucket(rows.get(s)),
            )
        )
    return tuple(out)


def _holds(
    then: int | None, current: int | None, planned: int | None, admit_new: bool, settled: bool
) -> bool:
    """Whether one dependency still supports its plan (see `dependencies_hold`)."""
    if current is None:
        return then is None  # a measurement that disappeared took its evidence with it
    if then is None:
        if not admit_new:
            return True  # newly measured, but past this quantity's learning rounds
        # Judged against what the plan was built on: a value that lands where the plan
        # already stood would re-plan into the same plan.
        return planned is not None and abs(current - planned) < 2
    if settled or not admit_new or planned is None:
        # Only drift from what it measured when the plan was built re-plans it now.
        return abs(current - then) < 2
    return abs(current - planned) < 2


def dependencies_hold(hub: Any, snapshot: Snapshot, rounds: int = 0, settled: bool = False) -> bool:
    """Whether every dependency in `snapshot` still supports the plan built on it.

    A measured value supports a plan while it stays within a bucket of the value the plan
    was built on. That is the question that decides whether re-planning could change
    anything. Asking instead whether a measurement *appeared* or *moved* re-planned TPC-DS q37
    on each of its second, third and fourth runs, for ~45 ms each against a 23 ms query, and
    every one of those re-plans produced the plan it replaced: the first measurements landed
    near the estimates, and a correction factor kept drifting while the structural estimate it
    multiplies sharpened beneath it, so the corrected row count never moved. For a correctable
    operator the dependency is therefore its row count (the latest measured output against the
    corrected estimate the plan used), not the factor.

    Args:
        hub: The metadata hub the measurements are folded from.
        snapshot: What `dependency_snapshot` recorded when the plan was built.
        rounds: How many times this plan's key was already re-planned for its dependencies.
            Past its quantity's rounds (`LEARNING_ROUNDS` for a selectivity, one for a width
            or correction), a newly appeared measurement no longer invalidates.
        settled: The plan's last re-plan reproduced it (`plan_cache.memo`), so the values it
            was built on are known not to steer it: a dependency measured then re-plans only
            by drifting a full bucket. One measured since is still judged as usual.

    Returns:
        False when any dependency lost a measurement, or measures a value more than a bucket
        from what the plan was built on (a new measurement only while the key is within its
        learning rounds).
    """
    if hub is None or not snapshot:
        return True
    admit_sel = rounds < LEARNING_ROUNDS
    admit_other = rounds < _WIDTH_CORRECTION_ROUNDS
    sel, corr, width, rows = _measured(hub)
    for entry in snapshot:
        sig, sel_then, corr_then, width_then = entry[:4]
        sel_used, rows_used, width_used, rows_then = entry[4:8] or (None,) * 4
        if not _holds(sel_then, _bucket(sel.get(sig)), sel_used, admit_sel, settled):
            return False
        if not _holds(width_then, _bucket(width.get(sig)), width_used, admit_other, settled):
            return False
        if rows_used is not None:
            if not _holds(rows_then, _bucket(rows.get(sig)), rows_used, admit_other, settled):
                return False
        elif not _holds(corr_then, _bucket(corr.get(sig)), None, admit_other, settled):
            return False
    return True
