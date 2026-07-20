"""Findings about how the work was distributed and how well it was predicted.

`dominant_operator` and `long_tail` are deliberately opposites: one says fix this step,
the other says there is no step to fix. Keeping them together makes that pairing visible."""

from __future__ import annotations

from typing import Any

from .kinds import (
    _BOTTLENECK_SHARE,
    _EST_ERROR_HIGH,
    _EST_ERROR_LOW,
    _LONG_TAIL_MAX_SHARE,
    _LONG_TAIL_MIN_MS,
    _LONG_TAIL_MIN_OPS,
    _PLANNING_MIN_MS,
    _PLANNING_SHARE,
    _TRIVIAL_MS,
    Insight,
    count,
)

__all__ = ["bad_estimates", "dominant_operator", "long_tail", "planning_dominates"]


def bad_estimates(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A wrong cardinality estimate is why a plan is wrong, not merely slow."""
    out: list[Insight] = []
    for op in ops:
        error = op.get("est_error")
        if error is None or not (error > _EST_ERROR_HIGH or error < _EST_ERROR_LOW):
            continue
        if float(op.get("elapsed_ms", 0.0)) < 1.0:
            continue  # a sub-millisecond operator's misestimate changed no decision
        direction = "more" if error > 1 else "fewer"
        out.append(
            Insight(
                severity="warning",
                rule="cardinality-misestimate",
                title=f"{op.get('kind')} produced {error:.0f}x the planned rows"
                if error > 1
                else f"{op.get('kind')} produced {1 / error:.0f}x fewer rows than planned",
                evidence=(
                    f"Planned {count(op.get('est_rows'))} rows, produced "
                    f"{count(op.get('rows_out'))} — {direction} than Kyber costed the plan for."
                ),
                action=(
                    "Run it again: the measured cardinality is recorded, so the next plan for "
                    "this shape is costed from fact. To adapt within a single run, enable "
                    "adaptive re-optimization (collect(adaptive=True))."
                ),
                op=str(op.get("kind", "")),
                detail={"est_error": error, "op_id": op.get("op_id")},
            )
        )
    return out[:3]  # the worst few; a wrong scan estimate poisons every operator above it


def dominant_operator(
    _profile: dict[str, Any], ops: list[dict[str, Any]], total_ms: float
) -> list[Insight]:
    """When one operator is most of the *operator* time, it is the only one worth tuning.

    The denominator is the sum of per-operator elapsed time, **not** the query's wall time.
    Operators run concurrently across morsel threads, so their elapsed times overlap and
    routinely sum past wall-clock — on a 16-thread box a measured run summed to 267% of it,
    which made a wall-time denominator report "hash_join is 199% of the runtime". A share
    over 100% is self-evidently broken, and one visibly wrong number teaches a user to
    distrust every other number on the page.
    """
    if total_ms < _TRIVIAL_MS:
        return []
    op_total = sum(float(op.get("elapsed_ms", 0.0)) for op in ops)
    if op_total <= 0:
        return []
    slowest = max(ops, key=lambda op: float(op.get("elapsed_ms", 0.0)))
    elapsed = float(slowest.get("elapsed_ms", 0.0))
    share = elapsed / op_total
    if share < _BOTTLENECK_SHARE:
        return []
    return [
        Insight(
            severity="info",
            rule="dominant-operator",
            title=f"{slowest.get('kind')} is {share * 100:.0f}% of operator time",
            evidence=(
                f"{slowest.get('kind')} took {elapsed:.0f}ms of {op_total:.0f}ms summed across "
                f"operators, producing {count(slowest.get('rows_out'))} rows. (Operators run "
                f"concurrently, so this sum exceeds the {total_ms:.0f}ms wall time.)"
            ),
            action=(
                "Tune this operator first — everything else is rounding error. If it is a join, "
                "check the build side; if a sort, whether a top-N would do; if a scan, whether "
                "the predicate is reaching the reader."
            ),
            op=str(slowest.get("kind", "")),
            detail={"share": share, "elapsed_ms": elapsed},
        )
    ]


def long_tail(
    _profile: dict[str, Any], ops: list[dict[str, Any]], total_ms: float
) -> list[Insight]:
    """Time spread thinly across very many steps rather than concentrated in one.

    The opposite shape from a dominant operator, and it needs the opposite advice: there is
    no single step to fix, so the win is in doing fewer steps rather than a faster one.
    """
    if total_ms < _LONG_TAIL_MIN_MS or len(ops) < _LONG_TAIL_MIN_OPS:
        return []
    summed = sum(float(op.get("elapsed_ms", 0.0) or 0.0) for op in ops)
    if summed <= 0:
        return []
    worst = max(float(op.get("elapsed_ms", 0.0) or 0.0) for op in ops)
    if worst > summed * _LONG_TAIL_MAX_SHARE:
        return []
    return [
        Insight(
            severity="info",
            rule="long-tail",
            title=f"No single step dominates — {len(ops)} steps share the time",
            evidence=(
                f"The largest step is only {worst / summed:.0%} of operator time across "
                f"{len(ops)} measured steps, so there is no one bottleneck to attack."
            ),
            action=(
                "Look for work to remove rather than work to speed up: fewer columns carried, "
                "fewer intermediate steps, or a filter that lets several later steps see less."
            ),
            detail={"steps": len(ops), "top_share": round(worst / summed, 3)},
        )
    ]


def planning_dominates(
    _profile: dict[str, Any], ops: list[dict[str, Any]], total_ms: float
) -> list[Insight]:
    """More time spent choosing a plan than running it.

    Normal and fine for a tiny query — but worth saying out loud, because the reader is
    otherwise looking for a slow operator that does not exist.
    """
    executed = sum(float(op.get("elapsed_ms", 0.0) or 0.0) for op in ops)
    overhead = total_ms - executed
    if total_ms < _PLANNING_MIN_MS or overhead < total_ms * _PLANNING_SHARE:
        return []
    return [
        Insight(
            severity="info",
            rule="planning-dominates",
            title="Most of this run was not spent in the steps",
            evidence=(
                f"The steps account for {executed:.0f} ms of a {total_ms:.0f} ms run, leaving "
                f"{overhead:.0f} ms in planning, setup and result delivery."
            ),
            action=(
                "For a query this small that is expected — fixed overhead dominates. Re-running "
                "the same shape reuses the plan, so a second run is usually faster."
            ),
            detail={"executed_ms": round(executed, 1), "overhead_ms": round(overhead, 1)},
        )
    ]
