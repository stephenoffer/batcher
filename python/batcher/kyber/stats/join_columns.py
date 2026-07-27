"""Join column-statistics propagation.

Split from `stats.columns` because a join is the one operator whose output statistics are
not a per-side transformation: an equi-join relates the two inputs' key columns to each
other, so a key's surviving values are the *intersection* of the two sides' value sets and
each side's bounds may be tightened by the other's. That cross-side reasoning is what lives
here; the single-input propagators stay in `stats.columns`.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
from typing import Any

from batcher.plan.logical import AsofJoin, Join, RangeJoin
from batcher.plan.stats import ColumnStat, Provenance, RelStats, ambiguous_float_bound

__all__ = ["asof_join_columns", "join_columns", "range_join_columns"]


def join_columns(
    node: Join, left: RelStats, right: RelStats, out_rows: float | None = None
) -> dict[str, ColumnStat]:
    """Column stats for a join's output: each side's values as downgraded *bounds*.

    A join only removes rows from a side or repeats them (an FK match), never invents
    a new value, so a preserved column's `min`/`max` stay valid *bounds* — but the
    extremes may be dropped and provenance must fall from `EXACT` (a join is a
    row-shrinking/duplicating operator). `null_count` is dropped (a match may duplicate
    rows or an outer join add nulls). The membership bloom survives — a value absent
    from a side stays absent in any join output of it.

    `ndv` is carried forward as `min(ndv_in, out_rows)`. Because a join invents no
    values, the output's distinct count for a preserved column can only *fall* (rows
    are dropped) — never rise — and it is trivially bounded by the output row count;
    the minimum of the two is therefore a sound upper bound and the standard Selinger
    estimate. Dropping it instead (the previous behaviour) left every join *above* a
    join with no key NDV, so `_estimate_join` fell back to `max(|L|, |R|)` and could
    not see that an upstream selective join shrinks the pipeline — which is what
    steered TPC-H Q9 into multi-gigabyte `lineitem ⋈ partsupp ⋈ orders` intermediates
    before joining the 5%-selective `part`. `out_rows` is the join's estimated output
    cardinality; when unknown, `ndv` is dropped as before.
    """
    keys = _matched_key_stats(node, left, right)
    out: dict[str, ColumnStat] = {}
    for o in node.output:
        side = left if o.side == "left" else right
        src = side.columns.get(o.name)
        if src is not None:
            paired = keys.get((o.side, o.name))
            if paired is not None:
                src = _intersect_key_stat(src, paired)
            out[o.alias] = dataclasses.replace(
                src.downgrade(Provenance.DEFAULT),
                null_count=None,
                ndv=_join_ndv(src.ndv, out_rows),
                total_sum=None,  # matching duplicates rows, so a recorded sum no longer holds
                mean=None,
                # The measured *width* survives — a join changes which rows are present, not
                # how many bytes one holds — and it is what keeps byte-true memory and
                # broadcast sizing alive above a join. Frequencies (mcv) do not: a join
                # re-weights the value distribution by its match multiplicity.
                mcv=None,
            )
    return out


def asof_join_columns(node: AsofJoin, left: RelStats, right: RelStats) -> dict[str, ColumnStat]:
    """Column stats for an ASOF join's output.

    ASOF is **left-style**: every left row is emitted exactly once, so the left columns'
    values are exactly the left input's — their full stats (EXACT included) carry through
    unchanged. A right column is preserved where matched and NULL where not, and the nearest
    match can repeat or skip right rows, so it survives only as downgraded *bounds* with the
    null count dropped — the same treatment `join_columns` gives a preserved side. Without
    this the estimator returned no columns at all, blinding every operator above an ASOF join
    to the very statistics (bounds, ndv, byte width) that drive its cost and broadcast sizing.
    """
    out: dict[str, ColumnStat] = {}
    for o in node.output:
        if o.side == "left":
            src = left.columns.get(o.name)
            if src is not None:
                out[o.alias] = src  # every left row emitted once → values unchanged
        else:
            src = right.columns.get(o.name)
            if src is not None:
                out[o.alias] = dataclasses.replace(
                    src.downgrade(Provenance.DEFAULT), null_count=None, mcv=None
                )
    return out


def _join_ndv(ndv: float | None, out_rows: float | None) -> float | None:
    """A preserved column's distinct count after a join: `min(ndv, out_rows)`."""
    if ndv is None or out_rows is None:
        return None
    return max(1.0, min(ndv, out_rows))


# Join types whose output holds only *matched* key values on both sides, so an equi-key's
# surviving values are the set intersection of the two inputs'. An outer join preserves its
# outer side's unmatched rows, so no intersection may be claimed there.
_MATCHED_ONLY_JOINS = frozenset({"inner", "semi"})


def _matched_key_stats(
    node: Join, left: RelStats, right: RelStats
) -> dict[tuple[str, str], ColumnStat]:
    """`{(side, key_column): the opposite side's stat for its paired key}`.

    Only for a join whose output holds matched rows exclusively. This is what lets an
    equi-key's statistics be *intersected* rather than merely carried through: `l.k = r.k`
    means the surviving values of both columns are the same set, so each side's bounds may be
    tightened by the other's.
    """
    if node.join_type not in _MATCHED_ONLY_JOINS:
        return {}
    paired: dict[tuple[str, str], ColumnStat] = {}
    for lk, rk in zip(node.left_keys, node.right_keys, strict=False):
        lstat, rstat = left.columns.get(lk), right.columns.get(rk)
        if lstat is not None and rstat is not None:
            paired[("left", lk)] = rstat
            paired[("right", rk)] = lstat
    return paired


def _intersect_key_stat(stat: ColumnStat, other: ColumnStat) -> ColumnStat:
    """Tighten one equi-join key's statistics with its partner's.

    An equi-join emits a row only where both keys hold the *same* value, so the surviving
    values lie in the intersection of the two columns' value sets. Two consequences are
    exact, and both are free — they need no execution, only the bounds each side already
    carries:

    * **bounds** — `[max(min_L, min_R), min(max_L, max_R)]`. A three-year fact table joined
      to a one-year dimension emits only that year, and a range predicate above the join
      should see the narrow range, not the wide one.
    * **ndv** — at most `min(d_L, d_R)`, since a value must appear on both sides to survive.
      This is the containment bound, and it is much sharper than `min(d_side, out_rows)` for
      a fan-out join, where the output has many more rows than distinct keys.

    Bounds are only intersected when both are ordinal-comparable and unambiguous; a
    NaN/signed-zero float bound or a (possibly byte-truncated) string bound is left alone,
    since the engine's ordering and Python's can disagree there.
    """
    updates: dict[str, Any] = {}
    if stat.ndv is not None and other.ndv is not None:
        updates["ndv"] = max(1.0, min(stat.ndv, other.ndv))
    if _comparable_bound(stat.min) and _comparable_bound(other.min):
        updates["min"] = max(stat.min, other.min)
    if _comparable_bound(stat.max) and _comparable_bound(other.max):
        updates["max"] = min(stat.max, other.max)
    if not updates:
        return stat
    tightened = dataclasses.replace(stat, **updates)
    # An empty intersection means the join provably matches nothing; the row estimator
    # reports that separately, and emitting inverted bounds here would corrupt every
    # downstream selectivity. Keep the original bounds in that case.
    if (
        tightened.min is not None
        and tightened.max is not None
        and _comparable_bound(tightened.min)
        and _comparable_bound(tightened.max)
        and tightened.min > tightened.max
    ):
        return stat
    return tightened


def _comparable_bound(value: Any) -> bool:
    """Whether a bound may be compared against another with Python's own ordering."""
    return (
        value is not None
        and isinstance(value, (int, float, datetime.date, datetime.datetime, decimal.Decimal))
        and not isinstance(value, bool)
        and not ambiguous_float_bound(value)
    )


def range_join_columns(
    node: RangeJoin, left: RelStats, right: RelStats, out_rows: float | None = None
) -> dict[str, ColumnStat]:
    """Column stats for a range join's output, on the same reasoning as `join_columns`.

    A range join removes rows from a side or repeats them and never invents a value, so a
    preserved column's `min`/`max` stay valid *bounds* with provenance downgraded, and the
    membership bloom survives. What it does *not* share with the equi-join case is the
    cross-side key reasoning: an inequality relates the two key columns by *order*, not by
    equality, so neither side's value set is narrowed to the other's — the intersection
    tightening `join_columns` applies to a matched key pair would be unsound here.
    """
    out: dict[str, ColumnStat] = {}
    for o in node.output:
        side = left if o.side == "left" else right
        src = side.columns.get(o.name)
        if src is not None:
            out[o.alias] = dataclasses.replace(
                src.downgrade(Provenance.DEFAULT),
                null_count=None,
                ndv=_join_ndv(src.ndv, out_rows),
                total_sum=None,
                mean=None,
                mcv=None,
            )
    return out
