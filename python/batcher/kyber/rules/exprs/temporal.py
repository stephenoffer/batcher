"""Temporal identities: reading a date part through a truncation, and offset fusion.

`extra/temporal_extra` collapses nested truncations and folds a date function over a
literal, and `extra/temporal_sargable` turns `year(t) = 2021` into a range. The gap
this module fills is the *interaction* between a truncation and the part read back
out of it, which DuckDB handles as `date_part_simplification`.

`year(date_trunc('day', t))` is the shape a dashboard query produces when a
day-granularity bucket is grouped by year. Truncating to a day zeroes the time fields
and leaves everything from the day upward untouched, so the year is the year of `t`
and the truncation is pure work -- a per-row calendar decomposition, a rebuild, and a
second decomposition, to arrive where a single field read would have. Removing it
also un-blocks the sargable normalizer, which recognizes `year(t)` and not
`year(trunc(t))`, so the predicate can become a range and prune row groups.

The soundness condition is a granularity comparison, and it is stated as a rank
table rather than inferred, because the calendar units do not nest as cleanly as they
look. `week` sits between `day` and `month` on purpose: a week does not divide a
month, so truncating to a month genuinely changes which week a timestamp falls in,
while truncating to a day does not. `iso_year` is ranked with `week` for the same
reason -- ISO year boundaries follow week boundaries, not January the first. `epoch`
is excluded entirely: it reports microseconds, so every truncation changes it.

Timezone conversions are deliberately absent from this module, including the two
rewrites that look safest. A same-zone `convert_timezone` is *not* the identity here:
the engine nulls out DST-ambiguous and nonexistent local times, so dropping the call
would resurrect those rows' original values. Merging a `a -> b -> c` chain into
`a -> c` fails for the same reason one level in -- the intermediate zone can null a
row that the direct conversion would keep. `tests/unit/test_temporal_extra.py` pins
the refusal.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.func_nodes import DateFunc, DateOffset, DateTrunc
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "DATE_PART_THROUGH_TRUNC_RULES",
    "combine_adjacent_date_offsets",
    "drop_zero_date_offset",
    "last_day_idempotent",
]

#: Calendar granularity, coarsening upward. A truncation to a unit leaves every field
#: at this rank or above untouched, and zeroes everything below. `week` sits above
#: `day` and below `month` because a week nests inside neither a month nor a quarter.
_GRANULARITY = {
    "microsecond": 0,
    "microseconds": 0,
    "millisecond": 1,
    "milliseconds": 1,
    "second": 2,
    "minute": 3,
    "hour": 4,
    "day": 5,
    "week": 6,
    "month": 7,
    "quarter": 8,
    "year": 9,
    "decade": 10,
    "century": 11,
    "millennium": 12,
    "millenium": 12,
}

#: The coarsest granularity each date function's answer depends on -- read as "this
#: function is unaffected by any truncation at or below this rank". `epoch` is absent
#: because it reports microseconds since the epoch, which every truncation moves.
_PART_GRANULARITY = {
    "second": 2,
    "minute": 3,
    "hour": 4,
    "day": 5,
    "day_of_week": 5,
    "day_of_year": 5,
    "isodow": 5,
    "dayname": 5,
    "week": 6,
    "iso_year": 6,
    "month": 7,
    "monthname": 7,
    "days_in_month": 7,
    "last_day": 7,
    "quarter": 8,
    "year": 9,
    "is_leap_year": 9,
    "decade": 10,
    "century": 11,
    "millennium": 12,
}


#: Truncation units a part may be lifted through. Rank order alone is **not** enough: the
#: truncation's boundaries must also *align* with every coarser calendar unit, or
#: truncating moves the instant across a boundary the part can see.
#:
#: Three families are excluded, and each was caught returning wrong answers by the
#: exhaustive guard in `tests/property/test_prop_optimizer_result_invariance.py`:
#:
#: * **`week`** -- a week does not nest inside a month, quarter, or year.
#:   `date_trunc('week', 2021-01-01)` is `2020-12-28`, so `month(...)` is 12 while the
#:   untruncated month is 1. Ranking `week` between `day` and `month` let the rule fire
#:   and return the *original* month. A week aligns with days and below, nothing above.
#: * **`century` / `millennium`** -- the engine's truncation and its extraction disagree
#:   on where the period starts. `date_trunc('century', 1969)` is `1900-01-01`, but
#:   `century()` uses the 1901-2000 convention, so the truncated instant reads as century
#:   19 where the original reads 20.
#: * **`decade`** -- excluded alongside them. It happens to agree today, but it is the
#:   same class of convention mismatch and nothing pins it.
#:
#: What remains is the chain that genuinely nests: sub-second through day, then month,
#: quarter, and year. A day never spans two months, a month never spans two quarters, and
#: a quarter never spans two years.
_LIFTABLE_TRUNC_UNITS = frozenset(
    {
        "microsecond",
        "microseconds",
        "millisecond",
        "milliseconds",
        "second",
        "minute",
        "hour",
        "day",
        "month",
        "quarter",
        "year",
    }
)


def _part_through_trunc(part: str):
    """Build the leaf rewrite lifting one date part through a fine-enough truncation."""
    part_rank = _PART_GRANULARITY[part]

    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, DateFunc) and expr.fn == part and isinstance(expr.input, DateTrunc):
            unit = expr.input.unit
            trunc_rank = _GRANULARITY.get(unit)
            if unit in _LIFTABLE_TRUNC_UNITS and trunc_rank is not None and trunc_rank <= part_rank:
                return DateFunc(part, expr.input.input)
        return expr

    return leaf


def _make_part_rule(part: str):
    """The node-local `f(node, ctx)` closure for one date part."""
    leaf = _part_through_trunc(part)

    def apply(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
        return rewrite_node(node, leaf)

    return apply


# One registered rule per date part, over a single shared body -- the pattern
# `extra/temporal_sargable` established for its `(extraction, operator)` cross-product.
# Registering per part rather than as one blanket rule is what makes `explain` name the
# part that actually fired, and each is independently indexed on `Filter`/`Project`.
#
# `year(date_trunc('day', t)) -> year(t)`: truncating to a unit zeroes the fields below
# it and leaves the rest alone, so a part at or above that rank reads the same value
# either way, and the truncation is a calendar decomposition plus a rebuild plus a
# second decomposition where one field read would do.
#
# Removing it matters beyond the cycles: the sargable normalizer recognizes `year(t)`
# and turns it into a half-open range zone maps can prune with, and does not recognize
# `year(trunc(t))`. Null propagates identically through both forms.
DATE_PART_THROUGH_TRUNC_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"{part}_through_finer_trunc",
            Phase.NORMALIZE,
            _make_part_rule(part),
            matches=(Filter, Project),
            expr_fn=_part_through_trunc(part),
        )
    )
    for part in sorted(_PART_GRANULARITY)
]


def _last_day_idempotent(expr: Expr) -> Expr:
    if (
        isinstance(expr, DateFunc)
        and expr.fn == "last_day"
        and isinstance(expr.input, DateFunc)
        and expr.input.fn == "last_day"
    ):
        return expr.input
    return expr


@rule(
    name="last_day_idempotent",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_last_day_idempotent,
)
def last_day_idempotent(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`last_day(last_day(t)) -> last_day(t)`. The last day of a month is itself in
    that month, so a second application returns the same date -- confirmed against the
    engine, which keeps the null row null through both."""
    return rewrite_node(node, _last_day_idempotent)


def _zero_offset(expr: Expr) -> Expr:
    if isinstance(expr, DateOffset) and expr.months == 0 and expr.days == 0 and expr.micros == 0:
        return expr.input
    return expr


@rule(
    name="drop_zero_date_offset",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_zero_offset,
)
def drop_zero_date_offset(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A `DateOffset` of zero months, days, and microseconds `-> its input`. Shifting a
    timestamp by nothing returns it, including at a daylight-saving boundary, since
    there is no calendar arithmetic to perform. These arise from a parameterized
    offset whose argument constant-folded to zero."""
    return rewrite_node(node, _zero_offset)


def _combine_offsets(expr: Expr) -> Expr:
    if (
        isinstance(expr, DateOffset)
        and isinstance(expr.input, DateOffset)
        and expr.months == 0
        and expr.input.months == 0
    ):
        inner = expr.input
        return DateOffset(inner.input, 0, inner.days + expr.days, inner.micros + expr.micros)
    return expr


@rule(
    name="combine_adjacent_date_offsets",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_combine_offsets,
)
def combine_adjacent_date_offsets(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Fuse two stacked day/microsecond offsets into one by adding their components.

    Fires only when *both* offsets carry zero months, and that restriction is the
    correctness argument rather than a simplification. Month arithmetic clamps to the
    end of the target month, which makes it non-associative: January 31 plus one month
    is February 28, and one further month is March 28, while January 31 plus two
    months is March 31. Days and microseconds are exact durations with no clamping, so
    they add freely and the fused offset is the same function as the pair."""
    return rewrite_node(node, _combine_offsets)
