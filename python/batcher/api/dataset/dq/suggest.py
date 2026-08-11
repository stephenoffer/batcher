"""Read a contract off the data, so the first version of one is not written from memory.

The hardest part of a data contract is the blank page. Nobody knows, without looking, which
of two hundred columns are never null, which are keys, which are really enums with nine
values, and which are 3% missing by design. So the contract that gets written is the one
somebody remembered, and the columns nobody thought of go unchecked for years.

This reads the shape of the data and proposes the constraints it already satisfies — Deequ's
constraint suggestion and Great Expectations' profiler. Everything it proposes is *true of
this data right now*, which is the point and also the limit: a suggestion is a starting
point to edit, not a contract to trust. A column that happens to be complete in today's
sample gets a `not_null` it may not deserve.

Deliberately conservative. It proposes what a mistake would be cheap to notice
(completeness, keys, sign, small enumerations, an observed null rate with headroom) and
never proposes a bound it cannot defend: no `in_range` off an observed min and max, because
tomorrow's legitimate value is outside today's, and a contract that cries wolf is one that
gets deleted.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher.api.dataset.dq.checks import aggregates, strings, values
from batcher.api.dataset.dq.constraints import Constraint, UniqueConstraint

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["suggest"]

#: Below this many distinct values, a text column is treated as an enumeration rather than
#: as free text. Twenty-five covers currencies, statuses, country codes and the like without
#: pulling in a column of names.
DEFAULT_MAX_CATEGORIES = 25

#: The number of enumeration candidates whose values are actually read. Each one costs its
#: own `distinct()` pass, so the budget is what keeps "profile this table" from turning into
#: a hundred scans on a wide one.
_CATEGORY_BUDGET = 8


def _is_numeric(dtype: pa.DataType) -> bool:
    """Whether the column holds a number this module is willing to reason about."""
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)


def _is_text(dtype: pa.DataType) -> bool:
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _null_rate_bound(fraction: float) -> float | None:
    """A null-rate bound with headroom above what was observed, or None if there is no room.

    Rounded up to the next 5% so a contract written off a 3.1% sample does not fail the first
    time the feed reaches 3.2%. Above half missing there is nothing worth asserting.
    """
    if fraction <= 0.0 or fraction > 0.5:
        return None
    return min(0.5, math.ceil(fraction * 20 + 1) / 20)


def _profile_rows(ds: Dataset, columns: list[str]) -> list[dict[str, Any]]:
    """One profile row per column: count, null_count, null_fraction, approx_distinct."""
    profiled = ds.select(*columns) if columns else ds
    table = profiled.profile().to_pydict()
    return [dict(zip(table, row, strict=True)) for row in zip(*table.values(), strict=True)]


def _numeric_bounds(ds: Dataset, numeric: list[str]) -> dict[str, float | None]:
    """The minimum of every numeric column, measured in one keyless aggregate."""
    if not numeric:
        return {}
    from batcher.plan.expr_ir import Col

    row = ds.agg(**{f"__dq_min_{i}": Col(c).min() for i, c in enumerate(numeric)}).to_pydict()
    out: dict[str, float | None] = {}
    for i, c in enumerate(numeric):
        value = row[f"__dq_min_{i}"][0]
        out[c] = None if value is None else float(value)
    return out


def _categories(ds: Dataset, column: str, limit: int) -> list[Any] | None:
    """The distinct non-null values of `column`, or None if there are more than `limit`."""
    from batcher.plan.expr_ir import Col

    seen = (
        ds.select(column).filter(Col(column).is_not_null()).distinct().limit(limit + 1).to_pydict()
    )
    observed = seen[column]
    if not observed or len(observed) > limit:
        return None
    return sorted(observed)


def suggest(
    ds: Dataset,
    columns: list[str] | None = None,
    *,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> list[Constraint]:
    """Constraints this dataset already satisfies, in declaration-friendly order.

    Args:
        ds: The dataset to profile.
        columns: The columns to consider; defaults to every column.
        max_categories: Below this many distinct values, a text column is proposed as an
            enumeration rather than left unconstrained.

    Returns:
        The proposed constraints, schema-shaped ones first.
    """
    schema = ds.schema
    names = list(columns) if columns else list(schema.names)
    proposed: list[Constraint] = []
    rows = _profile_rows(ds, names)
    numeric = [c for c in names if _is_numeric(schema.field(c).type)]
    minimums = _numeric_bounds(ds, numeric)
    budget = _CATEGORY_BUDGET
    for row in rows:
        column = str(row["column"])
        if column not in names:
            continue
        dtype = schema.field(column).type
        count = int(row["count"] or 0)
        nulls = int(row["null_count"] or 0)
        distinct = int(row["approx_distinct"] or 0)
        if nulls == 0 and count:
            proposed.append(values.not_null((column,)))
        elif (bound := _null_rate_bound(float(row["null_fraction"] or 0.0))) is not None:
            proposed.append(aggregates.null_rate_below(column, bound))
        # A key is a column whose distinct count reaches its non-null row count. The count is
        # a HyperLogLog estimate, so this is a *proposal* — on a real key it is exact often
        # enough to be worth making, and wrong in the safe direction (a missed key) otherwise.
        if count > 1 and nulls == 0 and distinct == count:
            proposed.append(UniqueConstraint(f"unique({column})", (column,)))
        if column in minimums and minimums[column] is not None:
            low = minimums[column]
            if low >= 0:
                proposed.append(values.positive(column, strict=low > 0))
        if pa.types.is_floating(dtype):
            proposed.append(values.is_finite(column))
        # An enumeration is a *small vocabulary each of whose values recurs*. Testing only
        # "few distinct values" makes every short table's free-text column an enum — a
        # six-row table of six distinct notes is under any sane category ceiling — and the
        # resulting `accepted_values` would reject the seventh note. Requiring each value to
        # appear about twice separates a vocabulary from an identifier at any table size.
        categorical = 0 < distinct <= max_categories and distinct * 2 <= count
        if _is_text(dtype) and budget and categorical:
            members = _categories(ds, column, max_categories)
            budget -= 1
            if members:
                proposed.append(values.accepted_values(column, members))
                if all(isinstance(v, str) and v.strip() for v in members):
                    proposed.append(strings.not_empty(column))
    return proposed
