"""Turning a timestamp into features a model can use.

A raw timestamp is the least useful column in a feature table. A tree model splits it into
"before and after some instant", which generalizes to nothing; a linear model treats it as
a number that grows forever. What a model can actually learn from is the *parts* — the hour
of the day, the day of the week, whether it is a weekend — because those repeat.

Two preprocessors, and the difference between them is what a model can express:

`DateTimeFeaturizer`
    Expands a timestamp into its calendar parts as ordinary integer columns. This is what a
    tree model wants: it can split on "hour >= 18" directly.
`CyclicalEncoder`
    Encodes a periodic part as a ``(sin, cos)`` pair. This is what a linear model, a
    distance metric, or a neural net needs, because the integer encoding puts hour 23 and
    hour 0 twenty-three units apart when they are one hour apart — and no amount of scaling
    fixes that. On the unit circle they are adjacent, which is the whole point.

Both are stateless: the same expression applies to training and serving data with nothing
fitted in between, so there is no state to persist and no way for the two to skew.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir.constructors import col, lit
from batcher.plan.functions.temporal import date_part

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["CyclicalEncoder", "DateTimeFeaturizer"]

#: The calendar parts `DateTimeFeaturizer` can extract, and how to build each one.
_PARTS: dict[str, str] = {
    "year": "date_part",
    "quarter": "date_part",
    "month": "date_part",
    "day": "date_part",
    "hour": "date_part",
    "minute": "date_part",
    "second": "date_part",
    "weekday": "accessor",
    "day_of_year": "accessor",
    "week_of_year": "accessor",
    "is_weekend": "accessor",
    "is_month_start": "accessor",
    "is_month_end": "accessor",
}

#: The default part set: everything a model can usually learn from, without the ones that
#: only make sense for a particular grain (seconds on a daily table, year on a one-year one).
DEFAULT_PARTS = ("year", "month", "day", "hour", "weekday", "week_of_year", "is_weekend")

#: The period of each cyclical part — how many distinct values before it wraps.
CYCLE_PERIODS: dict[str, float] = {
    "month": 12.0,
    "day": 31.0,
    "hour": 24.0,
    "minute": 60.0,
    "second": 60.0,
    "weekday": 7.0,
    "day_of_year": 366.0,
    "week_of_year": 53.0,
    "quarter": 4.0,
}


def _part_expr(column: str, part: str) -> Expr:
    """The expression extracting one calendar `part` from a timestamp `column`."""
    source = col(column)
    if _PARTS.get(part) == "date_part":
        return date_part(part, source)
    if part == "weekday":
        return source.dt.weekday()
    if part == "day_of_year":
        return source.dt.ordinal_day()
    if part == "week_of_year":
        return source.dt.weekofyear()
    if part == "is_weekend":
        return source.dt.is_weekend()
    if part == "is_month_start":
        return source.dt.is_month_start()
    return source.dt.is_month_end()


def _check_parts(parts: Sequence[str], known: dict[str, float] | dict[str, str]) -> list[str]:
    """Validate a requested part list against what is available."""
    names = list(parts)
    if not names:
        raise PlanError("at least one calendar part is required")
    for name in names:
        if name not in known:
            from batcher._internal.errors import suggestion

            hint = suggestion(name, sorted(known))
            tail = f" {hint}" if hint else ""
            raise PlanError(
                f"unknown calendar part {name!r}; expected one of {sorted(known)}.{tail}"
            )
    return names


class DateTimeFeaturizer(Preprocessor):
    """Expand a timestamp column into its calendar parts as separate columns.

    The first thing to do with any timestamp in a feature table. The raw column stays, so
    it is still available for sorting, joining, or a time-series split; the parts are
    appended as ``{column}_{part}``.

    Stateless — nothing is learned, so the same expression applies to training and serving
    data and there is no state to keep in sync.

    Examples:
        .. doctest::

            >>> import datetime as dt
            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import DateTimeFeaturizer
            >>> ds = bt.from_pydict({"t": [dt.datetime(2024, 3, 16, 14, 30)]})
            >>> out = DateTimeFeaturizer("t", parts=["hour", "is_weekend"]).fit_transform(ds)
            >>> out.to_pydict()["t_hour"], out.to_pydict()["t_is_weekend"]
            ([14], [True])

    Args:
        columns: The timestamp columns to expand.
        parts: Which calendar parts to extract; `DEFAULT_PARTS` when omitted.
        drop_original: Remove the source timestamp column after expanding it.
    """

    __slots__ = ("columns", "drop_original", "parts")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        parts: Sequence[str] = DEFAULT_PARTS,
        drop_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="DateTimeFeaturizer")
        self.parts = _check_parts(parts, _PARTS)
        self.drop_original = drop_original

    def transform(self, ds: Dataset) -> Dataset:
        """Append one column per ``(column, part)`` pair.

        Examples:
            .. doctest::

                >>> import datetime as dt
                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import DateTimeFeaturizer
                >>> ds = bt.from_pydict({"t": [dt.datetime(2024, 7, 4)]})
                >>> DateTimeFeaturizer("t", parts=["month"]).transform(ds).columns
                ['t', 't_month']

        Args:
            ds: The dataset to expand.

        Returns:
            A new lazy `Dataset` with the calendar-part columns appended.
        """
        projections = {
            f"{name}_{part}": _part_expr(name, part) for name in self.columns for part in self.parts
        }
        out = ds.with_columns(**projections)
        return out.drop(*self.columns) if self.drop_original else out


class CyclicalEncoder(Preprocessor):
    """Encode a periodic calendar part as a ``(sin, cos)`` pair on the unit circle.

    The fix for the wrap-around problem that no scaler solves. Encoded as an integer, hour
    23 and hour 0 are 23 units apart while being one hour apart, so every distance-based
    model, every linear model, and every neural net learns a discontinuity at midnight that
    is not there. Two coordinates on a circle put them adjacent, which is the truth.

    Appends ``{column}_{part}_sin`` and ``{column}_{part}_cos``. Stateless.

    Examples:
        .. doctest::

            >>> import datetime as dt
            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import CyclicalEncoder
            >>> midnight, late = dt.datetime(2024, 1, 1, 0), dt.datetime(2024, 1, 1, 23)
            >>> ds = bt.from_pydict({"t": [midnight, late]})
            >>> out = CyclicalEncoder("t", parts=["hour"]).fit_transform(ds).to_pydict()
            >>> round(out["t_hour_cos"][0], 4), round(out["t_hour_cos"][1], 4)
            (1.0, 0.9659)

    Args:
        columns: The timestamp columns to encode.
        parts: Which periodic parts to encode; ``("hour", "weekday", "month")`` by default.
        drop_original: Remove the source timestamp column after encoding it.
    """

    __slots__ = ("columns", "drop_original", "parts")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        parts: Sequence[str] = ("hour", "weekday", "month"),
        drop_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="CyclicalEncoder")
        self.parts = _check_parts(parts, CYCLE_PERIODS)
        self.drop_original = drop_original

    def transform(self, ds: Dataset) -> Dataset:
        """Append a sine and cosine column for each ``(column, part)`` pair.

        Examples:
            .. doctest::

                >>> import datetime as dt
                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import CyclicalEncoder
                >>> ds = bt.from_pydict({"t": [dt.datetime(2024, 1, 1, 6)]})
                >>> CyclicalEncoder("t", parts=["hour"]).transform(ds).columns
                ['t', 't_hour_sin', 't_hour_cos']

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with the circular coordinates appended.
        """
        projections = {}
        for name in self.columns:
            for part in self.parts:
                angle = _part_expr(name, part).cast("float64") * lit(
                    2.0 * math.pi / CYCLE_PERIODS[part]
                )
                projections[f"{name}_{part}_sin"] = angle.sin()
                projections[f"{name}_{part}_cos"] = angle.cos()
        out = ds.with_columns(**projections)
        return out.drop(*self.columns) if self.drop_original else out
