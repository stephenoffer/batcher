"""What the engine types as a calendar day, and how a temporal value is built from numbers.

Split from `temporal`, which moves an *existing* instant around the calendar. This module
answers the two questions that come before that: which expressions the engine presents as an
Arrow DATE rather than a timestamp, and how an epoch count or a `(year, month, day)` triple
becomes a temporal value at all.

The typing question is not cosmetic and cannot be read off the computed column: **neither
dataframe library has a calendar-day type**. A date becomes a datetime the moment it enters a
frame, and the host backend's `astype` happens to land back on `date32` while the device's
cannot — so a computed date came back as `timestamp[ms]` from a real GPU and as `date32` from
CI. The expression has to say what it is.
"""

from __future__ import annotations

from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.temporal import (
    _I32_MAX,
    _date32,
    _int32,
    _int64,
    _timestamp_us,
    epoch_micros,
)

__all__ = ["date_typed", "eval_make_temporal", "eval_window_start"]

#: Epoch conversions that are a scaling into microseconds, as their multiplier. `from_unix_nanos`
#: is absent because it *divides*, and with a floor rather than a truncation.
_UNIX_SCALE = {"from_unix_seconds": 1_000_000, "from_unix_millis": 1_000, "from_unix_micros": 1}


#: Expressions that keep their input's DATE-ness. A day offset over a calendar day is another
#: calendar day; over an instant it is another instant.
_DATE_PRESERVING = frozenset({"date_offset"})


def date_typed(ir: dict, be) -> bool:
    """Whether the expression `ir` produces an Arrow DATE rather than a timestamp.

    Asked by the projection so it can tell the backend to present the column as a date. Neither
    dataframe library has a calendar-day type, so the answer cannot be read off the computed
    column — it has to come from the expression, and from which of its inputs were dates.

    Deliberately narrow: it answers for the three shapes the engine actually types as DATE and
    returns False for everything else, so an expression nobody classified is presented as what
    the library produced rather than cast to something it is not.

    Args:
        ir: The expression's JSON IR node.
        be: The dataframe backend, which knows which columns arrived as dates.

    Returns:
        True when the engine would type this expression as a DATE.
    """
    kind = ir.get("e")
    if kind == "cast":
        return ir.get("dtype") == "date"
    if kind == "date":
        # `last_day` is the one date *function* that returns a calendar day; every other one
        # returns a number.
        return ir.get("fn") == "last_day"
    if kind == "col":
        return be.is_date_column(ir["name"])
    if kind in _DATE_PRESERVING:
        return date_typed(ir["input"], be)
    return False


def eval_make_temporal(fn: str, args: list, be):
    """The epoch constructors — an integer column read as an instant.

    Only the `from_unix_*` family is translated. `make_date` and `make_timestamp` are not: the
    engine builds them through a calendar type that *validates*, so a February 30 or a month 13
    comes back null, and a construction that computed a day count instead would silently return
    the following month's first day. `days_from_civil` is a mapping, not a validator, and the
    validation is the whole contract of those two.

    Args:
        fn: The constructor's name.
        args: Its evaluated arguments, positionally.
        be: The dataframe backend to compute on.

    Returns:
        The constructed column, typed as the engine types it.

    Raises:
        Unsupported: For a constructor outside the `from_unix_*` family.
    """
    if fn not in _UNIX_SCALE and fn not in ("from_unix_nanos", "from_unix_date"):
        raise Unsupported(f"make_temporal {fn}")
    value = args[0].astype(be.dtype(_int64()))
    if fn == "from_unix_nanos":
        # Floor, not truncation, so a pre-1970 sub-microsecond instant lands in the microsecond
        # that contains it rather than one later — the same rule `epoch` follows.
        return (value // 1_000).astype(be.dtype(_timestamp_us()))
    if fn == "from_unix_date":
        # A day count, so it is already a DATE's own representation and needs no scaling —
        # which is just as well, because the widest Date32 is a year in the millions and
        # multiplying it up to microseconds would not fit in an instant at all. Outside the
        # 32-bit range the engine yields null rather than a wrapped date.
        inside = ((value <= _I32_MAX) & (value >= -_I32_MAX - 1)).fillna(False)
        narrowed = value.where(inside, 0).astype(be.dtype(_int32()))
        return narrowed.where(inside & value.notna(), None).astype(be.dtype(_date32()))
    # The engine scales *checked*, so a value too large to be an instant is null rather than a
    # wrapped one — a plausible date in the wrong millennium.
    scale = _UNIX_SCALE[fn]
    return _scaled(value, scale, (2**63 - 1) // scale).astype(be.dtype(_timestamp_us()))


def _scaled(value, scale: int, limit: int):
    """`value * scale`, null wherever the product would not fit, without ever computing it.

    The out-of-range rows are replaced with zero *before* the multiply rather than masked
    after it. Arrow's multiply is checked and evaluates the slots under a null mask too, so
    masking first still raises on the value it was told to ignore — which turns a row the
    engine reports as null into a failure of the whole column, and on the device into a silent
    fallback of the whole query.
    """
    inside = ((value <= limit) & (value >= -limit - 1)).fillna(False)
    return (value.where(inside, 0) * scale).where(inside & value.notna(), None)


def eval_window_start(x, width_micros: int, origin_micros: int, be):
    """`window_start` — the start of the fixed-width tumbling window containing each instant.

    The bucketing key a streaming aggregate groups by, and pure integer arithmetic: the offset
    from the origin, floored to a multiple of the width, put back on the origin. Flooring is
    what makes an instant before the origin land in the window that contains it rather than the
    one after, which is the only case truncation would get wrong.

    Args:
        x: The timestamp column to bucket.
        width_micros: The window width; must be positive.
        origin_micros: The instant the window grid is anchored to.
        be: The dataframe backend to compute on.

    Returns:
        Each instant's window start, as `timestamp[us]`.

    Raises:
        Unsupported: For a non-positive width, which the engine rejects outright.
    """
    if width_micros <= 0:
        raise Unsupported("window_start with a non-positive width")
    us = epoch_micros(x, be)
    floored = ((us - origin_micros) // width_micros) * width_micros
    return (floored + origin_micros).astype(be.dtype(_timestamp_us()))
