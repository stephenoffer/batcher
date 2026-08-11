"""SQL temporal *construction* — parsing text, reading epoch counts, and bucketing time.

The sibling `literals` module owns temporal *literals* and interval arithmetic, and
`functions` owns the field extractions (`year`, `month`, ...). This module is the third
part of the temporal surface: the functions that build a timestamp rather than read one.

Each entry is a name a migrating DuckDB or Spark query types (`strptime`, `to_timestamp`,
`epoch_ms`, `make_timestamp`, `time_bucket`) mapped onto the engine node that already
implements it. Nothing here invents a semantic: where DuckDB's answer depends on a session
time zone (`to_timestamp` returns TIMESTAMPTZ, `make_timestamptz`) the instant is the same
but the rendering is not, and where the bucket origin is a calendar unit rather than a
fixed width (`time_bucket(INTERVAL 1 MONTH, ...)`) the call is refused rather than answered
with an epoch-aligned bucket that is off by DuckDB's 2000-01-01 origin.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Binary, Cast, Expr, lit
from batcher.plan.expr_ir.func_nodes import DateOffset, WindowStart
from batcher.plan.functions.temporal import (
    current_timestamp,
    from_epoch,
    from_unix_date,
    make_timestamp,
)
from batcher.plan.ir_tags import MICROS_PER_DAY

__all__ = ["temporal_function"]

# sqlglot records `epoch_ms`'s scale as the decimal exponent of the unit (3 for
# milliseconds); `to_timestamp` carries no scale at all.
#
# Scale 0 is seconds, and it used to be missing. Every unlisted scale fell through to a
# default of `"ms"`, so `TO_TIMESTAMP(n, 0)` and `TO_TIMESTAMP(n, 3)` returned the *same*
# instant — a silent 1000x error on the one spelling a Snowflake port is most likely to
# use. There is no safe default here, so an unrecognized scale now raises.
_SCALE_UNIT = {"0": "s", "3": "ms", "6": "us", "9": "ns"}

# The `.dt` reader for each unit. Seconds is spelled `epoch`, not `epoch_s`, so the unit
# name cannot be interpolated into the method name for all four.
_UNIT_EPOCH_METHOD = {"s": "epoch", "ms": "epoch_ms", "us": "epoch_us", "ns": "epoch_ns"}

# `make_timestamp_ms(n)` — DuckDB's epoch constructors that sqlglot leaves anonymous.
_EPOCH_NAME_UNIT = {"make_timestamp_ms": "ms", "make_timestamp_ns": "ns"}

# Spark's epoch constructors, which sqlglot also leaves anonymous. Same function as
# DuckDB's `make_timestamp_*` under the names a ported Spark job types.
_SPARK_EPOCH_UNIT = {
    "timestamp_seconds": "s",
    "timestamp_millis": "ms",
    "timestamp_micros": "us",
    "timestamp_nanos": "ns",
}

# Spark/Java datetime pattern letters → the chrono/strftime specifier the engine's
# `strftime`/`to_datetime` take. Longest pattern first, so `yyyy` is not read as two
# `yy`s. Only the unambiguous letters are here: a pattern using one that is absent is
# refused rather than formatted with a specifier that means something else.
_JAVA_PATTERN = [
    ("yyyy", "%Y"),
    ("YYYY", "%Y"),
    ("MMMM", "%B"),
    ("MMM", "%b"),
    ("EEEE", "%A"),
    ("EEE", "%a"),
    ("SSS", "%3f"),
    ("yy", "%y"),
    ("MM", "%m"),
    ("dd", "%d"),
    ("HH", "%H"),
    ("hh", "%I"),
    ("mm", "%M"),
    ("ss", "%S"),
    ("DDD", "%j"),
    ("y", "%Y"),
    ("M", "%m"),
    ("d", "%d"),
    ("H", "%H"),
    ("m", "%M"),
    ("s", "%S"),
    ("a", "%p"),
]

# The Julian day of 1970-01-01T00:00 as DuckDB counts it. (The astronomical Julian day
# begins at noon, so the textbook constant is 2440587.5; DuckDB's `julian` reports the
# day that *contains* the instant, which is a half-day later.)
_JULIAN_EPOCH = 2440588.0

# `time_bucket` widths, in microseconds, for the fixed-length interval units. MONTH and
# larger are absent on purpose: DuckDB aligns calendar buckets to 2000-01-01, which an
# epoch-aligned width cannot express.
_BUCKET_MICROS = {
    "DAY": MICROS_PER_DAY,
    "HOUR": 3_600_000_000,
    "MINUTE": 60_000_000,
    "SECOND": 1_000_000,
    "MILLISECOND": 1_000,
    "MICROSECOND": 1,
}

# DuckDB anchors `time_bucket` at 2000-01-03 00:00:00, not at the Unix epoch; that is
# 10,959 days later. `WindowStart` is epoch-anchored, so the two agree only when the bucket
# width divides the gap between the origins evenly — which is why the units above looked
# correct: 1 DAY, 2 HOUR and 5 MINUTE all do. A width that does not (2 DAY, 7 DAY) puts
# every boundary on the wrong instant, silently: `time_bucket(INTERVAL 2 DAY, DATE
# '2021-01-01')` answered 2021-01-01 where DuckDB answers 2020-12-31, and a whole week's
# rows land in the neighbouring bucket. Such a width is refused, the same way MONTH already
# is, rather than answered with a shifted grid.
_BUCKET_ORIGIN_MICROS = 10_959 * MICROS_PER_DAY


def temporal_function(tr, node) -> Expr | None:
    """Translate a temporal construction call, or None when the name is not one of them."""

    from batcher._sql.parser.expressions.literals import _const_str_arg

    if isinstance(node, exp.TimeStrToTime):
        # Spark's implicit "read this text as a timestamp" wrapper, which `date_format`
        # and friends wrap their argument in. A plain cast is what it means.
        return Cast(tr._scalar(node.this), "timestamp")
    if isinstance(node, exp.StrToTime):  # strptime(s, fmt)
        fmt = _const_str_arg(node.args.get("format"), "strptime()", "format")
        return tr._scalar(node.this).str.to_datetime(fmt)

    if isinstance(node, exp.UnixToTime):  # to_timestamp(n), epoch_ms(n)
        return _unix_to_time(tr, node)

    if isinstance(node, exp.TimestampFromParts):  # make_timestamp(y, m, d, h, mi, s)
        parts = ("year", "month", "day", "hour", "min", "sec")
        if all(node.args.get(p) is not None for p in parts):
            return make_timestamp(*(tr._scalar(node.args[p]) for p in parts))
        if node.args.get("year") is not None and all(node.args.get(p) is None for p in parts[1:]):
            # `make_timestamp(1234567890000000)` — the one-argument overload reads a
            # microsecond count, not a year.
            return from_epoch(tr._scalar(node.args["year"]), "us")
        return None

    if isinstance(node, exp.DateBin):  # time_bucket(INTERVAL n unit, ts)
        return _time_bucket(tr, node)

    built = _spark_temporal(tr, node)
    if built is not None:
        return built

    if not isinstance(node, exp.Anonymous):
        return None
    name = node.name.lower()
    args = list(node.expressions)
    if len(args) == 1:
        unit = _EPOCH_NAME_UNIT.get(name)
        if unit is not None:
            return from_epoch(tr._scalar(args[0]), unit)
        if name == "julian":
            # The cast is load-bearing: `epoch_us` of a Date32 is not its microsecond
            # count, and a DATE is the argument `julian` is most often given.
            micros = Cast(tr._scalar(args[0]), "timestamp").dt.epoch_us()
            return micros / lit(float(MICROS_PER_DAY)) + lit(_JULIAN_EPOCH)
        if name == "era":
            # 1 for the Common Era, 0 before it — the year sign is the whole test. Cast
            # the comparison rather than branching on it: a null date has a null year, so
            # `year > 0` is null and a `when/otherwise` would take the else branch and
            # report a missing date as BCE.
            return (tr._scalar(args[0]).dt.year() > lit(0)).cast("int64")
    if len(args) == 1:
        unit = _SPARK_EPOCH_UNIT.get(name)
        if unit is not None:
            return from_epoch(tr._scalar(args[0]), unit)
    if not args and name in ("now", "getdate"):
        return _query_now(tr)
    if len(args) == 1 and name == "unix_nanos":
        return tr._scalar(args[0]).dt.epoch_us() * lit(1_000)
    if len(args) == 2 and name == "try_strptime":
        # `strptime` raises on unparseable text in DuckDB and `try_strptime` returns
        # null; the engine's parser returns null either way, so this is the exact one.
        fmt = _const_str_arg(args[1], "try_strptime()", "format")
        return tr._scalar(args[0]).str.to_datetime(fmt)
    return None


def _query_now(tr):
    """The one instant a query's `now()` reads, memoized on the translator.

    SQL requires `now()` to be constant *within a statement*: `SELECT now() AS a, now()
    AS b` must give `a == b`, and a predicate comparing a column to `now()` must not see
    the clock move between morsels. Folding each call separately gave two different
    timestamps, which is the sort of thing that only shows up as a flaky result later.
    """
    cached = getattr(tr, "_query_now_lit", None)
    if cached is None:
        cached = current_timestamp()
        tr._query_now_lit = cached
    return cached


def _spark_temporal(tr, node) -> Expr | None:
    """The Spark temporal nodes: month shifts, epoch readings, and zone conversion."""

    from batcher._sql.parser.expressions.literals import _const_int_arg, _const_str_arg

    if isinstance(node, (exp.CurrentTimestamp, exp.Localtimestamp)):
        # `now()` / `current_timestamp()` / `localtimestamp()`. Engine timestamps are
        # tz-naive UTC, so all three name the same instant; the constant is bound once at
        # plan-build time, which is also what makes a query using it deterministic across
        # the morsels and partitions it runs on.
        return _query_now(tr)
    if isinstance(node, exp.CurrentTimezone):
        # Not a lookup: engine timestamps are tz-naive UTC by construction, so this is
        # the only answer that is true of them.
        return lit("UTC")
    if isinstance(node, exp.NextDay):
        return _next_day(tr, node)
    if isinstance(node, exp.MonthsBetween):
        return _months_between(tr, node)
    if isinstance(node, exp.AddMonths):
        months = _const_int_arg(node.expression, "add_months(): months")
        return DateOffset(Cast(tr._scalar(node.this), "date"), months, 0, 0)
    if isinstance(node, exp.TsOrDsAdd):  # date_add(d, n) / date_sub(d, n)
        days = _const_int_arg(node.expression, "date_add(): days")
        return DateOffset(Cast(tr._scalar(node.this), "date"), 0, days, 0)
    if isinstance(node, exp.UnixDate):
        # Whole days since the epoch. `epoch` is seconds, and a DATE is midnight, so the
        # division is exact.
        return Cast(tr._scalar(node.this), "timestamp").dt.epoch() // lit(86_400)
    if isinstance(node, (exp.UnixSeconds, exp.UnixMillis, exp.UnixMicros)):
        method = {
            "UnixSeconds": "epoch",
            "UnixMillis": "epoch_ms",
            "UnixMicros": "epoch_us",
        }[type(node).__name__]
        return getattr(Cast(tr._scalar(node.this), "timestamp").dt, method)()
    if isinstance(node, exp.UnixToStr):  # from_unixtime(n, fmt)
        fmt = node.args.get("format")
        raw = _const_str_arg(fmt, "from_unixtime()", "format") if fmt is not None else None
        pattern = datetime_pattern(raw) if raw is not None else None
        if raw is not None and pattern is None:
            return None
        stamp = from_epoch(tr._scalar(node.this), "s")
        return stamp.dt.strftime(pattern or "%Y-%m-%d %H:%M:%S")
    if isinstance(node, exp.StrToUnix):  # to_unix_timestamp(s, fmt)
        if isinstance(node.this, exp.CurrentTimestamp):
            # The nullary `unix_timestamp()`: sqlglot writes it as
            # `to_unix_timestamp(current_timestamp(), fmt)`, and formatting a literal
            # timestamp back into text to re-parse it is a round trip with nothing in it.
            # Fold to the second count at plan-build time, where the constant already is.
            # The constant is a *naive local* wall-clock, which is what `.timestamp()`
            # reads it as; forcing UTC on it would shift the answer by the local offset.
            return lit(int(_query_now(tr).value.timestamp()))
        fmt = node.args.get("format")
        if fmt is None:
            return None
        pattern = datetime_pattern(_const_str_arg(fmt, "to_unix_timestamp()", "format"))
        if pattern is None:
            return None
        return tr._scalar(node.this).str.to_datetime(pattern).dt.epoch()
    if isinstance(node, (exp.AtTimeZone, exp.FromTimeZone)):
        # `from_utc_timestamp(ts, tz)` reads a UTC wall-clock in `tz`;
        # `to_utc_timestamp(ts, tz)` is the inverse.
        zone = _const_str_arg(node.args.get("zone"), "timezone conversion", "time zone")
        value = Cast(tr._scalar(node.this), "timestamp")
        if isinstance(node, exp.AtTimeZone):
            return value.dt.convert_timezone("UTC", zone)
        return value.dt.convert_timezone(zone, "UTC")
    if isinstance(node, exp.ConvertTimezone):
        source = node.args.get("source_tz")
        target = _const_str_arg(node.args.get("target_tz"), "convert_timezone()", "target zone")
        stamp = Cast(tr._scalar(node.args["timestamp"]), "timestamp")
        from_zone = (
            _const_str_arg(source, "convert_timezone()", "source zone")
            if source is not None
            else "UTC"
        )
        return stamp.dt.convert_timezone(from_zone, target)
    return None


# `next_day(d, 'TU')` — Spark's day-of-week abbreviations, as ISO day numbers (Mon = 1).
_NEXT_DAY = {
    "MO": 1,
    "MON": 1,
    "MONDAY": 1,
    "TU": 2,
    "TUE": 2,
    "TUESDAY": 2,
    "WE": 3,
    "WED": 3,
    "WEDNESDAY": 3,
    "TH": 4,
    "THU": 4,
    "THURSDAY": 4,
    "FR": 5,
    "FRI": 5,
    "FRIDAY": 5,
    "SA": 6,
    "SAT": 6,
    "SATURDAY": 6,
    "SU": 7,
    "SUN": 7,
    "SUNDAY": 7,
}


def _next_day(tr, node) -> Expr | None:
    """`next_day(d, 'TU')` — the first date *strictly after* `d` with that weekday.

    Composed from the ISO day number rather than a kernel: the shift is
    ``((target - today + 6) mod 7) + 1`` days, which is 7 when the two coincide (the
    "strictly after" rule) and never 0.
    """
    from batcher._sql.parser.expressions.literals import _const_str_arg

    name = _const_str_arg(node.expression, "next_day()", "day of week").strip().upper()
    target = _NEXT_DAY.get(name)
    if target is None:
        return None
    date = Cast(tr._scalar(node.this), "date")
    shift = ((lit(target) - date.dt.isodow() + lit(6)) % lit(7)) + lit(1)
    # `offset_by` takes a constant duration, and this shift is per row, so the arithmetic
    # runs on the day count and is read back as a date.
    return from_unix_date(Cast(date, "timestamp").dt.epoch() // lit(86_400) + shift)


def _seconds_of_day(ts: Expr) -> Expr:
    """The instant's time of day in whole seconds — the fractional part of a month."""
    return ts.dt.hour() * lit(3600) + ts.dt.minute() * lit(60) + ts.dt.second()


def _months_between(tr, node) -> Expr:
    """`months_between(a, b)` — Spark's fractional month difference.

    Spark's definition, exactly: whole months from the calendar fields, plus the
    leftover days and time-of-day divided by 31 (a fixed divisor, not the month's
    length). Composed here because the parts are all field extractions the engine has.
    """
    left = Cast(tr._scalar(node.this), "timestamp")
    right = Cast(tr._scalar(node.expression), "timestamp")
    months = (left.dt.year() - right.dt.year()) * lit(12) + (left.dt.month() - right.dt.month())
    day_part = (left.dt.day() - right.dt.day()) + (
        (_seconds_of_day(left) - _seconds_of_day(right)) / lit(86400.0)
    )
    # Spark rounds the result to 8 decimal places unless the third argument says not to.
    total = months + day_part / lit(31.0)
    round_off = node.args.get("roundoff")
    exact = isinstance(round_off, exp.Boolean) and not round_off.this
    return total if exact else total.round(8)


def datetime_pattern(fmt: str) -> str | None:
    """The chrono pattern a user's format string denotes, whichever dialect wrote it.

    sqlglot's Spark dialect already rewrites a Java pattern into a `%`-style one, but
    marks the numeric fields it parses strictly by appending the word `strict`
    (`yyyy-MM-dd` becomes `%Y-%mstrict-%dstrict`). Left in place, those markers are
    emitted as literal text — `date_format('2016-04-08', 'yyyy-MM-dd')` returned
    `2016-04strict-08strict`. Stripping them is what makes the Spark spelling work.

    A pattern with no `%` at all never came from sqlglot's rewrite, so it is read as a
    Java one directly; that returns None when the table cannot express it, and the caller
    refuses rather than formatting with the wrong field.
    """
    import re

    if "%" in fmt:
        stripped = re.sub(r"(%-?[A-Za-z])strict", r"\1", fmt)
        # Java quotes a literal section (`yyyy'T'MM`), and sqlglot leaves the quotes in
        # the pattern, so they were emitted as text: `1970'T'01` where Spark writes
        # `1970T01`. `''` is Java's escape for a literal apostrophe.
        return re.sub(r"'([^']*)'", lambda m: m.group(1) or "'", stripped)
    return _java_pattern(fmt)


def _java_pattern(fmt: str) -> str | None:
    """Rewrite a Spark/Java datetime pattern as a chrono one, or None if it cannot be.

    Quoted literal sections (`'T'`) and any letter with no entry in the table are what
    make a pattern untranslatable; those return None so the caller refuses the call
    rather than formatting with the wrong field.
    """
    out: list[str] = []
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        if not ch.isalpha():
            if ch == "'":
                return None
            out.append("%%" if ch == "%" else ch)
            i += 1
            continue
        for token, spec in _JAVA_PATTERN:
            if fmt.startswith(token, i):
                out.append(spec)
                i += len(token)
                break
        else:
            return None
    return "".join(out)


def _unix_to_time(tr, node) -> Expr:
    """`to_timestamp(n)` / `epoch_ms(x)` — an epoch count in, a timestamp out.

    `epoch_ms` is two functions under one name: `epoch_ms(1234)` builds a timestamp from
    a millisecond count, and `epoch_ms(TIMESTAMP '...')` reads the count back out. The
    argument decides, and only a *numeric literal* is unambiguous, so a temporal literal
    or anything else takes the extraction reading — which is what the `.dt` table this
    replaces already meant by the name. `to_timestamp` carries no scale and is never
    ambiguous: its argument is always a second count.
    """
    scale = node.args.get("scale")
    if scale is None:
        return from_epoch(tr._scalar(node.this), "s")
    key = str(scale.name if hasattr(scale, "name") else scale)
    if key not in _SCALE_UNIT:
        raise NotImplementedError(f"epoch scale {key} is not supported; use 0 (seconds), 3, 6 or 9")
    unit = _SCALE_UNIT[key]
    if _is_integer_literal(node.this):
        return from_epoch(tr._scalar(node.this), unit)
    return getattr(tr._scalar(node.this).dt, _UNIT_EPOCH_METHOD[unit])()


def _is_integer_literal(node) -> bool:
    """True for an integer literal, or an arithmetic expression over integer literals."""
    if isinstance(node, exp.Literal):
        return not node.is_string
    if isinstance(node, exp.Neg):
        return _is_integer_literal(node.this)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        return _is_integer_literal(node.this) and _is_integer_literal(node.expression)
    return False


def _time_bucket(tr, node) -> Expr | None:
    """`time_bucket(INTERVAL n unit, ts)` → the start of the bucket containing each row."""
    interval = node.this
    if not isinstance(interval, exp.Interval):
        return None
    unit = (interval.text("unit") or "DAY").upper().removesuffix("S")
    micros = _BUCKET_MICROS.get(unit)
    if micros is None:
        return None
    width = int(interval.this.name) * micros
    if width <= 0:
        return None
    if _BUCKET_ORIGIN_MICROS % width:
        raise NotImplementedError(
            f"time_bucket(INTERVAL {interval.this.name} {unit}, ...) is not supported: "
            "buckets here start from the Unix epoch, DuckDB starts them from 2000-01-03, "
            "and this width does not divide the gap — every boundary would land on a "
            "different instant. Use a width that divides a day evenly (1 DAY, 6 HOUR, "
            "15 MINUTE), or date_trunc for calendar buckets"
        )
    return WindowStart(tr._scalar(node.expression), width)


#: Fixed-width `date_diff` units, in microseconds.
#:
#: `date_diff` counts **boundary crossings**, not elapsed time: DuckDB answers
#: `date_diff('hour', '00:59', '01:00')` with 1 (one minute apart, but one hour boundary
#: between them) and `date_diff('hour', '00:00', '00:59')` with 0. So the unit is a grid to
#: snap both endpoints onto, and the answer is the number of grid cells between them —
#: not the elapsed span divided by the unit, which gets both of those cases backwards.
#:
#: Snapping is `floor_div` on the epoch microsecond count rather than a `date_trunc`,
#: because the Unix epoch is itself aligned on every boundary in this table, so the two
#: agree exactly. Integer division also keeps it exact where a float divide would not, and
#: *floor* (not truncate) is what keeps a pre-1970 timestamp snapping to the period that
#: contains it.
_DIFF_MICROS = {
    "MICROSECOND": 1,
    "MILLISECOND": 1_000,
    "SECOND": 1_000_000,
    "MINUTE": 60_000_000,
    "HOUR": 3_600_000_000,
    "DAY": MICROS_PER_DAY,
}

#: Calendar `date_diff` units, as (periods per year, `.dt` accessor for the period).
#: These cannot use a microsecond grid because months and quarters are not a fixed width;
#: the field arithmetic below counts calendar boundaries directly, which is the same thing
#: DuckDB does.
_DIFF_CALENDAR = {"MONTH": (12, "month"), "QUARTER": (4, "quarter")}


def _epoch_cell(value: Expr, micros: int) -> Expr:
    """Which `micros`-wide cell of the epoch grid `value` falls in."""
    epoch_us = Cast(value, "timestamp").dt.epoch_us()
    if micros == 1:
        return epoch_us
    return Binary("floor_div", epoch_us, lit(micros))


def _date_diff(tr, node) -> Expr:
    """`date_diff(unit, a, b)` — the number of `unit` boundaries crossed going a → b."""
    unit = (node.text("unit") or "DAY").upper().rstrip("S")
    # sqlglot: this=end (b), expression=start (a).
    end, start = tr._scalar(node.this), tr._scalar(node.expression)

    micros = _DIFF_MICROS.get(unit)
    if micros is not None:
        return Cast(_epoch_cell(end, micros) - _epoch_cell(start, micros), "int64")

    if unit == "WEEK":
        # The one unit that is *not* boundary-crossing: DuckDB reports the whole number of
        # 7-day spans, truncated toward zero (so -6 days is 0, not -1). Verified against
        # DuckDB across the Monday boundary, which a week-grid reading would count and
        # this correctly does not.
        days = _epoch_cell(end, _DIFF_MICROS["DAY"]) - _epoch_cell(start, _DIFF_MICROS["DAY"])
        return Cast((days / lit(7)).trunc(), "int64")

    if unit == "YEAR":
        return Cast(end.dt.year() - start.dt.year(), "int64")

    if unit in _DIFF_CALENDAR:
        per, field = _DIFF_CALENDAR[unit]
        ordinal = lambda v: v.dt.year() * lit(per) + getattr(v.dt, field)()  # noqa: E731
        return Cast(ordinal(end) - ordinal(start), "int64")

    raise NotImplementedError(
        f"date_diff unit {unit} is not supported; use one of "
        f"{', '.join(sorted({*_DIFF_MICROS, *_DIFF_CALENDAR, 'WEEK', 'YEAR'}))}"
    )
