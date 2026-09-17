"""SQL temporal *construction* — parsing text, reading epoch counts, and bucketing time.

The sibling `literals` module owns temporal *literals* and interval arithmetic, and
`functions` owns the field extractions (`year`, `month`, ...). This module is the third
part of the temporal surface: the functions that build a timestamp rather than read one.

Each entry is a name a migrating DuckDB or Spark query types (`strptime`, `to_timestamp`,
`epoch_ms`, `make_timestamp`, `time_bucket`) mapped onto the engine node that already
implements it. Nothing here invents a semantic: where DuckDB's answer depends on a session
time zone (`to_timestamp` returns TIMESTAMPTZ, `make_timestamptz`) the instant is the same
but the rendering is not, and a calendar bucket width
(`time_bucket(INTERVAL 1 MONTH, ...)`) is answered on the *month index* rather than on an
epoch-aligned microsecond width, which no number of microseconds can express.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.lowering.buckets import time_bucket
from batcher.plan.expr_ir import Binary, Cast, Expr, lit
from batcher.plan.expr_ir.func_nodes import DateOffset
from batcher.plan.functions.partitioning import partition_days
from batcher.plan.functions.temporal import (
    current_timestamp,
    from_epoch,
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
        return time_bucket(tr, node)

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
    if len(args) == 3 and name in ("date_sub", "datesub", "date_diff", "datediff"):
        # DuckDB's three-argument `date_sub(part, start, end)` is `date_diff` under another
        # name — both answer 60 for `('day', '2020-01-01', '2020-03-01')`. sqlglot leaves
        # this spelling anonymous, so it reached the scalar translator as an unknown
        # function. (The *two*-argument `date_sub(date, days)` is Spark's and parses to a
        # typed `DateSub`, handled by the interval path.)
        return _date_diff(tr, exp.DateDiff(this=args[2], expression=args[1], unit=args[0]))
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
        weekday = _const_str_arg(node.expression, "next_day()", "day of week")
        return Cast(tr._scalar(node.this), "date").dt.next_day(weekday)
    if isinstance(node, exp.MonthsBetween):
        round_off = node.args.get("roundoff")
        exact = isinstance(round_off, exp.Boolean) and not round_off.this
        left = Cast(tr._scalar(node.this), "timestamp")
        return left.dt.months_between(
            Cast(tr._scalar(node.expression), "timestamp"), round_off=not exact
        )
    if isinstance(node, exp.AddMonths):
        months = _const_int_arg(node.expression, "add_months(): months")
        return DateOffset(Cast(tr._scalar(node.this), "date"), months, 0, 0)
    if isinstance(node, exp.TsOrDsAdd):  # date_add(d, n) / date_sub(d, n)
        days = _const_int_arg(node.expression, "date_add(): days")
        return DateOffset(Cast(tr._scalar(node.this), "date"), 0, days, 0)
    if isinstance(node, exp.UnixDate):
        # Spark `unix_date` is the Iceberg day transform: whole days since the epoch.
        return partition_days(tr._scalar(node.this))
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
    value = tr._scalar(node.this)
    if _reads_as_integer(tr, value, node.this):
        return from_epoch(value, unit)
    return getattr(value.dt, _UNIT_EPOCH_METHOD[unit])()


def _reads_as_integer(tr, value: Expr, node) -> bool:
    """Whether `epoch_ms`'s argument is a count to build from, rather than a time to read.

    Asked of the argument's *inferred type*, not of its syntax. The syntactic reading — a
    bare integer literal, or a bare column the scope calls an integer — could not see past
    either one: `epoch_ms(n * 1000)`, `epoch_ms(n + 0)`, `epoch_ms(abs(n))` and
    `epoch_ms(CAST(n AS BIGINT))` all took the *extraction* branch on an integer column and
    silently returned a meaningless number where DuckDB builds a timestamp. Inference
    answers all four, because it is the same analysis `Dataset.schema` is answered from
    rather than a second statement of what an integer expression looks like.

    The syntactic check survives only as the fallback for when inference is uncertain
    (`None`), so the behavior can improve but never regress.

    Args:
        tr: The translator, for its scope types.
        value: The built argument expression.
        node: The argument's AST node, for the fallback.

    Returns:
        True to read the argument as an epoch count.
    """
    import pyarrow as pa

    inferred = tr.expr_type(value)
    if inferred is not None:
        return pa.types.is_integer(inferred) or pa.types.is_floating(inferred)
    return _is_integer_literal(node)


def _is_integer_literal(node) -> bool:
    """True for an integer literal, or an arithmetic expression over integer literals."""
    if isinstance(node, exp.Literal):
        return not node.is_string
    if isinstance(node, exp.Neg):
        return _is_integer_literal(node.this)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
        return _is_integer_literal(node.this) and _is_integer_literal(node.expression)
    return False


def _epoch_cell(value: Expr, micros: int) -> Expr:
    """Which `micros`-wide cell of the epoch grid `value` falls in."""
    epoch_us = Cast(value, "timestamp").dt.epoch_us()
    if micros == 1:
        return epoch_us
    return Binary("floor_div", epoch_us, lit(micros))


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

#: `date_diff` units that are a difference of one `.dt` field. Each was refused as "unit
#: not supported" where the accessor to answer it already existed.
_DIFF_FIELD = {"DECADE": "decade", "ISOYEAR": "iso_year"}

#: `date_diff` units counted as a difference of `year / n`, **not** of the same-named
#: field. DuckDB's `century(DATE '2000-01-01')` is 20 and its `century(DATE '0001-01-01')`
#: is 1 — the 1-based ordinal — but `date_diff('century', …)` between those two dates is
#: 20, not 19, because the difference is taken on the *zero-based* period number. The two
#: readings agree everywhere except across the year-1 boundary, which is exactly the kind
#: of divergence a plausible answer hides.
_DIFF_YEAR_SCALE = {"CENTURY": 100, "MILLENNIUM": 1000}


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

    field = _DIFF_FIELD.get(unit)
    if field is not None:
        return Cast(getattr(end.dt, field)() - getattr(start.dt, field)(), "int64")

    scale = _DIFF_YEAR_SCALE.get(unit)
    if scale is not None:
        period = lambda v: Binary("floor_div", v.dt.year(), lit(scale))  # noqa: E731
        return Cast(period(end) - period(start), "int64")

    if unit in _DIFF_CALENDAR:
        per, field = _DIFF_CALENDAR[unit]
        ordinal = lambda v: v.dt.year() * lit(per) + getattr(v.dt, field)()  # noqa: E731
        return Cast(ordinal(end) - ordinal(start), "int64")

    known = {
        *_DIFF_MICROS,
        *_DIFF_CALENDAR,
        *_DIFF_FIELD,
        *_DIFF_YEAR_SCALE,
        "WEEK",
        "YEAR",
    }
    raise NotImplementedError(
        f"date_diff unit {unit} is not supported; use one of {', '.join(sorted(known))}"
    )
