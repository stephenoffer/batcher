"""Literals, temporal handling, dtype mapping, and SQL dispatch tables.

The constant dispatch tables and the stateless literal/temporal/dtype helpers
used across the SQL translator live here. Everything in this module is a pure
function or a module-level constant — no translator state is required.
"""

from __future__ import annotations

import datetime as _dt
import re as _re

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Binary, Cast, Expr, lit
from batcher.plan.expr_ir.func_nodes import DateOffset
from batcher.plan.types import resolve_dtype

_AGG_FUNCS = {
    "sum": "sum",
    "count": "count",
    "avg": "mean",
    "min": "min",
    "max": "max",
    # keyed by the lowercased sqlglot node class name
    "variance": "var",
    "stddev": "stddev",
    "stddevsamp": "stddev",
    "median": "median",
    # `mode() WITHIN GROUP (ORDER BY x)` is rewritten to `Mode(this=x)` up front
    # (see clauses._within_group_to_agg) so it reaches here as a plain aggregate.
    "mode": "mode",
    # sqlglot parses `BOOL_AND`/`BOOL_OR` to LogicalAnd/LogicalOr. They map straight
    # to the native mergeable aggregates, which ignore NULLs and yield NULL for a group
    # with no non-null input — exactly DuckDB's semantics. (A `COUNT(*) FILTER`
    # rewrite cannot reproduce that NULL result and silently answered TRUE/FALSE.)
    "logicaland": "bool_and",
    "logicalor": "bool_or",
}

# sqlglot DataType.Type names that fold a string literal into a temporal literal.
_TEMPORAL_KINDS = {
    "DATE",
    "TIMESTAMP",
    "TIMESTAMPNTZ",
    "TIMESTAMPTZ",
    "TIMESTAMPLTZ",
    "DATETIME",
}
_DATE_KINDS = {"DATE"}

# Scalar functions dispatched by sqlglot node class name. Unary forms map to a
# method on the (numeric/string/date) argument expression.
_UNARY_MATH = {
    "Ln": "ln",
    "Exp": "exp",
    "Sqrt": "sqrt",
    "Abs": "abs",
    "Sign": "sign",
    "Floor": "floor",
    "Ceil": "ceil",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Cbrt": "cbrt",
    "Trunc": "trunc",
    "Degrees": "degrees",
    "Radians": "radians",
    # The remaining DuckDB math builtins sqlglot promotes to typed nodes. Each has an
    # identically-named `Expr` method, so the only thing that was missing was the row
    # in this table — `SELECT cot(x)` raised "unsupported SQL expression: Cot".
    "Atan": "atan",
    "Asin": "asin",
    "Acos": "acos",
    "Sinh": "sinh",
    "Cosh": "cosh",
    "Tanh": "tanh",
    "Asinh": "asinh",
    "Acosh": "acosh",
    "Atanh": "atanh",
    "Cot": "cot",
    "Factorial": "factorial",
    # Spark spellings sqlglot gives a typed node; each is an existing `Expr` method.
    "Sec": "sec",
    "Csc": "csc",
    "Rint": "rint",
    "BitwiseCount": "bit_count",
    # Not math *functions*, but the same shape: a method on the value expression.
    "IsNan": "is_nan",
    "IsInf": "is_infinite",
}
_UNARY_STR = {
    "Upper": "upper",
    "Lower": "lower",
    "Length": "len",
    "Reverse": "reverse",
    "Ascii": "ascii",
    # `unicode(s)` is DuckDB's spelling of `ascii(s)` — the first character's codepoint.
    "Unicode": "ascii",
    "Hex": "hex",
    "Unhex": "unhex",
    "MD5": "md5",
    "SHA": "sha1",
    # `SHA2` is deliberately absent: it carries a digest width, and matching it here
    # (this table is consulted first) made `sha2(s, 512)` silently return sha256.
    # `functions._scalar_function` handles it and rejects any width but 256.
    "BitLength": "bit_length",
    "Initcap": "initcap",
    "Soundex": "soundex",
    "ToBase64": "base64",
    "FromBase64": "from_base64",
    "ToBinary": "to_binary",
    "FromBinary": "from_binary",
}
_DATE_PART = {
    "Year": "year",
    "Month": "month",
    "Day": "day",
    "DayOfMonth": "day",
    "Hour": "hour",
    "Minute": "minute",
    "Second": "second",
    "Quarter": "quarter",
    "Week": "week",
    "DayOfWeek": "dayofweek",
    "DayOfYear": "dayofyear",
    # DuckDB date-part builtins sqlglot promotes to typed nodes; each already has a
    # `.dt` method, so only the row was missing.
    "Dayname": "dayname",
    "Monthname": "monthname",
    "LastDay": "last_day",
    "WeekOfYear": "weekofyear",
    "DayOfWeekIso": "isodow",
}
# EXTRACT(<part> FROM ts) field name (lowercased) → `.dt` method.
_EXTRACT_PART = {
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "quarter": "quarter",
    "week": "week",
    "dow": "dayofweek",
    "dayofweek": "dayofweek",
    "doy": "dayofyear",
    "dayofyear": "dayofyear",
    "epoch": "epoch",
    # Fields the `.dt` namespace already computes but this table never listed, so
    # `EXTRACT(isodow FROM ...)` — ISO-8601 weekday numbering, the one a report actually
    # wants — raised "field not supported" while `.dt.isodow()` sat right there. Each is
    # checked against DuckDB's answer for the same instant, not assumed equivalent by name:
    # `isodow` counts Monday as 1 where `dow` counts Sunday as 0, and `isoyear` is the year
    # the ISO *week* belongs to, which differs from `year` around New Year.
    "isodow": "isodow",
    "isoyear": "iso_year",
    "weekofyear": "week_of_year",
    "dayofmonth": "day",
    "century": "century",
    "millennium": "millennium",
    "decade": "decade",
}

# EXTRACT fields that are *not* a single `.dt` accessor. DuckDB (and Postgres) report
# `microsecond` and `millisecond` as the whole seconds field scaled — `03:04:05.123456`
# gives 5123456 microseconds, not 123456 — while `.dt.microsecond()` is the sub-second
# remainder alone. Mapping them by name would have looked right and been off by the
# seconds, so they are built here instead of listed above.
_EXTRACT_COMPOSITE = {
    "microsecond": lambda dt: dt.second() * 1_000_000 + dt.microsecond(),
    "microseconds": lambda dt: dt.second() * 1_000_000 + dt.microsecond(),
    "millisecond": lambda dt: dt.second() * 1_000 + dt.millisecond(),
    "milliseconds": lambda dt: dt.second() * 1_000 + dt.millisecond(),
    # `weekday` is DuckDB's Sunday=0 numbering, *not* the ISO Monday=1 one that
    # `.dt.weekday()` spells — the two agree on every day but Sunday, which is the kind of
    # near-miss that reaches production. It is `dayofweek` under another name.
    "weekday": lambda dt: dt.dayofweek(),
    # The ISO year and week as one number, `yyyyww` — and the year has to be the *ISO*
    # one, which differs from the calendar year in the days around New Year that are the
    # only reason anybody asks for this.
    "yearweek": lambda dt: dt.iso_year() * 100 + dt.week_of_year(),
    # 1 in the Common Era, 0 before it. Built from a comparison rather than a branch so a
    # null date stays null instead of being reported as BCE.
    "era": lambda dt: (dt.year() > 0).cast("int64"),
    # `date_part('epoch', ...)` is DOUBLE and keeps the fraction — `23:59:59.999999` on
    # 1969-12-31 is -1e-06 seconds, not -1. `.dt.epoch()` is the integer-seconds accessor
    # (DuckDB spells that one `epoch(...)` with no `date_part`), so pointing this field at
    # it discarded the sub-second part of every timestamp.
    "epoch": lambda dt: dt.epoch_us().cast("float64") / 1_000_000.0,
    "epoch_seconds": lambda dt: dt.epoch_us().cast("float64") / 1_000_000.0,
}


def _like_to_regex(pattern: str, escape: str | None = None) -> str:
    """Convert a SQL LIKE pattern to an anchored regex (`%`→`.*`, `_`→`.`).

    Literal characters are regex-escaped; `escape` (if given) quotes the next
    char as a literal. The result is anchored so it matches the whole string.

    `(?s)` makes `.` match a newline too: SQL's `%`/`_` are "any character", with no
    exception for `\\n`, and the native matcher (`bc_expr`'s `like_regex`) anchors the
    same way. Without it `'a\\nb' LIKE 'a%b'` was false here and true in DuckDB.
    """
    import re

    out = ["(?s)^"]
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if escape is not None and c == escape:
            i += 1
            if i < len(pattern):
                out.append(re.escape(pattern[i]))
        elif c == "%":
            out.append(".*")
        elif c == "_":
            out.append(".")
        else:
            out.append(re.escape(c))
        i += 1
    out.append("$")
    return "".join(out)


def _regexp_flags_prefix(flag: str | None) -> str:
    """Translate a DuckDB regex option string into an inline `(?…)` flag prefix.

    DuckDB's ``regexp_*`` functions take an option string (e.g. ``'i'``); the Rust
    ``regex`` crate honours the same letters as an inline prefix on the pattern. Only
    the options whose Rust mapping is verified bit-identical to DuckDB are accepted —
    ``'i'`` (case-insensitive) and ``'s'`` (``.`` matches newline); ``'c'`` is the
    (default) case-sensitive mode and is a no-op. Any other option (``'m'``/``'n'``
    line-anchoring, ``'l'`` literal, ``'g'`` global, …) raises rather than being
    silently dropped, which would return a wrong (e.g. case-sensitive) result.

    Args:
        flag: The DuckDB option string, or None when no options were given.

    Returns:
        An inline flag prefix such as ``"(?i)"``, or ``""`` for no active flags.
    """
    if not flag:
        return ""
    active = ""
    for opt in flag:
        if opt == "c":
            continue  # case-sensitive is the default
        if opt in "is":
            if opt not in active:
                active += opt
        else:
            raise NotImplementedError(
                f"regexp option {opt!r} is not supported (only 'i', 's', 'c')"
            )
    return f"(?{active})" if active else ""


def _literal(node) -> Expr:
    if node.is_string:
        return lit(node.this)
    text = node.this
    return lit(float(text) if ("." in text or "e" in text.lower()) else int(text))


def _int_literal(node) -> int | None:
    """The integer a constant node denotes, or `None` if it is not a constant integer.

    A negative number is not a literal in the parse tree: sqlglot renders `-2` as a `Neg`
    wrapping the literal `2`. Matching only `Literal` would therefore reject every negative
    argument, and `ROUND(x, -2)` is a legal query.

    Constant *arithmetic* is folded for the same reason. A parser may rewrite a call into
    one — Spark's `date_sub(d, 1)` arrives as `date_add(d, 1 * -1)` — and refusing it
    reported "must be an integer literal" for an argument the user did write as a literal.
    """
    if isinstance(node, exp.Paren):
        return _int_literal(node.this)
    if isinstance(node, exp.Neg):
        inner = _int_literal(node.this)
        return None if inner is None else -inner
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        left, right = _int_literal(node.this), _int_literal(node.expression)
        if left is None or right is None:
            return None
        if isinstance(node, exp.Add):
            return left + right
        return left - right if isinstance(node, exp.Sub) else left * right
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(node.this)
        except (TypeError, ValueError):
            return None
    return None


def _const_str_arg(node, what: str, role: str = "argument") -> str:
    """The string a constant string-literal argument denotes, or raise.

    Many SQL scalar functions (`replace`, `split_part`, `regexp_extract`, …) can only
    lower a pattern/delimiter/prefix when it is a constant string literal. `what` names
    the function and `role` the argument, so the rejection reads e.g. `split_part()
    requires a constant string delimiter`.
    """
    if not (isinstance(node, exp.Literal) and node.is_string):
        raise NotImplementedError(f"{what} requires a constant string {role}")
    return node.this


def _const_int_arg(node, what: str) -> int:
    """The integer a constant integer-literal argument denotes, or raise.

    Wraps `_int_literal` (which folds a `Neg`-wrapped negative literal) with the uniform
    `must be an integer literal` rejection the scalar path repeats. `what` names the
    offending argument for the error message.
    """
    value = _int_literal(node)
    if value is None:
        raise NotImplementedError(f"{what} must be an integer literal")
    return value


def _fold_const_arith(node) -> Expr | None:
    """Constant-fold ``literal <op> literal`` arithmetic with exact decimal semantics.

    SQL numeric literals are exact decimals, so ``0.06 + 0.01`` is ``0.07`` — not the
    IEEE ``0.0699999…`` that folding two ``float`` literals yields. Folding through
    ``Decimal`` (whenever a decimal literal is involved) makes boundary comparisons
    like ``l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`` (TPC-H Q6) agree with
    DuckDB/Spark. Pure-integer arithmetic keeps its integer type. Returns the folded
    ``lit``, or ``None`` when the node isn't foldable literal arithmetic.
    """
    from decimal import Decimal, InvalidOperation

    op = {exp.Add: "+", exp.Sub: "-", exp.Mul: "*", exp.Div: "/"}.get(type(node))
    if op is None:
        return None
    a, b = node.this, node.expression
    if not (isinstance(a, exp.Literal) and isinstance(b, exp.Literal)):
        return None
    if a.is_string or b.is_string:
        return None
    if not any(("." in x.this) or ("e" in x.this.lower()) for x in (a, b)):
        return None  # pure-integer arithmetic keeps its integer type
    try:
        da, db = Decimal(a.this), Decimal(b.this)
        if op == "+":
            r = da + db
        elif op == "-":
            r = da - db
        elif op == "*":
            r = da * db
        elif db == 0:
            return None
        else:
            r = da / db
    except InvalidOperation:
        return None
    return lit(float(r))


# INTERVAL units, by which component of `offset_by` they contribute to. Calendar
# months clamp at end-of-month; days and microseconds are exact. Every DuckDB unit
# spelling is here, singular and plural (`INTERVAL 2 HOURS`), so a unit that the
# engine can express is never refused for a spelling.
_INTERVAL_MONTHS = {
    "MONTH": 1,
    "QUARTER": 3,
    "YEAR": 12,
    "DECADE": 120,
    "CENTURY": 1200,
    "MILLENNIUM": 12000,
}
_INTERVAL_DAYS = {"DAY": 1, "WEEK": 7}
_INTERVAL_MICROS = {
    "HOUR": 3_600_000_000,
    "MINUTE": 60_000_000,
    "SECOND": 1_000_000,
    "MILLISECOND": 1_000,
    "MICROSECOND": 1,
}


def _apply_interval(date_expr: Expr, interval, *, subtract: bool) -> Expr:
    """`ts/date +/- INTERVAL n <unit>` for a DATE or TIMESTAMP operand.

    Calendar units (MONTH/QUARTER/YEAR/DECADE/CENTURY/MILLENNIUM) add months,
    DAY/WEEK add exact days, and the sub-day units add exact microseconds — all via
    the type-preserving `offset_by` (`DateOffset`) node so the shift is applied
    correctly whether the operand is a Date32 (epoch days) or a Timestamp
    (microseconds).

    A sub-day offset is not representable on a Date32, and DuckDB promotes the
    operand to TIMESTAMP for exactly that reason (`DATE '2024-03-05' + INTERVAL 1
    HOUR` is a timestamp there), so the cast is applied here rather than letting the
    engine reject the shift. On a timestamp operand the cast is a no-op.
    """
    if isinstance(interval, exp.Interval):
        n = int(interval.this.name)
        unit = (interval.text("unit") or "DAY").upper()
    elif isinstance(interval, exp.Literal) and not interval.is_string:
        n, unit = int(interval.name), "DAY"  # date_add(d, 5) — bare day count
    else:
        raise NotImplementedError("only constant interval literals are supported")
    if subtract:
        n = -n
    unit = unit.removesuffix("S")  # `INTERVAL 2 HOURS` — no unit name ends in S

    if unit in _INTERVAL_MONTHS:
        return DateOffset(date_expr, n * _INTERVAL_MONTHS[unit], 0, 0)
    if unit in _INTERVAL_DAYS:
        return DateOffset(date_expr, 0, n * _INTERVAL_DAYS[unit], 0)
    if unit in _INTERVAL_MICROS:
        return DateOffset(Cast(date_expr, "timestamp"), 0, 0, n * _INTERVAL_MICROS[unit])
    raise NotImplementedError(f"INTERVAL unit {unit} is not supported")


def _temporal_literal(text: str, kind: str) -> Expr:
    """Parse a DATE/TIMESTAMP string literal into a temporal `lit`."""
    if kind in _DATE_KINDS:
        return lit(_dt.date.fromisoformat(text))
    # TIMESTAMP (and TIMESTAMPTZ): accept 'YYYY-MM-DD' or full datetime.
    normalized = text.replace("T", " ")
    try:
        return lit(_dt.datetime.fromisoformat(normalized))
    except ValueError:
        return lit(_dt.datetime.combine(_dt.date.fromisoformat(text), _dt.time()))


#: SQL type names with no engine dtype of their own, mapped to the nearest one that exists.
#: Everything else is handed to `resolve_dtype` verbatim, because the registry already
#: spells every SQL name at its true width (``tinyint`` is int8, ``usmallint`` is uint16,
#: ``decimal(10,2)`` is decimal128).
#:
#: A lookup table here used to flatten all twelve integer widths onto ``int64``, which made
#: a narrowing cast a no-op: ``CAST(32768 AS TINYINT)`` returned 32768 instead of raising,
#: and ``TRY_CAST(32768 AS TINYINT)`` — whose entire purpose is to NULL what does not fit —
#: returned it too, so the idiom used to *filter* out-of-range values filtered nothing.
#:
#: The keys are what **sqlglot renders**, not what the user wrote, and the two differ more
#: than they look: the duckdb dialect normalizes ``UINTEGER`` to ``UINT``, ``HUGEINT`` to
#: ``INT128``, and — the one that matters most — plain ``TIMESTAMP`` to ``TIMESTAMPNTZ``.
#: Every entry below was arrived at by rendering the type through each supported dialect and
#: checking what actually came out, not by reading the SQL the user types.
_DTYPE_ALIAS = {
    "uint": "uint32",  # duckdb dialect's rendering of UINTEGER
    "int128": "int64",  # HUGEINT — int64 is the widest integer the engine has
    "varbinary": "binary",  # BLOB
    "decimal": "float64",  # bare DECIMAL/NUMERIC with no precision; a parametrized
    "numeric": "float64",  # ``decimal(p,s)`` resolves properly and is left alone
    "uuid": "string",  # no native type, and text is how both DuckDB and Arrow carry it
    "json": "string",
    "char": "string",  # CHAR/CHAR(n) — Batcher does not pad to a fixed width
    "nchar": "string",
    # `CAST(x AS TIMESTAMP)` reaches here spelled `TIMESTAMPNTZ`, and `TIMESTAMPTZ` is how
    # both duckdb and spark render their tz-aware forms. All three map to the naive
    # microsecond timestamp, which is what this did before the widths were fixed — a
    # tz-carrying cast target is `timestamp(us, <tz>)` and resolves on its own.
    "timestampntz": "timestamp",
    "timestamptz": "timestamp",
    "timestampltz": "timestamp",
}


def _dtype_name(to) -> str:
    """Map a SQL type name onto the engine dtype name of the same width."""
    name = to.sql().lower().strip()
    if resolve_dtype(name) is not None:
        return name
    # ``VARCHAR(10)``, ``TIMESTAMP WITH TIME ZONE``, ``DECIMAL``: reduce to the head word
    # and try again. The alias table is consulted first so a bare ``DECIMAL`` cannot
    # resolve as something else.
    head = _re.split(r"[(\s]", name, maxsplit=1)[0]
    if head in _DTYPE_ALIAS:
        return _DTYPE_ALIAS[head]
    if resolve_dtype(head) is not None:
        return head
    # Refusing beats the ``string`` default this used to end in. That default made an
    # unknown type name a *silent* cast to text — ``CAST(i AS UINTEGER)`` returned
    # ``['1', '300', '-5']`` — which is a wrong answer wearing the wrong type, and no
    # value comparison against DuckDB catches it because the query still returns rows.
    raise NotImplementedError(f"CAST to {to.sql()} is not supported; Batcher has no dtype for it")


def _sql_int_div(tr, a: Expr, b: Expr) -> Expr:
    """SQL ``//`` (and its ``divide()`` spelling), which is two operators under one symbol.

    On **integers** it is division truncating toward zero (`divide(-7, 2)` is -3), NULL on a
    zero divisor. On **floating** operands DuckDB does not round at all — `7.0 // 2.0` is
    `3.5` — while still answering NULL rather than `inf` when the divisor is zero. Applying
    the integer rule to a float pair returned `3.0`, a silently different number.

    The operand types decide, and they are read from the control plane's own static
    inference; when it cannot state one (an opaque sub-expression), the integer rule stands,
    which is what the operator means on the types SQL most often applies it to.

    Args:
        tr: The translator, for the in-scope schema.
        a: The dividend.
        b: The divisor.

    Returns:
        The quotient expression.
    """
    import pyarrow as pa

    from batcher.plan.expr_ir.constructors import nullif

    types = [tr.expr_type(a), tr.expr_type(b)]
    if any(t is not None and pa.types.is_floating(t) for t in types):
        # `a / b` already propagates nulls; only the zero divisor needs a guard, and
        # `nullif(b, 0)` supplies it without evaluating the quotient twice.
        return a / nullif(b, lit(0.0))
    return _trunc_div(a, b)


def _trunc_div(a: Expr, b: Expr) -> Expr:
    """SQL `//` — integer division truncating *toward zero*, built on the engine's floor.

    The obvious spelling, `(a / b).trunc()`, is wrong in three ways at once, and all three
    are silent. True division casts to Float64, so the result type is DOUBLE where DuckDB
    gives BIGINT; the cast happens *before* the truncation, so `9223372036854775807 // 2`
    came back as `4.611686018427388e+18` instead of the exact `4611686018427387903`; and a
    zero divisor produced `inf` rather than the NULL every SQL engine returns.

    `floor_div` has none of those problems -- it is type-preserving, exact above 2^53, and
    NULL on a zero divisor -- so the only thing left is to convert its *floor* into a
    *truncation*. They differ by exactly one, and only when the division is inexact and the
    operands have opposite signs: `-7 // 3` floors to -3 and truncates to -2.

    The inexactness test uses the engine's `%`, which truncates toward zero, so a non-zero
    remainder means "not exact" regardless of sign.
    """
    from batcher.plan.expr_ir.constructors import when

    floor = Binary("floor_div", a, b)
    inexact = Binary("mod", a, b) != lit(0)
    opposite_signs = (a < lit(0)) != (b < lit(0))
    return when(inexact & opposite_signs).then(floor + lit(1)).otherwise(floor)


def _build_binops():
    return {
        exp.Add: lambda a, b: a + b,
        exp.Sub: lambda a, b: a - b,
        exp.Mul: lambda a, b: a * b,
        exp.Div: lambda a, b: a / b,
        # SQL `//` is integer division that truncates *toward zero* (DuckDB/C
        # semantics), not Python's floor: `-7 // 3` is `-2`, not `-3`.
        # `//` is type-dependent; the wrapper is applied at the call site, which has the
        # translator. This entry is the integer reading, kept so the table stays total.
        exp.IntDiv: _trunc_div,
        exp.Pow: lambda a, b: a**b,  # SQL `^` / power() / `**`
        exp.Mod: lambda a, b: a % b,
        exp.EQ: lambda a, b: a == b,
        exp.NEQ: lambda a, b: a != b,
        exp.GT: lambda a, b: a > b,
        exp.GTE: lambda a, b: a >= b,
        exp.LT: lambda a, b: a < b,
        exp.LTE: lambda a, b: a <= b,
        exp.And: lambda a, b: a & b,
        exp.Or: lambda a, b: a | b,
        exp.DPipe: lambda a, b: Binary("concat", a, b),  # SQL `||` string concat
        exp.BitwiseAnd: lambda a, b: Binary("bit_and", a, b),
        exp.BitwiseOr: lambda a, b: Binary("bit_or", a, b),
        exp.BitwiseXor: lambda a, b: Binary("bit_xor", a, b),
        exp.BitwiseLeftShift: lambda a, b: Binary("shift_left", a, b),
        exp.BitwiseRightShift: lambda a, b: Binary("shift_right", a, b),
    }


try:
    _BINOPS = _build_binops()
except Exception:  # pragma: no cover - sqlglot missing at import-time tooling
    _BINOPS = {}
