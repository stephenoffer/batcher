"""Literals, temporal handling, dtype mapping, and SQL dispatch tables.

The constant dispatch tables and the stateless literal/temporal/dtype helpers
used across the SQL translator live here. Everything in this module is a pure
function or a module-level constant — no translator state is required.
"""

from __future__ import annotations

import datetime as _dt

from batcher.plan.expr_ir import Binary, Expr, lit
from batcher.plan.expr_ir.func_nodes import DateOffset

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
    """The integer a literal node denotes, or `None` if it isn't an integer literal.

    A negative number is not a literal in the parse tree: sqlglot renders `-2` as a `Neg`
    wrapping the literal `2`. Matching only `Literal` would therefore reject every negative
    argument, and `ROUND(x, -2)` is a legal query.
    """
    from sqlglot import expressions as exp

    if isinstance(node, exp.Neg):
        inner = _int_literal(node.this)
        return None if inner is None else -inner
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
    from sqlglot import expressions as exp

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

    from sqlglot import expressions as exp

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


def _apply_interval(date_expr: Expr, interval, *, subtract: bool) -> Expr:
    """`ts/date +/- INTERVAL n <unit>` for a DATE or TIMESTAMP operand.

    DAY/WEEK add exact days and MONTH/YEAR add calendar months, both via the
    type-preserving `offset_by` (`DateOffset`) node so the shift is applied
    correctly whether the operand is a Date32 (epoch days) or a Timestamp
    (microseconds). Returns the operand's own type (DuckDB promotes a DATE to
    timestamp, but the calendar value is the same).
    """
    from sqlglot import expressions as exp

    if isinstance(interval, exp.Interval):
        n = int(interval.this.name)
        unit = (interval.text("unit") or "DAY").upper()
    elif isinstance(interval, exp.Literal) and not interval.is_string:
        n, unit = int(interval.name), "DAY"  # date_add(d, 5) — bare day count
    else:
        raise NotImplementedError("only constant interval literals are supported")
    if subtract:
        n = -n

    if unit.startswith("DAY"):
        return DateOffset(date_expr, 0, n, 0)
    if unit.startswith("WEEK"):
        return DateOffset(date_expr, 0, n * 7, 0)
    if unit.startswith("MONTH"):
        return DateOffset(date_expr, n, 0, 0)
    if unit.startswith("YEAR"):
        return DateOffset(date_expr, n * 12, 0, 0)
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


def _dtype_name(to) -> str:
    name = to.sql().lower()
    # Longest-prefix wins so that e.g. ``bigint`` isn't shadowed by ``int`` — a
    # dict iteration order is not a reliable tiebreak, and ``smallint`` must not
    # fall through to the ``string`` default (which silently cast integers to text).
    table = {
        "tinyint": "int64",
        "smallint": "int64",
        "bigint": "int64",
        "hugeint": "int64",
        "int128": "int64",
        "int": "int64",
        "integer": "int64",
        "long": "int64",
        "ubigint": "int64",
        "uinteger": "int64",
        "usmallint": "int64",
        "utinyint": "int64",
        "double": "float64",
        "decimal": "float64",
        "numeric": "float64",
        "float": "float64",
        "real": "float64",
        "varchar": "string",
        "text": "string",
        "string": "string",
        "boolean": "bool",
        "bool": "bool",
        "date": "date",
        "timestamp": "timestamp",
        "datetime": "timestamp",
    }
    best = None
    for k, v in table.items():
        if name.startswith(k) and (best is None or len(k) > len(best[0])):
            best = (k, v)
    return best[1] if best is not None else "string"


def _build_binops():
    from sqlglot import expressions as exp

    return {
        exp.Add: lambda a, b: a + b,
        exp.Sub: lambda a, b: a - b,
        exp.Mul: lambda a, b: a * b,
        exp.Div: lambda a, b: a / b,
        # SQL `//` is integer division that truncates *toward zero* (DuckDB/C
        # semantics), not Python's floor: `-7 // 3` is `-2`, not `-3`. The engine's
        # `//` floors, so build it as a truncated true-division instead.
        exp.IntDiv: lambda a, b: (a / b).trunc(),
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
