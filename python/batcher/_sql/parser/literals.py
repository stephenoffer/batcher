"""Literals, temporal handling, dtype mapping, and SQL dispatch tables.

The constant dispatch tables and the stateless literal/temporal/dtype helpers
used across the SQL translator live here. Everything in this module is a pure
function or a module-level constant — no translator state is required.
"""

from __future__ import annotations

import datetime as _dt

from batcher.plan.expr_ir import Binary, Cast, Expr, lit

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
}
_UNARY_STR = {"Upper": "upper", "Lower": "lower", "Length": "len", "Reverse": "reverse"}
_DATE_PART = {
    "Year": "year",
    "Month": "month",
    "Day": "day",
    "Hour": "hour",
    "Minute": "minute",
    "Second": "second",
    "Quarter": "quarter",
    "Week": "week",
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
    """
    import re

    out = ["^"]
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


def _literal(node) -> Expr:
    if node.is_string:
        return lit(node.this)
    text = node.this
    return lit(float(text) if ("." in text or "e" in text.lower()) else int(text))


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
    """`date +/- INTERVAL n <unit>` for a DATE operand.

    DAY/WEEK use the day-count representation (Date32 = epoch days); MONTH/YEAR
    use calendar arithmetic via the engine's `add_months`. Returns a DATE
    (DuckDB promotes to timestamp, but the calendar value is the same).
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
        return Cast(Cast(date_expr, "int64") + lit(n), "date")
    if unit.startswith("WEEK"):
        return Cast(Cast(date_expr, "int64") + lit(n * 7), "date")
    if unit.startswith("MONTH"):
        return Binary("add_months", date_expr, lit(n))
    if unit.startswith("YEAR"):
        return Binary("add_months", date_expr, lit(n * 12))
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
    table = {
        "bigint": "int64",
        "int": "int64",
        "integer": "int64",
        "long": "int64",
        "double": "float64",
        "float": "float64",
        "real": "float64",
        "varchar": "string",
        "text": "string",
        "string": "string",
        "boolean": "bool",
        "date": "date",
        "timestamp": "timestamp",
        "datetime": "timestamp",
    }
    for k, v in table.items():
        if name.startswith(k):
            return v
    return "string"


def _build_binops():
    from sqlglot import expressions as exp

    return {
        exp.Add: lambda a, b: a + b,
        exp.Sub: lambda a, b: a - b,
        exp.Mul: lambda a, b: a * b,
        exp.Div: lambda a, b: a / b,
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
