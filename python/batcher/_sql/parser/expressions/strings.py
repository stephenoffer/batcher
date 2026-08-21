"""SQL string functions whose translation is more than a name lookup.

The name-keyed tables in `literals` (`_UNARY_STR`) and `anonymous` cover the string
functions that map onto a `.str` method one-for-one. This module holds the rest: the ones
that need the argument rewritten (`regexp_full_match` anchors its pattern) or a shape the
tables cannot express.

The template family (`format`, `printf`, `format_string`) is the largest of those. All
three interpolate values into a constant template and differ only in how the template
marks its holes — DuckDB writes `{}`, Spark and C write `%s`/`%d` — so all three reach the
one `format_string` builder, and the printf spellings are rewritten into brace form at
plan-build time. The rewrite is deliberately narrow: a conversion carrying flags, a width
or a precision (`%05.2f`) is **refused**, because the engine has no per-conversion
formatting and answering it with an unpadded value would be a plausible wrong string
rather than an error.
"""

from __future__ import annotations

import re

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Binary, Expr, StrFunc, lit, nullif, when
from batcher.plan.functions.string import concat, format_string

__all__ = ["string_function"]

# DuckDB's three spellings of "a byte count as text", and whether each uses SI units.
_FORMAT_BYTES = {
    "format_bytes": False,
    "formatreadablesize": False,
    "formatreadabledecimalsize": True,
}


def string_function(tr, node) -> Expr | None:
    """Translate a string call needing a rewrite, or None when the name is not one."""

    from batcher._sql.parser.expressions.literals import _const_int_arg

    if isinstance(node, exp.Chr):
        # `chr(65)` / Spark `char(65)`. sqlglot keeps the argument list, not `this`.
        args = node.expressions or ([node.this] if node.this is not None else [])
        return tr._scalar(args[0]).chr() if len(args) == 1 else None
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        args = list(node.expressions)
        if name == "bin" and len(args) == 1:
            # Not `to_base(x, 2)`: DuckDB's `bin` renders a negative as its 64-bit
            # two's-complement pattern where `to_base` refuses one outright, so mapping
            # them together answered `bin(-3)` with `-11`.
            return StrFunc("bin", tr._scalar(args[0]))
        if name == "to_base" and len(args) == 2:
            return tr._scalar(args[0]).to_base(_const_int_arg(args[1], "to_base(): radix"))
        if name in _FORMAT_BYTES and len(args) == 1:
            return tr._scalar(args[0]).format_bytes(si=_FORMAT_BYTES[name])
        if name == "conv" and len(args) == 3:
            return _conv(tr, args)
        if name in ("printf", "format_string") and args:
            return _format_call(tr, args[0], args[1:])
        if name == "quote" and len(args) == 1:
            return _quote(tr._scalar(args[0]))
    if isinstance(node, exp.Format):
        # DuckDB's `format('{} and {}', ...)` and Spark's `format_string('a %d', ...)`
        # both parse to this node; the template's own syntax says which one it is.
        return _format_call(tr, node.this, node.expressions)
    if isinstance(node, exp.RegexpFullMatch):
        # DuckDB's `regexp_full_match` requires the pattern to match the *whole* string,
        # where `regexp_matches` is a search. Anchoring is the difference, and the
        # non-capturing group is load-bearing: `^a|b$` would otherwise anchor only the
        # first alternative.
        from batcher._sql.parser.expressions.lowering.dynamic import const_str, str_call

        pat = const_str(node.expression)
        if pat is not None:
            return tr._scalar(node.this).str.regexp_matches(f"^(?:{pat})$")
        anchored = Binary(
            "concat",
            Binary("concat", lit("^(?:"), tr._scalar(node.expression)),
            lit(")$"),
        )
        return str_call(tr, "regexp_matches", node.this, pattern=anchored)
    return None


def _conv(tr, args) -> Expr | None:
    """Spark `conv(text, from_base, to_base)` — re-base a number written as text.

    Only a *constant* source base is served: the digits have to be parsed before they can
    be re-written, and the engine's cast reads decimal. Base 10 in is the common call
    (`conv('100', 10, 2)`); any other source base is declined rather than misparsed.
    """
    from batcher._sql.parser.expressions.literals import _const_int_arg

    from_base = _const_int_arg(args[1], "conv(): source base")
    to_base = _const_int_arg(args[2], "conv(): target base")
    if from_base != 10:
        return None
    return tr._scalar(args[0]).cast("int64").to_base(to_base)


#: A printf conversion: an optional argument index, then flags/width/precision, then the
#: conversion character. Only a *bare* conversion is representable here, so the groups
#: exist to detect the ones that are not rather than to honour them.
_PRINTF = re.compile(r"%(\d+\$)?([-+ #0]*)(\d+|\*)?(\.\d+|\.\*)?([a-zA-Z%])")

#: printf conversions the brace template can express, and the cast each one implies.
#: `d`/`i` truncate toward zero the way C does; `s` takes the value as written.
_PRINTF_CAST = {"s": None, "d": "int64", "i": "int64", "f": "float64", "b": None}

#: The conversions whose C definition *truncates* toward zero rather than rounding.
#: `.cast("int64")` rounds, so `%d` of 3.7 came out as 4 where C writes 3. DuckDB
#: refuses a float for `%d` outright; truncating is the more useful answer and is
#: identical to DuckDB's for the integer arguments the conversion is actually for.
_PRINTF_TRUNCATES = frozenset({"d", "i"})


def _format_call(tr, template, values) -> Expr:
    """`format`/`printf`/`format_string` — interpolate `values` into a constant template.

    Args:
        tr: The translator, for the recursive scalar translation of each value.
        template: The template node. Must be a string literal: the number and order of
            the holes decides the plan shape, so it cannot come from a column.
        values: The value nodes to interpolate.

    Returns:
        The concatenation expression the template denotes.

    Raises:
        NotImplementedError: If the template is not a literal, or carries a printf
            conversion the engine cannot reproduce exactly.
    """
    from batcher._sql.parser.expressions.literals import _const_str_arg

    raw = _const_str_arg(template, "format()", "template")
    braces, casts = _to_brace_template(raw, len(values))
    args = [
        tr._scalar(v) if cast is None else _converted(tr._scalar(v), cast)
        for v, cast in zip(values, casts, strict=True)
    ]
    return _null_if_any_null(format_string(braces, *args), args)


def _to_brace_template(raw: str, count: int) -> tuple[str, list[str | None]]:
    """A printf or brace template as `(brace_template, per-argument cast)`.

    A template with no ``%`` conversion is already brace-form and is returned unchanged,
    which is what makes one function serve both dialects. Otherwise every conversion is
    rewritten to ``{}`` and contributes the cast its letter implies.
    """
    if not _PRINTF.search(raw):
        return raw, [None] * count
    casts: list[str | None] = []
    out: list[str] = []
    last = 0
    for m in _PRINTF.finditer(raw):
        index, flags, width, precision, letter = m.groups()
        out.append(raw[last : m.start()])
        last = m.end()
        if letter == "%":
            out.append("%")
            continue
        if index or flags or width or precision:
            raise NotImplementedError(
                f"printf conversion {m.group(0)!r} carries a flag, width or precision, "
                "which the engine cannot reproduce; format the value explicitly instead"
            )
        if letter not in _PRINTF_CAST:
            raise NotImplementedError(f"printf conversion %{letter} is not supported")
        casts.append(_PRINTF_CAST[letter])
        out.append("{}")
    out.append(raw[last:])
    template = "".join(out)
    if len(casts) != count:
        raise PlanError(f"format(): {count} argument(s) but {len(casts)} conversion(s)")
    # The literal `%%` became a single `%`; a brace that was already in the text would
    # now read as a hole, so escaping is not attempted — a template mixing the two
    # syntaxes is ambiguous and the caller should pick one.
    return template, casts


def _quote(value: Expr) -> Expr:
    """Spark `quote(s)` — wrap in single quotes, backslash-escaping any the value holds.

    Spark's convention is a **backslash** before an embedded quote, not the doubled quote
    SQL uses for its own literals. The two produce different text for the same input, and
    the function exists to be pasted back into a Spark query, so it follows Spark.

    Composed rather than given a kernel because it is two string operations, and the
    escaping has to happen *before* the wrapping quotes are added or it would escape
    those too.
    """
    quoted = concat(lit("'"), value.str.replace("'", "\\'"), lit("'"))
    return when(value.is_not_null()).then(quoted).otherwise(nullif(quoted, quoted))


def _converted(value: Expr, cast: str) -> Expr:
    """Apply a printf conversion's implied cast, truncating where C truncates."""
    if cast == "int64":
        return value.trunc().cast("int64")
    return value.cast(cast)


def _null_if_any_null(built: Expr, args: list[Expr]) -> Expr:
    """`built`, or NULL when any of `args` is null — the template family's null rule.

    `format`/`printf` return NULL if any argument is NULL, which is *not* what the
    underlying `format_string` builder does: it composes `concat`, and `concat` treats a
    null as empty (that is DuckDB's `concat` too, and the reason `||` and `concat` differ
    there). Without this guard `format('{}', NULL)` returned the template with a hole
    where DuckDB returns NULL — a plausible string in place of a null, on every row that
    has one.
    """
    if not args:
        return built
    guard = args[0].is_not_null()
    for extra in args[1:]:
        guard = guard & extra.is_not_null()
    return when(guard).then(built).otherwise(nullif(built, built))
