"""SQL string functions whose translation is more than a name lookup.

The name-keyed tables in `literals` (`_UNARY_STR`) and `anonymous` cover the string
functions that map onto a `.str` method one-for-one. This module holds the rest: the ones
that need the argument rewritten (`regexp_full_match` anchors its pattern) or a shape the
tables cannot express.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Expr

__all__ = ["string_function"]

# DuckDB's three spellings of "a byte count as text", and whether each uses SI units.
_FORMAT_BYTES = {
    "format_bytes": False,
    "formatreadablesize": False,
    "formatreadabledecimalsize": True,
}


def string_function(tr, node) -> Expr | None:
    """Translate a string call needing a rewrite, or None when the name is not one."""

    from batcher._sql.parser.expressions.literals import _const_int_arg, _const_str_arg

    if isinstance(node, exp.Chr):
        # `chr(65)` / Spark `char(65)`. sqlglot keeps the argument list, not `this`.
        args = node.expressions or ([node.this] if node.this is not None else [])
        return tr._scalar(args[0]).chr() if len(args) == 1 else None
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        args = list(node.expressions)
        if name == "bin" and len(args) == 1:
            return tr._scalar(args[0]).to_base(2)
        if name == "to_base" and len(args) == 2:
            return tr._scalar(args[0]).to_base(_const_int_arg(args[1], "to_base(): radix"))
        if name in _FORMAT_BYTES and len(args) == 1:
            return tr._scalar(args[0]).format_bytes(si=_FORMAT_BYTES[name])
        if name == "conv" and len(args) == 3:
            return _conv(tr, args)
    if isinstance(node, exp.RegexpFullMatch):
        # DuckDB's `regexp_full_match` requires the pattern to match the *whole* string,
        # where `regexp_matches` is a search. Anchoring is the difference, and the
        # non-capturing group is load-bearing: `^a|b$` would otherwise anchor only the
        # first alternative.
        pat = _const_str_arg(node.expression, "regexp_full_match()", "pattern")
        return tr._scalar(node.this).str.regexp_matches(f"^(?:{pat})$")
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
