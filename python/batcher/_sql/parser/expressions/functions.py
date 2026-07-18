"""Named-function dispatch for the SQL translator's scalar path.

Maps a SQL scalar function call node (math/string/date builtins, list/array
operations, `regexp_replace`, `date_diff`) to its `Expr` builder. Split out of
the sibling `scalar` dispatch so each stays under the ceiling; every function takes the
translator instance (`tr`) as its first argument and recurses through `tr._scalar`.
"""

from __future__ import annotations

from batcher._sql.parser.expressions.literals import (
    _DATE_PART,
    _EXTRACT_PART,
    _UNARY_MATH,
    _UNARY_STR,
    _int_literal,
)
from batcher.plan.expr_ir import Cast, Expr, lit


def _scalar_function(tr, node):
    """Map a SQL scalar function call to its `Expr` builder, or None."""
    from sqlglot import expressions as exp

    name = type(node).__name__
    if name == "Trunc" and node.args.get("decimals") is not None:
        # `trunc(x, n)` truncates to `n` decimal places; the one-arg `.trunc()`
        # ignores `n` and silently truncated to a whole number. Scale, truncate,
        # unscale so `trunc(2.567, 1)` is `2.5`, not `2.0`.
        digits = _int_literal(node.args["decimals"])
        if digits is None:
            raise NotImplementedError("trunc(x, n): n must be an integer literal")
        factor = lit(10.0**digits)
        return (tr._scalar(node.this) * factor).trunc() / factor
    if name in _UNARY_MATH:
        return getattr(tr._scalar(node.this), _UNARY_MATH[name])()
    if name in _UNARY_STR:
        return getattr(tr._scalar(node.this).str, _UNARY_STR[name])()
    if name in _DATE_PART:
        return getattr(tr._scalar(node.this).dt, _DATE_PART[name])()
    if name == "Round":
        # `decimals` is the digit count; dropping it rounded to a whole number instead.
        decimals = node.args.get("decimals")
        if decimals is None:
            return tr._scalar(node.this).round()
        digits = _int_literal(decimals)
        if digits is None:
            raise NotImplementedError("ROUND(x, n): n must be an integer literal")
        return tr._scalar(node.this).round(digits)
    if name == "Log":
        # log(x) → log10(x); log10(x)/log2(x) parse as log(base, value) with
        # the base in `this` and the value in `expression`.
        value = node.args.get("expression")
        if value is None:
            return tr._scalar(node.this).log10()
        base = node.this
        if isinstance(base, exp.Literal) and base.this == "10":
            return tr._scalar(value).log10()
        if isinstance(base, exp.Literal) and base.this == "2":
            return tr._scalar(value).log2()
        # General base: log_b(x) = ln(x) / ln(b).
        return tr._scalar(value).ln() / tr._scalar(base).ln()
    if name == "Trim":
        # ltrim/rtrim (and `TRIM(LEADING/TRAILING …)`) carry a `position`; treating
        # them all as a both-sided trim silently stripped the wrong end. An optional
        # `expression` is the character set to strip (defaults to whitespace).
        base = tr._scalar(node.this).str
        chars_node = node.args.get("expression")
        chars = chars_node.this if isinstance(chars_node, exp.Literal) else None
        position = node.args.get("position") or "BOTH"
        if position == "LEADING":
            return base.lstrip(chars)
        if position == "TRAILING":
            return base.rstrip(chars)
        return base.trim(chars)
    if name == "Replace":
        pat, repl = node.expression, node.args.get("replacement")
        if not (isinstance(pat, exp.Literal) and pat.is_string):
            raise NotImplementedError("replace() requires a constant string pattern")
        if not (isinstance(repl, exp.Literal) and repl.is_string):
            raise NotImplementedError("replace() requires a constant string replacement")
        return tr._scalar(node.this).str.replace(pat.this, repl.this)
    if name == "SplitPart":
        delim, part = node.args.get("delimiter"), node.args.get("part_index")
        if not (isinstance(delim, exp.Literal) and delim.is_string):
            raise NotImplementedError("split_part() requires a constant string delimiter")
        idx = _int_literal(part)
        if idx is None:
            raise NotImplementedError("split_part() requires an integer field index")
        return tr._scalar(node.this).str.split_part(delim.this, idx)
    if name == "StartsWith":
        pat = node.expression
        if not (isinstance(pat, exp.Literal) and pat.is_string):
            raise NotImplementedError("starts_with() requires a constant string prefix")
        return tr._scalar(node.this).str.starts_with(pat.this)
    if name == "Repeat":
        times = _int_literal(node.args.get("times"))
        if times is None:
            raise NotImplementedError("repeat() requires an integer repeat count")
        return tr._scalar(node.this).str.repeat(times)
    if name == "Substring":
        base = tr._scalar(node.this).str
        # A negative start (`substr(s, -2)`) parses as a `Neg`, not a bare literal —
        # `int(node.args["start"].this)` then raised a TypeError. `_int_literal`
        # folds the sign; the engine's `.substr` handles negative offsets (matching
        # DuckDB: it counts from the string end).
        start = _int_literal(node.args["start"])
        if start is None:
            raise NotImplementedError("substr(): start must be an integer literal")
        length_node = node.args.get("length")
        if length_node is None:
            return base.substr(start)
        length = _int_literal(length_node)
        if length is None:
            raise NotImplementedError("substr(): length must be an integer literal")
        return base.substr(start, length)
    if name in ("Left", "Right"):
        n = _int_literal(node.expression)
        if n is None:
            raise NotImplementedError(f"{name.lower()}(): length must be an integer literal")
        method = "left" if name == "Left" else "right"
        return getattr(tr._scalar(node.this).str, method)(n)
    if name in ("EndsWith", "Contains"):
        pat = node.expression
        if not (isinstance(pat, exp.Literal) and pat.is_string):
            raise NotImplementedError(f"{name.lower()}() requires a constant string argument")
        method = "ends_with" if name == "EndsWith" else "contains"
        return getattr(tr._scalar(node.this).str, method)(pat.this)
    if name == "RegexpExtract":
        pat = node.expression
        if not (isinstance(pat, exp.Literal) and pat.is_string):
            raise NotImplementedError("regexp_extract() requires a constant string pattern")
        group = _int_literal(node.args.get("group")) if node.args.get("group") is not None else 0
        if group is None:
            raise NotImplementedError("regexp_extract() capture group must be an integer literal")
        return tr._scalar(node.this).str.regexp_extract(pat.this, group)
    if name == "StrPosition":
        pat = node.args["substr"]
        if not isinstance(pat, exp.Literal) or not pat.is_string:
            raise NotImplementedError("position() requires a string literal pattern")
        return tr._scalar(node.this).str.position(pat.this)
    if name == "Pad":
        # A negative width (`lpad(s, -1, '*')`) parses as a `Neg` wrapping the
        # literal, so `int(node.args["expression"].this)` raised a TypeError on the
        # `Neg` node. `_int_literal` folds the sign; the engine's `.lpad`/`.rpad`
        # clamp a non-positive width to the empty string, matching DuckDB.
        width = _int_literal(node.args["expression"])
        if width is None:
            raise NotImplementedError("lpad()/rpad(): width must be an integer literal")
        fill_node = node.args.get("fill_pattern")
        fill = fill_node.this if fill_node is not None else " "
        base = tr._scalar(node.this).str
        is_left = bool(node.args.get("is_left"))
        return base.lpad(width, fill) if is_left else base.rpad(width, fill)
    if isinstance(node, exp.Anonymous) and node.name.lower() == "date_part":
        # `date_part('unit', ts)` — the field-name spelling of EXTRACT. sqlglot keeps
        # it Anonymous (unit literal first, then the temporal argument).
        args = node.expressions
        if len(args) != 2 or not (isinstance(args[0], exp.Literal) and args[0].is_string):
            raise NotImplementedError("date_part(unit, ts): unit must be a string literal")
        method = _EXTRACT_PART.get(args[0].this.lower())
        if method is None:
            raise NotImplementedError(f"date_part field {args[0].this!r} is not supported")
        return getattr(tr._scalar(args[1]).dt, method)()
    return None


# Typed `Array*`/`SortArray` reduction nodes → `.list` method name.
_LIST_REDUCE = {
    "ArrayMin": "min",
    "ArrayMax": "max",
    "ArraySum": "sum",
    "ArrayDistinct": "unique",
    "SortArray": "sort",
}
# `list_*` functions that sqlglot parses as `Anonymous` → `.list` method name.
_LIST_ANON = {
    "list_sum": "sum",
    "list_avg": "mean",
    "list_mean": "mean",
    "list_product": "product",
    "list_reverse": "reverse",
    "list_unique": "unique",
    "list_count": "len",
    "list_min": "min",
    "list_max": "max",
}


def _list_function(tr, node):
    """List/array operations dispatched to the `.list` namespace, or None."""
    from sqlglot import expressions as exp

    if isinstance(node, exp.ArraySize):  # array_length / len(list)
        return tr._scalar(node.this).list.len()
    if isinstance(node, exp.ArrayContains):  # list_contains(a, v)
        return tr._scalar(node.this).list.contains(_raw_value(node.expression))
    if isinstance(node, exp.Bracket):  # a[i] — sqlglot already 0-bases the index
        idxs = node.expressions
        if len(idxs) == 1 and not isinstance(idxs[0], exp.Slice):
            return tr._scalar(node.this).list.get(int(idxs[0].name))
        return None  # slices (a[lo:hi]) not supported
    reduce = _LIST_REDUCE.get(type(node).__name__)
    if reduce is not None:
        return getattr(tr._scalar(node.this).list, reduce)()
    if isinstance(node, exp.Anonymous):
        method = _LIST_ANON.get(node.name.lower())
        if method is not None and node.expressions:
            return getattr(tr._scalar(node.expressions[0]).list, method)()
    return None


def _raw_value(node):
    """The Python value of a literal node (for `.list.contains`)."""
    from sqlglot import expressions as exp

    if not isinstance(node, exp.Literal):
        raise NotImplementedError("list_contains requires a constant value")
    if node.is_string:
        return node.name
    text = node.name
    return float(text) if ("." in text or "e" in text.lower()) else int(text)


def _regexp_replace(tr, node) -> Expr:
    """`regexp_replace(s, pattern, replacement[, options])` — replace the first match, or
    every match with the ``'g'`` option (DuckDB default is first-only; constant args).

    The ``options`` string is honoured, not dropped: ``'g'`` selects the global
    (replace-all) variant, and ``'i'``/``'s'``/``'c'`` map to an inline regex flag prefix
    (`_regexp_flags_prefix`). Previously the whole options arg was ignored, so
    ``regexp_replace(s, 'abc', 'X', 'i')`` matched case-sensitively (wrong vs DuckDB)."""
    from sqlglot import expressions as exp

    from batcher._sql.parser.expressions.literals import _regexp_flags_prefix

    pat = node.expression
    repl = node.args.get("replacement")
    if not (isinstance(pat, exp.Literal) and pat.is_string):
        raise NotImplementedError("regexp_replace requires a constant string pattern")
    if not (isinstance(repl, exp.Literal) and repl.is_string):
        raise NotImplementedError("regexp_replace requires a constant string replacement")
    mods = node.args.get("modifiers")
    global_replace = False
    prefix = ""
    if mods is not None:
        if not (isinstance(mods, exp.Literal) and mods.is_string):
            raise NotImplementedError("regexp_replace options must be a constant string")
        flags = mods.this
        global_replace = "g" in flags
        # 'g' controls all-vs-first here (not a regex flag); the rest map to the inline
        # prefix, which raises on any option it can't reproduce bit-identically.
        prefix = _regexp_flags_prefix(flags.replace("g", ""))
    s = tr._scalar(node.this)
    if global_replace:
        return s.str.regexp_replace_all(prefix + pat.this, repl.this)
    return s.str.regexp_replace(prefix + pat.this, repl.this)


def _date_diff(tr, node) -> Expr:
    """`date_diff(unit, a, b)` = (b - a) in `unit` (DAY/WEEK), for DATE inputs."""
    unit = (node.text("unit") or "DAY").upper()
    # sqlglot: this=end (b), expression=start (a).
    days = Cast(tr._scalar(node.this), "int64") - Cast(tr._scalar(node.expression), "int64")
    if unit.startswith("DAY"):
        return days
    if unit.startswith("WEEK"):
        # DuckDB's week difference is the whole number of 7-day spans, truncated
        # *toward zero* (so `-6` days is `0`, not `-1`), returned as an integer —
        # not the fractional `days / 7` this used to yield.
        return Cast((days / lit(7)).trunc(), "int64")
    raise NotImplementedError(f"date_diff unit {unit} not supported (only DAY/WEEK)")
