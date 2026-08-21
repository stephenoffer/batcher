"""Named-function dispatch for the SQL translator's scalar path.

Maps a SQL scalar function call node (math/string/date builtins, list/array
operations, `regexp_replace`, `date_diff`) to its `Expr` builder. Split out of
the sibling `scalar` dispatch so each stays under the ceiling; every function takes the
translator instance (`tr`) as its first argument and recurses through `tr._scalar`.
"""

from __future__ import annotations

import math

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.collections import collection_function
from batcher._sql.parser.expressions.json import json_function
from batcher._sql.parser.expressions.literals import (
    _DATE_PART,
    _EXTRACT_COMPOSITE,
    _EXTRACT_PART,
    _UNARY_MATH,
    _UNARY_STR,
    _const_int_arg,
    _const_str_arg,
    _int_literal,
)
from batcher._sql.parser.expressions.lowering.dynamic import (
    const_str,
    dynamic_left,
    str_call,
)
from batcher._sql.parser.expressions.maps import map_function
from batcher._sql.parser.expressions.spark import spark_function
from batcher._sql.parser.expressions.strings import string_function
from batcher._sql.parser.expressions.temporal import datetime_pattern, temporal_function
from batcher.plan.expr_ir import Binary, Cast, Expr, Math2Expr, atan2, lit, when
from batcher.plan.functions.temporal import current_date, make_date

# sqlglot node names for the nullary constant functions → the literal they denote.
_NULLARY_CONST = {
    "Pi": lambda: lit(math.pi),
    "CurrentDate": current_date,
}

# Typed sqlglot nodes of the shape `f(value, constant-string)`: the engine's method
# takes the second operand as a Python `str` (a pattern, delimiter or comparison
# string), not an expression. The value is the `.str` method and the role name used in
# the rejection message when the argument is not a literal.
_STR_CONST_ARG = {
    "Split": ("split", "delimiter"),
    "Levenshtein": ("levenshtein", "comparison string"),
    "RegexpSplit": ("regexp_split", "pattern"),
    "JarowinklerSimilarity": ("jaro_winkler_similarity", "comparison string"),
    "JaroSimilarity": ("jaro_similarity", "comparison string"),
}


def _empty_string_is_minus_one(value: Expr) -> Expr:
    """`ord(s)`/`unicode(s)` — the first code point, or -1 for the empty string.

    DuckDB draws a line the kernel does not: `ascii('')` is 0 while `ord('')` and
    `unicode('')` are -1. A null argument stays null under both, which the `otherwise`
    branch gives for free.

    Args:
        value: The string expression.

    Returns:
        The code-point expression.
    """
    return when(value.str.len() == lit(0)).then(lit(-1)).otherwise(value.str.ascii())


def _collection_len(tr, arg):
    """`len(x)` for a list or map column, or None when `x` is not one (or is unknown).

    Args:
        tr: The translator, for the in-scope column types.
        arg: The argument node.

    Returns:
        The element-count expression, or None to fall through to the string reading.
    """
    import pyarrow as pa

    t = tr.column_type(arg)
    if t is None:
        return None
    if pa.types.is_map(t):
        return tr._scalar(arg).map.len()
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return tr._scalar(arg).list.len()
    return None


def _scalar_function(tr, node):
    """Map a SQL scalar function call to its `Expr` builder, or None."""
    name = type(node).__name__
    if name in _NULLARY_CONST:
        # `pi()` / `today()` — DuckDB spells them as nullary functions, and sqlglot has a
        # typed node for each. There is nothing per-row to compute, so they lower to a
        # literal at plan-build time (which also makes them constant-foldable downstream).
        return _NULLARY_CONST[name]()
    if name == "Trunc" and node.args.get("decimals") is not None:
        # `trunc(x, n)` truncates to `n` decimal places; the one-arg `.trunc()`
        # ignores `n` and silently truncated to a whole number. Scale, truncate,
        # unscale so `trunc(2.567, 1)` is `2.5`, not `2.0`.
        #
        # `n` may be a column: the whole rewrite is arithmetic, so nothing about it needs
        # a plan-time constant, and requiring one refused `trunc(x, scale)` outright.
        decimals = node.args["decimals"]
        digits = _int_literal(decimals)
        factor = lit(10.0**digits) if digits is not None else lit(10.0) ** tr._scalar(decimals)
        value = tr._scalar(node.this)
        scaled = value * factor
        # A magnitude that overflows when scaled has no representable fractional part at
        # that scale, so it truncates to itself. Without the guard `trunc(1e308, 1)`
        # scaled to +inf and came back as inf where DuckDB answers 1e308.
        return when(scaled.is_finite()).then(scaled.trunc() / factor).otherwise(value)
    if name in _UNARY_MATH:
        return getattr(tr._scalar(node.this), _UNARY_MATH[name])()
    if name == "Length":
        # DuckDB's `len`/`length` is defined on strings, lists and maps alike; the
        # argument's type picks the reading. Dispatching on the name alone sent a list
        # column into the string kernel, which refused the whole query ("string function
        # Len expected a Utf8 argument, got List").
        collection = _collection_len(tr, node.this)
        if collection is not None:
            return collection
    if name == "Unicode":
        # `unicode('')`/`ord('')` is -1 where `ascii('')` is 0 — the kernel implements
        # `ascii`, so the empty-string case is layered on here rather than forked there.
        return _empty_string_is_minus_one(tr._scalar(node.this))
    if name in _UNARY_STR:
        if name == "ToBinary" and node.args.get("format") is not None:
            # DuckDB's `to_binary(s)` is a `0`/`1` *bit string*; Spark's
            # `to_binary(s, charset)` is the encoded *bytes*. Same name, different
            # function — so the two-argument form is refused rather than answered with
            # the bit string, which is what it silently did.
            raise NotImplementedError(
                "to_binary(value, charset) is Spark's binary encoding, not DuckDB's "
                "bit-string to_binary; the two-argument form is not supported"
            )
        return getattr(tr._scalar(node.this).str, _UNARY_STR[name])()
    if name in _DATE_PART:
        return getattr(tr._scalar(node.this).dt, _DATE_PART[name])()
    if name == "Round":
        # `decimals` is the digit count; dropping it rounded to a whole number instead.
        # It may be a column — the kernel takes the digit count per row — and demanding a
        # literal refused `round(x, scale)`, an ordinary shape in a currency table.
        decimals = node.args.get("decimals")
        if decimals is None:
            return tr._scalar(node.this).round()
        digits = _int_literal(decimals)
        if digits is not None:
            return tr._scalar(node.this).round(digits)
        return Math2Expr("round", tr._scalar(node.this), tr._scalar(decimals))
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
        chars = node.args.get("expression")
        position = node.args.get("position") or "BOTH"
        tag = {"LEADING": "l_trim", "TRAILING": "r_trim"}.get(position, "trim")
        return str_call(tr, tag, node.this, pattern=chars)
    if name == "Replace":
        return str_call(
            tr,
            "replace",
            node.this,
            pattern=node.expression,
            replacement=node.args.get("replacement"),
        )
    if name == "SplitPart":
        return str_call(
            tr,
            "split_part",
            node.this,
            pattern=node.args.get("delimiter"),
            start=node.args.get("part_index"),
        )
    if name == "StartsWith":
        return str_call(tr, "starts_with", node.this, pattern=node.expression)
    if name == "Repeat":
        return str_call(tr, "repeat", node.this, start=node.args.get("times"))
    if name == "Substring":
        # A negative start (`substr(s, -2)`) parses as a `Neg`, not a bare literal, and
        # either bound may be a column; `str_call` folds the sign and picks the constant
        # or per-row form.
        return str_call(
            tr,
            "substr",
            node.this,
            start=node.args["start"],
            length=node.args.get("length"),
        )
    if name in ("Left", "Right"):
        if name == "Right":
            return str_call(tr, "right", node.this, start=node.expression)
        n = _int_literal(node.expression)
        if n is None:
            return dynamic_left(tr, node.this, node.expression)
        return tr._scalar(node.this).str.left(n)
    if name in ("EndsWith", "Contains"):
        tag = "ends_with" if name == "EndsWith" else "contains"
        return str_call(tr, tag, node.this, pattern=node.expression)
    if name in _STR_CONST_ARG:
        # `jaro_winkler_similarity(a, b, scale)` carries a prefix scale factor the kernel
        # has no parameter for (sqlglot parks it under `case_insensitive`). Dropping it
        # answered the *default*-scale similarity — a plausible number that is not the one
        # asked for — so the extra argument is refused instead.
        extra = node.args.get("case_insensitive") or node.expressions
        if extra:
            raise NotImplementedError(
                f"{name.lower()}() takes two arguments here; the third (the prefix scale "
                "factor) is not a parameter the engine's kernel has"
            )
        # `f(s, t)` where the second operand fills the kernel's `pattern` slot:
        # `split`/`str_split`, the edit-distance metrics, `regexp_split`.
        tag, _role = _STR_CONST_ARG[name]
        return str_call(tr, tag, node.this, pattern=node.expression)
    if name == "Translate":
        # sqlglot names the source set `from_`, not `expression`.
        return str_call(
            tr,
            "translate",
            node.this,
            pattern=node.args.get("from_"),
            replacement=node.args.get("to"),
        )
    if name == "TimeToStr":  # strftime(ts, fmt) / Spark date_format(ts, fmt)
        raw = _const_str_arg(node.args.get("format"), "strftime()", "format")
        pattern = datetime_pattern(raw)
        if pattern is None:
            raise NotImplementedError(f"datetime format {raw!r} is not supported")
        return tr._scalar(node.this).dt.strftime(pattern)
    if name == "TsOrDsToDate":
        # Spark's implicit "this is a date" wrapper — `year('2016-07-30')` parses as
        # `Year(TsOrDsToDate('2016-07-30'))`. With a format it is `to_date(s, fmt)`;
        # without, it is a plain cast, which is what Spark means by it.
        fmt = node.args.get("format")
        value = tr._scalar(node.this)
        if fmt is None:
            return Cast(value, "date")
        raw = _const_str_arg(fmt, "to_date()", "format")
        pattern = datetime_pattern(raw)
        if pattern is None:
            raise NotImplementedError(f"datetime format {raw!r} is not supported")
        return value.str.to_date(pattern)
    if name == "TimeToUnix":
        # `epoch(ts)` is DuckDB's *fractional* seconds since the epoch (a DOUBLE), not
        # the whole seconds `.dt.epoch()` returns — `epoch('…:08.123456')` is
        # `1709618828.123456`. Divide the microsecond count instead of truncating.
        return tr._scalar(node.this).dt.epoch_us() / lit(1_000_000.0)
    if name == "Atan2":
        return atan2(tr._scalar(node.this), tr._scalar(node.expression))
    if name == "DateFromParts":  # make_date(y, m, d)
        return make_date(
            tr._scalar(node.args["year"]),
            tr._scalar(node.args["month"]),
            tr._scalar(node.args["day"]),
        )
    if name == "SHA2":
        # `sha256(s)` parses as SHA2 with a digest length; only 256 is implemented, so
        # any other width is refused rather than silently answered with sha256.
        width = node.args.get("length")
        bits = _const_int_arg(width, "sha2(): digest length") if width is not None else 256
        if bits != 256:
            raise NotImplementedError(f"sha2 digest length {bits} is not supported (only 256)")
        return tr._scalar(node.this).str.sha256()
    if name == "ArrayIntersect":
        # sqlglot gives both operands in `expressions` with no `this`.
        operands = node.expressions
        if len(operands) == 2:
            return tr._scalar(operands[0]).list.intersect(tr._scalar(operands[1]))
    if name in ("RegexpExtract", "RegexpExtractAll"):
        # Both carry an optional capture-group index. `regexp_extract_all` used to drop
        # it, so `regexp_extract_all('100-200', '(\\d+)-(\\d+)', 1)` collected the whole
        # matches (`['100-200']`) where DuckDB collects the group (`['100']`).
        tag = "regexp_extract" if name == "RegexpExtract" else "regexp_extract_all"
        grp = node.args.get("group")
        return str_call(
            tr, tag, node.this, pattern=node.expression, start=0 if grp is None else grp
        )
    if name == "StrPosition":
        return str_call(tr, "position", node.this, pattern=node.args["substr"])
    if name == "Pad":
        # A negative width (`lpad(s, -1, '*')`) parses as a `Neg` wrapping the
        # literal, so `int(node.args["expression"].this)` raised a TypeError on the
        # `Neg` node. `_int_literal` folds the sign; the engine's `.lpad`/`.rpad`
        # clamp a non-positive width to the empty string, matching DuckDB.
        fill = node.args.get("fill_pattern")
        tag = "lpad" if bool(node.args.get("is_left")) else "rpad"
        return str_call(
            tr,
            tag,
            node.this,
            start=node.args["expression"],
            pattern=" " if fill is None else fill,
        )
    if isinstance(node, exp.Anonymous):
        ml = _UNARY_ML.get(node.name.lower())
        if ml is not None and len(node.expressions) == 1:
            return getattr(tr._scalar(node.expressions[0]), ml)()
    if isinstance(node, exp.Anonymous) and node.name.lower() == "date_part":
        # `date_part('unit', ts)` — the field-name spelling of EXTRACT. sqlglot keeps
        # it Anonymous (unit literal first, then the temporal argument).
        args = node.expressions
        if len(args) != 2 or not (isinstance(args[0], exp.Literal) and args[0].is_string):
            raise NotImplementedError("date_part(unit, ts): unit must be a string literal")
        part = args[0].this.lower()
        composite = _EXTRACT_COMPOSITE.get(part)
        if composite is not None:
            return composite(tr._scalar(args[1]).dt)
        method = _EXTRACT_PART.get(part)
        if method is None:
            raise NotImplementedError(f"date_part field {args[0].this!r} is not supported")
        return getattr(tr._scalar(args[1]).dt, method)()
    # Families that carry enough of their own dispatch to live in a module of their own.
    # Each returns None for a name it does not serve, so the caller's "unknown function"
    # error still names it.
    for family in (
        json_function,
        temporal_function,
        string_function,
        collection_function,
        spark_function,
        map_function,
    ):
        built = family(tr, node)
        if built is not None:
            return built
    return None


# Elementwise ML activation functions callable in SQL → the `Expr` method. These make a
# feature-engineering / scoring step expressible in the query itself (a small ML-in-SQL
# surface): ``SELECT sigmoid(logit_score) AS p FROM t``.
_UNARY_ML = {
    "sigmoid": "sigmoid",
    "relu": "relu",
    "softplus": "softplus",
    "logit": "logit",
    "silu": "silu",
    "swish": "silu",  # SiLU and Swish are the same activation
    "gelu": "gelu",
    "mish": "mish",
    "hardsigmoid": "hardsigmoid",
    "hardswish": "hardswish",
}


def _regexp_replace(tr, node) -> Expr:
    """`regexp_replace(s, pattern, replacement[, options])` — replace the first match, or
    every match with the ``'g'`` option (DuckDB default is first-only; constant args).

    The ``options`` string is honoured, not dropped: ``'g'`` selects the global
    (replace-all) variant, and ``'i'``/``'s'``/``'c'`` map to an inline regex flag prefix
    (`_regexp_flags_prefix`). Previously the whole options arg was ignored, so
    ``regexp_replace(s, 'abc', 'X', 'i')`` matched case-sensitively (wrong vs DuckDB)."""

    from batcher._sql.parser.expressions.literals import _regexp_flags_prefix

    pat_node, repl_node = node.expression, node.args.get("replacement")
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
    tag = "regexp_replace_all" if global_replace else "regexp_replace"
    pat = const_str(pat_node)
    if pat is not None:
        pat_node = prefix + pat
    elif prefix:
        # A per-row pattern cannot carry the flag prefix through `str_call`'s literal
        # slot, so the two are concatenated as an expression instead.
        pat_node = Binary("concat", lit(prefix), tr._scalar(pat_node))
    return str_call(tr, tag, node.this, pattern=pat_node, replacement=repl_node)
