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
    _const_int_arg,
    _const_str_arg,
    _int_literal,
)
from batcher.plan.expr_ir import Cast, Expr, atan2, lit
from batcher.plan.functions.temporal import make_date

# Typed sqlglot nodes of the shape `f(value, constant-string)`: the engine's method
# takes the second operand as a Python `str` (a pattern, delimiter or comparison
# string), not an expression. The value is the `.str` method and the role name used in
# the rejection message when the argument is not a literal.
_STR_CONST_ARG = {
    "Split": ("split", "delimiter"),
    "Levenshtein": ("levenshtein", "comparison string"),
    "RegexpExtractAll": ("regexp_extract_all", "pattern"),
    "RegexpSplit": ("regexp_split", "pattern"),
    "JarowinklerSimilarity": ("jaro_winkler_similarity", "comparison string"),
    "JaroSimilarity": ("jaro_similarity", "comparison string"),
}


def _scalar_function(tr, node):
    """Map a SQL scalar function call to its `Expr` builder, or None."""
    from sqlglot import expressions as exp

    name = type(node).__name__
    if name == "Trunc" and node.args.get("decimals") is not None:
        # `trunc(x, n)` truncates to `n` decimal places; the one-arg `.trunc()`
        # ignores `n` and silently truncated to a whole number. Scale, truncate,
        # unscale so `trunc(2.567, 1)` is `2.5`, not `2.0`.
        digits = _const_int_arg(node.args["decimals"], "trunc(x, n): n")
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
        return tr._scalar(node.this).round(_const_int_arg(decimals, "ROUND(x, n): n"))
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
        pat = _const_str_arg(node.expression, "replace()", "pattern")
        repl = _const_str_arg(node.args.get("replacement"), "replace()", "replacement")
        return tr._scalar(node.this).str.replace(pat, repl)
    if name == "SplitPart":
        delim = _const_str_arg(node.args.get("delimiter"), "split_part()", "delimiter")
        idx = _int_literal(node.args.get("part_index"))
        if idx is None:
            raise NotImplementedError("split_part() requires an integer field index")
        return tr._scalar(node.this).str.split_part(delim, idx)
    if name == "StartsWith":
        pat = _const_str_arg(node.expression, "starts_with()", "prefix")
        return tr._scalar(node.this).str.starts_with(pat)
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
        start = _const_int_arg(node.args["start"], "substr(): start")
        length_node = node.args.get("length")
        if length_node is None:
            return base.substr(start)
        return base.substr(start, _const_int_arg(length_node, "substr(): length"))
    if name in ("Left", "Right"):
        n = _const_int_arg(node.expression, f"{name.lower()}(): length")
        method = "left" if name == "Left" else "right"
        return getattr(tr._scalar(node.this).str, method)(n)
    if name in ("EndsWith", "Contains"):
        pat = _const_str_arg(node.expression, f"{name.lower()}()")
        method = "ends_with" if name == "EndsWith" else "contains"
        return getattr(tr._scalar(node.this).str, method)(pat)
    if name in _STR_CONST_ARG:
        # `f(s, t)` where the engine's method takes `t` as a Python string, not an
        # expression: `split`/`str_split`, the edit-distance metrics, and
        # `regexp_extract_all`. Each already exists on `.str`; only the row was missing.
        method, role = _STR_CONST_ARG[name]
        text = _const_str_arg(node.expression, f"{name.lower()}()", role)
        return getattr(tr._scalar(node.this).str, method)(text)
    if name == "Translate":
        # `translate(s, from, to)` — both character sets must be constants. sqlglot
        # names the source set `from_`, not `expression`.
        frm = _const_str_arg(node.args.get("from_"), "translate()", "source character set")
        to = _const_str_arg(node.args.get("to"), "translate()", "target character set")
        return tr._scalar(node.this).str.translate(frm, to)
    if name == "TimeToStr":  # strftime(ts, fmt)
        fmt = _const_str_arg(node.args.get("format"), "strftime()", "format")
        return tr._scalar(node.this).dt.strftime(fmt)
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
    if name == "RegexpExtract":
        pat = _const_str_arg(node.expression, "regexp_extract()", "pattern")
        grp = node.args.get("group")
        group = _const_int_arg(grp, "regexp_extract() capture group") if grp is not None else 0
        return tr._scalar(node.this).str.regexp_extract(pat, group)
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
        width = _const_int_arg(node.args["expression"], "lpad()/rpad(): width")
        fill_node = node.args.get("fill_pattern")
        fill = fill_node.this if fill_node is not None else " "
        base = tr._scalar(node.this).str
        is_left = bool(node.args.get("is_left"))
        return base.lpad(width, fill) if is_left else base.rpad(width, fill)
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
        method = _EXTRACT_PART.get(args[0].this.lower())
        if method is None:
            raise NotImplementedError(f"date_part field {args[0].this!r} is not supported")
        return getattr(tr._scalar(args[1]).dt, method)()
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


# Typed `Array*` reduction nodes → `.list` method name. `SortArray` is *not* here: it
# carries a direction (`list_reverse_sort` parses as `SortArray(asc=False)`) that a bare
# method name cannot express, and folding it in here sorted descending calls ascending.
_LIST_REDUCE = {
    "ArrayMin": "min",
    "ArrayMax": "max",
    "ArraySum": "sum",
    "ArrayDistinct": "unique",
}
# `list_*` functions that sqlglot parses as `Anonymous` → `.list` method name. DuckDB
# spells every one of these `array_*` as well; `_list_anon_method` strips either prefix,
# so the keys are the bare operation.
#
# `unique` is the trap: DuckDB's `list_unique` returns the **count** of distinct
# elements, and `list_distinct` returns the distinct elements. Mapping both to the
# distinct list returned a list where DuckDB returns an integer.
_LIST_ANON = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "product": "product",
    "reverse": "reverse",
    "unique": "n_unique",
    "distinct": "unique",
    "count": "len",
    "length": "len",
    "min": "min",
    "max": "max",
}

# Two-argument vector functions → the binary `.list` method. Both DuckDB's canonical
# `list_*` spellings and the bare names are accepted, so a vector search reads naturally in
# SQL: ``ORDER BY cosine_similarity(emb, [0.1, 0.2]) DESC LIMIT 10``. This is the SQL-level
# vector-search / embedding-math surface (cf. DuckDB's list functions, BigQuery `ML.DISTANCE`).
_LIST_BINARY_ANON = {
    "list_cosine_similarity": "cosine_similarity",
    "cosine_similarity": "cosine_similarity",
    "list_cosine_distance": "cosine_distance",
    "cosine_distance": "cosine_distance",
    "list_distance": "l2_distance",  # DuckDB's L2 spelling
    "l2_distance": "l2_distance",
    "euclidean_distance": "l2_distance",
    "l1_distance": "l1_distance",
    "manhattan_distance": "l1_distance",
    "hamming_distance": "hamming_distance",
    "list_dot_product": "dot",
    "list_inner_product": "dot",
    "inner_product": "dot",
    "dot_product": "dot",
    "list_jaccard": "jaccard",
    # Element-wise vector arithmetic.
    "list_add": "add",
    "list_subtract": "subtract",
    "list_multiply": "multiply",
}

# sqlglot expression *types* (not `Anonymous`) for two-arg vector functions → binary method.
_LIST_TYPED_BINARY = {
    "EuclideanDistance": "l2_distance",
    "CosineDistance": "cosine_distance",
    "DotProduct": "dot",
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
    if isinstance(node, exp.SortArray):
        # `list_sort(l)` ascending; `list_reverse_sort(l)` descending — the latter
        # parses as the same node with `asc=False`, which used to be dropped, so
        # `list_reverse_sort` returned the ascending order.
        sorted_list = tr._scalar(node.this).list.sort()
        asc = node.args.get("asc")
        descending = asc is not None and not _boolean_arg(asc)
        return sorted_list.list.reverse() if descending else sorted_list
    # sqlglot promotes a few vector functions to typed nodes (two args in `this`/`expression`)
    # rather than `Anonymous`; dispatch them to the same binary `.list` methods.
    typed_binary = _LIST_TYPED_BINARY.get(type(node).__name__)
    if typed_binary is not None:
        return getattr(tr._scalar(node.this).list, typed_binary)(tr._scalar(node.expression))
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        method = _list_anon_method(name)
        if method is not None and node.expressions:
            return getattr(tr._scalar(node.expressions[0]).list, method)()
        binary = _LIST_BINARY_ANON.get(name)
        if binary is not None and len(node.expressions) == 2:
            left = tr._scalar(node.expressions[0])
            right = tr._scalar(node.expressions[1])
            return getattr(left.list, binary)(right)
    return None


def _list_anon_method(name: str) -> str | None:
    """The `.list` method for a `list_*`/`array_*` DuckDB spelling, or None.

    DuckDB gives every list operation both prefixes; stripping either here keeps one
    row per operation instead of two tables that can drift apart.
    """
    for prefix in ("list_", "array_"):
        if name.startswith(prefix):
            return _LIST_ANON.get(name.removeprefix(prefix))
    return None


def _boolean_arg(node) -> bool:
    """The boolean a sqlglot `Boolean`/literal argument denotes."""
    from sqlglot import expressions as exp

    if isinstance(node, exp.Boolean):
        return bool(node.this)
    return str(node.this).lower() not in ("false", "0")


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

    pat = _const_str_arg(node.expression, "regexp_replace", "pattern")
    repl = _const_str_arg(node.args.get("replacement"), "regexp_replace", "replacement")
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
        return s.str.regexp_replace_all(prefix + pat, repl)
    return s.str.regexp_replace(prefix + pat, repl)


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
