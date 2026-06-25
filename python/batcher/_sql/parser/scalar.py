"""Scalar expression dispatch — translate a sqlglot value node into an `Expr`.

The functions here take the translator instance (`tr`) as their first argument so
they can recurse through `tr._scalar`, resolve aggregate output columns, and run
nested subqueries via `tr.statement`. They hold no state of their own.
"""

from __future__ import annotations

from batcher._sql.parser.literals import (
    _BINOPS,
    _DATE_PART,
    _EXTRACT_PART,
    _TEMPORAL_KINDS,
    _UNARY_MATH,
    _UNARY_STR,
    _apply_interval,
    _dtype_name,
    _fold_const_arith,
    _like_to_regex,
    _literal,
    _temporal_literal,
)
from batcher.plan.expr_ir import (
    Array,
    Binary,
    Cast,
    Expr,
    ListJoin,
    Lit,
    coalesce,
    col,
    greatest,
    least,
    lit,
    nullif,
    when,
)


def _scalar(tr, node) -> Expr:
    from sqlglot import expressions as exp

    # Inside an aggregate query, an aggregate sub-expression refers to its
    # pre-computed output column.
    if tr._agg_map is not None and isinstance(node, exp.AggFunc):
        entry = tr._agg_map.get(node.sql())
        if entry is not None:
            # string_agg collects into a list (array_agg); join it here with the
            # separator (DuckDB default ',').
            if isinstance(node, exp.GroupConcat):
                sep = node.args.get("separator")
                sep = sep.name if isinstance(sep, exp.Literal) else ","
                return ListJoin(col(entry[0]), sep)
            return col(entry[0])
    if isinstance(node, exp.Paren):
        return tr._scalar(node.this)
    if isinstance(node, exp.Column):
        return col(node.name)
    if isinstance(node, exp.Literal):
        return _literal(node)
    if isinstance(node, exp.Boolean):
        return lit(bool(node.this))
    if isinstance(node, exp.Neg):
        return Lit(0) - tr._scalar(node.this)
    if isinstance(node, exp.Not):
        return ~tr._scalar(node.this)
    if isinstance(node, exp.Cast):
        # DATE '..' / TIMESTAMP '..' / CAST('..' AS DATE) parse as a cast of a
        # string literal to a temporal type — fold to a real temporal literal.
        inner = node.this
        kind = node.to.this.name if node.to and node.to.this else ""
        if isinstance(inner, exp.Literal) and inner.is_string and kind in _TEMPORAL_KINDS:
            return _temporal_literal(inner.this, kind)
        return Cast(tr._scalar(inner), _dtype_name(node.to))
    if isinstance(node, exp.Case):
        return _case(tr, node)
    if isinstance(node, exp.Null):
        # A bare NULL literal — a typed NULL (`nullif(c, c)` is null for all
        # rows). Used for `SELECT NULL`, `coalesce(x, NULL)`, etc.
        return nullif(lit(1), lit(1))
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        # x IS NULL  (x IS NOT NULL parses as Not(Is(...)), handled above)
        return tr._scalar(node.this).is_null()
    if isinstance(node, exp.Subquery):
        return _scalar_subquery(tr, node.this)
    if isinstance(node, (exp.Select, exp.Union)):
        return _scalar_subquery(tr, node)
    if isinstance(node, exp.In):
        return _in(tr, node)
    if isinstance(node, exp.Between):
        return _between(tr, node)
    if isinstance(node, exp.Escape):
        # `x [I]LIKE p ESCAPE e` → the inner Like/ILike with the escape char.
        inner = node.this
        return _like(
            tr,
            inner,
            case_insensitive=isinstance(inner, exp.ILike),
            escape=node.expression.this,
        )
    if isinstance(node, exp.ILike):
        return _like(tr, node, case_insensitive=True)
    if isinstance(node, exp.Like):
        return _like(tr, node)
    if isinstance(node, exp.Coalesce):
        return _coalesce(tr, node)
    if isinstance(node, exp.Nullif):
        return nullif(tr._scalar(node.this), tr._scalar(node.expression))
    if isinstance(node, exp.Greatest):
        return greatest(*_scalar_args(tr, node))
    if isinstance(node, exp.Least):
        return least(*_scalar_args(tr, node))
    if isinstance(node, exp.Array):
        return Array([tr._scalar(e) for e in node.expressions])
    list_fn = _list_function(tr, node)
    if list_fn is not None:
        return list_fn
    if isinstance(node, (exp.Concat, exp.ConcatWs)):
        return _concat(tr, node)
    if isinstance(node, exp.NullSafeNEQ):  # a IS DISTINCT FROM b
        return _is_distinct_from(tr, node)
    if isinstance(node, exp.NullSafeEQ):  # a IS NOT DISTINCT FROM b
        return ~_is_distinct_from(tr, node)
    if isinstance(node, exp.Extract):
        part = node.this.name.lower()
        method = _EXTRACT_PART.get(part)
        if method is None:
            raise NotImplementedError(f"EXTRACT field {part!r} is not supported")
        return getattr(tr._scalar(node.expression).dt, method)()
    if isinstance(node, exp.RegexpReplace):
        return _regexp_replace(tr, node)

    # Date ± INTERVAL, date_add/date_sub, date_diff (DATE operands).
    if isinstance(node, (exp.Add, exp.Sub)) and isinstance(node.expression, exp.Interval):
        return _apply_interval(
            tr._scalar(node.this), node.expression, subtract=isinstance(node, exp.Sub)
        )
    if isinstance(node, (exp.DateAdd, exp.DateSub)):
        return _apply_interval(
            tr._scalar(node.this), node.expression, subtract=isinstance(node, exp.DateSub)
        )
    if isinstance(node, exp.DateDiff):
        return _date_diff(tr, node)

    # Fold `literal <op> literal` arithmetic with exact decimal semantics before the
    # generic binop path (so `0.06 + 0.01` is `0.07`, not IEEE `0.0699…`).
    folded = _fold_const_arith(node)
    if folded is not None:
        return folded

    binop = _BINOPS.get(type(node))
    if binop is not None:
        return binop(tr._scalar(node.this), tr._scalar(node.expression))

    fn = _scalar_function(tr, node)
    if fn is not None:
        return fn

    # An unknown function call (parsed as Anonymous) is the most common cause —
    # name it and point at registration rather than a generic node-type error.
    if isinstance(node, exp.Anonymous):
        raise NotImplementedError(
            f"unknown function {node.name!r}: it is not a supported SQL function and "
            f"is not registered (use bt.register_function to call a Python function)"
        )
    raise NotImplementedError(f"unsupported SQL expression: {type(node).__name__}")


def _scalar_function(tr, node):
    """Map a SQL scalar function call to its `Expr` builder, or None."""
    from sqlglot import expressions as exp

    name = type(node).__name__
    if name in _UNARY_MATH:
        return getattr(tr._scalar(node.this), _UNARY_MATH[name])()
    if name in _UNARY_STR:
        return getattr(tr._scalar(node.this).str, _UNARY_STR[name])()
    if name in _DATE_PART:
        return getattr(tr._scalar(node.this).dt, _DATE_PART[name])()
    if name == "Round":
        return tr._scalar(node.this).round()
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
        return tr._scalar(node.this).str.trim()
    if name == "Substring":
        base = tr._scalar(node.this).str
        start = int(node.args["start"].this)
        length = node.args.get("length")
        return base.substr(start, int(length.this)) if length is not None else base.substr(start)
    if name == "StrPosition":
        pat = node.args["substr"]
        if not isinstance(pat, exp.Literal) or not pat.is_string:
            raise NotImplementedError("position() requires a string literal pattern")
        return tr._scalar(node.this).str.position(pat.this)
    if name == "Pad":
        width = int(node.args["expression"].this)
        fill_node = node.args.get("fill_pattern")
        fill = fill_node.this if fill_node is not None else " "
        base = tr._scalar(node.this).str
        is_left = bool(node.args.get("is_left"))
        return base.lpad(width, fill) if is_left else base.rpad(width, fill)
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
    """`regexp_replace(s, pattern, replacement)` — replace the first match (DuckDB
    default; constant pattern/replacement)."""
    from sqlglot import expressions as exp

    pat = node.expression
    repl = node.args.get("replacement")
    if not (isinstance(pat, exp.Literal) and pat.is_string):
        raise NotImplementedError("regexp_replace requires a constant string pattern")
    if not (isinstance(repl, exp.Literal) and repl.is_string):
        raise NotImplementedError("regexp_replace requires a constant string replacement")
    return tr._scalar(node.this).str.regexp_replace(pat.this, repl.this)


def _date_diff(tr, node) -> Expr:
    """`date_diff(unit, a, b)` = (b - a) in `unit` (DAY/WEEK), for DATE inputs."""
    unit = (node.text("unit") or "DAY").upper()
    # sqlglot: this=end (b), expression=start (a).
    days = Cast(tr._scalar(node.this), "int64") - Cast(tr._scalar(node.expression), "int64")
    if unit.startswith("DAY"):
        return days
    if unit.startswith("WEEK"):
        return days / lit(7)
    raise NotImplementedError(f"date_diff unit {unit} not supported (only DAY/WEEK)")


def _case(tr, node) -> Expr:
    # Simple CASE `CASE x WHEN v THEN …` compares the operand to each WHEN
    # value; searched CASE `CASE WHEN cond THEN …` has no operand.
    operand = node.this
    subject = tr._scalar(operand) if operand is not None else None
    builder = None
    first_then = None
    for if_ in node.args.get("ifs", []):
        when_val = tr._scalar(if_.this)
        cond = (subject == when_val) if subject is not None else when_val
        then = tr._scalar(if_.args["true"])
        if first_then is None:
            first_then = then
        builder = (when(cond) if builder is None else builder.when(cond)).then(then)
    if builder is None:
        raise NotImplementedError("CASE without WHEN is unsupported")
    default = node.args.get("default")
    if default is not None:
        return builder.otherwise(tr._scalar(default))
    # No ELSE → SQL yields NULL (typed as the THEN value) where nothing
    # matches. `nullif(x, x)` is exactly that typed NULL.
    return builder.otherwise(nullif(first_then, first_then))


def _scalar_subquery(tr, select_node) -> Expr:
    """Uncorrelated scalar subquery → a literal.

    Translate the inner SELECT, collect it **eagerly** (this executes the
    subquery now, not lazily), assert it is exactly 1 row x 1 column, and
    substitute the scalar value as a literal in the enclosing expression.
    """
    tr._reject_correlated(select_node)
    # Detach from the outer AST so ancestor walks (e.g. _has_aggregate's
    # Subquery/Window checks) stay within the subquery's own scope.
    select_node = select_node.copy()
    # The subquery may itself aggregate, which resets the translator's aggregate
    # bookkeeping (``_agg_map`` / ``_agg_n``). Save and restore it so the enclosing
    # query's aggregate columns still resolve after the subquery is evaluated — e.g.
    # ``HAVING sum(x) > (SELECT sum(x) * k FROM ...)`` (TPC-H Q11).
    saved_agg_map, saved_agg_n = tr._agg_map, tr._agg_n
    try:
        inner_ds = tr.statement(select_node)
        if len(inner_ds.columns) != 1:
            raise NotImplementedError("scalar subquery must project exactly one column")
        table = inner_ds.collect()
    finally:
        tr._agg_map, tr._agg_n = saved_agg_map, saved_agg_n
    if table.num_rows != 1:
        raise NotImplementedError(
            f"scalar subquery must return exactly one row, got {table.num_rows}"
        )
    value = table.column(0)[0].as_py()
    return lit(value)


def _in(tr, node) -> Expr:
    items = node.expressions
    if node.args.get("query") is not None:
        raise NotImplementedError(
            "IN (subquery) must be handled at the WHERE level, not as a scalar"
        )
    if not items:
        raise NotImplementedError("IN requires an explicit value list")
    target = tr._scalar(node.this)
    # x IN (a, b, c)  →  (x == a) | (x == b) | (x == c)
    result: Expr | None = None
    for item in items:
        eq = target == tr._scalar(item)
        result = eq if result is None else (result | eq)
    return result


def _between(tr, node) -> Expr:
    # x BETWEEN lo AND hi  →  (x >= lo) & (x <= hi)
    target = tr._scalar(node.this)
    low = tr._scalar(node.args["low"])
    high = tr._scalar(node.args["high"])
    return (target >= low) & (target <= high)


def _is_distinct_from(tr, node) -> Expr:
    """`a IS DISTINCT FROM b` — null-safe inequality (NULL is a comparable
    value). Built as the negation of null-safe *equality* (both null, or both
    non-null and equal); that form is null-free (the `a == b` term is masked by
    `~an & ~bn`, so it never leaks a NULL into the boolean result).
    """
    a = tr._scalar(node.this)
    b = tr._scalar(node.expression)
    # `a == b` is NULL when either side is NULL; `coalesce` then falls back to
    # "are both NULL?" — giving a null-free null-safe-equality without relying
    # on Kleene `and`/`or` (which the engine does not implement).
    not_distinct = coalesce(a == b, a.is_null() & b.is_null())
    return ~not_distinct


def _scalar_args(tr, node) -> list[Expr]:
    """All argument sub-expressions of a variadic node (`this` + `expressions`)."""
    args = [node.this, *node.expressions] if node.this is not None else list(node.expressions)
    return [tr._scalar(a) for a in args if a is not None]


def _concat(tr, node) -> Expr:
    """`concat(a, b, …)` / `concat_ws(sep, a, b, …)` → chained `||`.

    Unlike the `||` operator, the SQL concat functions ignore NULL arguments
    (DuckDB semantics), so each argument is coalesced to ''.
    """
    from sqlglot import expressions as exp

    empty = lit("")
    parts = [coalesce(p, empty) for p in _scalar_args(tr, node)]
    if isinstance(node, exp.ConcatWs):
        sep, parts = parts[0], parts[1:]
        out = None
        for p in parts:
            out = p if out is None else Binary("concat", Binary("concat", out, sep), p)
        return out if out is not None else empty
    out = parts[0]
    for p in parts[1:]:
        out = Binary("concat", out, p)
    return out


def _like(tr, node, case_insensitive: bool = False, escape: str | None = None) -> Expr:
    from sqlglot import expressions as exp

    pattern_node = node.expression
    if not isinstance(pattern_node, exp.Literal) or not pattern_node.is_string:
        raise NotImplementedError("LIKE supports only constant string patterns")
    pattern = pattern_node.this
    target = tr._scalar(node.this)
    # ILIKE: fold both sides to lower case for a case-insensitive match.
    if case_insensitive:
        target = target.str.lower()
        pattern = pattern.lower()

    # Simple patterns (no escape, no `_`, `%` only at the ends) use the fast
    # starts_with/ends_with/contains kernels; anything richer compiles to a
    # regex (handles `_`, interior `%`, and ESCAPE).
    simple = escape is None and "_" not in pattern and "%" not in pattern.strip("%")
    if simple:
        result = _like_simple(target, pattern)
    else:
        result = target.str.regexp_matches(_like_to_regex(pattern, escape))

    # `x NOT LIKE p` parses as a Like node with negate=True.
    if node.args.get("negate"):
        result = ~result
    return result


def _like_simple(target: Expr, pattern: str) -> Expr:
    starts = pattern.startswith("%")
    ends = pattern.endswith("%")
    inner = pattern[1:] if starts else pattern
    inner = inner[:-1] if ends else inner
    if starts and ends:
        return target.str.contains(inner)
    if ends:  # 'abc%'
        return target.str.starts_with(inner)
    if starts:  # '%abc'
        return target.str.ends_with(inner)
    return target == lit(inner)  # no wildcards → exact match


def _coalesce(tr, node) -> Expr:
    # COALESCE(a, b, ..., z)  →
    #   when(a.is_not_null()).then(a).when(b.is_not_null()).then(b)...otherwise(z)
    args = [node.this, *node.expressions]
    exprs = [tr._scalar(a) for a in args if a is not None]
    if not exprs:
        raise NotImplementedError("COALESCE requires at least one argument")
    if len(exprs) == 1:
        return exprs[0]
    builder = None
    for e in exprs[:-1]:
        cond = e.is_not_null()
        builder = (when(cond) if builder is None else builder.when(cond)).then(e)
    return builder.otherwise(exprs[-1])
