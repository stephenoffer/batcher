"""Scalar expression dispatch — translate a sqlglot value node into an `Expr`.

The functions here take the translator instance (`tr`) as their first argument so
they can recurse through `tr._scalar`, resolve aggregate output columns, and run
nested subqueries via `tr.statement`. They hold no state of their own.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.core_utils import _columns_selector
from batcher._sql.parser.expressions.aggregates import is_agg_node
from batcher._sql.parser.expressions.anonymous import anonymous_scalar
from batcher._sql.parser.expressions.functions import (
    _date_diff,
    _list_function,
    _regexp_replace,
    _scalar_function,
)
from batcher._sql.parser.expressions.json import json_extract
from batcher._sql.parser.expressions.literals import (
    _BINOPS,
    _EXTRACT_PART,
    _TEMPORAL_KINDS,
    _apply_interval,
    _const_str_arg,
    _dtype_name,
    _fold_const_arith,
    _like_to_regex,
    _literal,
    _regexp_flags_prefix,
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
from batcher.plan.functions.scalar import nanvl


def _scalar(tr, node) -> Expr:
    # Inside an aggregate query, an aggregate sub-expression refers to its
    # pre-computed output column.
    # `is_agg_node` rather than `isinstance(node, exp.AggFunc)`: the DuckDB aggregates
    # sqlglot leaves anonymous (`product`, `sem`, `count_star`, …) are registered by the
    # grouping too, so they must resolve to their output column here as well.
    if tr._agg_map is not None and is_agg_node(node):
        entry = tr._agg_map.get(node.sql())
        if entry is not None:
            # string_agg collects into a list (array_agg); join it here with the
            # separator (DuckDB default ',').
            if isinstance(node, exp.GroupConcat):
                sep = node.args.get("separator")
                if isinstance(sep, exp.Null):
                    # DuckDB: an explicit NULL separator makes the whole aggregate NULL
                    # (concatenating through a NULL delimiter is NULL). A plain `exp.Literal`
                    # check misses this — a SQL NULL parses to `exp.Null`, not `exp.Literal`,
                    # so it used to fall through to the default ',' and wrongly join the values.
                    joined = ListJoin(col(entry[0]), ",")
                    return nullif(joined, joined)  # a string-typed NULL for every group
                sep = sep.name if isinstance(sep, exp.Literal) else ","
                return ListJoin(col(entry[0]), sep)
            return col(entry[0])
    if isinstance(node, exp.Paren):
        return tr._scalar(node.this)
    if isinstance(node, exp.Columns):
        return _columns_selector(node)
    if isinstance(node, exp.Column):
        return col(node.name)
    if isinstance(node, exp.Literal):
        return _literal(node)
    if isinstance(node, exp.ByteString):
        # An `E'…'` escape string: sqlglot has already decoded the C-escapes
        # (`\n`, `\t`, …) into `node.this`; it is a text value, not binary.
        return lit(node.this)
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
        # TRY_CAST (sqlglot `exp.TryCast`, a subclass of Cast) returns NULL on an
        # unconvertible value instead of erroring — carry that through, or a plain
        # Cast is built and the query errors on the first bad row.
        try_cast = isinstance(node, exp.TryCast)
        if isinstance(inner, exp.Literal) and inner.is_string and kind in _TEMPORAL_KINDS:
            return _temporal_literal(inner.this, kind)
        return Cast(tr._scalar(inner), _dtype_name(node.to), try_cast=try_cast)
    if isinstance(node, exp.Case):
        return _case(tr, node)
    if isinstance(node, exp.Null):
        # A bare NULL literal — a typed NULL (`nullif(c, c)` is null for all
        # rows). Used for `SELECT NULL`, `coalesce(x, NULL)`, etc.
        return nullif(lit(1), lit(1))
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        # x IS NULL  (x IS NOT NULL parses as Not(Is(...)), handled above)
        return tr._scalar(node.this).is_null()
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Boolean):
        # x IS TRUE / x IS FALSE — a total (never-NULL) test: a NULL operand is
        # neither true nor false. `IS NOT TRUE` / `IS NOT FALSE` parse as
        # Not(Is(...)) and are handled by the `exp.Not` branch above.
        inner = tr._scalar(node.this)
        want = inner if bool(node.expression.this) else ~inner
        return coalesce(want, lit(False))
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
    # sqlglot parses `nanvl` into a *typed* node, so the Anonymous fallback table that
    # also lists it (`anonymous.py`) is never consulted for this spelling.
    if isinstance(node, exp.Nanvl):
        return nanvl(tr._scalar(node.this), tr._scalar(node.expression))
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
    if isinstance(node, (exp.DateTrunc, exp.TimestampTrunc)):
        # DATE_TRUNC('unit', ts) → floor the timestamp to `unit`. sqlglot puts the timestamp
        # in `this` and the unit literal in `args['unit']` (e.g. Literal 'MINUTE').
        unit = node.args.get("unit")
        if unit is None:
            raise NotImplementedError("date_trunc requires an explicit unit")
        return tr._scalar(node.this).dt.truncate(unit.name.lower())
    if isinstance(node, exp.RegexpReplace):
        return _regexp_replace(tr, node)
    if isinstance(node, exp.RegexpLike):  # regexp_matches(s, pattern[, options])
        pat = _const_str_arg(node.expression, "regexp_matches", "pattern")
        flag_node = node.args.get("flag")
        is_str_lit = isinstance(flag_node, exp.Literal) and flag_node.is_string
        if flag_node is not None and not is_str_lit:
            raise NotImplementedError("regexp_matches options must be a constant string")
        prefix = _regexp_flags_prefix(flag_node.this if is_str_lit else None)
        return tr._scalar(node.this).str.regexp_matches(prefix + pat)
    if isinstance(node, (exp.JSONExtract, exp.JSONExtractScalar)):
        return json_extract(tr, node)

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
        named = anonymous_scalar(tr, node)
        if named is not None:
            return named
        raise NotImplementedError(
            f"unknown function {node.name!r}: it is not a supported SQL function and "
            f"is not registered (use bt.register_function to call a Python function)"
        )
    raise NotImplementedError(f"unsupported SQL expression: {type(node).__name__}")


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
    if table.num_rows == 0:
        # SQL: a scalar subquery with no rows is NULL (typed as its output column),
        # not an error — e.g. `(SELECT sal FROM emp WHERE id=999)` is NULL per row.
        return _typed_null(table.schema.field(0).type)
    if table.num_rows > 1:
        raise NotImplementedError(
            f"scalar subquery must return at most one row, got {table.num_rows}"
        )
    value = table.column(0)[0].as_py()
    return lit(value)


def _typed_null(arrow_type) -> Expr:
    """A NULL literal typed to match `arrow_type` (the subquery's output column).

    Built as `NULLIF(1, 1)` (a typed NULL of int) cast to the target type, so the
    output schema matches DuckDB's — a scalar subquery yields a column of its own
    type even when it produces no row.
    """
    import pyarrow as pa

    typed = nullif(lit(1), lit(1))
    if pa.types.is_floating(arrow_type):
        return Cast(typed, "float64")
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return Cast(typed, "string")
    if pa.types.is_boolean(arrow_type):
        return Cast(typed, "bool")
    if pa.types.is_date(arrow_type):
        return Cast(typed, "date")
    if pa.types.is_timestamp(arrow_type):
        return Cast(typed, "timestamp")
    return typed  # integer (and any other) → the int-typed NULL


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
    (DuckDB semantics). `concat` drops each NULL (coalesce to ''); `concat_ws`
    additionally emits **no separator** for a dropped argument — so
    `concat_ws(',', NULL, 'x', NULL)` is `'x'`, not `',x,'`.
    """
    empty = lit("")
    arg_nodes = [node.this, *node.expressions] if node.this is not None else list(node.expressions)
    arg_nodes = [a for a in arg_nodes if a is not None]

    if isinstance(node, exp.ConcatWs):
        sep_node, val_nodes = arg_nodes[0], arg_nodes[1:]
        sep = tr._scalar(sep_node)
        # A constant separator lets us emit `sep || arg` only for non-null args and
        # strip the single leading separator — the exact DuckDB null-skip semantics.
        if isinstance(sep_node, exp.Literal) and sep_node.is_string:
            sep_len = len(sep_node.this)
            raw: Expr | None = None
            for vn in val_nodes:
                v = tr._scalar(vn)
                piece = when(v.is_not_null()).then(Binary("concat", sep, v)).otherwise(empty)
                raw = piece if raw is None else Binary("concat", raw, piece)
            if raw is None:
                return empty
            stripped = raw.str.substr(sep_len + 1)  # drop the one leading separator
            return when(raw.str.len() > lit(0)).then(stripped).otherwise(empty)
        # Non-constant separator: best-effort (coalesce dropped args to '').
        parts = [coalesce(tr._scalar(vn), empty) for vn in val_nodes]
        out = None
        for p in parts:
            out = p if out is None else Binary("concat", Binary("concat", out, sep), p)
        return out if out is not None else empty

    # DuckDB's `concat` casts every argument to text (so `concat(id, name)` works on
    # a numeric column) and skips NULLs (treats them as ''). Coalescing to `''`
    # up front raised "arguments need the same data type" on a non-string column, so
    # concatenate through the kernel (which casts) and drop nulls with a guard.
    out: Expr = empty
    for a in arg_nodes:
        v = tr._scalar(a)
        out = when(v.is_not_null()).then(Binary("concat", out, v)).otherwise(out)
    return out


def _like(tr, node, case_insensitive: bool = False, escape: str | None = None) -> Expr:
    pattern_node = node.expression
    if not isinstance(pattern_node, exp.Literal) or not pattern_node.is_string:
        raise NotImplementedError("LIKE supports only constant string patterns")
    pattern = pattern_node.this
    target = tr._scalar(node.this)
    # ILIKE: fold both sides to lower case for a case-insensitive match.
    if case_insensitive:
        target = target.str.lower()
        pattern = pattern.lower()

    # Boundary-only `%` with no `_`/ESCAPE lowers to the anchored
    # starts_with/ends_with/contains kernels — leanest, and the shape Kyber's
    # `like_prefix_to_range` can further turn into a zone-map-prunable range.
    #
    # Anything richer goes to the native `like`, whose Rust matcher classifies the
    # pattern *once per morsel* into the cheapest shape (prefix/suffix/ordered
    # `memmem` segment scan) and falls back to a cached anchored regex only for `_`.
    # It must not be spelled as `regexp_matches(_like_to_regex(...))`: that runs a
    # regex automaton per row for `%a%b%`, which measured ~7x DuckDB on TPC-H q13's
    # `o_comment NOT LIKE '%special%requests%'` (76ms vs 11ms) — and, lacking `(?s)`,
    # also made `%` stop at a newline, which SQL says it must not.
    #
    # ESCAPE keeps the Python-desugared regex: the native matcher has no escape char.
    simple = escape is None and "_" not in pattern and "%" not in pattern.strip("%")
    if simple:
        result = _like_simple(target, pattern)
    elif escape is None:
        result = target.str.like(pattern)
    else:
        result = target.str.regexp_matches(_like_to_regex(pattern, escape))

    # `x NOT LIKE p` parses as a Like node with negate=True.
    if node.args.get("negate"):
        result = ~result
    return result


def _like_simple(target: Expr, pattern: str) -> Expr:
    starts = pattern.startswith("%")
    ends = pattern.endswith("%")
    # Strip *all* boundary `%`, not just one: a pattern like `%%c` / `a%%` carries
    # consecutive leading/trailing wildcards, and the caller's `simple` guard already
    # proved the stripped core holds no `%`/`_`, so it is a pure literal. Peeling only a
    # single `%` left an interior `%` in `inner` that `starts_with`/`ends_with`/`contains`
    # then matched literally (`'abc' LIKE '%%c'` → false instead of true).
    inner = pattern.strip("%")
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
