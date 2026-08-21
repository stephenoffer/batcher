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
from batcher._sql.parser.expressions.collections import list_function
from batcher._sql.parser.expressions.functions import (
    _regexp_replace,
    _scalar_function,
)
from batcher._sql.parser.expressions.json import json_extract
from batcher._sql.parser.expressions.literals import (
    _BINOPS,
    _EXTRACT_COMPOSITE,
    _EXTRACT_PART,
    _TEMPORAL_KINDS,
    _apply_interval,
    _dtype_name,
    _fold_const_arith,
    _literal,
    _regexp_flags_prefix,
    _sql_int_div,
    _temporal_literal,
)
from batcher._sql.parser.expressions.lowering import (
    between,
    binop_with_null,
    const_str,
    in_membership,
    is_distinct_from,
    like,
    null_boolean,
    positional_null,
    str_call,
    typed_null,
)
from batcher._sql.parser.expressions.temporal import _date_diff
from batcher.plan.expr_ir import (
    Array,
    Binary,
    Cast,
    Expr,
    ListJoin,
    Lit,
    StrFuncDyn,
    StructField,
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
                if sep is not None and not isinstance(sep, exp.Literal):
                    # Falling back to ',' answered with the *default* separator where the
                    # query asked for a computed one — a wrong string, not a refusal.
                    raise NotImplementedError(
                        "string_agg() needs a constant separator; got " + sep.sql()
                    )
                sep = sep.name if sep is not None else ","
                return ListJoin(col(entry[0]), sep)
            return col(entry[0])
    if isinstance(node, exp.Paren):
        return tr._scalar(node.this)
    if isinstance(node, exp.Columns):
        return _columns_selector(node)
    if isinstance(node, exp.Column):
        # `st.a` where `st` is a *struct column* in scope, not a table alias. sqlglot
        # cannot tell the two apart — both parse as a qualified column — and the
        # qualified reading looked for a column `a`, so the dotted spelling of a struct
        # field failed with "unknown column" while `struct_extract(st, 'a')` worked.
        field = _struct_field_reference(tr, node)
        if field is not None:
            return field
        return col(node.name)
    if isinstance(node, exp.Dot):
        # `(expr).field` — the parenthesized spelling, which nests for a struct of structs.
        return _dot_field(tr, node)
    if isinstance(node, exp.Literal):
        return _literal(node)
    if isinstance(node, exp.ByteString):
        # An `E'…'` escape string: sqlglot has already decoded the C-escapes
        # (`\n`, `\t`, …) into `node.this`; it is a text value, not binary.
        return lit(node.this)
    if isinstance(node, exp.Boolean):
        return lit(bool(node.this))
    # `exp.Neg` is the unary minus (`-x`); `exp.Negative` is the *function* spelling
    # (`negative(x)`, which Spark and DuckDB both have). sqlglot parses them to two
    # different nodes and only the operator one was handled, so `negative(x)` reached the
    # fallthrough as "unsupported SQL expression: Negative". They mean the same thing.
    if isinstance(node, (exp.Neg, exp.Negative)):
        return Lit(0) - tr._scalar(node.this)
    if isinstance(node, exp.Not):
        # `NOT NULL` is a boolean NULL, not the negation of the Int64 typed null.
        if isinstance(node.this, exp.Null):
            return null_boolean()
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
        # rows). Used for `SELECT NULL`, `coalesce(x, NULL)`, etc. The IR has no
        # untyped null, so the type is read off the position the literal sits in:
        # `upper(NULL)` needs a *string*-typed one, and the Int64 default reached the
        # engine as "string function Upper expected a Utf8 argument, got Int64".
        return positional_null(node)
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
        return in_membership(tr, node)
    if isinstance(node, exp.Between):
        return between(tr, node)
    if isinstance(node, exp.Escape):
        # `x [I]LIKE p ESCAPE e` → the inner Like/ILike with the escape char.
        inner = node.this
        return like(
            tr,
            inner,
            case_insensitive=isinstance(inner, exp.ILike),
            escape=node.expression.this,
        )
    if isinstance(node, exp.ILike):
        return like(tr, node, case_insensitive=True)
    if isinstance(node, exp.Like):
        return like(tr, node)
    if isinstance(node, exp.Coalesce):
        return _coalesce(tr, node)
    if isinstance(node, exp.Nullif):
        # `nullif(x, NULL)` is `x`: the comparison is never true, and building it would
        # pair the value's type against the untyped null's Int64.
        if isinstance(node.expression, exp.Null):
            return tr._scalar(node.this)
        if isinstance(node.this, exp.Null):
            return positional_null(node.this)
        return nullif(tr._scalar(node.this), tr._scalar(node.expression))
    # sqlglot parses `nanvl` into a *typed* node, so the Anonymous fallback table that
    # also lists it (`anonymous.py`) is never consulted for this spelling.
    if isinstance(node, exp.Nanvl):
        return nanvl(tr._scalar(node.this), tr._scalar(node.expression))
    if isinstance(node, (exp.Greatest, exp.Least)):
        # A bare `NULL` argument is dropped for the same reason `COALESCE` drops one: it
        # carries no type, and DuckDB's `greatest`/`least` ignore nulls anyway.
        args = [a for a in _arg_nodes(node) if not isinstance(a, exp.Null)]
        if not args:
            return nullif(lit(1), lit(1))
        built = [tr._scalar(a) for a in args]
        return greatest(*built) if isinstance(node, exp.Greatest) else least(*built)
    if isinstance(node, exp.Array):
        return Array([tr._scalar(e) for e in node.expressions])
    list_fn = list_function(tr, node)
    if list_fn is not None:
        return list_fn
    if isinstance(node, (exp.Concat, exp.ConcatWs)):
        return _concat(tr, node)
    if isinstance(node, exp.NullSafeNEQ):  # a IS DISTINCT FROM b
        return is_distinct_from(tr, node)
    if isinstance(node, exp.NullSafeEQ):  # a IS NOT DISTINCT FROM b
        return ~is_distinct_from(tr, node)
    if isinstance(node, exp.Extract):
        part = node.this.name.lower()
        composite = _EXTRACT_COMPOSITE.get(part)
        if composite is not None:
            return composite(tr._scalar(node.expression).dt)
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
    if isinstance(node, exp.SimilarTo):
        # `x SIMILAR TO p` matches the *whole* string against `p` as a regex — DuckDB's
        # reading, and the one that matters here: `'abc' SIMILAR TO 'a%'` is False, so
        # this is not LIKE with regex syntax bolted on. `regexp_matches` is unanchored,
        # hence the wrapping. The non-capturing group keeps an alternation in `p` from
        # binding past the anchors (`a|b` would otherwise mean `^a` or `b$`).
        pattern = const_str(node.expression)
        if pattern is not None:
            return tr._scalar(node.this).str.regexp_matches(f"^(?:{pattern})$")
        anchored = Binary(
            "concat", Binary("concat", lit("^(?:"), tr._scalar(node.expression)), lit(")$")
        )
        return str_call(tr, "regexp_matches", node.this, pattern=anchored)
    if isinstance(node, exp.RegexpLike):  # regexp_matches(s, pattern[, options])
        flag_node = node.args.get("flag")
        is_str_lit = isinstance(flag_node, exp.Literal) and flag_node.is_string
        if flag_node is not None and not is_str_lit:
            raise NotImplementedError("regexp_matches options must be a constant string")
        prefix = _regexp_flags_prefix(flag_node.this if is_str_lit else None)
        pat = const_str(node.expression)
        # A per-row pattern carries the flag prefix as an expression, since the constant
        # slot cannot hold it.
        pattern = (
            prefix + pat
            if pat is not None
            else Binary("concat", lit(prefix), tr._scalar(node.expression))
        )
        return str_call(tr, "regexp_matches", node.this, pattern=pattern)
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

    # A bare `NULL` operand has no type of its own — it takes the one its context needs.
    # The IR has no untyped null (a bare NULL lowers to the Int64 `nullif(1, 1)`), so
    # `NULL OR x` reached the engine as `or(Int64, Bool)` and `CASE s WHEN NULL` as
    # `Utf8 == Int64`, both of which failed the query outright where SQL simply answers
    # NULL.
    if type(node) in _BINOPS and (
        isinstance(node.this, exp.Null) or isinstance(node.expression, exp.Null)
    ):
        return binop_with_null(tr, node)

    # Fold `literal <op> literal` arithmetic with exact decimal semantics before the
    # generic binop path (so `0.06 + 0.01` is `0.07`, not IEEE `0.0699…`).
    folded = _fold_const_arith(node)
    if folded is not None:
        return folded

    if isinstance(node, exp.IntDiv):
        # `//` means truncating *integer* division on integers and plain division on
        # floats; the operand types decide, so it cannot ride the operator table.
        return _sql_int_div(tr, tr._scalar(node.this), tr._scalar(node.expression))

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


def _struct_field_reference(tr, node) -> Expr | None:
    """`st.a` (or `db.st.a.b`) read as a struct field chain, or None if it is not one.

    A qualified column and a dotted struct field parse identically, so which one a name
    denotes is decided by the schema: the *first* part that names an in-scope struct column
    starts the chain, and the parts after it are field names. A leading table qualifier
    (`l.st.a`) is skipped for free, because `l` is not a column.

    Args:
        tr: The translator, for the in-scope column types.
        node: The qualified `Column` node.

    Returns:
        The field expression, or None to keep the ordinary qualified-column reading.
    """
    import pyarrow as pa

    parts = [p.name for p in (node.args.get(k) for k in ("catalog", "db", "table")) if p]
    if not parts:
        return None
    parts.append(node.name)
    for start, name in enumerate(parts[:-1]):
        dtype = tr._scope_types.get(name)
        if dtype is None or not pa.types.is_struct(dtype):
            continue
        built, current = col(name), dtype
        for field in parts[start + 1 :]:
            if not pa.types.is_struct(current) or current.get_field_index(field) < 0:
                return None
            built = StructField(built, field)
            current = current.field(current.get_field_index(field)).type
        return built
    return None


def _dot_field(tr, node) -> Expr:
    """`(expr).field` — a struct field of a parenthesized expression.

    Args:
        tr: The translator.
        node: The `Dot` node.

    Returns:
        The field expression.

    Raises:
        NotImplementedError: The right-hand side is not a plain field name.
    """
    name = node.expression
    if not isinstance(name, (exp.Identifier, exp.Column)):
        raise NotImplementedError(f"`.{name.sql()}` is not a struct field reference")
    return StructField(tr._scalar(node.this), name.name)


def _case(tr, node) -> Expr:
    # Simple CASE `CASE x WHEN v THEN …` compares the operand to each WHEN
    # value; searched CASE `CASE WHEN cond THEN …` has no operand.
    operand = node.this
    subject = tr._scalar(operand) if operand is not None else None
    builder = None
    first_then = None
    for if_ in node.args.get("ifs", []):
        if subject is not None and isinstance(if_.this, exp.Null):
            # `CASE x WHEN NULL THEN …` compares with `=`, which is NULL for every row,
            # so the branch can never be taken. Building the comparison instead reached
            # the engine as `Utf8 == Int64` (the untyped NULL lowers to Int64) and failed.
            continue
        when_val = tr._scalar(if_.this)
        cond = (subject == when_val) if subject is not None else when_val
        then = tr._scalar(if_.args["true"])
        if first_then is None:
            first_then = then
        builder = (when(cond) if builder is None else builder.when(cond)).then(then)
    default = node.args.get("default")
    if builder is None:
        # Every WHEN was a bare NULL (skipped above), so nothing can ever match: the
        # CASE is its ELSE, or a NULL typed like the first THEN.
        if not node.args.get("ifs"):
            raise NotImplementedError("CASE without WHEN is unsupported")
        if default is not None:
            return tr._scalar(default)
        first = tr._scalar(node.args["ifs"][0].args["true"])
        return nullif(first, first)
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
        return typed_null(table.schema.field(0).type)
    if table.num_rows > 1:
        raise NotImplementedError(
            f"scalar subquery must return at most one row, got {table.num_rows}"
        )
    value = table.column(0)[0].as_py()
    if value is None:
        # One row whose value *is* NULL — a different case from the no-rows one above, and
        # the one an ordinary threshold query hits: `WHERE x > (SELECT AVG(x) FROM t)` over
        # an empty or all-null column returns a single NULL row, not zero rows. `lit(None)`
        # has no wire form (the IR has no untyped null literal), so this raised a bare
        # `TypeError: unsupported literal type: NoneType` from deep inside `to_ir` where
        # DuckDB simply returns no rows.
        return typed_null(table.schema.field(0).type)
    return lit(value)


def _arg_nodes(node) -> list:
    """All argument nodes of a variadic call (`this` + `expressions`), skipping absent ones."""
    args = [node.this, *node.expressions] if node.this is not None else list(node.expressions)
    return [a for a in args if a is not None]


def _scalar_args(tr, node) -> list[Expr]:
    """All argument sub-expressions of a variadic node (`this` + `expressions`)."""
    return [tr._scalar(a) for a in _arg_nodes(node)]


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
        if isinstance(sep_node, exp.Null):
            # DuckDB: a NULL separator makes the whole result NULL.
            return nullif(empty, empty)
        sep = tr._scalar(sep_node)
        # Emit `sep || arg` for each non-null arg and strip the single leading separator —
        # the exact DuckDB null-skip semantics (a dropped argument takes its separator with
        # it, so `concat_ws(',', NULL, 'x')` is `'x'`, not `',x'`).
        raw: Expr | None = None
        for vn in val_nodes:
            v = tr._scalar(vn)
            piece = when(v.is_not_null()).then(Binary("concat", sep, v)).otherwise(empty)
            raw = piece if raw is None else Binary("concat", raw, piece)
        if raw is None:
            return empty
        constant = const_str(sep_node)
        if constant is not None:
            stripped = raw.str.substr(len(constant) + 1)
        else:
            # A computed separator's length is a per-row value, so the strip is too. The
            # old fallback coalesced a dropped argument to `''` and kept its separator,
            # answering `'a,,b'` where DuckDB answers `'a,b'` — a wrong string, not a
            # refusal, and only on the spelling that takes a column.
            stripped = StrFuncDyn("substr", raw, start=sep.str.len() + lit(1))
            joined = when(raw.str.len() > lit(0)).then(stripped).otherwise(empty)
            # A NULL separator makes the whole result NULL, including when every value was
            # dropped — which the length test above cannot see, since `raw` is `''` there.
            return when(sep.is_null()).then(nullif(empty, empty)).otherwise(joined)
        return when(raw.str.len() > lit(0)).then(stripped).otherwise(empty)

    # DuckDB's `concat` casts every argument to text (so `concat(id, name)` works on
    # a numeric column) and skips NULLs (treats them as ''). Coalescing to `''`
    # up front raised "arguments need the same data type" on a non-string column, so
    # concatenate through the kernel (which casts) and drop nulls with a guard.
    out: Expr = empty
    for a in arg_nodes:
        v = tr._scalar(a)
        out = when(v.is_not_null()).then(Binary("concat", out, v)).otherwise(out)
    return out


def _coalesce(tr, node) -> Expr:
    # COALESCE(a, b, ..., z)  →
    #   when(a.is_not_null()).then(a).when(b.is_not_null()).then(b)...otherwise(z)
    args = [a for a in (node.this, *node.expressions) if a is not None]
    # A bare `NULL` argument can never be the chosen value, and it has no type of its own —
    # including it built a `CASE` mixing the Int64 typed null with the real arguments, so
    # `coalesce(NULL, s)` on a text column died on "arguments need to have the same data
    # type" where SQL simply answers `s`.
    typed = [a for a in args if not isinstance(a, exp.Null)]
    if not typed:
        if not args:
            raise NotImplementedError("COALESCE requires at least one argument")
        return nullif(lit(1), lit(1))
    exprs = [tr._scalar(a) for a in typed]
    if not exprs:
        raise NotImplementedError("COALESCE requires at least one argument")
    if len(exprs) == 1:
        return exprs[0]
    builder = None
    for e in exprs[:-1]:
        cond = e.is_not_null()
        builder = (when(cond) if builder is None else builder.when(cond)).then(e)
    return builder.otherwise(exprs[-1])
