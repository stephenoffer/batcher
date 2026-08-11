"""SQL list/array functions — the Spark-shaped half, including the lambda forms.

`anonymous._LIST_PAIR` covers the DuckDB `list_*`/`array_*` names that map onto a binary
`.list` method. This module covers the rest: the typed nodes sqlglot promotes
(`ArrayAppend`, `ArrayCompact`, `Transform`, `ArrayFilter`, ...) and, with them, the
**higher-order** forms — `transform(xs, x -> x + 1)`, `filter`, `exists`, `forall` — whose
argument is a lambda rather than a value.

A lambda is translated by rewriting its body against `element()`, the engine's placeholder
for "the current element" inside `.list.transform` / `.list.filter`. That is the whole
mechanism: bind the lambda's single parameter name to `element()` and translate the body
with the ordinary scalar path, so every expression the engine already has is available
inside a lambda without a second evaluator.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.literals import _const_int_arg, _const_str_arg
from batcher._sql.parser.expressions.maps import map_subscript
from batcher.plan.expr_ir import Expr, array, lit, nullif, when
from batcher.plan.functions.collection import element, sequence

__all__ = ["collection_function", "list_function"]

# `f(vector, other)` → a binary `.list` method. Spark's `vector_*` family is the same
# arithmetic as the engine's embedding methods under different names.
_LAMBDA_LIST = {
    "list_transform": "transform",
    "array_transform": "transform",
    "list_apply": "transform",
    "list_filter": "filter",
    "array_filter": "filter",
}

_VECTOR_PAIR = {
    "vector_cosine_similarity": "cosine_similarity",
    "vector_inner_product": "dot",
    "vector_l2_distance": "l2_distance",
}


def collection_function(tr, node) -> Expr | None:
    """Translate a list/array call, or None when the node is not one of them."""
    if isinstance(node, exp.GenerateSeries):
        return _generate_series(tr, node)
    if isinstance(node, exp.ArrayAppend):
        return tr._scalar(node.this).list.concat(array(tr._scalar(node.expression)))
    if isinstance(node, exp.ArrayPrepend):
        # sqlglot keeps Spark's argument order (`array_prepend(xs, elem)`), so the
        # element is `expression` here even though it lands on the left of the concat.
        return array(tr._scalar(node.expression)).list.concat(tr._scalar(node.this))
    if isinstance(node, exp.ArrayInsert):
        return _array_insert(tr, node)
    if isinstance(node, exp.ArrayCompact):
        return tr._scalar(node.this).list.filter(element().is_not_null())
    if isinstance(node, exp.ArrayExcept):
        return tr._scalar(node.this).list.difference(tr._scalar(node.expression))
    if isinstance(node, exp.ArrayRemove):
        # The null guard is load-bearing: `element() != v` is *null* for a null element,
        # which the filter drops, so `array_remove(array(1, null), 1)` lost the null that
        # Spark keeps.
        value = tr._scalar(node.expression)
        return tr._scalar(node.this).list.filter(element().is_null() | (element() != value))
    if isinstance(node, exp.Transform):
        body = _lambda_body(tr, node.expression)
        return None if body is None else tr._scalar(node.this).list.transform(body)
    if isinstance(node, exp.ArrayFilter):
        body = _lambda_body(tr, node.expression)
        return None if body is None else tr._scalar(node.this).list.filter(body)
    if isinstance(node, exp.Exists) and isinstance(node.expression, exp.Lambda):
        body = _lambda_body(tr, node.expression)
        return None if body is None else tr._scalar(node.this).list.filter(body).list.len() > lit(0)

    if not isinstance(node, exp.Anonymous):
        return None
    name = node.name.lower()
    args = list(node.expressions)

    if name in _LAMBDA_LIST and len(args) == 2:
        # DuckDB's spellings of the same two higher-order functions.
        body = _lambda_body(tr, args[1])
        if body is None:
            return None
        return getattr(tr._scalar(args[0]).list, _LAMBDA_LIST[name])(body)
    if name == "forall" and len(args) == 2:
        body = _lambda_body(tr, args[1])
        if body is None:
            return None
        # Every element satisfies the predicate exactly when none fails it.
        return tr._scalar(args[0]).list.filter(~body).list.len() == lit(0)
    if name == "get" and len(args) == 2:
        # Spark's `get` is 0-based and yields null out of range, which is `.list.get`.
        from batcher._sql.parser.expressions.literals import _const_int_arg

        return tr._scalar(args[0]).list.get(_const_int_arg(args[1], "get(): index"))
    if name == "array_repeat" and len(args) == 2:
        # `array_repeat(v, n)` — a list of `n` copies. The count has to be constant: a
        # per-row length would need a kernel, and a literal is the shape Spark's own
        # examples and nearly every real call use.
        from batcher._sql.parser.expressions.literals import _int_literal

        count = _int_literal(args[1])
        if count is None:
            return None
        value = tr._scalar(args[0])
        return array(*([value] * count)) if count > 0 else None
    if name == "arrays_overlap" and len(args) == 2:
        return _arrays_overlap(tr._scalar(args[0]), tr._scalar(args[1]))
    if name in _VECTOR_PAIR and len(args) == 2:
        method = _VECTOR_PAIR[name]
        return getattr(tr._scalar(args[0]).list, method)(tr._scalar(args[1]))
    if name == "vector_norm" and len(args) == 2:
        return _vector_norm(tr, args)
    return None


def _lambda_body(tr, lam) -> Expr | None:
    """Translate a one-parameter lambda body with its parameter bound to `element()`.

    Returns None for a lambda the engine cannot express — more than one parameter (the
    `(acc, x)` folds and `zip_with`'s `(x, y)`), which needs a second placeholder the
    `.list` kernels do not have.
    """
    if not isinstance(lam, exp.Lambda):
        return None
    params = [p.name for p in lam.expressions]
    if len(params) != 1:
        return None
    body = lam.this.copy()
    param = params[0]
    placeholder = exp.Anonymous(this="element", expressions=[])
    if isinstance(body, (exp.Column, exp.Identifier)) and body.name == param:
        return element()
    # A lambda parameter reaches the body as a bare `Identifier` (`x -> x + 1`) or, when
    # it is qualified, as a `Column`. Both spellings have to be rewritten, or the body
    # translates as a reference to a column that does not exist.
    for ref in list(body.find_all(exp.Column, exp.Identifier)):
        if ref.name == param and ref.parent is not None:
            ref.replace(placeholder.copy())
    return tr._scalar(body)


def _arrays_overlap(left: Expr, right: Expr) -> Expr:
    """Spark `arrays_overlap`: true on a shared element, null when only nulls could hide one.

    The three-valued rule is the whole difficulty. True wins outright; otherwise, if
    either side contains a null, the answer is unknown rather than false, because the
    null *might* have been the shared element. Composed from the set intersection and a
    null count so each branch is exact rather than approximated by `has_any`, whose null
    rule is a different one (it is null when a whole *list* is null).
    """
    shared = left.list.intersect(right).list.len() > lit(0)
    has_null = (left.list.filter(element().is_null()).list.len() > lit(0)) | (
        right.list.filter(element().is_null()).list.len() > lit(0)
    )
    unknown = nullif(lit(True), lit(True))
    return when(shared).then(lit(True)).when(has_null).then(unknown).otherwise(lit(False))


def _vector_norm(tr, args) -> Expr | None:
    """Spark `vector_norm(v, p)` for the two norms the engine implements (p = 1 or 2)."""
    order = args[1]
    if not isinstance(order, exp.Literal) or order.is_string:
        return None
    p = float(order.this)
    if p == 1.0:
        return tr._scalar(args[0]).list.l1_norm()
    if p == 2.0:
        return tr._scalar(args[0]).list.l2_norm()
    return None


def _generate_series(tr, node) -> Expr | None:
    """`range(...)` / `generate_series(...)` / Spark `sequence(...)` → the engine's series.

    One node serves three functions whose *bounds* differ, and the difference is not
    cosmetic: `generate_series(1, 5)` and `sequence(1, 5)` are `[1,2,3,4,5]` where
    `range(1, 5)` is `[1,2,3,4]`, and `range(3)` is `[0,1,2]`. The engine's `sequence` is
    the inclusive form, so the exclusive one is expressed by pulling the stop in by a
    step — which is also the only reading that stays correct for a step other than 1.
    """
    start, stop = node.args.get("start"), node.args.get("end")
    if start is None:
        return None
    step = node.args.get("step")
    step_expr = tr._scalar(step) if step is not None else lit(1)
    if stop is None:
        # The one-argument `range(n)`: 0 .. n-1.
        return sequence(lit(0), tr._scalar(start) - lit(1), step_expr)
    exclusive = bool(node.args.get("is_end_exclusive"))
    stop_expr = tr._scalar(stop)
    if exclusive:
        stop_expr = stop_expr - step_expr
    return sequence(tr._scalar(start), stop_expr, step_expr)


def _array_insert(tr, node) -> Expr | None:
    """Spark `array_insert(xs, pos, value)` — `value` spliced in at 1-based `pos`.

    Only a *constant* position is served, because the splice is expressed as two slices
    around it and a per-row position would need a kernel. Spark's rule that a position
    past the end pads with nulls is not expressible that way either, so a position beyond
    the list's length is declined rather than answered with a shorter list.
    """
    from batcher._sql.parser.expressions.literals import _const_int_arg

    position = _const_int_arg(node.args.get("position"), "array_insert(): position")
    if position < 1:
        return None  # Spark counts a negative position from the end; not expressible here
    xs = tr._scalar(node.this)
    value = array(tr._scalar(node.args["expression"]))
    head = xs.list.slice(0, position - 1)
    tail = xs.list.slice(position - 1, 2**31 - 1)
    return head.list.concat(value).list.concat(tail)


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
    # `count` is deliberately absent: it ignores nulls and is handled above, where the
    # two-step composition it needs can be expressed.
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


def list_function(tr, node):
    """List/array operations dispatched to the `.list` namespace, or None."""
    if isinstance(node, exp.ArraySize):  # array_length / len(list)
        return tr._scalar(node.this).list.len()
    if isinstance(node, exp.ArrayContainsAll):  # array_has_all / arrays_contain_all
        return tr._scalar(node.this).list.has_all(tr._scalar(node.expression))
    if isinstance(node, exp.ArrayOverlaps):  # list_has_any / arrays_overlap
        return tr._scalar(node.this).list.has_any(tr._scalar(node.expression))
    if isinstance(node, exp.ArrayConcat):
        # `list_concat`/`array_cat` — sqlglot puts the first operand in `this` and the
        # rest in `expressions`, so a three-way concat folds left to right.
        result = tr._scalar(node.this)
        for operand in node.expressions:
            result = result.list.concat(tr._scalar(operand))
        return result
    if isinstance(node, exp.Flatten):  # flatten(list-of-lists) — one level
        return tr._scalar(node.this).list.flatten()
    if isinstance(node, exp.ArraySort):  # array_sort(l) without a comparator
        if node.expression is not None:
            raise NotImplementedError("array_sort with a comparator is not supported")
        return tr._scalar(node.this).list.sort()
    if isinstance(node, exp.ArrayToString):  # array_join(l, sep) / list_aggr concat
        sep = _const_str_arg(node.expression, "array_join()", "separator")
        return tr._scalar(node.this).list.join(sep)
    if isinstance(node, exp.ArrayPosition):
        # Spark's `array_position(l, v)` is the 1-based index of `v`, 0 when absent —
        # which is what `.list.position` returns.
        return tr._scalar(node.this).list.position(_raw_value(node.expression))
    if isinstance(node, exp.ArraySlice):
        # `slice(l, start, length)`. SQL counts the start from 1 and `.list.slice` from
        # 0, so the index is shifted; sqlglot names the second operand `end`, but Spark's
        # is a *length*, so it passes through as the length rather than as an index.
        # Getting either wrong returns a plausible window one element along.
        start = _const_int_arg(node.args["start"], "slice(): start")
        size = node.args.get("end")
        length = _const_int_arg(size, "slice(): length") if size is not None else None
        return tr._scalar(node.this).list.slice(start - 1, length)
    if isinstance(node, exp.ArrayContains):  # list_contains(a, v)
        return tr._scalar(node.this).list.contains(_raw_value(node.expression))
    if isinstance(node, exp.Bracket):
        # `a[i]`. sqlglot 0-bases the index for the dialects whose subscript is 1-based
        # (duckdb, postgres) and leaves `offset` unset; where it cannot, it keeps the
        # written index and records the base in `offset` — Spark's `element_at(a, 2)`
        # becomes `Bracket(expressions=[2], offset=1)`. Ignoring `offset` made every such
        # subscript return the *next* element: `element_at(array(1,2,3), 2)` answered 3.
        idxs = node.expressions
        if len(idxs) == 1 and not isinstance(idxs[0], exp.Slice):
            # A map subscript is the same node as a list one, so `maps` reads it first.
            if (as_map := map_subscript(tr, node)) is not None:
                return as_map
            offset = int(node.args.get("offset") or 0)
            index = _subscript_value(idxs[0])
            if index < 0:
                # A negative subscript counts from the end (`a[-1]` is the last element),
                # which is what `.list.get` already means — but sqlglot 0-bases a 1-based
                # dialect by subtracting one from *whatever was written*, negatives
                # included. Undo it there, and only there.
                index = index + 1 if offset == 0 else index
                return tr._scalar(node.this).list.get(index)
            return tr._scalar(node.this).list.get(index - offset)
        if len(idxs) == 1 and isinstance(idxs[0], exp.Slice):
            return _list_slice(tr, node, idxs[0])
        return None
    reduce = _LIST_REDUCE.get(type(node).__name__)
    if reduce is not None:
        return getattr(tr._scalar(node.this).list, reduce)()
    if isinstance(node, exp.SortArray):
        # `list_sort(l)` ascending; `list_reverse_sort(l)` descending — the latter
        # parses as the same node with `asc=False`, which used to be dropped, so
        # `list_reverse_sort` returned the ascending order.
        value = tr._scalar(node.this)
        asc = node.args.get("asc")
        descending = asc is not None and not _boolean_arg(asc)
        # Descending is its own kernel rather than `sort().reverse()`. Ascending places
        # nulls last, so reversing lands them at the *front*, where DuckDB keeps them at
        # the back — `list_reverse_sort([4, NULL, 6])` is `[6, 4, NULL]`, not
        # `[NULL, 6, 4]`.
        return value.list.sort_desc() if descending else value.list.sort()
    # sqlglot promotes a few vector functions to typed nodes (two args in `this`/`expression`)
    # rather than `Anonymous`; dispatch them to the same binary `.list` methods.
    typed_binary = _LIST_TYPED_BINARY.get(type(node).__name__)
    if typed_binary is not None:
        return getattr(tr._scalar(node.this).list, typed_binary)(tr._scalar(node.expression))
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        if name in ("list_count", "array_count") and node.expressions:
            # `list_count` is a COUNT, not a length: DuckDB ignores nulls, so
            # `list_count([NULL, 4])` is 1 and `list_count([NULL, NULL])` is 0. This was
            # mapped straight to `len`, which returned the element count including nulls —
            # the same number for a list of four values and a list of four nulls.
            return tr._scalar(node.expressions[0]).list.drop_nulls().list.len()
        method = _list_anon_method(name)
        if method is not None and node.expressions:
            return getattr(tr._scalar(node.expressions[0]).list, method)()
        binary = _LIST_BINARY_ANON.get(name)
        if binary is not None and len(node.expressions) == 2:
            left = tr._scalar(node.expressions[0])
            right = tr._scalar(node.expressions[1])
            return getattr(left.list, binary)(right)
    return None


def _subscript_value(node) -> int:
    """The signed integer a subscript node carries.

    More than ``int(node.name)`` because a negative subscript parses as ``exp.Neg``
    wrapping the magnitude, and ``.name`` reads straight through to the child — so the
    sign was dropped and ``a[-1]`` asked for element 2 instead of the last one.
    """
    if isinstance(node, exp.Neg):
        return -int(node.this.name)
    return int(node.name)


def _list_slice(tr, node, sl) -> Expr:
    """``a[lo:hi]`` — a list slice, in SQL's 1-based, both-ends-inclusive convention.

    ``list.slice`` is 0-based and takes a *length*, so ``a[2:3]`` is ``slice(1, 2)``.
    Either bound may be omitted (``a[2:]``, ``a[:3]``).

    A negative bound is rejected rather than translated: DuckDB counts it back from the
    end, while `list.slice` clamps it to the start and returns the whole list — so
    ``a[-2:]`` would answer the entire list instead of its last two elements. Declining
    costs an error; translating it would cost a wrong answer.
    """
    lo_node, hi_node = sl.this, sl.expression
    lo = _const_int_arg(lo_node, "list slice: lower bound") if lo_node is not None else 1
    hi = _const_int_arg(hi_node, "list slice: upper bound") if hi_node is not None else None
    for bound, value in (("lower", lo), ("upper", hi)):
        if value is not None and value < 0:
            raise NotImplementedError(
                f"a negative {bound} bound in a list slice ({node.sql()}) is not supported; "
                "index from the start, or use list_reverse first"
            )
    if lo < 1:
        # DuckDB treats `a[0:n]` as `a[1:n]`; 0-basing it here would take one element
        # too many.
        lo = 1
    length = None if hi is None else max(hi - lo + 1, 0)
    return tr._scalar(node.this).list.slice(lo - 1, length)


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
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    return str(node.this).lower() not in ("false", "0")


def _raw_value(node):
    """The Python value of a literal node (for `.list.contains`)."""
    if not isinstance(node, exp.Literal):
        raise NotImplementedError("list_contains requires a constant value")
    if node.is_string:
        return node.name
    text = node.name
    return float(text) if ("." in text or "e" in text.lower()) else int(text)
