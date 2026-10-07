"""Aggregate pre-pass rewrites for the SQL translator.

A rewrite that reshapes the *input* so an aggregate the engine cannot express directly
becomes one it can: DISTINCT aggregates (dedup first), and the ``FILTER (WHERE ...)``
clause, which is carried to the aggregate builder and lowered there by the same
`aggregate_semantics.filter_aggregate` that `AggExpr.filter` uses. An ordered aggregate
(`array_agg(x ORDER BY y)`) needs no rewrite; it lowers to the engine's ordered list
aggregate, which carries its own keys.

`<agg>(DISTINCT x)` per group is `<agg>(x)` over the rows left once duplicate `x` values
are removed *within* each group. Three shapes are handled:

* **Only distinct aggregates**, all over one expression — dedup on `(group keys, x)` once
  up front and every aggregate becomes an ordinary one.
* **One distinct expression mixed with plain aggregates that have a single-column
  partial** — a two-level aggregate. Grouping by `(group keys, x)` dedups `x` implicitly
  while pre-aggregating the plain aggregates into mergeable partials; the second level
  then aggregates `x` directly and *combines* those partials.
* **Anything else** — several distinct expressions, or a plain aggregate with no
  single-column partial (`avg`, a quantile, `count(DISTINCT)` beside `sum(DISTINCT)`) —
  uses Spark's *Expand* rewrite. The input is unioned once per distinct expression plus
  once for the plain aggregates, each copy tagged with a `__bc_gid` and carrying only its
  own value; each distinct copy is deduplicated on `(group keys, value)`. One aggregate
  over the union then computes every output, each restricted to its own copy with the
  `FILTER` lowering above. Every group keeps its raw rows in the plain copy, so an empty
  group, a NULL key and `count(*)` need no special case. The price is one scan of the
  input per copy, which is Spark's price for the same query.

None of the three joins two aggregates on the group keys: a join drops rows whose key is
NULL, which would silently lose the NULL group that `GROUP BY` legitimately produces.

Split from `grouping.py` because it is a self-contained rewrite with its own correctness
argument — and because that module is at its size limit.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import AggExpr, Expr, coalesce, col, lit
from batcher.plan.expr_ir.walk import referenced_columns
from batcher.plan.expr_rewrite.traverse import transform_expr_up
from batcher.plan.functions.aggregate_semantics import filter_aggregate_leaves, masked

__all__ = [
    "distinct_value",
    "hoist_grouped_udfs",
    "mark_filter",
    "name_udf_items",
    "rewrite_distinct_aggs",
    "split_filter",
]

#: The `ELSE` branch that marks a CASE as a carried `FILTER (WHERE ...)` guard rather than a
#: CASE the user wrote. It is a call no catalog defines, so nothing else can produce it.
_FILTERED = "__bc_filtered"

#: The aggregate wrappers sqlglot puts *around* an aggregate call (`any_value(x)` parses as
#: `IgnoreNulls(AnyValue(x))`); the argument lives on the call inside.
_WRAPPERS = (exp.IgnoreNulls, exp.RespectNulls)


def _call(agg):
    """The aggregate call inside any null-treatment wrapper."""
    while isinstance(agg, _WRAPPERS):
        agg = agg.this
    return agg


def _argument(agg, *, strict: bool = False):
    """The node holding `agg`'s first argument, or None for an argument-less call.

    Looks through the shapes sqlglot gives an argument: a `DISTINCT` (`count(DISTINCT x)`),
    an `ORDER BY` riding the call (`array_agg(x ORDER BY y)`), and an anonymous call, which
    keeps its arguments in `expressions` rather than `this`.
    """
    call = _call(agg)
    if isinstance(call, exp.Anonymous):
        arg = call.expressions[0] if call.expressions else None
    else:
        arg = call.this
    if isinstance(arg, exp.Order):
        arg = arg.this
    if isinstance(arg, exp.Distinct):
        if len(arg.expressions) != 1 and not strict:
            return None
        if len(arg.expressions) != 1:
            raise NotImplementedError(
                "FILTER (WHERE ...) on an aggregate of a multi-argument DISTINCT is not "
                "supported; filter in a subquery first"
            )
        arg = arg.expressions[0]
    return None if arg is None or isinstance(arg, exp.Star) else arg


def mark_filter(node):
    """Carry ``agg(arg) FILTER (WHERE c)`` to the aggregate builder, as a marked guard.

    The argument becomes ``CASE WHEN c THEN arg ELSE __bc_filtered() END``. That keeps the
    node an ordinary aggregate call for every pass that collects or matches aggregates by
    their SQL text, gives the filtered and unfiltered forms different texts, and leaves
    `split_filter` an unambiguous shape to take apart. ``COUNT(*)`` guards the constant 1.

    Args:
        node: Any node of a SELECT tree; only a `Filter` is rewritten.

    Returns:
        The rewritten aggregate, or `node` itself when it is not a `Filter`.
    """
    if not isinstance(node, exp.Filter):
        return node
    agg = node.this.copy()
    arg = _argument(agg, strict=True)
    guard = exp.Case(
        ifs=[
            exp.If(
                this=node.expression.this.copy(),
                true=exp.Literal.number(1) if arg is None else arg.copy(),
            )
        ],
        default=exp.Anonymous(this=_FILTERED),
    )
    if arg is not None:
        arg.replace(guard)
        return agg
    call = _call(agg)
    if isinstance(call, exp.Anonymous):
        call.set("expressions", [guard])
    else:
        call.set("this", guard)
    return agg


def split_filter(node):
    """Take a `mark_filter` guard back off an aggregate: ``(plain aggregate, condition)``.

    Args:
        node: An aggregate call node.

    Returns:
        The aggregate with its original argument restored and the `FILTER` condition, or
        ``(node, None)`` when it carries no filter.
    """
    arg = _argument(node)
    default = arg.args.get("default") if isinstance(arg, exp.Case) else None
    if not (isinstance(default, exp.Anonymous) and default.name == _FILTERED):
        return node, None
    node = node.copy()
    guard = _argument(node)
    branch = guard.args["ifs"][0]
    guard.replace(branch.args["true"].copy())
    return node, branch.this


def distinct_value(tr, entry) -> Expr:
    """The value a recorded DISTINCT argument deduplicates, with its FILTER guard applied.

    Args:
        tr: The translator.
        entry: A ``(key, node)`` or ``(key, node, predicate)`` record from aggregate
            registration.

    Returns:
        The value expression.
    """
    value = tr._scalar(entry[1])
    return masked(value, entry[2]) if len(entry) > 2 else value


# Plain aggregates that survive pre-aggregation, as {level-1 partial: level-2 combine}.
#
# The condition is that the aggregate has a **single-column mergeable partial**: the level-1
# value computed per sub-group can be combined into the group's true answer by one aggregate
# over one column. `count` is the interesting one, and the reason this is a map rather than a
# set: a group's total is the SUM of its sub-groups' counts, not their count.
#
# The rest combine with *themselves*, because level 1 partitions the group's rows — every row
# lands in exactly one sub-group — and each of these operations is associative and commutative
# over that partition. `min`/`max`/`sum`/`product` and the bitwise folds are associative
# outright; `bit_xor` needs the partition to be exact (a row counted twice would cancel) and it
# is; `bool_and`/`bool_or` are idempotent as well, so they hold regardless. `any_value` holds
# because any value of any sub-group is a value of the group, which is all it promises.
# NULL handling needs no special case: a sub-group with nothing to aggregate yields NULL and
# the level-2 aggregate skips it, exactly as the one-level form skips the same rows.
#
# Anything absent here — `mean`, `stddev`, `var`, the quantiles, `count_distinct` — has no
# single-column partial (a mean needs a sum *and* a count) and goes through the Expand
# rewrite instead.
_DECOMPOSABLE = {
    "count": "sum",
    "count_star": "sum",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "any_value": "any_value",
    "bool_and": "bool_and",
    "bool_or": "bool_or",
    "bit_and": "bit_and",
    "bit_or": "bit_or",
    "bit_xor": "bit_xor",
    "product": "product",
}


def _redirect(agg: AggExpr | Expr, value: Expr, target: Expr) -> AggExpr | Expr:
    """`agg` with every aggregate operand equal to `value` replaced by `target`.

    A distinct aggregate was built over the DISTINCT argument itself; after the dedup it
    reads the deduplicated column instead. Composite aggregates (`kurtosis_pop`) are
    walked to their leaves, and every other field of each leaf is carried over.

    Raises:
        NotImplementedError: If no operand matched, which would aggregate the wrong rows.
    """
    wanted = repr(value)
    hits: list[bool] = []

    def swap(e: Expr) -> Expr:
        if repr(e) == wanted:
            hits.append(True)
            return target
        return e

    def rule(node: Expr) -> Expr:
        return node.map_operands(swap) if isinstance(node, AggExpr) else node  # type: ignore[return-value]

    out = agg.map_operands(swap) if isinstance(agg, AggExpr) else transform_expr_up(agg, rule)
    if not hits:
        raise NotImplementedError(
            f"the DISTINCT aggregate {agg!r} could not be matched to its deduplicated input; "
            "compute it in a separate subquery"
        )
    return out


def rewrite_distinct_aggs(
    tr, ds: Dataset, group_cols, group_exprs, agg_kwargs: dict[str, AggExpr | Expr]
) -> tuple[Dataset, dict[str, AggExpr | Expr]]:
    """Rewrite a query containing DISTINCT aggregates into an equivalent plain one.

    Args:
        tr: The translator, carrying the distinct expressions collected during registration.
        ds: The dataset to aggregate.
        group_cols: Plain column group keys.
        group_exprs: Computed group keys, by output alias.
        agg_kwargs: Every aggregate in the query, by output name. A *composite*
            aggregate (`stddev_pop`, `regr_slope`, `sem`) is an `Expr` over several
            partials rather than an `AggExpr`, so the values are not all one type.

    Returns:
        The dataset to group and the aggregates to apply to it. The caller groups by
        `group_cols` plus the *aliases* of `group_exprs`, which this has materialized.
    """
    keys = [*group_cols, *group_exprs]
    ds = ds.with_columns(**group_exprs) if group_exprs else ds
    values: dict[str, Expr] = {}
    for entry in tr._agg_distinct.values():
        values.setdefault(entry[0], distinct_value(tr, entry))
    plain = {name: a for name, a in agg_kwargs.items() if name not in tr._agg_distinct}
    distinct = {name: tr._agg_distinct[name][0] for name in agg_kwargs if name not in plain}
    one_column = all(isinstance(agg_kwargs[name], AggExpr) for name in distinct)
    if len(values) == 1 and one_column:
        value = next(iter(values.values()))
        if not plain:
            return _dedup_once(ds, keys, value, agg_kwargs)
        if all(isinstance(a, AggExpr) and a.func in _DECOMPOSABLE for a in plain.values()):
            return _two_level(ds, keys, value, plain, agg_kwargs, distinct)
    return _expand(ds, keys, values, plain, agg_kwargs, distinct)


def _dedup_once(ds: Dataset, keys, value: Expr, agg_kwargs) -> tuple[Dataset, dict]:
    """Every aggregate is DISTINCT over one expression: one dedup, then plain aggregates."""
    dv = "__distinct_agg_v"
    deduped = ds.with_columns(**{dv: value}).select(*keys, dv).distinct()
    return deduped, {name: _redirect(a, value, col(dv)) for name, a in agg_kwargs.items()}


def _two_level(ds: Dataset, keys, value: Expr, plain, agg_kwargs, distinct):
    """One distinct expression beside plain aggregates that pre-aggregate into partials.

    Level 1 groups by the keys PLUS the distinct expression, which dedups it implicitly
    while each plain aggregate becomes its per-sub-group partial. Level 2 is the caller's
    group-by: distinct aggregates read the deduplicated column, plain ones combine their
    partials. A count combines by `sum`, which is null over no sub-groups at all, so it is
    coalesced back to the 0 an ungrouped count over an empty input answers.
    """
    dv = "__distinct_agg_v"
    level1 = ds.with_columns(**{dv: value}).group_by(*keys, dv).agg(**plain)
    level2: dict[str, AggExpr | Expr] = {
        name: _redirect(a, value, col(dv)) for name, a in agg_kwargs.items() if name in distinct
    }
    for name, a in plain.items():
        combined = AggExpr(_DECOMPOSABLE[a.func], col(name))
        counting = a.func in ("count", "count_star")
        level2[name] = coalesce(combined, lit(0)) if counting else combined
    return level1, level2


def _expand(ds: Dataset, keys, values: dict[str, Expr], plain, agg_kwargs, distinct):
    """Spark's Expand rewrite: one tagged, deduplicated copy per distinct expression."""
    gid = "__bc_gid"
    dvs = {key: f"__bc_dv{i}" for i, key in enumerate(values)}
    tags = {key: i for i, key in enumerate(values)}
    plain_tag = len(values)
    carried: set[str] = set()
    for a in plain.values():
        carried |= _columns_read(a)
    carried -= set(keys)

    def copy(tag: int) -> Dataset:
        cols: dict[str, Expr] = {k: col(k) for k in keys}
        cols[gid] = lit(tag)
        for key, value in values.items():
            cols[dvs[key]] = value if tags[key] == tag else masked(value, lit(False))
        for c in sorted(carried):
            cols[c] = col(c) if tag == plain_tag else masked(col(c), lit(False))
        out = ds.select(**cols)
        return out if tag == plain_tag else out.distinct()

    expanded = copy(0).union(*(copy(tag) for tag in range(1, plain_tag + 1)))
    level2: dict[str, AggExpr | Expr] = {}
    for name, a in agg_kwargs.items():
        if name in distinct:
            key = distinct[name]
            reads = _redirect(a, values[key], col(dvs[key]))
            level2[name] = filter_aggregate_leaves(reads, col(gid) == lit(tags[key]))
        else:
            level2[name] = filter_aggregate_leaves(a, col(gid) == lit(plain_tag))
    return expanded, level2


def _columns_read(agg: AggExpr | Expr) -> set[str]:
    """The input columns every aggregate inside `agg` reads."""
    found: set[str] = set()

    def rule(node: Expr) -> Expr:
        if isinstance(node, AggExpr):
            for operand in node.operands():
                found.update(referenced_columns(operand))
        return node

    if isinstance(agg, AggExpr):
        rule(agg)  # type: ignore[arg-type]
    else:
        transform_expr_up(agg, rule)
    return found


_UDF_REFUSAL = (
    "a registered scalar function over an aggregate or window result is supported only in "
    "the SELECT list, outside any window; compute it in a subquery or a projected alias "
    "first, then filter or order on that column"
)


def hoist_grouped_udfs(tr, ds: Dataset, node, projections, has_agg: bool):
    """Materialize the registered scalar functions of an aggregate or window query.

    Python cannot run inside the engine's expressions, so a registered function becomes a
    `map_batches` stage that appends its result as a column (`udf._hoist_udfs`). Where
    that stage goes depends on what the call reads:

    * **Before the aggregate or window pass**, a call over input rows: inside an aggregate
      (``SUM(f(x))``), a GROUP BY key (``GROUP BY f(x)``, which every other spelling of the
      same call then reads), or anywhere in a window query that does not read a window.
      Identical calls are computed once, so a key and the SELECT item naming it match.
    * **After the aggregate**, a call over the grouped rows (``f(SUM(x))``, ``f(k)`` for a
      key ``k``), in the SELECT list: each argument becomes a hidden grouped column and
      the call runs over those once the grouping is done.
    * **After the window pass**, a call over a window's result, in a window query's SELECT
      list; the caller hoists those once the window columns exist.

    Anywhere else (HAVING, QUALIFY or ORDER BY over a grouped value) is refused.

    Args:
        tr: The translator.
        ds: The relation the query reads.
        node: The SELECT node.
        projections: The SELECT list, rewritten in place.
        has_agg: Whether this is an aggregate query (otherwise a window-only one).

    Returns:
        ``(ds, hidden, finish)``: the relation with the input-row calls appended, the hidden
        SELECT items to aggregate alongside the user's, and a function that takes the
        aggregate's ``(ds, named)`` and returns them with the post-aggregate calls applied.
    """
    from batcher._sql.parser.udf import _is_registered_scalar

    name_udf_items(tr, projections)
    clauses = [*projections, node.args.get("group"), node.args.get("having")]
    clauses += [node.args.get("qualify"), node.args.get("order")]
    calls = [
        n
        for c in clauses
        if c is not None
        for n in c.find_all(exp.Anonymous)
        if _is_registered_scalar(tr, n)
        and n.find_ancestor(exp.Select) is node
        and not any(_is_registered_scalar(tr, a) for a in _ancestors(n))
    ]
    if not calls:
        return ds, [], _no_post
    keys = _group_key_texts(node, projections) if has_agg else set()
    pre: dict[str, list] = {}
    post: list = []
    for call in calls:
        if _runs_on_input_rows(call, keys, has_agg):
            pre.setdefault(call.sql(), []).append(call)
        elif call.find_ancestor(exp.Window) is None and _in_projection(call, projections):
            post.append(call)
        else:
            raise PlanError(_UDF_REFUSAL)
    for same in pre.values():
        ds, (replacement,) = tr._hoist_udfs(ds, [same[0]])
        for other in same[1:]:
            other.replace(replacement.copy())
    if not has_agg:
        return ds, [], _no_post
    return ds, *_defer_post_calls(tr, post)


def _no_post(ds: Dataset, named: dict) -> tuple[Dataset, dict]:
    return ds, named


def _ancestors(node):
    """`node`'s strict ancestors, nearest first."""
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def name_udf_items(tr, projections) -> None:
    """Give each unaliased SELECT item holding a registered call its as-written name.

    Hoisting replaces the call with a reference to an internal column, and an unaliased
    item is named after its expression, so without this the output column was named
    ``__bc_udf_0`` instead of ``f(x)``.
    """
    from batcher._sql.parser.core_utils import _alias_of
    from batcher._sql.parser.udf import contains_registered_scalar

    for i, p in enumerate(list(projections)):
        if not isinstance(p, exp.Alias) and contains_registered_scalar(tr, p):
            named = exp.alias_(p.copy(), _alias_of(p), quoted=True)
            p.replace(named)
            projections[i] = named


def _group_key_texts(node, projections) -> set[str]:
    """The SQL text of every GROUP BY key, resolving ordinals, select aliases and ``ALL``."""
    from batcher._sql.parser.core_utils import _alias_of, _has_aggregate, _unwrap_alias

    group = node.args.get("group")
    if group is None:
        return set()
    by_alias = {_alias_of(p): _unwrap_alias(p) for p in projections}
    items = list(group.expressions)
    if group.args.get("all") and not items:
        items = [_unwrap_alias(p) for p in projections if not _has_aggregate(p)]
    texts = set()
    for g in items:
        if isinstance(g, exp.Literal) and not g.is_string and 0 < int(g.name) <= len(projections):
            g = _unwrap_alias(projections[int(g.name) - 1])
        elif isinstance(g, exp.Column) and g.name in by_alias:
            texts.add(by_alias[g.name].sql())
        texts.add(g.sql())
    return texts


def _runs_on_input_rows(call, keys: set[str], has_agg: bool) -> bool:
    """Whether `call` reads input rows (hoist before the pass) rather than its results."""
    from batcher._sql.parser.expressions.aggregates import is_agg_node, iter_agg_nodes

    if call.find(exp.Window) is not None:
        return False
    if not has_agg:
        return True
    if any(True for _ in iter_agg_nodes(call)):
        return False
    within = list(_ancestors(call))
    if any(isinstance(a, exp.Window) for a in within):
        return any(is_agg_node(a) for a in within[: _index_of_window(within)])
    return any(is_agg_node(a) for a in within) or any(n.sql() in keys for n in [call, *within])


def _index_of_window(ancestors) -> int:
    return next(i for i, a in enumerate(ancestors) if isinstance(a, exp.Window))


def _in_projection(call, projections) -> bool:
    return any(any(call is a for a in p.find_all(exp.Anonymous)) for p in projections)


def _defer_post_calls(tr, calls):
    """Split post-aggregate calls into hidden grouped arguments and a finishing step."""
    hidden = []
    deferred = []
    for call in calls:
        out = f"__bc_udfpost{len(deferred)}"
        for arg in list(call.expressions):
            if isinstance(arg, (exp.Literal, exp.Boolean, exp.Null)):
                continue
            name = f"__bc_udfarg{len(hidden)}"
            hidden.append(exp.alias_(arg.copy(), name))
            arg.replace(exp.column(name))
        detached = call.copy()
        call.replace(exp.column(out))
        deferred.append((detached, out))
    names = [h.alias for h in hidden]

    def finish(ds: Dataset, named: dict) -> tuple[Dataset, dict]:
        if names:
            ds = ds.with_columns(**{n: named.pop(n) for n in names})
        for detached, out in deferred:
            ds, (column,) = tr._hoist_udfs(ds, [detached])
            ds = ds.with_columns(**{out: col(column.name)})
        return ds, named

    return hidden, finish
