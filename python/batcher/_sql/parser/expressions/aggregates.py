"""DuckDB aggregate spellings → the Batcher aggregate surface.

`literals._AGG_FUNCS` maps the handful of sqlglot aggregate nodes whose Batcher
equivalent is a single named `AggExpr` tag. This module covers the two shapes that map
does not:

* **Two-input aggregates** — `corr`, `covar_pop/samp`, `arg_min/arg_max` (and their
  `min_by`/`max_by` spellings), and the nine `regr_*` functions. They carry a second
  expression, which a name-to-tag table has nowhere to put.
* **Composite aggregates** — `stddev_pop`, `var_pop`, `sem` and the `regr_*` family are
  built *from* aggregates rather than being one (`sem` is ``stddev / sqrt(n)``). They
  lower to an `Expr` over aggregate leaves, which `GroupBy.agg` already hoists into
  hidden columns and re-evaluates in a following projection.

Plus the names sqlglot does not recognize as aggregates at all. `product(x)`,
`histogram(x)`, `mean(x)`, `sem(x)` and `count_star()` parse as `exp.Anonymous`, which
`find_all(exp.AggFunc)` never yields — so they were not merely unmapped, they were
invisible to aggregate collection and fell through to the scalar translator with an
"unknown function" error. `is_agg_node`/`iter_agg_nodes` are the widened predicate the
collection sites use so those names are seen.

A DuckDB aggregate whose closest Batcher equivalent has *different* semantics is
deliberately absent. `first`/`last` name a row *in scan order*, which a mergeable
aggregate cannot promise, so they keep raising. (`fsum`/`kahan_sum` were in that
list until the engine grew a compensated sum of its own; they now map to it.)

`any_value`/`arbitrary` are here, and are not the same case: DuckDB documents the chosen
row as *unspecified*, so the group minimum — which a commutative combine can compute
identically on one node and a hundred — conforms.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.plan.expr_ir import AggExpr, Expr
from batcher.plan.functions.regression import (
    regr_avgx,
    regr_avgy,
    regr_count,
    regr_intercept,
    regr_r2,
    regr_slope,
    regr_sxx,
    regr_sxy,
    regr_syy,
)
from batcher.plan.functions.statistics import sem, stddev_pop, var_pop

__all__ = [
    "build_anon_agg",
    "build_typed_agg",
    "distinct_input",
    "is_agg_node",
    "iter_agg_nodes",
]


# Single-input aggregates sqlglot promotes to a typed node → the `AggExpr` tag.
_TYPED_UNARY = {
    "approxdistinct": "approx_count_distinct",
    "bitwiseandagg": "bit_and",
    "bitwiseoragg": "bit_or",
    "bitwisexoragg": "bit_xor",
    "kurtosis": "kurtosis",
    "skewness": "skewness",
}

# Single-input aggregates that lower to an expression *over* aggregates.
_TYPED_UNARY_COMPOSITE = {
    "stddevpop": stddev_pop,
    "variancepop": var_pop,
}

# Two-input aggregates. `this` is the first SQL argument, `expression` the second, in
# the order DuckDB documents: `arg_max(value, ordering_key)`, `regr_slope(y, x)`,
# `covar_pop(y, x)`.
_TYPED_BINARY = {
    "argmax": lambda a, b: a.arg_max(b),
    "argmin": lambda a, b: a.arg_min(b),
    "corr": lambda a, b: AggExpr("corr", a, input2=b),
    "covarpop": lambda a, b: AggExpr("covar_pop", a, input2=b),
    "covarsamp": lambda a, b: AggExpr("covar_samp", a, input2=b),
    "regrslope": regr_slope,
    "regrintercept": regr_intercept,
    "regrr2": regr_r2,
    "regrcount": regr_count,
    "regravgx": regr_avgx,
    "regravgy": regr_avgy,
    "regrsxx": regr_sxx,
    "regrsxy": regr_sxy,
    "regrsyy": regr_syy,
}

# DuckDB aggregate names sqlglot leaves as `exp.Anonymous` → a builder over the single
# translated argument (`None` for the nullary `count_star()`).
_ANON: dict[str, object] = {
    "product": lambda x: x.product(),
    "entropy": lambda x: x.entropy(),
    "fsum": lambda x: x.kahan_sum(),
    "kahan_sum": lambda x: x.kahan_sum(),
    "sumkahan": lambda x: x.kahan_sum(),
    "mad": lambda x: x.mad(),
    "kurtosis_pop": lambda x: x.kurtosis(bias=True),
    # DuckDB leaves the choice of row unspecified, and so does the engine's `any_value`
    # (it takes the group minimum, which is what a commutative combine can promise).
    "any_value": lambda x: x.any_value(),
    "arbitrary": lambda x: x.any_value(),
    "histogram": lambda x: x.histogram(),
    "mean": lambda x: x.mean(),
    "favg": lambda x: x.mean(),
    "sem": sem,
    "count_star": lambda _x: AggExpr("count_star", None),
}

# Anonymous aggregates that take a *second, constant* argument: a quantile or a count.
# They cannot live in `_ANON`, whose builders take exactly one translated argument.
_ANON_PARAM: dict[str, object] = {
    "quantile_disc": lambda x, p: x.quantile_disc(p),
    "percentile_disc": lambda x, p: x.quantile_disc(p),
    "approx_top_k": lambda x, p: x.top_k(int(p)),
}

#: Every DuckDB aggregate name that arrives as `exp.Anonymous`. The collection sites
#: test membership here to decide whether an anonymous call is an aggregate.
ANON_AGG_NAMES = frozenset(_ANON) | frozenset(_ANON_PARAM)


def is_agg_node(node) -> bool:
    """Whether `node` is an aggregate call — typed or one of the anonymous names.

    Args:
        node: A sqlglot expression node.

    Returns:
        True for `exp.AggFunc` and for an `exp.Anonymous` naming a known aggregate.
    """
    if isinstance(node, exp.AggFunc):
        return True
    if isinstance(node, exp.IgnoreNulls):
        # `any_value(x)` parses as `IgnoreNulls(AnyValue(x))`, and `IgnoreNulls` is not
        # an `AggFunc` subclass — so without this the aggregate was invisible to
        # collection and the name reached the scalar translator instead.
        return is_agg_node(node.this)
    if not isinstance(node, exp.Anonymous):
        return False
    if node.name.lower() in ANON_AGG_NAMES:
        return True
    # The rest of the public function library. An aggregate has to be *recognized here*,
    # before the query is built, or collection never sees it and the `GROUP BY` is built
    # without it — which is what made `all_caps_rate(s)` answer correctly with no grouping
    # and then fail under `GROUP BY g` with "window function '__bt_win_0' references
    # unknown column". The curated names above stay first, so none of them changes meaning.
    from batcher._sql.parser.expressions.lowering.families import library_aggregate

    return library_aggregate(node.name) is not None


def iter_agg_nodes(root):
    """Yield every aggregate call in `root`'s subtree, typed or anonymous.

    The widened counterpart of ``root.find_all(exp.AggFunc)``: an anonymous aggregate
    (`product(x)`, `sem(x)`, …) is not an `AggFunc` subclass, so the narrow walk skipped
    it entirely and the name reached the scalar translator instead.

    Args:
        root: The sqlglot node to walk.

    Returns:
        An iterator over the aggregate call nodes, in walk order.
    """
    return (n for n in root.find_all(exp.AggFunc, exp.Anonymous, exp.IgnoreNulls) if is_agg_node(n))


def distinct_input(tr, node, args=None):
    """The single expression a ``DISTINCT`` aggregate argument holds, or `None`.

    `<agg>(DISTINCT x)` is `<agg>(x)` over rows deduplicated on the group keys plus `x`.
    The engine's aggregates carry no per-group dedup flag, so the rewrite in
    `agg_rewrites.rewrite_distinct_aggs` dedups the *input* once up front and the
    aggregate itself becomes an ordinary one. This records the expression for that pass —
    the same protocol `literals._AGG_FUNCS` aggregates use — and hands back the argument
    to build over.

    Without it every aggregate reaching this module rejected a `DISTINCT` argument by
    falling through to the scalar translator, which reported the sqlglot node it could not
    lower (``unsupported SQL expression: Distinct``) rather than the aggregate the user
    wrote. Eleven aggregates DuckDB supports with `DISTINCT` — `bit_and`, `bit_or`,
    `bit_xor`, `kurtosis`, `skewness`, `approx_count_distinct`, `product`, `entropy`,
    `mad`, `any_value`, `quantile_cont` — could not be spelled that way at all.

    Args:
        tr: The translator instance, carrying the pending-distinct list.
        node: The aggregate node.
        args: The argument list to read instead of ``node.this`` — anonymous calls park
            their arguments in ``node.expressions``.

    Returns:
        The sqlglot expression the `DISTINCT` wraps, or `None` when there is no
        `DISTINCT` to unwrap.

    Raises:
        NotImplementedError: If the `DISTINCT` holds more than one expression, which
            needs a dedup per expression rather than the single pass this rewrite makes.
    """
    arg = args[0] if args else node.args.get("this")
    if not isinstance(arg, exp.Distinct):
        return None
    exprs = arg.expressions
    if len(exprs) != 1:
        raise NotImplementedError(
            f"{_agg_label(node)}(DISTINCT ...) supports exactly one expression"
        )
    tr._agg_pending_distinct.append((exprs[0].sql(), exprs[0]))
    return exprs[0]


def _reject_distinct(node, args=None) -> None:
    """Refuse a `DISTINCT` argument the pre-dedup rewrite cannot serve.

    A *composite* aggregate (`stddev_pop`, `var_pop`, `sem`) is an expression over several
    aggregate leaves, and a *two-input* one (`corr`, `covar_*`, `regr_*`, `arg_min/max`)
    reads two columns. Neither has the single input the dedup redirects, so both are
    declined by name here instead of reaching the scalar translator, which reported only
    that it could not lower a ``Distinct`` node.

    Args:
        node: The aggregate node.
        args: The argument list to read instead of ``node.this``.

    Raises:
        NotImplementedError: If the argument is a `DISTINCT`.
    """
    arg = args[0] if args else node.args.get("this")
    if isinstance(arg, exp.Distinct):
        label = _agg_label(node)
        raise NotImplementedError(
            f"{label}(DISTINCT ...) is not supported: {label} is built from more than one "
            "aggregate over more than one input, so there is no single column to "
            "deduplicate. Deduplicate in a subquery first — "
            f"SELECT {label}(x) FROM (SELECT DISTINCT g, x FROM t) GROUP BY g"
        )


def _agg_label(node) -> str:
    """The aggregate's name as the user spelled it, for a message.

    An anonymous call carries its name in `node.name`; `sql_name()` would answer
    ``"ANONYMOUS"`` for it, which names the sqlglot node rather than the function and
    tells the reader nothing about what to change.
    """
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    name = getattr(node, "sql_name", None)
    return (name() if callable(name) else str(node.name)).lower()


def build_typed_agg(tr, node) -> AggExpr | Expr | None:
    """Build the aggregate a typed sqlglot node denotes, or None if it is not one here.

    Args:
        tr: The translator instance (for recursive `_scalar` calls).
        node: The `exp.AggFunc` node.

    Returns:
        The aggregate (or expression over aggregates), or `None` when the node's kind is
        not served by this module and the caller should fall through to `_AGG_FUNCS`.
    """
    kind = type(node).__name__.lower()
    _reject_top_n(node, kind)
    if kind in _TYPED_UNARY:
        return AggExpr(_TYPED_UNARY[kind], tr._scalar(distinct_input(tr, node) or node.this))
    composite = _TYPED_UNARY_COMPOSITE.get(kind)
    if composite is not None:
        _reject_distinct(node)
        return composite(tr._scalar(node.this))
    binary = _TYPED_BINARY.get(kind)
    if binary is not None:
        _reject_distinct(node)
        return binary(tr._scalar(node.this), tr._scalar(node.expression))
    if kind == "countif":
        # `count_if(cond)` counts the rows where `cond` is true — the condition is the
        # aggregate's whole argument, not a value column.
        return _count_if(tr._scalar(node.this))
    if kind == "ignorenulls" and type(node.this).__name__.lower() == "anyvalue":
        # `any_value(x)` parses as `IgnoreNulls(AnyValue(x))` — the ignore-nulls wrapper
        # is what every value aggregate already does, so only the inner node matters.
        inner = node.this
        return tr._scalar(distinct_input(tr, inner) or inner.this).any_value()
    if kind == "anyvalue":
        return tr._scalar(distinct_input(tr, node) or node.this).any_value()
    if kind == "quantile":
        # DuckDB's bare `quantile(x, p)` is `quantile_disc`, not the continuous one.
        # `bt.quantile(...)` is the *continuous* quantile and is spelled `quantile_cont` in
        # SQL, so routing this to the library function by name would answer 2.5 where
        # DuckDB answers 2.0 — a name shared by the two front ends that means two different
        # things is worse than the "unsupported aggregate" it used to raise.
        return AggExpr(
            "quantile_disc",
            tr._scalar(distinct_input(tr, node) or node.this),
            param=_fraction(node.args.get("quantile")),
        )
    if kind == "percentiledisc":
        return AggExpr(
            "quantile_disc",
            tr._scalar(distinct_input(tr, node) or node.this),
            param=_fraction(node.args.get("expression")),
        )
    if kind == "approxtopk":
        count = node.args.get("expression")
        return tr._scalar(node.this).top_k(int(_fraction(count)))
    if kind == "approxquantile":
        return AggExpr(
            "approx_quantile",
            tr._scalar(distinct_input(tr, node) or node.this),
            param=_fraction(node.args.get("quantile")),
        )
    return None


#: Aggregates with a DuckDB "top N" overload — `max(x, n)`, `arg_max(x, y, n)` — whose
#: result is a **list** of the n best values, not the single best. sqlglot parks the count
#: in a different slot per node, so both are checked.
_TOP_N_KINDS = {"max": "expressions", "min": "expressions", "argmax": "count", "argmin": "count"}


def _reject_top_n(node, kind: str) -> None:
    """Refuse the `max(x, n)` / `arg_max(x, y, n)` overloads rather than drop the count.

    DuckDB's three-argument `arg_max` and two-argument `max` return a **list** of the n
    best values. The count sits in a slot the two-input builder never reads, so the call
    was answered with the plain one-value aggregate — a scalar where SQL asks for a list,
    silently, on every row.

    Args:
        node: The aggregate node.
        kind: Its lower-cased sqlglot class name.

    Raises:
        NotImplementedError: When the top-n overload is used.
    """
    slot = _TOP_N_KINDS.get(kind)
    if slot is None or not node.args.get(slot):
        return
    raise NotImplementedError(
        f"the top-N form of {kind}() returns a list of the N best values, which Batcher "
        "has no aggregate for; use ORDER BY ... LIMIT N, or list(x ORDER BY ...)"
    )


def build_anon_agg(tr, node) -> AggExpr | Expr:
    """Build the aggregate an anonymous call denotes (the name must be in `ANON_AGG_NAMES`).

    Args:
        tr: The translator instance (for recursive `_scalar` calls).
        node: The `exp.Anonymous` node.

    Returns:
        The aggregate, or an expression over aggregates for the composite ones.
    """
    name = node.name.lower()
    args = list(node.expressions)
    if name == "count_star":
        return AggExpr("count_star", None)
    parametric = _ANON_PARAM.get(name)
    if parametric is not None:
        if len(args) != 2:
            raise NotImplementedError(f"{name}() takes a value and a constant")
        return parametric(tr._scalar(distinct_input(tr, node, args) or args[0]), _fraction(args[1]))
    if name not in _ANON:
        return _library_agg(tr, node, name, args)
    if len(args) != 1:
        raise NotImplementedError(f"{name}() takes exactly one argument")
    if name == "sem":
        # `sem` is composite (stddev / sqrt(n)) — no single input for the dedup.
        _reject_distinct(node, args)
    return _ANON[name](tr._scalar(distinct_input(tr, node, args) or args[0]))


def _library_agg(tr, node, name: str, args: list) -> AggExpr | Expr:
    """Build a library aggregate — the ones outside `_ANON`'s curated table.

    Args:
        tr: The translator instance.
        node: The `exp.Anonymous` node, for the error messages.
        name: The lowercased function name.
        args: The call's argument nodes.

    Returns:
        The aggregate, or an expression over aggregates for the composite ones — which most
        of these are, and which `GroupBy.agg` already hoists into hidden columns.

    Raises:
        NotImplementedError: On the wrong argument count, or on `DISTINCT`, which has no
            single input to de-duplicate once the call reduces more than one column.
    """
    from batcher._sql.parser.expressions.lowering.families import (
        library_aggregate,
        positional_arity,
    )
    from batcher._sql.parser.expressions.lowering.signatures import build_arguments

    fn = library_aggregate(name)
    if fn is None:  # pragma: no cover - `is_agg_node` already answered for this name
        raise NotImplementedError(f"unknown aggregate {name!r}")
    required, total = positional_arity(fn)
    if len(args) < required or (total is not None and len(args) > total):
        if total is None:
            expected = f"at least {required}"
        else:
            expected = str(required) if required == total else f"{required} to {total}"
        raise NotImplementedError(f"{name}() takes {expected} argument(s), got {len(args)}")
    # `DISTINCT` de-duplicates *one* input before reducing; a composite aggregate reduces
    # several columns and there is no single one to de-duplicate, so it is refused rather
    # than silently ignored — which is the same rule `sem` already follows above.
    _reject_distinct(node, args)
    # `build_arguments`, not `tr._scalar` per argument: several of these take a plan-time
    # constant (`char_repetition_rate(text, n)`, `approx_quantile(x, q)`), and lowering it
    # to a `Lit` fails inside the function with an error naming an internal node.
    return fn(*build_arguments(tr, fn, args, name))


def _count_if(condition: Expr) -> AggExpr:
    """`count_if(cond)` as the engine spells it."""
    from batcher.plan.functions.aggregate import count_if

    return count_if(condition)


def _fraction(node) -> float:
    """The constant fraction a quantile argument denotes."""
    if not isinstance(node, exp.Literal) or node.is_string:
        raise NotImplementedError("approx_quantile requires a constant fraction")
    return float(node.name)
