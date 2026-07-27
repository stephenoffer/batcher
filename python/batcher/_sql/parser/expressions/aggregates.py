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

__all__ = ["build_anon_agg", "build_typed_agg", "is_agg_node", "iter_agg_nodes"]


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
    "kurtosis_pop": lambda x: x.kurtosis_pop(),
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
    return isinstance(node, exp.Anonymous) and node.name.lower() in ANON_AGG_NAMES


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
    if kind in _TYPED_UNARY:
        return AggExpr(_TYPED_UNARY[kind], tr._scalar(node.this))
    composite = _TYPED_UNARY_COMPOSITE.get(kind)
    if composite is not None:
        return composite(tr._scalar(node.this))
    binary = _TYPED_BINARY.get(kind)
    if binary is not None:
        return binary(tr._scalar(node.this), tr._scalar(node.expression))
    if kind == "countif":
        # `count_if(cond)` counts the rows where `cond` is true — the condition is the
        # aggregate's whole argument, not a value column.
        return _count_if(tr._scalar(node.this))
    if kind == "ignorenulls" and type(node.this).__name__.lower() == "anyvalue":
        # `any_value(x)` parses as `IgnoreNulls(AnyValue(x))` — the ignore-nulls wrapper
        # is what every value aggregate already does, so only the inner node matters.
        return tr._scalar(node.this.this).any_value()
    if kind == "anyvalue":
        return tr._scalar(node.this).any_value()
    if kind == "percentiledisc":
        return AggExpr(
            "quantile_disc", tr._scalar(node.this), param=_fraction(node.args.get("expression"))
        )
    if kind == "approxtopk":
        count = node.args.get("expression")
        return tr._scalar(node.this).top_k(int(_fraction(count)))
    if kind == "approxquantile":
        return AggExpr(
            "approx_quantile", tr._scalar(node.this), param=_fraction(node.args.get("quantile"))
        )
    return None


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
        return parametric(tr._scalar(args[0]), _fraction(args[1]))
    if len(args) != 1:
        raise NotImplementedError(f"{name}() takes exactly one argument")
    return _ANON[name](tr._scalar(args[0]))


def _count_if(condition: Expr) -> AggExpr:
    """`count_if(cond)` as the engine spells it."""
    from batcher.plan.functions.aggregate import count_if

    return count_if(condition)


def _fraction(node) -> float:
    """The constant fraction a quantile argument denotes."""
    if not isinstance(node, exp.Literal) or node.is_string:
        raise NotImplementedError("approx_quantile requires a constant fraction")
    return float(node.name)
