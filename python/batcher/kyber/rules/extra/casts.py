"""NORMALIZE-phase rules for `CAST` — the shapes a SQL front end and the type-coercion
layer leave behind.

A cast is a full pass over a column, so a cast that cannot change a value is pure cost;
and a cast that *can* change one is the single most dangerous thing to rewrite, because
it moves a query's output **type**. Every rule here removes or relocates a cast only when
its Arrow source and target types are both *provably known* (`plan.types.infer_type`) and
the conversion is one we can prove exact against the engine's own kernel
(`bc_expr::eval::cast::cast_expr`, which is `arrow::compute::cast_with_options` with
`safe = try_cast`, plus DuckDB half-to-even rounding for float→int).

What the neighbours already cover, and this module does not repeat: `normalize.simplify`
drops `cast(cast(x, T), T)` with matching `try_cast` flags; `normalize.fold` folds
`cast(<literal>)` for the *exact* conversions (identity, int↔int, temporal↔int);
`projection_scan.drop_self_cast_in_{filter,projection,sort_key}` drops `cast(col(c), T)`
when the **column** `c` is already `T`; `fusion.push_down_narrowing_cast` relocates a
narrowing cast toward its producer.

Two guards recur below:

* **Infallible conversions** (`_INFALLIBLE`). A cast is infallible when *no* value of the
  source type can fail to convert — int→float, and numeric/bool→string. For those, a
  strict `CAST` and a `TRY_CAST` compute the identical array (`safe` only decides what
  happens to a value that fails, and none does), and the cast introduces no null. This is
  what licenses `try_cast_to_strict_when_infallible` and
  `drop_infallible_cast_in_null_check`. Everything else — string→number (parses), float→int
  (rounds, and overflows), int→int (overflows) — is refused.
* **Type preservation.** A folded literal must land on the target type *exactly*. A `Lit`
  carries its Python type, so an `int` literal is Int64 — meaning a fold of
  `cast(x, 'int32')` to an integer literal would silently *widen* the expression to Int64.
  `_cast_literal` therefore only ever folds to a value whose literal type **is** the
  target, and refuses the narrow dtypes outright.

Refused, deliberately (an unprovable rule must not ship):

* folding a cast **of** or **to** a float (other than the exact int→float64) — float→int
  rounds half-to-even in the engine but *errors* in pyarrow's safe cast, and float→string
  formatting is not guaranteed identical across the two Arrow implementations;
* folding `cast('123', 'int64')` and every other string parse — the failure modes (and
  `TRY_CAST`'s NULL) are the engine's to decide at run time;
* `cast(a, T) OP cast(b, T) → a OP b` for a source type `S != T` — a common cast is not
  injective (Int64→Float64 collapses two values above 2^53 onto one double) nor
  order-preserving (Int64→String orders lexically), so stripping it can change the answer.
  The `S == T` case *is* the identity cast that `drop_cast_to_inferred_type` removes;
* pushing a cast into `COALESCE`'s arguments, or into a `CASE`'s non-literal arms — the
  cast still runs per row (it is not a saving), and under vectorized evaluation a strict
  cast on an arm the row never selects can raise an error the original never would;
* `cast(cast(x, W), S) → x` (the lossless widen-then-narrow round trip) — the direction
  that actually occurs, `cast(cast(x, 'int32'), 'int64')`, is the *unsound* one: the inner
  narrowing can overflow.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# `_rewrite_node` (a leaf `Expr → Expr` rule applied to every expression a node carries,
# returning None when nothing changed) is the boolean family's helper — imported, not
# re-implemented.
from batcher.kyber.rules.exprs.guards import schema_rule
from batcher.kyber.rules.extra.boolean_algebra import _rewrite_node
from batcher.plan.expr_ir import Binary, Case, Cast, Expr, IsNotNull, IsNull, Lit
from batcher.plan.expr_ir.core import IsInf, IsNan
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Sort, Window
from batcher.plan.schema import SchemaRef
from batcher.plan.types import DTYPE_REGISTRY, infer_type

__all__ = [
    "canonicalize_cast_dtype_alias",
    "drop_cast_to_inferred_type",
    "drop_infallible_cast_in_null_check",
    "drop_numeric_cast_in_float_predicate",
    "drop_string_cast_in_concat",
    "fold_cast_of_literal",
    "push_cast_into_case_literal_branches",
    "try_cast_to_strict_when_infallible",
]

# Every node whose expressions these rules rewrite (each has a single `.input`, whose
# schema types the expressions).
_NODES = (Filter, Project, Aggregate, Sort, Window)

_INT64, _FLOAT64, _BOOL, _STRING = pa.int64(), pa.float64(), pa.bool_(), pa.string()
# The engine's two numeric types after the FFI widens narrow numerics on input.
_NUMERIC = (_INT64, _FLOAT64)

# The one spelling this optimizer canonicalizes each Arrow type to. `DTYPE_REGISTRY` maps
# several names onto one type (`long`/`int64`, `double`/`float64`, …); two casts that
# differ only in that spelling are the same cast, but no structural comparison (CSE,
# de-duplication, this module's own guards) can see that until they agree textually.
_CANONICAL_NAME: dict[pa.DataType, str] = {
    _INT64: "int64",
    pa.int32(): "int32",
    _FLOAT64: "float64",
    pa.float32(): "float32",
    _BOOL: "bool",
    _STRING: "string",
    pa.date32(): "date",
    pa.timestamp("us"): "timestamp",
}
# alias → canonical spelling, for every registry name that is not already canonical.
_DTYPE_ALIAS: dict[str, str] = {
    name: _CANONICAL_NAME[dtype]
    for name, dtype in DTYPE_REGISTRY.items()
    if dtype in _CANONICAL_NAME and _CANONICAL_NAME[dtype] != name
}

# (source, target) conversions that **cannot fail for any value of the source type**, so a
# strict CAST and a TRY_CAST agree, and neither introduces a null. Int→float rounds but
# never fails; anything→string always has a representation. Everything absent is refused:
# string→number parses, float→int overflows (and rounds), int→int overflows.
_INFALLIBLE: frozenset[tuple[pa.DataType, pa.DataType]] = frozenset(
    {
        (_INT64, _FLOAT64),
        (_INT64, _STRING),
        (_FLOAT64, _STRING),
        (_BOOL, _STRING),
        (_BOOL, _INT64),
    }
)


def _target(expr: Cast) -> pa.DataType | None:
    """The Arrow type a `Cast` produces (`None` for a dtype name we don't know)."""
    return DTYPE_REGISTRY.get(expr.dtype)


def _source(expr: Cast, schema: SchemaRef | None) -> pa.DataType | None:
    """The Arrow type flowing *into* a `Cast` (`None` when not provable)."""
    return None if schema is None else infer_type(expr.input, schema)


def _cast_literal(value: object, target: pa.DataType) -> object | None:
    """The value of `cast(<literal>, target)`, or `None` where the fold is not provably
    exact *and* type-preserving.

    Only the conversions both Arrow implementations agree on bit-for-bit, and only where
    the folded Python value's own literal type **is** the target — so the fold can never
    move the expression's type (an `int` literal is Int64, so no fold to Int32 is possible
    here, and none is attempted).
    """
    if isinstance(value, bool):  # bool before int — bool is an int subclass
        if target == _INT64:
            return int(value)  # arrow: Boolean → 1 / 0
        if target == _STRING:
            return "true" if value else "false"  # arrow: Boolean → "true" / "false"
        return None
    if type(value) is int:
        if target == _STRING:
            return str(value)  # arrow: Int64 → its decimal spelling, sign included
        if target == _FLOAT64 and abs(value) <= 2**53:
            return float(value)  # exactly representable; no rounding to disagree about
        return None
    return None  # a float / string / temporal literal: refused (see the module docstring)


# --- identity casts ---------------------------------------------------------


def _drop_identity_cast(expr: Expr, schema: SchemaRef | None) -> Expr:
    if isinstance(expr, Cast):
        target = _target(expr)
        if target is not None and _source(expr, schema) == target:
            return expr.input
    return expr


@rule(
    name="drop_cast_to_inferred_type",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(Cast,),
    expr_schema=_drop_identity_cast,
)
def drop_cast_to_inferred_type(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop `cast(e, T)` when `e` **already has** type `T` — for any expression `e`, not
    just a bare column.

    The engine's cast kernel hands the array straight back when the types match, so the cast
    is provably a no-op: same values, same nulls, no possible error (a cast cannot fail
    converting a type to itself), and the `try_cast` flag is irrelevant. This generalizes
    `drop_self_cast_in_*` (which matches only `cast(col(c), T)`) to arithmetic, conditionals
    and nested casts — including `cast(try_cast(x, T), T)`, which `normalize.simplify`
    leaves alone because the two flags differ. Fires only when `infer_type` *proves* the
    type; an unknown one is left as it is.
    """
    return schema_rule(node, _drop_identity_cast, carries=(Cast,))


def _canonicalize_alias(expr: Expr) -> Expr:
    if isinstance(expr, Cast) and expr.dtype in _DTYPE_ALIAS:
        return Cast(expr.input, _DTYPE_ALIAS[expr.dtype], try_cast=expr.try_cast)
    return expr


@rule(
    name="canonicalize_cast_dtype_alias",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_canonicalize_alias,
    expr_matches=(Cast,),
)
def canonicalize_cast_dtype_alias(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Rewrite a cast's dtype name to the one canonical spelling of its Arrow type:
    `cast(x, 'long')` → `cast(x, 'int64')`, `'double'` → `'float64'`, `'utf8'` → `'string'`,
    `'datetime'` → `'timestamp'`, and the rest of `DTYPE_REGISTRY`'s aliases.

    Purely a change of name — the two spellings resolve to the *same* `pa.DataType` here and
    to the same type in `bc_arrow::dtype_from_name` on the wire, so the engine computes
    exactly what it did before. What it buys is structural: the IR key of `cast(x, 'long')`
    and `cast(x, 'int64')` now match, so CSE, the de-duplication rules, and every guard that
    compares `expr.dtype` see one cast where they used to see two.
    """
    return _rewrite_node(node, _canonicalize_alias)


# --- literal folding --------------------------------------------------------


def _fold_cast_lit(expr: Expr) -> Expr:
    if not (isinstance(expr, Cast) and isinstance(expr.input, Lit)):
        return expr
    target = _target(expr)
    if target is None:
        return expr
    folded = _cast_literal(expr.input.value, target)
    return expr if folded is None else Lit(folded)


@rule(
    name="fold_cast_of_literal",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_fold_cast_lit,
    expr_matches=(Cast, Lit),
)
def fold_cast_of_literal(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Evaluate `cast(<literal>, T)` at plan time for the conversions `normalize.fold`
    refuses but that are still provably exact: `bool → int64 / string`, `int → string`, and
    `int → float64` **within ±2^53** (where the conversion is exact and no rounding rule can
    differ). A `TRY_CAST` folds identically — none of these conversions can fail, so the
    strict and safe kernels compute the same value.

    Everything else is left to the engine: a *float* source (the engine rounds half-to-even
    to an integer, pyarrow's safe cast errors instead — and float→string formatting is not
    pinned across the two Arrow implementations), a *string* source (a parse, whose failure
    is the engine's to raise or NULL out), and every narrow target (`int32`/`float32`/…),
    because a Python literal cannot carry a narrow type and folding to one would silently
    widen the expression to Int64/Float64.
    """
    return _rewrite_node(node, _fold_cast_lit)


def _cast_arm(arm: Expr, target: pa.DataType, schema: SchemaRef) -> Lit | None:
    """A CASE arm rewritten as a literal of exactly `target`, or `None` if it isn't one."""
    if not isinstance(arm, Lit):
        return None
    if infer_type(arm, schema) == target:
        return arm  # already the target type: the cast of this arm is the identity
    folded = _cast_literal(arm.value, target)
    return None if folded is None else Lit(folded)


def _push_into_case(expr: Expr, schema: SchemaRef | None) -> Expr:
    if not (isinstance(expr, Cast) and isinstance(expr.input, Case)) or schema is None:
        return expr
    case, target = expr.input, _target(expr)
    if target is None:
        return expr
    arms = [_cast_arm(t, target, schema) for _c, t in case.branches]
    arms.append(_cast_arm(case.otherwise, target, schema))
    if any(arm is None for arm in arms):
        return expr
    branches = [(c, arm) for (c, _t), arm in zip(case.branches, arms[:-1], strict=True)]
    return Case(branches, arms[-1])


@rule(
    name="push_cast_into_case_literal_branches",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(Cast,),
    expr_schema=_push_into_case,
)
def push_cast_into_case_literal_branches(
    node: LogicalPlan, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`cast(CASE WHEN c THEN 1 ELSE 2 END, 'float64')` → `CASE WHEN c THEN 1.0 ELSE 2.0 END`
    — push a cast into a `CASE` whose arms are **all literals**, folding each one, so the
    per-row cast over the CASE's output disappears entirely.

    Sound because nothing is left to do at run time: every arm folds to a constant at plan
    time (via `_cast_literal`, so each conversion is exact), which is also what keeps the
    error behavior identical — the usual hazard of pushing a cast into a conditional is that
    a strict cast on an arm a row never selects can raise, and a constant arm cannot raise.
    The result type is unchanged: every arm now has type `T` exactly, so the CASE's type join
    is `T` — the type the cast produced. Fires only when *every* arm (each `then` and the
    `otherwise`) folds; one non-literal arm and the rule declines, because casting it per row
    would be work, not a saving.
    """
    return schema_rule(node, _push_into_case, carries=(Cast,))


# --- infallible-cast simplifications ----------------------------------------


def _infallible(source: pa.DataType | None, target: pa.DataType | None) -> bool:
    """Whether `source → target` cannot fail for any value — so a strict CAST and a TRY_CAST
    agree, and the cast introduces no null of its own."""
    if source is None or target is None:
        return False
    return source == target or (source, target) in _INFALLIBLE


def _try_to_strict(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, Cast)
        and expr.try_cast
        and _infallible(_source(expr, schema), _target(expr))
    ):
        return Cast(expr.input, expr.dtype, try_cast=False)
    return expr


@rule(
    name="try_cast_to_strict_when_infallible",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(Cast,),
    expr_schema=_try_to_strict,
)
def try_cast_to_strict_when_infallible(
    node: LogicalPlan, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`try_cast(x, T)` → `cast(x, T)` when the conversion **cannot fail** — int→float,
    numeric/bool→string, or a cast to the type `x` already has.

    `try_cast` differs from `cast` in exactly one way: arrow's `safe` mode turns a value that
    fails to convert into NULL where the strict kernel raises. When no value of the source
    type can fail, the two kernels compute the identical array, so the flag is meaningless —
    and dropping it canonicalizes the shape, letting the structural rules (CSE, the identity
    and nested-cast collapses, which key on `try_cast`) see one cast instead of two. Fires
    only when `infer_type` proves the source type; a string source (parse) or a float→int
    (overflow) target keeps its `TRY_CAST` semantics untouched.
    """
    return schema_rule(node, _try_to_strict, carries=(Cast,))


def _drop_cast_in_null_check(expr: Expr, schema: SchemaRef | None) -> Expr:
    if isinstance(expr, (IsNull, IsNotNull)) and isinstance(expr.input, Cast):
        inner = expr.input
        if _infallible(_source(inner, schema), _target(inner)):
            return type(expr)(inner.input)
    return expr


@rule(
    name="drop_infallible_cast_in_null_check",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(IsNull, IsNotNull),
    expr_schema=_drop_cast_in_null_check,
)
def drop_infallible_cast_in_null_check(
    node: LogicalPlan, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`is_null(cast(x, T))` → `is_null(x)` (and the `is_not_null` dual) when the cast cannot
    fail.

    An infallible cast maps null to null and every non-null value to a non-null value, so it
    cannot change *which rows are null* — the only thing the predicate looks at. The whole
    cast pass therefore disappears, and the predicate reduces to a bare `is_null(col)` that
    null-count metadata and zone-map pruning can answer without touching data. The guard is
    essential in both directions: a *fallible* strict cast would abort the query (dropping it
    would hide the error), and a fallible `TRY_CAST` **manufactures** nulls —
    `is_null(try_cast('x', 'int64'))` is TRUE where `is_null('x')` is FALSE.
    """
    return schema_rule(node, _drop_cast_in_null_check, carries=(IsNull, IsNotNull))


def _strip_string_cast(operand: Expr, schema: SchemaRef | None) -> Expr:
    """Peel a cast-to-string off a `||` operand when the concat kernel would do that very
    cast itself and the conversion cannot fail."""
    if isinstance(operand, Cast) and _target(operand) == _STRING:
        source = _source(operand, schema)
        if source == _STRING or (source is not None and (source, _STRING) in _INFALLIBLE):
            return operand.input
    return operand


def _drop_concat_cast(expr: Expr, schema: SchemaRef | None) -> Expr:
    if not (isinstance(expr, Binary) and expr.op == "concat"):
        return expr
    left = _strip_string_cast(expr.left, schema)
    right = _strip_string_cast(expr.right, schema)
    if left is expr.left and right is expr.right:
        return expr
    return Binary("concat", left, right)


@rule(
    name="drop_string_cast_in_concat",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(Binary,),
    expr_ops=("concat",),
    expr_schema=_drop_concat_cast,
)
def drop_string_cast_in_concat(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`cast(x, 'string') || y` → `x || y` for an Int64/Float64/Bool/String `x`.

    The engine's `||` casts **both** operands to Utf8 before concatenating, so an explicit
    cast-to-string on an operand is a second, redundant pass over the column: the concat
    kernel then casts an already-Utf8 array (a no-op) instead of the source array. Dropping
    it hands the concat exactly the array it would have produced. Restricted to sources whose
    string conversion **cannot fail**, which is what makes the two spellings agree: the
    explicit cast is strict (it would error) or safe (it would NULL), while the concat's
    internal cast is arrow's default safe cast — a distinction with no difference precisely
    when no value can fail. A temporal or nested source keeps its explicit cast.
    """
    return schema_rule(node, _drop_concat_cast, carries=(Binary,))


def _drop_float_cast_in_predicate(expr: Expr, schema: SchemaRef | None) -> Expr:
    if isinstance(expr, (IsNan, IsInf)) and isinstance(expr.input, Cast):
        inner = expr.input
        if _target(inner) == _FLOAT64 and _source(inner, schema) in _NUMERIC:
            return type(expr)(inner.input)
    return expr


@rule(
    name="drop_numeric_cast_in_float_predicate",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(IsNan, IsInf),
    expr_schema=_drop_float_cast_in_predicate,
)
def drop_numeric_cast_in_float_predicate(
    node: LogicalPlan, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`is_nan(cast(x, 'float64'))` → `is_nan(x)` (and the `is_inf` dual) for a numeric `x`.

    `bc_expr`'s `is_nan`/`is_inf` cast their argument to Float64 themselves before testing
    it, so an explicit Float64 cast in front of them is the same conversion done twice: the
    predicate receives an identical Float64 array either way, and the cast preserves nulls,
    so the null rows are the same too. (For an Int64 `x` both spellings are all-false, which
    is the correct answer — NaN is not an integer notion.) Restricted to a provably Int64 or
    Float64 source: a *string* source would make the explicit cast a parse, which can fail or
    NULL, and the predicate's internal cast would then be reading a different array.
    """
    return schema_rule(node, _drop_float_cast_in_predicate, carries=(IsNan, IsInf))
