"""Unsatisfiable predicates — empty out a filter no value can satisfy, from one conjunct alone.

The sibling of `predicate_infer`, reasoning the same way (top-level conjunction, 3VL filter
semantics, no statistics) but from a different source of truth. `predicate_infer` compares a
column's *bounds against each other*, so it needs two conjuncts to contradict; these compare a
single conjunct against **the set of values its own operator or function can produce at all**,
which needs no sibling and no metadata.

`filter_arithmetic_contradiction` covers the operators:

* **remainder magnitude.** Batcher's `%` is a truncated remainder — `7 % 3` is `1`, `-7 % 3`
  is `-1`, `7 % -3` is `1` — so `x % k` always lies strictly inside `(-|k|, |k|)`. A predicate
  comparing it against anything at or beyond that is unsatisfiable: `id % 10 = 15`,
  `id % 10 >= 10`, `bucket % 4 < -3`.
* **multiplication parity.** The engine's `*` wraps mod 2^64, and `x * k` for `k = 2^v * odd`
  is always a multiple of `2^v` — the odd factor is a bijection, so the reachable set is
  exactly the multiples of `2^v`. `x * 2 = 7` therefore matches no row at all, and neither
  does `x * 4 = 6`. (An *odd* `k` is a bijection outright, so it is never contradictory —
  `sargable.py` inverts those instead.)
* **bit masks.** `x & m` can only set bits `m` sets, and `x | m` always sets every bit `m`
  sets. So `flags & 12 = 3` and `flags | 12 = 3` are both unsatisfiable, whatever `flags` is.

Each is a statement about *every* value the left side can take, so a conjunct that fails it is
never TRUE — and a `Filter` keeps a row only where the predicate is TRUE. Nulls need no
separate argument for exactly that reason: a null conjunct is not TRUE either, so the filter
keeps nothing whether the row is null or merely non-matching. That is what makes the whole
predicate collapse to constant `FALSE` sound, which `filter_false_to_empty` then turns into the
canonical empty relation so the operators above it fold away too.

The type guard is load-bearing and is why this reads the node's schema: `%` on a float has the
same magnitude property but `*` has no parity one, and a bit operation over a non-integer is
not this shape at all. Anything not provably integer is left alone.

`filter_arithmetic_contradiction` also refutes a **widening cast against a literal no integer
can equal**: `cast(i AS DOUBLE) = 5.5`. That one is left alone by `exprs/cast_unwrap`, correctly
and for a stated reason -- the fold is to a constant, which differs from the original on a null
row, so it "is only correct at the top of a filter". This is the top of a filter.

A second rule, `filter_function_range_contradiction`, refutes from the **image of a function**
rather than the range of an operator: `month(ts) = 13`, `hour(ts) >= 24`, `length(s) < 0`,
`abs(x) = -5`, `sign(x) = 5`, and `upper(name) = 'john'` are unsatisfiable in every table there
has ever been. It is separate from the arithmetic one because it needs strictly less: an image
bound is a property of the function, so no schema is resolved at all.

**One rule per source of truth, not one per invariant.** The expression-algebra families split a
rewrite per operator because the driver fuses declared leaves into one shared traversal; a
`Filter` node rule gets no such fusion, so a name per invariant would mean walking the same
predicate once per invariant for one conclusion.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# `_oriented` resolves `cast(<int> AS float) <op> <float literal>` in either operand order --
# the sibling family's helper, imported rather than re-implemented.
from batcher.kyber.rules.exprs.cast_unwrap import _oriented
from batcher.kyber.rules.exprs.guards import is_integer, node_schema
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.logical import Filter, LogicalPlan
from batcher.plan.schema import SchemaRef

__all__ = ["filter_arithmetic_contradiction", "filter_function_range_contradiction"]

#: Comparisons whose operands may be swapped by flipping the operator, so a conjunct written
#: `15 = id % 10` is analyzed as the `id % 10 = 15` spelling without a second table.

#: The arithmetic operators this rule knows an invariant for.
_ARITH = frozenset({"mod", "mul", "bit_and", "bit_or"})

#: Two's-complement 64-bit mask, for reasoning about the bitwise operators the way the engine
#: computes them (a negative literal is its 64-bit pattern).
_U64 = (1 << 64) - 1


def _int_lit(expr: Expr) -> int | None:
    """The value of a plain integer literal, else ``None`` (a `bool` is not one)."""
    if isinstance(expr, Lit) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


def _decompose(conjunct: Expr) -> tuple[str, str, Expr, int, int] | None:
    """`(arith_op, cmp_op, operand, k, lit)` for a comparison between `operand <arith> k` and
    an integer literal, normalized so the arithmetic reads as the left-hand side."""
    if not isinstance(conjunct, Binary) or conjunct.op not in COMPARISON_FLIP:
        return None
    for inner, other, op in (
        (conjunct.left, conjunct.right, conjunct.op),
        (conjunct.right, conjunct.left, COMPARISON_FLIP[conjunct.op]),
    ):
        lit = _int_lit(other)
        if lit is None or not isinstance(inner, Binary) or inner.op not in _ARITH:
            continue
        # `mod` is not commutative, so its divisor must be on the right; `mul`/`bit_and`/
        # `bit_or` are, so the constant may sit on either side.
        candidates = (
            ((inner.left, inner.right),)
            if inner.op == "mod"
            else ((inner.left, inner.right), (inner.right, inner.left))
        )
        for operand, constant in candidates:
            k = _int_lit(constant)
            if k is not None:
                return inner.op, op, operand, k, lit
    return None


def _two_adic(value: int) -> int:
    """How many times 2 divides `value`, capped at 64 (and 64 for zero).

    64 is the cap because the arithmetic is mod 2^64: past that every value is congruent to a
    multiple of 2^64, which is 0. Treating zero as 64 is what makes `x * 0 = lit` come out
    right — the reachable set is `{0}`, so any non-zero `lit` has a strictly smaller valuation
    and is refuted, while `lit = 0` is satisfied by every row.
    """
    if value == 0:
        return 64
    count = 0
    while count < 64 and value % 2 == 0:
        value //= 2
        count += 1
    return count


def _is_equality(expr: Expr) -> bool:
    """Whether `expr` is an equality — the cheap shape gate for the cast refutation."""
    return isinstance(expr, Binary) and expr.op == "eq"


def _range_refutes(op: str, low: float | None, high: float | None, lit: float) -> bool:
    """Whether `v <op> lit` is unsatisfiable for every `v` in `[low, high]`.

    `None` at either end means unbounded there. `ne` is never refuted: a literal the range
    cannot reach makes "not equal" true on every row rather than false on all of them.

    Every comparison here is a plain Python one, which is what makes a NaN literal safe: each
    of them answers `False` against a NaN, so nothing is refuted — and that is the correct
    answer, since the engine's float comparisons put a NaN *above* every finite value, so
    `abs(x) < NaN` really is true rather than impossible.
    """
    if op == "eq":
        return (low is not None and lit < low) or (high is not None and lit > high)
    if op == "gt":
        return high is not None and lit >= high
    if op == "ge":
        return high is not None and lit > high
    if op == "lt":
        return low is not None and lit <= low
    if op == "le":
        return low is not None and lit < low
    return False


def _modulo_impossible(op: str, k: int, lit: int) -> bool:
    """Whether `x % k <op> lit` is unsatisfiable, from `|x % k| <= |k| - 1` alone.

    A zero divisor yields NULL on every row rather than a value, so it is not refuted here —
    NULL is already not TRUE, and leaving it alone keeps this rule's claim to be about the
    *range* of a real result.
    """
    if k == 0:
        return False
    high = abs(k) - 1  # the largest magnitude a truncated remainder can reach
    return _range_refutes(op, -high, high, lit)


def _multiply_impossible(op: str, k: int, lit: int) -> bool:
    """Whether `x * k = lit` is unsatisfiable, because `x * k` is always a multiple of the
    largest power of two dividing `k` and `lit` is not one."""
    return op == "eq" and _two_adic(k) > _two_adic(lit)


def _bitwise_impossible(arith: str, op: str, k: int, lit: int) -> bool:
    """Whether `x & k = lit` or `x | k = lit` is unsatisfiable, from the bits the result is
    forced to have. Compared as 64-bit patterns, which is how the engine computes them, so a
    negative constant behaves as its two's-complement bits."""
    if op != "eq":
        return False
    mask, value = k & _U64, lit & _U64
    if arith == "bit_and":
        return bool(value & ~mask & _U64)  # a bit set outside the mask is unreachable
    return bool(mask & ~value & _U64)  # `|` forces every mask bit, so a missing one is unreachable


def _cast_refutes(conjunct: Expr, schema: SchemaRef | None) -> bool:
    """Whether `cast(<integer col> AS double) = <fractional literal>` — never TRUE.

    `exprs/cast_unwrap` unwraps the *ordered* comparisons against a fractional literal
    (`i > 3.5` is `i > 3`) and deliberately leaves equality alone, for a reason its docstring
    states precisely: the fold would be to a constant, and the original yields NULL on a null
    row where a constant does not — so it "is only correct at the top of a filter". This is the
    top of a filter. No integer equals 3.5, so the conjunct is TRUE on no row, and NULL and
    FALSE both drop a row here, which is exactly the context that makes the refutation sound.

    `<>` is not refuted, and not by omission: `i <> 3.5` is TRUE on every non-null row, so
    rewriting it to a constant TRUE would *keep* the null rows a filter must drop.
    """
    if not isinstance(conjunct, Binary) or conjunct.op != "eq":
        return False
    resolved = _oriented(conjunct, schema)
    if resolved is None:
        return False
    _inner, op, value = resolved
    # A float64 above 2**53 has no fractional part, so `is_integer()` already confines this to
    # the window where the literal names a value strictly between two integers.
    return op == "eq" and not value.is_integer()


def _never_true(conjunct: Expr, schema: SchemaRef | None) -> bool:
    """Whether `conjunct` is TRUE on no row at all, by one of the arithmetic invariants."""
    if _cast_refutes(conjunct, schema):
        return True
    found = _decompose(conjunct)
    if found is None:
        return False
    arith, op, operand, k, lit = found
    # Every invariant here is about integers: `*` has no parity argument over floats, and a
    # bit operation over a non-integer is not this shape. `%` would survive on a float, but
    # gating the whole rule on one guard is what keeps the claim simple.
    if not is_integer(operand, schema):
        return False
    if arith == "mod":
        return _modulo_impossible(op, k, lit)
    if arith == "mul":
        return _multiply_impossible(op, k, lit)
    return _bitwise_impossible(arith, op, k, lit)


# --- the image of a function, rather than the range of an operator -------------------

#: The interval each function's result is confined to, whatever its input — `None` at an end
#: meaning unbounded there. These are properties of the function, not of the data, so they need
#: no schema and no statistics: `month(ts) = 13` matches nothing in any table there has ever
#: been. Ranges verified against the engine rather than assumed, which is why `day_of_week`
#: reads 0-6 (Sunday = 0) while `isodow` reads 1-7 (Monday = 1) — the two conventions live
#: side by side here, and guessing either would have made the rule wrong at one end.
#:
#: **No float function carries an upper bound here, and that is deliberate.** `sin`/`cos` look
#: like obvious entries at `[-1, 1]`, and the upper half would be wrong: `sin(NaN)` is NaN, and
#: the engine's total order places a NaN *above* every finite value, so `sin(x) > 1` is TRUE on a
#: NaN row rather than impossible. Refuting it would drop that row. Every float entry below is
#: therefore lower-bounded only — a direction NaN cannot violate, since a NaN is never *below*
#: anything. The integer-valued entries (the calendar parts, the lengths) have no NaN to worry
#: about, which is what lets them bound both ends.
#: Keyed by `Expr` type first, then function name, and the nesting is deliberate rather than
#: incidental. Probing for a function name with `getattr(expr, "fn", None)` looks harmless and is
#: not: `Expr.__getattr__` builds a "did you mean ..." suggestion with `difflib` before raising
#: the `AttributeError` that a default then swallows, so every miss pays a fuzzy match over the
#: whole expression vocabulary. Measured at **+95% planning time** for this one rule before the
#: index was reshaped. Looking the *type* up first means `.fn` is only ever read on a type that
#: has one. (`optimizer.expr_dispatch.discriminator` avoids the same trap by caching the answer
#: per type.)
_IMAGE: dict[type, dict[str, tuple[float | None, float | None]]] = {}


def _register_images() -> None:
    """Populate `_IMAGE`, keyed by `Expr` type and then by function name."""
    from batcher.plan.expr_ir.core import MathExpr
    from batcher.plan.expr_ir.func_nodes import DateFunc, ListFunc, StrFunc

    calendar = {
        "month": (1, 12),
        "quarter": (1, 4),
        "day": (1, 31),
        "hour": (0, 23),
        "minute": (0, 59),
        # No leap seconds in an Arrow timestamp, so 59 really is the maximum.
        "second": (0, 59),
        "week": (1, 53),  # ISO week
        "day_of_week": (0, 6),  # Sunday = 0
        "isodow": (1, 7),  # Monday = 1
        "day_of_year": (1, 366),  # a leap year has 366
    }
    _IMAGE[DateFunc] = dict(calendar)
    # Every counting function: a length or an occurrence count is never negative and has no
    # upper bound. `len` is the character length, `octet_length` the byte length, and
    # `regexp_count` the number of matches; a `ListFunc` `len` counts elements.
    _IMAGE[StrFunc] = dict.fromkeys(("len", "octet_length", "regexp_count"), (0, None))
    _IMAGE[ListFunc] = {"len": (0, None)}
    # `abs` is non-negative for every input the engine has: integer `abs` *saturates* rather
    # than wrapping (`abs(INT64_MIN)` answers `INT64_MAX`), and `abs(NaN)` is NaN, which the
    # engine's total order places above every finite value — so above zero either way.
    _IMAGE[MathExpr] = {"abs": (0, None)}
    # `sign` answers -1, 0, or 1 — including 0 for a **NaN**, so it never returns one and the
    # upper bound is safe here in a way it would not be for a trig function.
    _IMAGE[MathExpr]["sign"] = (-1, 1)
    # `sqrt` answers NaN for a negative input (again, above every finite value) and `-0.0` for
    # `-0.0`, which equals `0.0` — so nothing it returns is below zero. Measured, because the
    # signed zero is exactly where a guess would go wrong.
    _IMAGE[MathExpr]["sqrt"] = (0, None)
    # `exp` is mathematically positive and *reaches* zero by underflow (`exp(-1e300)` is `0.0`),
    # so the bound is the inclusive zero rather than an exclusive one.
    _IMAGE[MathExpr]["exp"] = (0, None)


_register_images()

#: Case-folding functions, and the character class their result can never contain. `upper(s)`
#: cannot contain an ASCII lowercase letter: an ASCII lowercase input maps to its uppercase,
#: and no other character's uppercase mapping *is* an ASCII lowercase letter (an uppercase
#: expansion is uppercase — `ß` becomes `SS`). Restricted to ASCII on purpose, since the
#: general Unicode case mappings are locale-sensitive at the edges and this needs no such
#: assumption to catch the shape that actually happens: `WHERE upper(name) = 'john'`.
_CASE_FOLD = {"upper": str.islower, "lower": str.isupper}


def _image_refutes(conjunct: Expr) -> bool:
    """Whether `conjunct` compares a function against a value outside its possible image."""
    if not isinstance(conjunct, Binary) or conjunct.op not in COMPARISON_FLIP:
        return False
    for call, other, op in (
        (conjunct.left, conjunct.right, conjunct.op),
        (conjunct.right, conjunct.left, COMPARISON_FLIP[conjunct.op]),
    ):
        # Type first, so `.fn` is read only on a type that has one -- see `_IMAGE`.
        by_fn = _IMAGE.get(type(call))
        if by_fn is None or not isinstance(other, Lit):
            continue
        fn = call.fn
        bounds = by_fn.get(fn)
        numeric = isinstance(other.value, (int, float)) and not isinstance(other.value, bool)
        if bounds is not None and numeric and _range_refutes(op, *bounds, other.value):
            return True
        # Any character of the forbidden class is enough: the result cannot contain one, so it
        # cannot equal a literal that does.
        if (
            op == "eq"
            and fn in _CASE_FOLD
            and isinstance(other.value, str)
            and any(_CASE_FOLD[fn](ch) and ch.isascii() for ch in other.value)
        ):
            return True
    return False


@rule(name="filter_function_range_contradiction", phase=Phase.NORMALIZE, matches=(Filter,))
def filter_function_range_contradiction(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Empty out a filter comparing a function against a value it can never return:
    `month(ts) = 13`, `hour(ts) >= 24`, `length(s) < 0`, `abs(x) = -5`, `sign(x) = 5`, and
    `upper(name) = 'john'` (an uppercased string contains no ASCII lowercase letter).

    These are properties of the *function*, not of the data, so unlike `zonemap_prune_filter`
    this needs no statistics and unlike `filter_arithmetic_contradiction` it needs no schema —
    the image bound holds for every input in every table. The conjunct is therefore TRUE on no
    row, and a Filter keeps a row only where the predicate is TRUE, so the whole conjunction
    collapses to constant `FALSE` and `filter_false_to_empty` makes it the empty relation.

    `<>` is deliberately not refuted: a value outside the image makes "not equal" true on
    *every* row, which is the opposite conclusion. Returns None when no conjunct violates an
    image bound; idempotent, since the `FALSE` it produces carries no call left to check.
    """
    if isinstance(node.predicate, Lit):
        return None
    if any(_image_refutes(conjunct) for conjunct in split_conjuncts(node.predicate)):
        return Filter(node.input, Lit(False))
    return None


@rule(name="filter_arithmetic_contradiction", phase=Phase.NORMALIZE, matches=(Filter,))
def filter_arithmetic_contradiction(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Empty out a filter one of whose conjuncts no value can satisfy: `id % 10 = 15`,
    `id % 10 >= 10`, `x * 2 = 7`, `flags & 12 = 3`, `flags | 12 = 3` → constant `FALSE`.

    Each conjunct is checked against the range or bit lattice its own arithmetic can produce,
    so no sibling conjunct and no statistics are needed. `WHERE p1 AND ... AND pn` keeps a row
    only where every conjunct is TRUE, so one never-TRUE conjunct empties the whole filter —
    including on null rows, which are not TRUE either. Returns None unless the operand is
    provably integer-typed and one invariant is actually violated; idempotent, because the
    `FALSE` it produces carries no arithmetic left to refute.
    """
    if isinstance(node.predicate, Lit):
        return None  # already constant-folded; nothing to refute and nothing to rebuild
    conjuncts = split_conjuncts(node.predicate)
    # The schema is resolved only once a conjunct has the right *shape*, since `node_schema`
    # rebuilds a pyarrow schema up the plan and almost no filter carries this shape.
    # The cast shape is admitted here as well as the arithmetic one; both need the schema, and
    # resolving it once for either is what keeps this rule to a single pass.
    shaped = [c for c in conjuncts if _decompose(c) is not None or _is_equality(c)]
    if not shaped:
        return None
    schema = node_schema(node)
    if schema is None:
        return None
    if any(_never_true(conjunct, schema) for conjunct in shaped):
        return Filter(node.input, Lit(False))
    return None
