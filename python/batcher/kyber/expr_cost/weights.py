"""Per-node evaluation costs, and the traversal that reaches every sub-expression.

The numbers are the data plane's per-row work for one `Expr` node, in abstract
work-units normalized so that **one interpreted numeric comparison over one row = 1.0**.
They are grouped by family (binary operator, string function, math function, date part,
and a by-class-name table for everything else) so a new node family is priced by adding
an entry, never by editing the traversal.

Costs describe the Tier-0 Arrow-kernel path. `jit` decides whether the Cranelift tier
runs the expression instead, and `model` applies the resulting speedup.

Where the numbers come from. The families below were **measured**, not guessed: each
function was run as the sole expression of a projection over a million rows in a fresh
process, and a bare column projection was subtracted so only the function's own work is
counted. Some results are unintuitive and are the reason a hand-guessed table is not good
enough — `regexp_matches` is only ~3x `contains` (Arrow's RE2 prefilters on literals),
while `sha256` is ~7x a regex and `levenshtein` ~5x, because their per-row work is a real
O(n) or O(n^2) loop. Even `length` costs ~40x a comparison, since decoding string offsets
dominates the operation itself.

Two caveats. String costs scale with string length (measured on 16-char values), and a
function's cost varies with its arguments (an anchored regex is far cheaper than a
backtracking one). A single scalar per function cannot capture that, so these are
*rankings* good to a small factor, not predictions. The systematic component of the
residual error is absorbed by `jit_speedup`, which `kyber.calibration` fits from the tier
tag the engine reports on every operator.
"""

from __future__ import annotations

import dataclasses

from batcher.plan.expr_ir import (
    Aliased,
    Binary,
    Case,
    Col,
    Expr,
    InList,
    Lit,
    Math2Expr,
    MathExpr,
)
from batcher.plan.expr_ir.node_base import IRNode

__all__ = ["own_cost", "sub_exprs"]

_LEAF_COST = 0.2  # a column buffer read; a literal broadcasts for free
_DEFAULT_COST = 5.0  # an unrecognized node: assume moderately expensive (safe direction)

# Binary operators. Comparisons/arithmetic are single instructions; `concat`
# allocates a string per row, `add_months` walks a calendar.
BINARY_COST: dict[str, float] = {
    "eq": 1.0,
    "ne": 1.0,
    "lt": 1.0,
    "le": 1.0,
    "gt": 1.0,
    "ge": 1.0,
    "add": 1.0,
    "sub": 1.0,
    "mul": 1.0,
    "div": 3.0,  # hardware divide is multi-cycle
    "mod": 3.0,
    "and": 0.5,
    "or": 0.5,
    "bit_and": 1.0,
    "bit_or": 1.0,
    "bit_xor": 1.0,
    "shift_left": 1.0,
    "shift_right": 1.0,
    "concat": 12.0,
    "add_months": 8.0,
}

# String functions. Measured (see the module docstring); the floor for *any* of them is
# the ~15 units it costs merely to decode a row's string offsets and bytes.
_STR_DEFAULT = 20.0
_STR_COST: dict[str, float] = {
    # Buffer/offset reads — dominated by string decoding, not by the operation.
    "len": 14.5,
    "bit_length": 14.5,
    "octet_length": 14.5,
    "ascii": 14.5,
    # Prefix/suffix tests short-circuit on the first bytes.
    "starts_with": 8.0,
    "ends_with": 8.0,
    # Substring search over the whole value.
    "contains": 20.0,
    "position": 20.0,
    "hash64": 12.0,
    "xxhash64": 12.0,
    # Allocating transforms: a new string per row.
    "upper": 14.5,
    "lower": 14.5,
    "initcap": 18.0,
    "trim": 12.0,
    "l_trim": 12.0,
    "r_trim": 12.0,
    "reverse": 14.5,
    "right": 12.0,
    "substr": 12.0,
    "repeat": 25.0,
    "lpad": 22.0,
    "rpad": 22.0,
    "replace": 22.0,
    "translate": 22.0,
    "overlay": 22.0,
    "split": 35.0,
    "split_part": 25.0,
    "substring_index": 25.0,
    # Glob matching walks the value; `ilike` also case-folds.
    "like": 28.0,
    "ilike": 34.0,
    # Encodings.
    "hex": 25.0,
    "unhex": 25.0,
    "base64": 25.0,
    "from_base64": 25.0,
    # Regex compiles once; the match is a per-row automaton walk. Cheaper than intuition
    # suggests because RE2 prefilters on required literals.
    "regexp_matches": 48.0,
    "regexp_count": 48.0,
    "regexp_extract": 55.0,
    "regexp_extract_all": 70.0,
    "regexp_replace": 65.0,
    "regexp_replace_all": 75.0,
    # Edit distance is a real O(len^2) inner loop — far pricier than a regex.
    "levenshtein": 230.0,
    "soundex": 40.0,
    # Cryptographic digests: a full compression function per row, the priciest string ops.
    "crc32": 60.0,
    "md5": 155.0,
    "sha1": 210.0,
    "sha256": 325.0,
    # JSON extraction parses a document per row.
    "json_extract_bool": 90.0,
    "json_extract_int": 90.0,
    "json_extract_float": 90.0,
    "json_extract_string": 100.0,
}

# Unary math. Measured: a hardware `sqrt` is barely more than an add, and even a libm
# `log` is only ~3x a comparison — nothing like the string family. The transcendentals
# that lower to a libm libcall are the priciest, and the ones the JIT cannot lower (see
# `jit`) pay the interpreter's dispatch on top.
_MATH_DEFAULT = 5.0
_MATH_COST: dict[str, float] = {
    "abs": 1.0,
    "sign": 1.0,
    "ceil": 1.2,
    "floor": 1.2,
    "trunc": 1.2,
    "round": 1.5,
    "degrees": 1.2,
    "radians": 1.2,
    "bit_count": 1.5,
    "sqrt": 1.5,
    "cbrt": 6.0,
    "exp": 4.0,
    "ln": 4.0,
    "log2": 4.0,
    "log10": 4.0,
    "sin": 6.0,
    "cos": 6.0,
    "tan": 6.0,
    "cot": 7.0,
    "asin": 6.0,
    "acos": 6.0,
    "atan": 6.0,
    "sinh": 6.0,
    "cosh": 6.0,
    "tanh": 6.0,
    "factorial": 15.0,
}

# Date-part extraction is a divmod on the epoch value, except where it consults a
# calendar table or formats a name.
_DATE_DEFAULT = 3.0
_DATE_COST: dict[str, float] = {
    "dayname": 8.0,
    "monthname": 8.0,
    "last_day": 5.0,
    "days_in_month": 5.0,
    "is_leap_year": 5.0,
}

# Flat per-node costs keyed by class name, so node families added later (list, struct,
# map, media) are priced without importing every class into this module.
_BY_CLASS_NAME: dict[str, float] = {
    "IsNull": 0.5,
    "IsNotNull": 0.5,
    "IsNan": 0.5,
    "IsInf": 0.5,
    "Not": 0.5,
    "Cast": 2.0,
    "Coalesce": 1.0,
    "NullIf": 1.5,
    "Greatest": 1.0,
    "Least": 1.0,
    "Array": 2.0,
    "MakeStruct": 2.0,
    "Sequence": 5.0,
    "StructField": 1.0,
    "ListJoin": 20.0,
    "MapFunc": 10.0,
    # Nested/list kernels walk a child buffer per row.
    "ListFunc": 15.0,
    "ListGet": 5.0,
    "ListContains": 15.0,
    "ListPosition": 15.0,
    "ListSlice": 10.0,
    "ListBinary": 20.0,
    "ListSet": 25.0,
    # These evaluate a sub-expression per *element*, not per row.
    "ListTransform": 40.0,
    "ListFilter": 40.0,
    # Temporal formatting/parsing walks a format string per row.
    "Strftime": 40.0,
    "Strptime": 45.0,
    "ConvertTimezone": 15.0,
    "DateTrunc": 5.0,
    "DateOffset": 8.0,
    "WindowStart": 5.0,
    "WindowBuckets": 5.0,
    # Media decode dwarfs every scalar op; costing it high is what makes Kyber push
    # filters below an image/audio/video expression.
    "ImageFunc": 500.0,
    "AudioFunc": 500.0,
    "VideoFunc": 500.0,
}


def own_cost(expr: Expr) -> float:
    """The cost of evaluating `expr`'s own node, excluding its sub-expressions.

    Args:
        expr: The scalar expression node to price.

    Returns:
        Cost in work-units, where one numeric comparison over one row is 1.0.
    """
    if isinstance(expr, (Col, Lit)):
        return _LEAF_COST
    if isinstance(expr, Binary):
        return BINARY_COST.get(expr.op, _DEFAULT_COST)
    if isinstance(expr, Aliased):
        return 0.0  # transparent in the IR
    if isinstance(expr, InList):
        # Lowered to a hash-set probe; a handful of values stays a compare chain.
        return min(6.0, 1.0 + 0.3 * len(expr.values))
    if isinstance(expr, Case):
        # Every branch condition and its result are evaluated (no short-circuit); the
        # per-branch selection itself is what is counted here.
        return 0.5 * (len(expr.branches) + 1)
    if isinstance(expr, MathExpr):
        return _MATH_COST.get(expr.fn, _MATH_DEFAULT)
    if isinstance(expr, Math2Expr):
        return 10.0
    cls = type(expr).__name__
    if cls == "StrFunc":
        return _STR_COST.get(expr.fn, _STR_DEFAULT)
    if cls == "DateFunc":
        return _DATE_COST.get(expr.fn, _DATE_DEFAULT)
    return _BY_CLASS_NAME.get(cls, _DEFAULT_COST)


def sub_exprs(expr: Expr) -> tuple[Expr, ...]:
    """The immediate sub-expressions of `expr`.

    `IRNode`s are dataclasses, so their sub-expressions are found by walking the
    declared fields (recursing into lists/tuples, which is how `Case.branches` and
    `MakeStruct.fields` carry their pairs). The three hand-written `Expr` classes use
    `__slots__` and are matched explicitly — a generic `vars()` walk would silently
    miss `InList.input` and report the node as a leaf.

    Args:
        expr: The scalar expression node to descend into.

    Returns:
        Its immediate sub-expressions, in declaration order.
    """
    if isinstance(expr, InList):
        return (expr.input,)
    if isinstance(expr, Aliased):
        return (expr.inner,)
    if isinstance(expr, Lit):
        return ()
    if isinstance(expr, IRNode):
        out: list[Expr] = []
        for f in dataclasses.fields(expr):
            _collect_exprs(getattr(expr, f.name, None), out)
        return tuple(out)
    return ()


def _collect_exprs(value: object, out: list[Expr]) -> None:
    """Append every `Expr` reachable from `value` (a field, or a list/tuple of them)."""
    if isinstance(value, Expr):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_exprs(item, out)
