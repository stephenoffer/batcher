"""`jit_compilable` must agree with the tier the engine actually runs.

Kyber's cost model prices an expression by asking `kyber.expr_cost.jit_compilable` whether
the Cranelift tier will compile it. That predicate is a Python-side mirror of
`crates/bc-codegen/src/analyze.rs`, and a mirror drifts: the JIT's supported subset grows
in Rust and the cost model keeps pricing an expression as interpreted (or, worse, the
reverse). The engine already reports the truth — it tags each operator with the tier that
ran its per-row work (`op_stats.backend`) — so the mirror can be checked against it.

The contract this pins is **one-directional and deliberately so**:

* `jit_compilable(e) == False` ⟹ the engine must NOT report `"jit"`. A false positive
  under-prices an expression, telling the optimizer a fast path exists where it does not.
  This is the direction that produces wrong plans, and it is asserted strictly.
* `jit_compilable(e) == True` does not guarantee `"jit"`. The JIT decides per *batch* and
  falls back on any batch with nulls in a referenced column. So a `True` here means
  "eligible", and the assertion is that the tier is `"jit"` on null-free input.

`analyze.rs` is the source of truth; when its subset changes, this test is what fails.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.kyber.expr_cost import jit_compilable
from batcher.plan.expr_ir import Binary, Cast

pytest.importorskip("batcher._native", reason="native engine not built")

_N = 4096


def _table() -> pa.Table:
    """Null-free columns of every dtype the JIT accepts, plus ones it does not.

    Values *span* each predicate's threshold. A predicate the zone-map can decide from
    the column's min/max is folded away by `zonemap_prune_filter` before execution, and
    then there is no filter operator left to inspect. `n` is the only nullable column, so
    a null check can be exercised without pushing every other predicate onto the
    interpreter's null fallback.
    """
    return pa.table(
        {
            "x": pa.array(list(range(_N)), pa.int64()),
            "y": pa.array([float(i) for i in range(_N)], pa.float64()),
            "d": pa.array(
                [dt.date(2020, 1, 1) + dt.timedelta(days=i % 28) for i in range(_N)], pa.date32()
            ),
            "s": pa.array([f"ab{i}" for i in range(_N)], pa.string()),
            "n": pa.array([None if i % 3 == 0 else i for i in range(_N)], pa.int64()),
        }
    )


def _expr_backend(expr) -> str:
    """The tier the engine reports for `expr` used as a computed group key, null-free input.

    A **group key**, not a `Filter`, and that is the whole point of the helper. The default
    (streaming) executor deliberately keeps filter and project on the interpreter -- a
    measured choice recorded in `stream::build_with`, where wiring Tier-1 into that path
    read 1.01x over TPC-H with five queries slower, because Arrow's compare and boolean
    kernels are already SIMD and a scalar Cranelift loop has nothing to win against them.
    Asking a filter which tier ran it therefore answers `"interp"` for *every* expression,
    which makes the mirror untestable in both directions rather than making it wrong.

    `ops::compile_agg` does compile, on exactly the subset `analyze.rs` accepts: it puts
    each computed group key and aggregate input through `try_compile_computed`. So a
    predicate spelled as `GROUP BY (x > 100)` reaches the same analysis the cost model
    mirrors, and the reported tier answers the question this file exists to ask.
    """
    ds = bt.from_arrow(_table()).group_by(k=expr).agg(n=bt.count())
    ds.collect()
    ops = [op for op in ds.stats().ops if op.kind == "aggregate"]
    assert ops, "expected an aggregate operator to survive optimization"
    return ops[0].backend


# Expressions `analyze.rs` compiles. Each must actually run on the JIT tier.
_COMPILED = [
    ("int compare", col("x") > 100),
    ("float compare", col("y") < 1000.0),
    ("compound and", (col("x") > 100) & (col("y") < 3000.0)),
    ("compound or", (col("x") > 4000) | (col("y") < 100.0)),
    ("negation", ~(col("x") > 100)),
    ("arithmetic", col("x") * 2 + 1 > 500),
    ("const int divisor", col("x") / 7 > 100),
    ("float divisor", col("x") / 2.5 > 100),
    ("widening cast", Cast(col("x"), "float64", False) > 100.0),
    ("date vs date literal", col("d") <= dt.date(2020, 1, 10)),
    # `analyze.rs` lowers sqrt/floor/ceil/trunc directly and the transcendentals to a
    # libm libcall; `abs` preserves its operand's type.
    ("sqrt", col("y").sqrt() > 10.0),
    ("libm transcendental", col("y").ln() > 1.0),
    ("abs", col("x").abs() > 100),
    ("pow", col("y").pow(2.0) > 100.0),
    # Float division is IEEE — it yields inf/nan and never traps — so a non-constant
    # divisor still compiles, unlike the integer case below.
    ("non-constant float divisor", col("x") / (col("y") + 1.0) > 0.5),
    # Under the Kleene ABI the answer is the operand's validity bit, so `analyze` compiles
    # both by recursing into the operand and yielding Bool.
    ("is_null", col("n").is_null()),
    ("is_not_null", col("n").is_not_null()),
]

# Expressions `analyze.rs` rejects. None may run on the JIT tier.
_INTERPRETED = [
    ("string literal compare", col("s") == "ab1"),
    ("string function", col("s").str.contains("b1")),
    ("regex", col("s").str.regexp_matches("^ab1")),
    # Integer `sdiv` traps on a zero divisor, so a non-constant integer divisor stays on
    # the interpreter. Built from the raw IR node: `Expr.__truediv__` casts to float64
    # first, which would make it the (compilable) IEEE float division above.
    ("non-constant int divisor", Binary("div", col("x"), col("x") + 1) > 0),
    # Math the JIT deliberately does not lower, to keep bit-for-bit parity with the
    # interpreter oracle: `round` (rounding mode), `sign` (select), `cbrt` (1 ULP).
    #
    # Spelled bare rather than compared to a literal. Kyber rewrites a monotonic function
    # under a comparison into a comparison on the bare column -- `round(y) > 100.0` becomes
    # `y >= 100.5` -- so the compared form deletes the very node this case exists to check
    # and the engine then (correctly) compiles the plain comparison left behind. The bare
    # key reaches the JIT as written.
    ("round", col("y").round()),
    ("sign", col("y").sign()),
    ("cbrt", col("y").cbrt()),
    # float64 -> int64: Arrow's rounding differs from cranelift's `fcvt`, so it falls
    # back. (An int64 -> int64 `try_cast` would be a self-cast the optimizer deletes.)
    ("try_cast", Cast(col("y"), "int64", True) > 100),
    ("cast to string", Cast(col("x"), "string", False) == "100"),
]


@pytest.mark.integration
@pytest.mark.parametrize(("label", "predicate"), _COMPILED, ids=[c[0] for c in _COMPILED])
def test_predicate_kyber_thinks_compiles_actually_compiles(label, predicate):
    assert jit_compilable(predicate), f"{label}: the cost model should call this compilable"
    assert _expr_backend(predicate) == "jit", (
        f"{label}: Kyber prices this as compiled, but the engine ran it on the interpreter"
    )


@pytest.mark.integration
@pytest.mark.parametrize(("label", "predicate"), _INTERPRETED, ids=[c[0] for c in _INTERPRETED])
def test_predicate_kyber_thinks_is_interpreted_never_compiles(label, predicate):
    assert not jit_compilable(predicate), f"{label}: the cost model should call this interpreted"
    assert _expr_backend(predicate) != "jit", (
        f"{label}: the engine compiled an expression Kyber prices as interpreted "
        f"(a false negative — safe, but the cost table is now stale)"
    )


@pytest.mark.integration
def test_no_expression_is_priced_as_compiled_but_interpreted_by_the_engine():
    """The unsafe direction, swept over every case at once.

    A `True` from `jit_compilable` that the engine does not honour makes Kyber under-price
    the expression by `jit_speedup`, which can suppress a filter split that would have
    paid for itself. Nothing in either list may do that.
    """
    for label, predicate in _COMPILED + _INTERPRETED:
        if jit_compilable(predicate):
            assert _expr_backend(predicate) == "jit", label
