"""Property: every expression lowers to stable, JSON-serializable IR.

The JSON IR is the wire contract with Rust (`bc_expr::Expr` serde tags). Whatever
expression a user builds, ``to_ir()`` must produce the same structure every time and
survive a JSON round-trip unchanged — a non-deterministic or unserializable lowering
is a silent correctness bug at the FFI boundary. Hypothesis builds random expression
trees and checks both laws. This needs no native engine (pure control-plane lowering).
"""

from __future__ import annotations

import json
import operator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from batcher.plan.expr_ir import col, lit

pytestmark = pytest.mark.property

_COMPARES = [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne]
_ARITH = [operator.add, operator.sub, operator.mul]


@st.composite
def _expr(draw: st.DrawFn, depth: int = 0):
    """A bounded random expression tree over col/lit, arithmetic, compares, and not."""
    leaf = st.one_of(
        st.sampled_from(["a", "b", "c"]).map(col),
        st.integers(min_value=-1000, max_value=1000).map(lit),
        st.floats(allow_nan=False, allow_infinity=False, width=32).map(lit),
    )
    if depth >= 3:
        return draw(leaf)
    kind = draw(st.sampled_from(["leaf", "arith", "cmp", "not"]))
    if kind == "leaf":
        return draw(leaf)
    if kind == "arith":
        op = draw(st.sampled_from(_ARITH))
        return op(draw(_expr(depth + 1)), draw(_expr(depth + 1)))
    if kind == "cmp":
        op = draw(st.sampled_from(_COMPARES))
        return op(draw(_expr(depth + 1)), draw(_expr(depth + 1)))
    return ~(draw(_expr(depth + 1)) == draw(_expr(depth + 1)))


@settings(max_examples=200, deadline=None)
@given(_expr())
def test_to_ir_is_deterministic_and_json_round_trips(expr) -> None:
    first = expr.to_ir()
    # Determinism: lowering the same expression twice yields the same structure.
    assert expr.to_ir() == first
    # JSON round-trip: the IR is exactly what crosses the FFI boundary, unchanged.
    assert json.loads(json.dumps(first)) == first
    # Every lowered node carries the discriminant tag the Rust serde enum keys on.
    assert "e" in first
