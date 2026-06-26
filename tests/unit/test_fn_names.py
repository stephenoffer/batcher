"""The function-name vocabularies catch typos at construction, and stay exhaustive.

`fn_names` is the documented home for the ``fn`` discriminator each function node
carries. The node base validates ``fn`` against the family vocabulary at build time,
turning an unknown function into a clear `PlanError` instead of an opaque engine
error — the guard that keeps the vocabulary honest as functions scale.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.expr_ir.fn_names import LIST_FNS, MATH_FNS, STR_FNS, MapFn, Math2Fn
from batcher.plan.expr_ir.func_nodes import ListFunc, MapFunc, StrFunc


def test_known_functions_construct() -> None:
    # A representative valid fn from each validated family builds without error.
    StrFunc("contains", _col(), pattern="x")
    ListFunc("sum", _col())
    MathExpr("sqrt", _col())
    MapFunc(MapFn.MAP_KEYS, _col())  # StrEnum member is accepted (it is a str)


@pytest.mark.parametrize(
    ("ctor", "bad"),
    [
        (lambda: StrFunc("uppercase", _col()), "uppercase"),  # typo for "upper"
        (lambda: ListFunc("total", _col()), "total"),  # not a real list reduction
        (lambda: MathExpr("squareroot", _col()), "squareroot"),
        (lambda: MapFunc("keys", _col()), "keys"),  # should be "map_keys"
    ],
)
def test_unknown_function_raises(ctor, bad: str) -> None:
    with pytest.raises(PlanError, match=bad):
        ctor()


def test_vocabularies_are_disjoint_from_typos() -> None:
    # Sanity: the documented sets are non-empty and the StrEnum families round-trip.
    assert "contains" in STR_FNS
    assert "sum" in LIST_FNS
    assert "sqrt" in MATH_FNS
    assert "pow" in frozenset(Math2Fn)
    assert MapFn.ELEMENT_AT == "element_at"  # StrEnum members are their wire string


def _col():
    from batcher.plan.expr_ir.nodes import Col

    return Col("x")
