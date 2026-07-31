"""Plan-shape unit tests for the sargable-predicate normalization rules.

Each rule gets: a fire test (the rewrite happens and yields `col OP literal`), an
idempotence test (apply twice == once), and a does-not-fire test for the unsafe cases
the engine's wrapping i64 arithmetic forbids (ordered comparisons, even multipliers,
non-divisible literals, and folded literals that would overflow i64).
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.sargable import (
    flip_comparison_literal,
    sarg_add_const,
    sarg_mul_const,
    sarg_rsub_const,
    sarg_sub_const,
    sarg_xor_const,
)
from batcher.plan.expr_ir import Binary, Col, Lit
from batcher.plan.logical import Filter

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

_SARGABLE_RULES = [
    "sarg_flip_comparison",
    "sarg_add_const",
    "sarg_sub_const",
    "sarg_rsub_const",
    "sarg_mul_const",
    "sarg_xor_const",
]


def _base():
    return bt.from_pydict({"x": [1, 2, 3], "v": [1, 2, 3]})._plan


def _filter(pred):
    return Filter(_base(), pred)


def _pred(node):
    return node.predicate.to_ir()


# --- registration -----------------------------------------------------------


@pytest.mark.parametrize("name", _SARGABLE_RULES)
def test_rules_registered(name):
    assert name in {r.name for r in DEFAULT_REGISTRY.rules()}


# --- comparison flip --------------------------------------------------------


def test_flip_puts_col_on_left():
    out = flip_comparison_literal(_filter(Binary("lt", Lit(5), Col("x"))))
    p = _pred(out)
    assert p["op"] == "gt"  # 5 < x  ->  x > 5
    assert p["left"] == {"e": "col", "name": "x"}
    assert p["right"]["value"] == {"int": 5}


def test_flip_eq_stays_eq():
    out = flip_comparison_literal(_filter(Binary("eq", Lit(5), Col("x"))))
    assert _pred(out)["op"] == "eq"


def test_flip_does_not_fire_when_canonical():
    plan = _filter(Binary("gt", Col("x"), Lit(5)))
    assert flip_comparison_literal(plan).to_ir() == plan.to_ir()


def test_flip_idempotent():
    plan = _filter(Binary("lt", Lit(5), Col("x")))
    once = flip_comparison_literal(plan)
    assert flip_comparison_literal(once).to_ir() == once.to_ir()


# --- additive: col + k ------------------------------------------------------


def test_add_reduces_eq():
    out = sarg_add_const(_filter((col("x") + 100) == 500))
    p = _pred(out)
    assert p["op"] == "eq"
    assert p["left"] == {"e": "col", "name": "x"}
    assert p["right"]["value"] == {"int": 400}


def test_add_reduces_commuted_and_ne():
    out = sarg_add_const(_filter((100 + col("x")) != 500))
    p = _pred(out)
    assert p["op"] == "ne" and p["right"]["value"] == {"int": 400}


def test_add_does_not_fire_on_ordered():
    plan = _filter((col("x") + 100) < 500)
    assert sarg_add_const(plan).to_ir() == plan.to_ir()


def test_add_does_not_fire_on_float():
    plan = _filter((col("x") + 1.0) == 5.0)
    assert sarg_add_const(plan).to_ir() == plan.to_ir()


def test_add_overflow_guard():
    # lit - k = INT64_MIN - 1 is out of i64 range -> must not rewrite.
    plan = _filter((col("x") + 1) == _INT64_MIN)
    assert sarg_add_const(plan).to_ir() == plan.to_ir()


def test_add_idempotent():
    plan = _filter((col("x") + 100) == 500)
    once = sarg_add_const(plan)
    assert sarg_add_const(once).to_ir() == once.to_ir()


# --- additive: col - k ------------------------------------------------------


def test_sub_reduces_eq():
    out = sarg_sub_const(_filter((col("x") - 3) == 10))
    assert _pred(out)["right"]["value"] == {"int": 13}


def test_sub_overflow_guard():
    plan = _filter((col("x") - 5) == _INT64_MAX)  # lit + 5 overflows i64
    assert sarg_sub_const(plan).to_ir() == plan.to_ir()


def test_sub_does_not_fire_on_ordered():
    plan = _filter((col("x") - 3) >= 10)
    assert sarg_sub_const(plan).to_ir() == plan.to_ir()


def test_sub_idempotent():
    plan = _filter((col("x") - 3) == 10)
    once = sarg_sub_const(plan)
    assert sarg_sub_const(once).to_ir() == once.to_ir()


# --- additive: k - col (and unary minus) ------------------------------------


def test_rsub_reduces_eq():
    out = sarg_rsub_const(_filter((5 - col("x")) == 2))
    assert _pred(out)["right"]["value"] == {"int": 3}  # x = 5 - 2


def test_rsub_covers_unary_minus():
    out = sarg_rsub_const(_filter((-col("x")) == 2))  # -x == 2  ->  x == -2
    p = _pred(out)
    assert p["left"] == {"e": "col", "name": "x"} and p["right"]["value"] == {"int": -2}


def test_rsub_unary_minus_int64_min_guard():
    # -x == INT64_MIN  ->  x == -INT64_MIN, which overflows i64 -> must not rewrite.
    plan = _filter((-col("x")) == _INT64_MIN)
    assert sarg_rsub_const(plan).to_ir() == plan.to_ir()


def test_rsub_idempotent():
    plan = _filter((5 - col("x")) == 2)
    once = sarg_rsub_const(plan)
    assert sarg_rsub_const(once).to_ir() == once.to_ir()


# --- multiplicative: col * k ------------------------------------------------


def test_mul_reduces_odd_exact_divide():
    out = sarg_mul_const(_filter((col("x") * 3) == 9))
    assert _pred(out)["right"]["value"] == {"int": 3}


def test_mul_negative_coefficient():
    out = sarg_mul_const(_filter((col("x") * -3) == -9))
    assert _pred(out)["right"]["value"] == {"int": 3}  # -9 / -3


def test_mul_does_not_fire_even_coefficient():
    # k = 2 is non-injective mod 2^64 (col*2==4 also matches col = 2 + 2^63) -> unsafe.
    plan = _filter((col("x") * 2) == 4)
    assert sarg_mul_const(plan).to_ir() == plan.to_ir()


def test_mul_does_not_fire_non_divisible():
    plan = _filter((col("x") * 3) == 7)  # 7 not a multiple of 3
    assert sarg_mul_const(plan).to_ir() == plan.to_ir()


def test_mul_does_not_fire_on_ordered():
    plan = _filter((col("x") * 3) < 9)
    assert sarg_mul_const(plan).to_ir() == plan.to_ir()


def test_mul_idempotent():
    plan = _filter((col("x") * 3) == 9)
    once = sarg_mul_const(plan)
    assert sarg_mul_const(once).to_ir() == once.to_ir()


# --- bitwise xor: col ^ k ---------------------------------------------------


def test_xor_reduces_eq():
    out = sarg_xor_const(_filter(col("x").bitwise_xor(5) == 7))
    assert _pred(out)["right"]["value"] == {"int": 2}  # 7 ^ 5


def test_xor_does_not_fire_on_ordered():
    plan = _filter(col("x").bitwise_xor(5) < 7)
    assert sarg_xor_const(plan).to_ir() == plan.to_ir()


def test_xor_idempotent():
    plan = _filter(col("x").bitwise_xor(5) == 7)
    once = sarg_xor_const(plan)
    assert sarg_xor_const(once).to_ir() == once.to_ir()


# --- full optimizer strips the arithmetic wrapper ---------------------------


def test_full_optimizer_exposes_raw_column():
    plan = bt.from_pydict({"x": [1, 2, 3], "v": [1, 2, 3]}).filter((col("x") + 100) == 500)._plan
    ir = json.dumps(Optimizer().optimize(plan).ir)
    assert '"add"' not in ir  # the wrapper is gone
    assert '{"int": 400}' in ir or '"int": 400' in ir


# --- the registered rules, not just the standalone functions ----------------

#: One query per registered sargable rule, each written in the shape only that rule
#: rewrites. The rewritten form is always `col OP literal`, which is what zone-map
#: pruning and source pushdown match on.
_FIRES = {
    "sarg_flip_comparison": lambda ds: ds.filter(bt.lit(5) == col("a")),
    "sarg_add_const": lambda ds: ds.filter(col("a") + bt.lit(2) == bt.lit(7)),
    "sarg_sub_const": lambda ds: ds.filter(col("a") - bt.lit(2) == bt.lit(7)),
    "sarg_rsub_const": lambda ds: ds.filter(bt.lit(9) - col("a") == bt.lit(7)),
    "sarg_mul_const": lambda ds: ds.filter(col("a") * bt.lit(3) == bt.lit(9)),
    "sarg_xor_const": lambda ds: ds.filter((col("a") ^ bt.lit(3)) == bt.lit(9)),
}


@pytest.mark.parametrize("name", sorted(_FIRES))
def test_registered_rule_still_rewrites(name):
    # These six are registered as node rules so the driver runs them inside its shared
    # expression traversal rather than each walking the whole plan. A rule that stops being
    # reached by that traversal is invisible: every result stays correct and only the
    # predicate silently keeps its un-sargable shape. So the assertion is that the *plan*
    # changes, driven through the registry rather than by calling the pass directly.
    from batcher.kyber.optimizer import optimize_logical

    plan = _FIRES[name](bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]}))._plan
    assert optimize_logical(plan).to_ir() != plan.to_ir(), f"{name} no longer fires"


def test_every_sargable_rule_is_covered_above():
    # If a rule is added to the family, this fails until it gets a firing case.
    #
    # Scoped to *this* module's family. The ordered-comparison family
    # (`sarg_bounded_*`, from `rules/extra/sargable_range`) is a separate module with the
    # same completeness guard over its own twelve rules, in
    # `tests/unit/test_sargable_ordered_bounds.py` — it has to live there because those
    # rewrites need recorded column bounds, which the plain fixture above does not supply.
    registered = {
        r.name
        for r in DEFAULT_REGISTRY.rules()
        if r.name.startswith("sarg_") and not r.name.startswith("sarg_bounded_")
    }
    assert registered == set(_FIRES)
