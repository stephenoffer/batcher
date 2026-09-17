"""Python-level ergonomics of `Expr`: dunders, builtins, repr, and absent spellings.

These are plan-build-time properties (no engine needed), so they live in `tests/unit`.
Three families:

1. **Builtins refuse clearly.** ``len``/``in``/``hash``/``bool``/``iter`` on an
   expression are all mistakes, and each must name the fix rather than surface
   Python's default message.
2. **One name per capability.** Ecosystem spellings that would be silently wrong
   (1-based SQL positions, `islower` on uncased strings) stay absent.
3. **`repr` round-trips visually.** The rendered form should read like the code that
   built the expression, including arguments.
"""

from __future__ import annotations

import copy
import math
import pickle

import pytest

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import col, lit

pytestmark = pytest.mark.unit


# --- 1. builtins refuse, and say what to do instead ----------------------------------
def test_len_names_the_three_alternatives():
    with pytest.raises(TypeError, match=r"\.str\.len\(\).*\.list\.len\(\).*ds\.count\(\)"):
        len(col("x"))


def test_in_operator_points_at_is_in():
    with pytest.raises(TypeError, match="is_in"):
        _ = 1 in col("x")


def test_in_operator_does_not_fall_through_to_iter():
    """``1 in expr`` must not surface the *iteration* message; it is a membership test."""
    with pytest.raises(TypeError) as exc:
        _ = 1 in col("x")
    assert "not iterable" not in str(exc.value)


def test_bool_points_at_the_bitwise_operators():
    with pytest.raises(PlanError, match=r"&"):
        bool(col("x") > 1)


def test_hash_explains_why_equality_is_unusable():
    with pytest.raises(TypeError, match="not hashable"):
        hash(col("x"))


def test_iter_mentions_the_horizontal_helpers():
    """``min(expr)`` reaches `__iter__`, so that message must cover the min/max case."""
    with pytest.raises(TypeError, match="least"):
        min(col("x"))


def test_iter_still_names_the_list_wrapping_fix():
    with pytest.raises(TypeError, match=r"wrap it in a list"):
        list(col("x"))


# --- builtins that should work -------------------------------------------------------
def test_divmod_returns_the_floordiv_mod_pair():
    q, r = divmod(col("x"), 3)
    assert q.to_ir() == (col("x") // 3).to_ir()
    assert r.to_ir() == (col("x") % 3).to_ir()


def test_rdivmod_supports_a_scalar_dividend():
    q, r = divmod(7, col("x"))
    assert q.to_ir() == (lit(7) // col("x")).to_ir()
    assert r.to_ir() == (lit(7) % col("x")).to_ir()


def test_matmul_is_the_list_dot_product():
    assert (col("a") @ col("b")).to_ir() == col("a").list.dot(col("b")).to_ir()


@pytest.mark.parametrize(
    ("builtin", "method"),
    [(abs, "abs"), (math.floor, "floor"), (math.ceil, "ceil"), (math.trunc, "trunc")],
)
def test_math_builtins_delegate_to_the_method(builtin, method):
    assert builtin(col("x")).to_ir() == getattr(col("x"), method)().to_ir()


def test_round_accepts_digits():
    assert round(col("x"), 2).to_ir() == col("x").round(2).to_ir()


def test_sum_builtin_folds_expressions():
    """``sum([...])`` starts at int 0, so it needs `__radd__` to fold expressions."""
    assert sum([col("a"), col("b")]).to_ir() == (0 + col("a") + col("b")).to_ir()


def test_expressions_survive_pickling():
    e = col("x").str.contains("a") & (col("y") > 1)
    assert pickle.loads(pickle.dumps(e)).to_ir() == e.to_ir()


def test_expressions_survive_deepcopy():
    e = col("x").fill_null(0)
    assert copy.deepcopy(e).to_ir() == e.to_ir()


# --- 2. no silently-wrong ecosystem spellings ---------------------------------------
def test_aliases_that_would_be_semantically_wrong_are_absent():
    """Guard the deliberate omissions in `expr_ir.compat.namespaces`.

    `position`/`substr` are 1-based SQL and `is_lower` is true for uncased strings,
    so these ecosystem names would each be a silently-wrong alias. If someone adds
    one, it must be a real implementation, not a delegation — this test should fail
    and make them prove the semantics.
    """
    for name in ("find", "index", "rfind", "substring", "islower", "isupper", "count"):
        assert not hasattr(col("s").str, name), f"str.{name} must not alias a 1-based/SQL primary"


# --- cast dtype names are case-insensitive -------------------------------------------
@pytest.mark.parametrize("spelling", ["int64", "Int64", "INT64", "InT64"])
def test_cast_accepts_any_case_and_canonicalizes(spelling):
    assert col("x").cast(spelling).to_ir()["dtype"] == "int64"


def test_cast_still_rejects_an_unknown_dtype_with_a_hint():
    with pytest.raises(PlanError, match="did you mean"):
        col("x").cast("Nt64")


# --- 3. repr round-trips visually ----------------------------------------------------
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: col("x") + 1, "(col('x') + lit(1))"),
        (lambda: col("s").str.contains("a"), "col('s').str.contains('a')"),
        (lambda: col("l").list.get(2), "col('l').list.get(2)"),
        (lambda: col("l").list.slice(1, 2), "col('l').list.slice(1, 2)"),
        (lambda: col("v").struct.field("a"), "col('v').struct.field('a')"),
        (lambda: col("t").dt.truncate("day"), "col('t').dt.truncate('day')"),
        (lambda: col("m").map.get("k"), "col('m').map.get('k')"),
        (lambda: col("a").list.dot(col("b")), "col('a').list.dot(col('b'))"),
        (lambda: col("a").list.union(col("b")), "col('a').list.union(col('b'))"),
    ],
)
def test_repr_reads_like_the_code_that_built_it(build, expected):
    assert repr(build()) == expected


def test_a_truncation_unit_alias_is_canonicalized_on_the_node():
    """`truncate('1d')` records `'day'`, and that is deliberate rather than cosmetic.

    The unit is normalized where the expression is built, not where it is lowered, because
    Kyber reasons over `DateTrunc.unit` by *name*: `temporal_extra._nested_trunc` gates on
    membership of `_TRUNC_ORDER` and `_collapse_same_unit` compares two units for equality.
    A node holding the raw `'1d'` would fall out of that set, so the nested-truncate
    collapse would silently stop firing for every aliased spelling -- a rule that quietly
    does nothing, which is worse than one that errors.

    So the repr reads like *equivalent* code rather than the exact characters typed. The
    parametrized repr case above uses the canonical spelling for that reason.
    """
    assert repr(col("t").dt.truncate("1d")) == "col('t').dt.truncate('day')"
    assert repr(col("t").dt.truncate("mo")) == "col('t').dt.truncate('month')"
    assert repr(col("t").dt.truncate("1h")) == "col('t').dt.truncate('hour')"


def test_two_column_list_nodes_repr_without_raising():
    """`ListBinary`/`ListSet` hold `left`/`right`, not `input`, and used to crash `repr`."""
    assert "list.dot" in repr(col("a").list.dot(col("b")))
    assert "list.union" in repr(col("a").list.union(col("b")))


def test_nested_accessor_keeps_its_arguments():
    """A nested `StrFunc` must not lose its pattern to the generic accessor renderer."""
    assert "contains('a')" in repr(col("s").str.contains("a") & lit(True))
