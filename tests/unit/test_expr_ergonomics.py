"""Python-level ergonomics of `Expr`: dunders, builtins, repr, and compat aliases.

These are plan-build-time properties (no engine needed), so they live in `tests/unit`.
Three families:

1. **Builtins refuse clearly.** ``len``/``in``/``hash``/``bool``/``iter`` on an
   expression are all mistakes, and each must name the fix rather than surface
   Python's default message.
2. **Aliases are delegations, not reimplementations.** Every compat spelling must
   produce an IR *identical* to the primary. Comparing `to_ir()` is what proves there
   is one implementation rather than two that can drift.
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
    with pytest.raises(TypeError, match="min_horizontal"):
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


# --- 2. compat aliases are exact delegations -----------------------------------------
@pytest.mark.parametrize(
    ("alias", "primary"),
    [
        ("isna", "is_null"),
        ("isnull", "is_null"),
        ("notna", "is_not_null"),
        ("notnull", "is_not_null"),
    ],
)
def test_nullary_alias_matches_primary(alias, primary):
    assert getattr(col("x"), alias)().to_ir() == getattr(col("x"), primary)().to_ir()


@pytest.mark.parametrize(
    ("alias", "primary"),
    [
        ("nunique", "n_unique"),
        ("skew", "skewness"),
        ("kurt", "kurtosis"),
        ("cumsum", "cum_sum"),
        ("cummax", "cum_max"),
        ("cummin", "cum_min"),
        ("cumcount", "cum_count"),
        ("prod", "product"),
        ("any", "bool_or"),
        ("all", "bool_and"),
        ("log", "ln"),
    ],
)
def test_aggregate_alias_matches_primary(alias, primary):
    """Aggregate/window nodes have no standalone `to_ir()`, so compare the built node.

    They are only serializable once hoisted into an `Aggregate`/`Window` plan node,
    so equality here is on the node's own rendered form.
    """
    assert repr(getattr(col("x"), alias)()) == repr(getattr(col("x"), primary)())


@pytest.mark.parametrize(
    ("alias", "primary", "arg"),
    [
        ("astype", "cast", "float64"),
        ("fillna", "fill_null", 0),
        ("isin", "is_in", [1, 2]),
        ("rename", "alias", "y"),
    ],
)
def test_unary_alias_matches_primary(alias, primary, arg):
    assert getattr(col("x"), alias)(arg).to_ir() == getattr(col("x"), primary)(arg).to_ir()


@pytest.mark.parametrize(
    ("method", "op"),
    [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("truediv", lambda a, b: a / b),
        ("div", lambda a, b: a / b),
        ("floordiv", lambda a, b: a // b),
        ("mod", lambda a, b: a % b),
        ("eq", lambda a, b: a == b),
        ("ne", lambda a, b: a != b),
        ("lt", lambda a, b: a < b),
        ("le", lambda a, b: a <= b),
        ("gt", lambda a, b: a > b),
        ("ge", lambda a, b: a >= b),
        ("and_", lambda a, b: a & b),
        ("or_", lambda a, b: a | b),
        ("xor", lambda a, b: a ^ b),
    ],
)
def test_operator_method_matches_the_operator(method, op):
    assert getattr(col("x"), method)(col("y")).to_ir() == op(col("x"), col("y")).to_ir()


def test_not_matches_the_invert_operator():
    assert col("x").not_().to_ir() == (~col("x")).to_ir()


@pytest.mark.parametrize(
    ("accessor", "alias", "primary"),
    [
        ("str", "isdigit", "is_numeric"),
        ("str", "isalpha", "is_alpha"),
        ("str", "isalnum", "is_alnum"),
        ("str", "isspace", "is_space"),
        ("dt", "day_of_week", "dayofweek"),
        ("dt", "day_of_year", "dayofyear"),
        ("dt", "week_of_year", "weekofyear"),
        ("list", "lengths", "len"),
        ("list", "argmin", "arg_min"),
        ("list", "argmax", "arg_max"),
    ],
)
def test_namespace_alias_matches_primary(accessor, alias, primary):
    ns = getattr(col("c"), accessor)
    assert getattr(ns, alias)().to_ir() == getattr(ns, primary)().to_ir()


@pytest.mark.parametrize(
    ("accessor", "alias", "primary", "arg"),
    [
        ("str", "strip_prefix", "removeprefix", "ab"),
        ("str", "strip_suffix", "removesuffix", "cd"),
        ("list", "element_at", "get", 1),
    ],
)
def test_namespace_alias_with_arg_matches_primary(accessor, alias, primary, arg):
    ns = getattr(col("c"), accessor)
    assert getattr(ns, alias)(arg).to_ir() == getattr(ns, primary)(arg).to_ir()


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
        (lambda: col("t").dt.truncate("1d"), "col('t').dt.truncate('1d')"),
        (lambda: col("m").map.get("k"), "col('m').map.get('k')"),
        (lambda: col("a").list.dot(col("b")), "col('a').list.dot(col('b'))"),
        (lambda: col("a").list.union(col("b")), "col('a').list.union(col('b'))"),
    ],
)
def test_repr_reads_like_the_code_that_built_it(build, expected):
    assert repr(build()) == expected


def test_two_column_list_nodes_repr_without_raising():
    """`ListBinary`/`ListSet` hold `left`/`right`, not `input`, and used to crash `repr`."""
    assert "list.dot" in repr(col("a").list.dot(col("b")))
    assert "list.union" in repr(col("a").list.union(col("b")))


def test_nested_accessor_keeps_its_arguments():
    """A nested `StrFunc` must not lose its pattern to the generic accessor renderer."""
    assert "contains('a')" in repr(col("s").str.contains("a") & lit(True))
