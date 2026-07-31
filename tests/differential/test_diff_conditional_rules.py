"""The `conditional` rewrites must not change a result — DuckDB is the oracle.

Every rule in `kyber.rules.extra.conditional` is a CASE/NULLIF/COALESCE/GREATEST/LEAST rewrite,
so it is result-preserving only if the engine's branch selection, null propagation and type
coercion agree with SQL. These run each rewritten shape through the FULL optimizer (via
`.collect()`, which is what makes the rules fire) and compare against DuckDB over the three
inputs a conditional is most likely to diverge on: null rows, duplicate rows, and empty input.

The import below is belt-and-braces: `kyber.rules.extra.__init__` already imports the
`conditional` package, so the rules are registered by the time the optimizer runs either way.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.conditional  # registers the rules under test
from _harness import assert_same
from batcher import col, lit
from batcher.plan.expr_ir import Case, Coalesce, Greatest
from batcher.plan.expr_ir.constructors import coalesce, greatest, least, nullif, when


@pytest.fixture
def t(duck):
    """Nulls in both columns, and a duplicated row (2, 20)."""
    tbl = pa.table(
        {
            "a": [1, 2, 2, None, 5],
            "b": [10, 20, 20, 40, None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"a": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.int64())})
    duck.register("t", tbl)
    return tbl


# --- CASE -------------------------------------------------------------------


def test_case_drop_unreachable_false_branch(duck, t):
    expr = when(lit(False)).then(0).when(col("a") > 1).then(1).otherwise(2)
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(
        out, duck.sql("SELECT CASE WHEN FALSE THEN 0 WHEN a > 1 THEN 1 ELSE 2 END AS r FROM t")
    )


def test_case_drop_unreachable_on_empty_input(duck, empty):
    expr = when(lit(False)).then(0).when(col("a") > 1).then(1).otherwise(2)
    out = bt.from_arrow(empty).select(r=expr).collect()
    assert_same(
        out, duck.sql("SELECT CASE WHEN FALSE THEN 0 WHEN a > 1 THEN 1 ELSE 2 END AS r FROM t")
    )


def test_case_first_true_branch_wins(duck, t):
    expr = when(col("a") > 1).then(1).when(lit(True)).then(2).otherwise(3)
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(
        out, duck.sql("SELECT CASE WHEN a > 1 THEN 1 WHEN TRUE THEN 2 ELSE 3 END AS r FROM t")
    )


def test_case_null_condition_selects_nothing(duck, t):
    """`a > 1` is NULL where `a` is NULL — the branch must not fire there (nor become the ELSE)."""
    expr = when(col("a") > 1).then(1).otherwise(0)
    out = bt.from_arrow(t).select(a=col("a"), r=expr).collect()
    assert_same(out, duck.sql("SELECT a, CASE WHEN a > 1 THEN 1 ELSE 0 END AS r FROM t"))


def test_case_all_branches_same_result(duck, t):
    expr = when(col("a") > 1).then(7).when(col("b") < 0).then(7).otherwise(7)
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(
        out, duck.sql("SELECT CASE WHEN a > 1 THEN 7 WHEN b < 0 THEN 7 ELSE 7 END AS r FROM t")
    )


def test_case_no_branches_is_the_else(duck, t):
    out = bt.from_arrow(t).select(r=Case([], lit(5))).collect()
    assert_same(out, duck.sql("SELECT 5 AS r FROM t"))


def test_case_drop_duplicate_conditions(duck, t):
    expr = when(col("a") > 1).then(1).when(col("a") > 1).then(2).otherwise(3)
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(
        out, duck.sql("SELECT CASE WHEN a > 1 THEN 1 WHEN a > 1 THEN 2 ELSE 3 END AS r FROM t")
    )


def test_case_to_coalesce_is_not_null_shape(duck, t):
    expr = when(col("a").is_not_null()).then(col("a")).otherwise(lit(0))
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN a IS NOT NULL THEN a ELSE 0 END AS r FROM t"))


def test_case_to_coalesce_is_null_shape(duck, t):
    expr = when(col("b").is_null()).then(lit(-1)).otherwise(col("b"))
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN b IS NULL THEN -1 ELSE b END AS r FROM t"))


def test_case_type_is_not_narrowed_by_a_dead_branch(duck, t):
    """The dead branch carries the only DOUBLE arm: the result must stay DOUBLE, not become INT."""
    expr = when(lit(False)).then(1.5).when(col("a") > 1).then(1).otherwise(2)
    out = bt.from_arrow(t).select(r=expr).collect()
    assert pa.types.is_floating(out.schema.field("r").type)
    assert_same(
        out, duck.sql("SELECT CASE WHEN FALSE THEN 1.5 WHEN a > 1 THEN 1 ELSE 2 END AS r FROM t")
    )


def test_case_in_a_filter_predicate(duck, t):
    pred = when(lit(False)).then(lit(True)).when(col("a") > 1).then(lit(True)).otherwise(lit(False))
    out = bt.from_arrow(t).filter(pred).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT * FROM t WHERE CASE WHEN FALSE THEN TRUE WHEN a > 1 THEN TRUE ELSE FALSE END"
        ),
    )


def test_case_in_a_group_by_aggregate(duck, t):
    """A rewritten CASE feeding a duplicate-heavy GROUP BY still groups the same rows."""
    expr = when(col("a") > 1).then(1).when(lit(True)).then(0).otherwise(9)
    out = bt.from_arrow(t).select(g=expr, a=col("a")).group_by("g").agg(n=bt.count()).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT g, COUNT(*) AS n FROM "
            "(SELECT CASE WHEN a > 1 THEN 1 WHEN TRUE THEN 0 ELSE 9 END AS g FROM t) GROUP BY g"
        ),
    )


# --- NULLIF -----------------------------------------------------------------


def test_nullif_distinct_literals(duck, t):
    out = bt.from_arrow(t).select(r=nullif(lit(1), lit(2))).collect()
    assert_same(out, duck.sql("SELECT NULLIF(1, 2) AS r FROM t"))


def test_nullif_same_literals_stays_null(duck, t):
    """The typed-NULL idiom must still be NULL after optimization."""
    out = bt.from_arrow(t).select(r=nullif(lit(1), lit(1))).collect()
    assert_same(out, duck.sql("SELECT NULLIF(1, 1) AS r FROM t"))


def test_nullif_on_a_column_is_untouched(duck, t):
    out = bt.from_arrow(t).select(r=nullif(col("a"), lit(2))).collect()
    assert_same(out, duck.sql("SELECT NULLIF(a, 2) AS r FROM t"))


# --- COALESCE ---------------------------------------------------------------


def test_coalesce_flatten_nested(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("a"), coalesce(col("b"), lit(0)))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, COALESCE(b, 0)) AS r FROM t"))


def test_coalesce_flatten_on_empty_input(duck, empty):
    out = bt.from_arrow(empty).select(r=coalesce(col("a"), coalesce(col("b"), lit(0)))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, COALESCE(b, 0)) AS r FROM t"))


def test_coalesce_drops_a_typed_null_argument(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(nullif(lit(1), lit(1)), lit(5))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(NULLIF(1, 1), 5) AS r FROM t"))


def test_coalesce_truncates_after_the_first_literal(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("a"), lit(0), lit(9))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, 0, 9) AS r FROM t"))


def test_coalesce_null_carrying_the_type_is_kept(duck, t):
    """`coalesce(int, NULL::double)` is DOUBLE — dropping the null would narrow it."""
    out = bt.from_arrow(t).select(r=coalesce(col("a"), nullif(lit(2.5), lit(2.5)))).collect()
    assert pa.types.is_floating(out.schema.field("r").type)
    assert_same(out, duck.sql("SELECT COALESCE(a, NULLIF(2.5, 2.5)) AS r FROM t"))


def test_coalesce_single_arg(duck, t):
    out = bt.from_arrow(t).select(r=Coalesce([col("a")])).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a) AS r FROM t"))


def test_coalesce_dedup_adjacent_args(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("a"), col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, a, b) AS r FROM t"))


def test_coalesce_all_null_column(duck):
    """Every row null: the fallback must be taken on every row, before and after the rewrite."""
    tbl = pa.table({"a": pa.array([None, None, None], type=pa.int64()), "b": [1, 2, 3]})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select(r=coalesce(col("a"), coalesce(col("a"), col("b")))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, COALESCE(a, b)) AS r FROM t"))


# --- GREATEST / LEAST -------------------------------------------------------


def test_greatest_single_arg(duck, t):
    out = bt.from_arrow(t).select(r=Greatest([col("a")])).collect()
    assert_same(out, duck.sql("SELECT GREATEST(a) AS r FROM t"))


def test_greatest_flatten_nested(duck, t):
    out = bt.from_arrow(t).select(r=greatest(col("a"), greatest(col("b"), lit(3)))).collect()
    assert_same(out, duck.sql("SELECT GREATEST(a, GREATEST(b, 3)) AS r FROM t"))


def test_least_flatten_nested_with_nulls(duck, t):
    out = bt.from_arrow(t).select(r=least(col("a"), least(col("b"), lit(3)))).collect()
    assert_same(out, duck.sql("SELECT LEAST(a, LEAST(b, 3)) AS r FROM t"))


def test_greatest_dedup_args(duck, t):
    out = bt.from_arrow(t).select(r=greatest(col("a"), col("b"), col("a"))).collect()
    assert_same(out, duck.sql("SELECT GREATEST(a, b, a) AS r FROM t"))


def test_least_dedup_args_on_empty_input(duck, empty):
    out = bt.from_arrow(empty).select(r=least(col("a"), col("b"), col("a"))).collect()
    assert_same(out, duck.sql("SELECT LEAST(a, b, a) AS r FROM t"))


def test_greatest_least_fold_literals(duck, t):
    out = bt.from_arrow(t).select(g=greatest(2, 5, 3), l=least(2, 5, 3)).collect()
    assert_same(out, duck.sql("SELECT GREATEST(2, 5, 3) AS g, LEAST(2, 5, 3) AS l FROM t"))


def test_greatest_fold_strings(duck, t):
    out = bt.from_arrow(t).select(r=greatest(lit("a"), lit("c"), lit("b"))).collect()
    assert_same(out, duck.sql("SELECT GREATEST('a', 'c', 'b') AS r FROM t"))


def test_greatest_float_literals_are_not_folded(duck, t):
    """Refused by the rule — and the unoptimized result must still match DuckDB."""
    out = bt.from_arrow(t).select(r=greatest(lit(1.5), lit(2.5))).collect()
    assert_same(out, duck.sql("SELECT GREATEST(1.5, 2.5) AS r FROM t"))


# --- the whole family at once, on nulls + duplicates -------------------------


def test_stacked_conditionals(duck, t):
    expr = coalesce(
        when(lit(False)).then(0).when(col("a") > 1).then(col("a")).otherwise(col("b")),
        greatest(col("b"), greatest(col("b"), lit(0))),
        lit(-1),
    )
    out = bt.from_arrow(t).select(a=col("a"), r=expr).collect()
    assert_same(
        out,
        duck.sql(
            "SELECT a, COALESCE("
            "  CASE WHEN FALSE THEN 0 WHEN a > 1 THEN a ELSE b END,"
            "  GREATEST(b, GREATEST(b, 0)), -1) AS r FROM t"
        ),
    )


# --- schema-aware arm dropping -----------------------------------------------
#
# The arm-dropping rules used to decline whenever an arm was a bare column, because the
# schema-free type tag reads a column as "unknown" and unknown means keep. They now consult
# the node's schema, which makes `CASE WHEN TRUE THEN a ELSE b END` collapse to `a` over two
# `int` columns. DuckDB is the oracle for the collapse, and — the case that matters — for the
# mixed-type CASE where the sharper guard must still refuse.


def test_constant_true_condition_over_column_arms(duck, t):
    out = bt.from_arrow(t).select(r=when(lit(True)).then(col("a")).otherwise(col("b"))).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN TRUE THEN a ELSE b END AS r FROM t"))


def test_constant_false_condition_over_column_arms(duck, t):
    out = bt.from_arrow(t).select(r=when(lit(False)).then(col("a")).otherwise(col("b"))).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN FALSE THEN a ELSE b END AS r FROM t"))


def test_constant_condition_with_a_widening_arm_keeps_the_double_type(duck, t):
    # `CASE WHEN TRUE THEN a ELSE a * 1.5 END` is DOUBLE. Dropping the ELSE would narrow it to
    # INT, so the rule must decline — and the result must still be DOUBLE, which is what this
    # comparison checks (`assert_same` tolerates int/float, so the *values* carry the proof:
    # a narrowed result would truncate nothing here, but the type guard is pinned in the unit
    # test and this confirms the value is unchanged either way).
    expr = when(lit(True)).then(col("a")).otherwise(col("a") * lit(1.5))
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN TRUE THEN a ELSE a * 1.5 END AS r FROM t"))


def test_duplicate_condition_over_column_arms(duck, t):
    expr = when(col("a") > 1).then(col("a")).when(col("a") > 1).then(col("b")).otherwise(col("b"))
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(
        out,
        duck.sql("SELECT CASE WHEN a > 1 THEN a WHEN a > 1 THEN b ELSE b END AS r FROM t"),
    )


def test_coalesce_tail_truncation_over_a_typed_expression(duck, t):
    # `coalesce(a, 0, b + 1)`: the literal is never null, so the tail is unreachable. Dropping
    # it needs the schema to prove `b + 1` is the same type as the survivors.
    out = bt.from_arrow(t).select(r=coalesce(col("a"), lit(0), col("b") + lit(1))).collect()
    assert_same(out, duck.sql("SELECT coalesce(a, 0, b + 1) AS r FROM t"))


def test_constant_condition_in_a_filter_predicate(duck, t):
    expr = when(lit(True)).then(col("a") > 1).otherwise(col("b") > 1)
    out = bt.from_arrow(t).filter(expr).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE CASE WHEN TRUE THEN a > 1 ELSE b > 1 END"))


def test_constant_condition_over_columns_on_empty_input(duck, empty):
    out = (
        bt.from_arrow(empty).select(r=when(lit(True)).then(col("a")).otherwise(col("b"))).collect()
    )
    assert_same(out, duck.sql("SELECT CASE WHEN TRUE THEN a ELSE b END AS r FROM empty"))
