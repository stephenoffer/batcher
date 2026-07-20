"""The schema-driven NULL rewrites must match DuckDB after optimization.

Every rule in `kyber.rules.extra.nullability` is proved from a *declared* NOT NULL — the
promise a source schema makes. These run each rewritten shape through the FULL optimizer
(via `.collect()`) and assert equality vs DuckDB, over the three places a null rewrite is
most likely to diverge: a table where the nullable columns really do carry NULLs, the same
query over an **empty** input, and the *projection* forms, where a wrongly-folded null check
is a wrong value rather than a wrongly-dropped row. Each query is also run against a
control column that is nullable, so a rule firing where it must not would show up as a
mismatch rather than as silence.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.nullability  # (importing registers the rules)
from _harness import assert_same, assert_same_ordered
from batcher import col, lit
from batcher.plan.expr_ir.constructors import coalesce

_SCHEMA = pa.schema(
    [
        pa.field("x", pa.int64(), nullable=False),  # declared NOT NULL — the proof
        pa.field("y", pa.int64()),  # nullable, and it does carry NULLs
        pa.field("f", pa.float64()),  # nullable float (NULL and NaN are different things)
    ]
)


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "x": pa.array([1, 2, 3, 2], type=pa.int64()),
            "y": pa.array([1, None, 3, None], type=pa.int64()),
            "f": pa.array([1.0, None, float("nan"), 4.0], type=pa.float64()),
        },
        schema=_SCHEMA,
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "x": pa.array([], type=pa.int64()),
            "y": pa.array([], type=pa.int64()),
            "f": pa.array([], type=pa.float64()),
        },
        schema=_SCHEMA,
    )
    duck.register("t", tbl)
    return tbl


# --- IS NULL / IS NOT NULL over a NOT NULL column ------------------------------


def test_is_null_on_non_nullable_is_empty(duck, t):
    out = bt.from_arrow(t).filter(col("x").is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NULL"))


def test_is_not_null_on_non_nullable_keeps_everything(duck, t):
    out = bt.from_arrow(t).filter(col("x").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NOT NULL"))


def test_is_null_on_nullable_column_is_untouched(duck, t):
    out = bt.from_arrow(t).filter(col("y").is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE y IS NULL"))


def test_is_null_on_a_derived_non_nullable_expression(duck, t):
    out = bt.from_arrow(t).filter((col("x") + 1).is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (x + 1) IS NULL"))


def test_null_check_over_empty_input(duck, empty):
    out = bt.from_arrow(empty).filter(col("x").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NOT NULL"))


# --- the projection forms: a folded null check is a VALUE -----------------------


def test_projected_null_check_on_non_nullable(duck, t):
    out = bt.from_arrow(t).select(r=col("x").is_null(), s=col("x").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT x IS NULL AS r, x IS NOT NULL AS s FROM t"))


def test_projected_null_check_on_nullable(duck, t):
    out = bt.from_arrow(t).select(r=col("y").is_null(), s=col("y").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT y IS NULL AS r, y IS NOT NULL AS s FROM t"))


def test_projected_null_check_of_a_literal(duck, t):
    out = bt.from_arrow(t).select(r=lit(1).is_null(), s=lit(1).is_not_null()).collect()
    assert_same(out, duck.sql("SELECT 1 IS NULL AS r, 1 IS NOT NULL AS s FROM t"))


def test_null_check_under_a_negation(duck, t):
    out = bt.from_arrow(t).select(r=~col("y").is_null(), s=~col("x").is_null()).collect()
    assert_same(out, duck.sql("SELECT NOT (y IS NULL) AS r, NOT (x IS NULL) AS s FROM t"))


def test_null_check_nested_in_a_disjunction(duck, t):
    out = bt.from_arrow(t).filter(col("x").is_null() | (col("y") == 3)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NULL OR y = 3"))


# --- COALESCE / fill_null ------------------------------------------------------


def test_fill_null_of_a_non_nullable_column(duck, t):
    out = bt.from_arrow(t).select(r=col("x").fill_null(0)).collect()
    assert_same(out, duck.sql("SELECT coalesce(x, 0) AS r FROM t"))


def test_fill_null_of_a_nullable_column(duck, t):
    out = bt.from_arrow(t).select(r=col("y").fill_null(0)).collect()
    assert_same(out, duck.sql("SELECT coalesce(y, 0) AS r FROM t"))


def test_coalesce_that_must_not_narrow_its_type(duck, t):
    # DOUBLE, not BIGINT: the rewrite must decline, so the result stays a float column.
    out = bt.from_arrow(t).select(r=coalesce(col("x"), lit(1.5))).collect()
    assert pa.types.is_floating(out.schema.field("r").type)
    assert_same(out, duck.sql("SELECT coalesce(x, 1.5) AS r FROM t"))


def test_coalesce_truncated_after_a_non_nullable_arg(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("y"), col("x"), lit(0))).collect()
    assert_same(out, duck.sql("SELECT coalesce(y, x, 0) AS r FROM t"))


def test_is_null_of_a_coalesce(duck, t):
    out = bt.from_arrow(t).filter(coalesce(col("y"), col("f")).is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE coalesce(y, f) IS NULL"))


def test_is_not_null_of_a_coalesce(duck, t):
    out = bt.from_arrow(t).filter(coalesce(col("y"), col("f")).is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE coalesce(y, f) IS NOT NULL"))


def test_is_null_of_a_coalesce_including_a_non_nullable_arg(duck, t):
    # Expands to `y IS NULL AND x IS NULL` → `… AND FALSE` → empty.
    out = bt.from_arrow(t).filter(coalesce(col("y"), col("x")).is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE coalesce(y, x) IS NULL"))


def test_projected_is_null_of_a_coalesce(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("y"), col("f")).is_null()).collect()
    assert_same(out, duck.sql("SELECT coalesce(y, f) IS NULL AS r FROM t"))


def test_coalesce_over_empty_input(duck, empty):
    out = bt.from_arrow(empty).select(r=col("x").fill_null(0)).collect()
    assert_same(out, duck.sql("SELECT coalesce(x, 0) AS r FROM t"))


# --- the null-safe comparison --------------------------------------------------


def test_null_safe_equality_on_non_nullable_columns(duck, t):
    out = bt.from_arrow(t).filter(col("x").eq_missing(col("x"))).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NOT DISTINCT FROM x"))


def test_null_safe_equality_with_a_nullable_operand(duck, t):
    # Two NULLs must still compare EQUAL here — the rewrite must not fire.
    out = bt.from_arrow(t).filter(col("y").eq_missing(col("y"))).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE y IS NOT DISTINCT FROM y"))


def test_projected_null_safe_equality_mixed_nullability(duck, t):
    out = bt.from_arrow(t).select(r=col("x").eq_missing(col("y"))).collect()
    assert_same(out, duck.sql("SELECT x IS NOT DISTINCT FROM y AS r FROM t"))


# --- COUNT(col) → COUNT(*) -----------------------------------------------------


def test_count_of_a_non_nullable_column_grouped(duck, t):
    out = bt.from_arrow(t).group_by("y").agg(c=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT y, count(x) AS c FROM t GROUP BY y"))


def test_count_of_a_nullable_column_grouped(duck, t):
    # COUNT(y) skips the NULL rows — it is NOT COUNT(*), and the rule must not fire.
    out = bt.from_arrow(t).group_by("x").agg(c=col("y").count()).collect()
    assert_same(out, duck.sql("SELECT x, count(y) AS c FROM t GROUP BY x"))


def test_global_count_of_a_non_nullable_column(duck, t):
    out = bt.from_arrow(t).agg(c=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT count(x) AS c FROM t"))


def test_global_count_of_a_non_nullable_column_over_empty_input(duck, empty):
    out = bt.from_arrow(empty).agg(c=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT count(x) AS c FROM t"))


# --- the sort's null placement -------------------------------------------------


def test_sort_nulls_first_on_a_non_nullable_key(duck, t):
    # Only the key is projected: `x` has a tie, and which tied *row* comes first is not a
    # property either engine promises — the ordering of the key values is.
    out = bt.from_arrow(t).sort("x", nulls_first=True).select("x").collect()
    assert_same_ordered(out, duck.sql("SELECT x FROM t ORDER BY x NULLS FIRST"))


def test_sort_nulls_first_on_a_nullable_key(duck, t):
    out = bt.from_arrow(t).sort("y", nulls_first=True).select("y").collect()
    assert_same_ordered(out, duck.sql("SELECT y FROM t ORDER BY y NULLS FIRST"))
