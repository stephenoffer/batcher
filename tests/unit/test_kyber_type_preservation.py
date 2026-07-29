"""Optimization may change the plan; it may never change the output schema.

Every Kyber rule is supposed to be semantics-preserving, and the *type* of each output
column is part of the semantics. This is easy to break in one specific way, and it has been
broken that way twice here: a rule deletes an argument it has proved unreachable, forgetting
that the constructs whose arguments it is deleting take their type from the **join** of all
of them. `coalesce(5, double_col)` is a DOUBLE because of the argument that never supplies a
value, and `CASE WHEN c THEN 1 ELSE (CASE WHEN c THEN 2.5 ELSE 3 END) END` is a DOUBLE
because of a branch no row can reach. Dropping either is correct by value and wrong by type,
and it hands the user a differently-typed column.

Nothing else catches it. The rewrite is semantics-preserving on every row, so a
result-comparing differential test passes; only the schema moves. So this module compares
schemas directly, across the constructs where a type join is at stake, and does it over the
whole optimizer rather than rule by rule -- the bug is in whichever rule fires, and a
per-rule test only ever covers the rules someone thought to list.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.plan.expr_ir.constructors import coalesce
from batcher.plan.types import infer_type


def _ds():
    return bt.from_pydict(
        {
            "i": [1, 2, 3],
            "j": [4, 5, 6],
            "f": [1.5, 2.5, 3.5],
            "s": ["a", "b", "c"],
            "t": [dt.datetime(2021, 1, 1), dt.datetime(2021, 6, 1), dt.datetime(2022, 1, 1)],
        }
    )


def _c():
    return col("i") > lit(1)


#: Expressions whose type comes from a join over several arguments, each written so that a
#: *later* argument is what widens the result. A rule that deletes such an argument as
#: unreachable narrows the column, which is exactly what this module exists to catch.
JOINS = {
    "coalesce_int_then_double": lambda: coalesce(lit(5), col("f")),
    "coalesce_nonnull_col_then_double": lambda: coalesce(col("i"), col("f")),
    "coalesce_three_args_widening_last": lambda: coalesce(col("i"), lit(7), col("f")),
    "coalesce_nested_widening": lambda: coalesce(coalesce(lit(1), col("i")), col("f")),
    "case_dead_branch_widens": lambda: (
        bt.when(_c()).then(lit(1)).otherwise(bt.when(_c()).then(lit(2.5)).otherwise(lit(3)))
    ),
    "case_duplicate_condition_widens": lambda: (
        bt.when(_c()).then(lit(1)).otherwise(bt.when(_c()).then(lit(2.5)).otherwise(col("f")))
    ),
    "case_otherwise_widens": lambda: bt.when(_c()).then(lit(1)).otherwise(col("f")),
    "greatest_widening_last": lambda: bt.greatest(col("i"), col("i"), col("f")),
    "least_widening_last": lambda: bt.least(col("i"), col("i"), col("f")),
    "greatest_duplicate_then_double": lambda: bt.greatest(lit(1), lit(1), col("f")),
    "coalesce_under_arithmetic": lambda: coalesce(lit(5), col("f")) + lit(1),
    "case_under_arithmetic": lambda: bt.when(_c()).then(lit(1)).otherwise(col("f")) * lit(2),
}


@pytest.mark.parametrize("name", sorted(JOINS))
def test_optimizing_preserves_the_output_type(name):
    plan = _ds().select(r=JOINS[name]())._plan
    before = plan.available_schema()
    after = optimize_logical(plan).available_schema()
    assert before is not None and after is not None, "the fixture must have a known schema"
    assert before.arrow.field("r").type == after.arrow.field("r").type, (
        f"{name}: optimization changed the output type from "
        f"{before.arrow.field('r').type} to {after.arrow.field('r').type}"
    )


@pytest.mark.parametrize("name", sorted(JOINS))
def test_optimizing_preserves_the_whole_schema(name):
    # The column above is the one at risk, but a rule that rewrites a neighbour has the same
    # opportunity, so the full schema is compared too.
    plan = _ds().select(r=JOINS[name](), keep=col("s"), n=col("i") + lit(1))._plan
    assert plan.available_schema().arrow == optimize_logical(plan).available_schema().arrow


def test_the_battery_actually_exercises_a_join():
    # A guard on the fixtures: if these expressions stopped being *mixed*-type, every
    # assertion above would hold vacuously. Each must produce a DOUBLE built from an
    # argument that is not the first.
    schema = _ds()._plan.available_schema()
    for name, build in JOINS.items():
        assert infer_type(build(), schema) == pa.float64(), (
            f"{name} no longer depends on a widening argument"
        )


#: A wider sweep than `JOINS`, one entry per rule family rather than per hazard. The join
#: constructs above are where the invariant has actually broken, but the invariant itself is
#: unconditional -- optimization changes the plan, never the schema -- so it is worth
#: asserting across the whole surface rather than only where a bug has already appeared.
FAMILIES = {
    "abs_then_floor": lambda: col("f").abs().floor(),
    "sign_of_integer": lambda: col("j").sign(),
    "round_to_digits": lambda: col("f").round(2),
    "cast_round_trip": lambda: col("i").cast("float64").cast("int64"),
    "string_case_and_trim": lambda: col("s").str.upper().str.trim(),
    "string_length": lambda: col("s").str.len(),
    "string_slice": lambda: col("s").str.substr(1, 2),
    "date_part": lambda: col("t").dt.year(),
    "date_truncation": lambda: col("t").dt.truncate("day"),
    "date_part_of_a_date": lambda: col("d").dt.month(),
    "coalesce_over_strings": lambda: coalesce(col("s"), col("u")),
    "least_of_integers": lambda: bt.least(col("i"), col("j")),
    "case_over_strings": lambda: bt.when(col("b")).then(col("s")).otherwise(lit("z")),
    "list_length": lambda: col("l").list.len(),
    "integer_arithmetic": lambda: col("i") + lit(1) - lit(1),
    "mixed_arithmetic": lambda: col("i") * col("f"),
    "null_check": lambda: col("i").is_null(),
    "membership": lambda: col("i").is_in([1, 2]),
    "negated_comparison": lambda: ~(col("i") == lit(1)),
    "float_division": lambda: col("f") / lit(2.0),
    "integer_modulo": lambda: col("i") % lit(3),
    "abs_of_a_negation": lambda: (lit(0) - col("i")).abs(),
    "case_inside_coalesce": lambda: coalesce(
        bt.when(col("b")).then(col("i")).otherwise(lit(2)), col("f")
    ),
}


def _wide_ds():
    return bt.from_pydict(
        {
            "i": [1, 2, 3],
            "j": [-4, 0, 5],
            "f": [1.5, -2.5, 3.5],
            "s": ["Abc", "b_b", "%c%"],
            "u": ["x", "y", "z"],
            "t": [
                dt.datetime(2021, 1, 1, 3, 4, 5),
                dt.datetime(2021, 6, 30),
                dt.datetime(2022, 2, 1),
            ],
            "d": [dt.date(2021, 1, 1), dt.date(2021, 6, 30), dt.date(2022, 2, 1)],
            "l": [[1, 2], [3], []],
            "b": [True, False, True],
        }
    )


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_optimizing_preserves_the_type_across_families(name):
    plan = _wide_ds().select(r=FAMILIES[name]())._plan
    before = plan.available_schema().arrow.field("r").type
    after = optimize_logical(plan).available_schema().arrow.field("r").type
    assert before == after, f"{name}: optimization changed the output type {before} -> {after}"
