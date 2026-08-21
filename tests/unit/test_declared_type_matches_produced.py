"""Every column's *declared* type must be the one the engine *produces*.

`LogicalPlan.available_schema` is the engine's own static type analysis: it answers
`Dataset.schema` without reading a row, Kyber plans against it, a `write` validates against
it, and it is the only check the device tier has that runs without a GPU
(`.claude/rules/device-tier.md`). A node it has no rule for declares ``null`` — a type no
execution ever produces — and that is not a harmless "unknown": it is a wrong answer to
`Dataset.schema`, and every consumer above believes it.

A sweep of the whole expression surface, comparing the declared type against a **non-empty**
execution, found the families below. They fall into two groups, and the second is worse than
the first:

* Nodes with no inference rule at all — the temporal *constructors* (`make_date`,
  `make_timestamp`, `from_unix_*`), four `.json` accessors, the two list higher-order
  operations, `.struct.keys()`, and every `decimal <op> int` (the shape `price * qty`
  produces, which is the single most ordinary thing a money column does). Each declared
  ``null`` while producing a real type.
* A node the *optimizer* retyped. `x * 1 -> x` and `x - 0 -> x` fired without checking that
  `x` keeps its type, and a **Boolean** `x` does not: the arithmetic promotes it to Int64,
  so dropping the operation returned a boolean column where the declared schema — computed
  before the rewrite ran — said Int64. No differential test could see it, because
  `assert_same` compares values and DuckDB rejects `b * 1` outright, so there was nothing to
  compare against.

The oracle here is a non-empty `collect()`. A *zero-row* execution degenerates derived
columns to ``null``, which is the degenerate answer this inference exists to replace, so it
cannot be the oracle — the same reason `test_available_schema.py` gives.
"""

from __future__ import annotations

import datetime as dt
import decimal
import itertools

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, lit
from batcher.plan.functions.collection import element
from batcher.plan.functions.temporal import (
    from_epoch,
    from_unix_date,
    make_date,
    make_timestamp,
)

D = decimal.Decimal


def _assert_declared_is_produced(ds, column: str = "r") -> pa.DataType:
    """The declared type equals the produced one; returns it."""
    declared = ds.schema.field(column).type
    produced = ds.collect().schema.field(column).type
    assert declared == produced, (
        f"declared {declared} but produced {produced} — Dataset.schema is answering with a "
        "type no execution makes"
    )
    return produced


# --- 1. nodes that had no inference rule -------------------------------------


def test_the_temporal_constructors_declare_the_type_they_build():
    """`make_date`/`make_timestamp`/`from_unix_*` all declared `null`."""
    ds = bt.from_pydict({"n": [1, 2, 3]})
    n = col("n")
    cases = {
        "from_unix_date": (from_unix_date(n), pa.date32()),
        "make_date": (make_date(n + 2020, n, n), pa.date32()),
        "from_epoch_s": (from_epoch(n, "s"), pa.timestamp("us")),
        "from_epoch_ms": (from_epoch(n, "ms"), pa.timestamp("us")),
        "make_timestamp": (make_timestamp(n + 2020, n, n, n, n, n), pa.timestamp("us")),
    }
    for name, (expr, want) in cases.items():
        produced = _assert_declared_is_produced(ds.select(r=expr))
        assert produced == want, f"{name}: {produced}"


def test_the_json_accessors_declare_their_types():
    ds = bt.from_pydict({"j": ['{"a": 1, "b": [1, 2]}', '{"a": 2, "b": []}']})
    j = col("j")
    for expr, want in (
        (j.json.type_of(), pa.string()),
        (j.json.keys(), pa.list_(pa.string())),
        (j.json.values(), pa.list_(pa.string())),
        (j.json.array_length("$.b"), pa.int64()),
    ):
        assert _assert_declared_is_produced(ds.select(r=expr)) == want


def test_struct_keys_declares_a_list_of_text():
    """`.struct.keys()` shares a node with `.map.keys()`, whose rule required a Map."""
    ds = bt.from_pydict({"s": [{"k": 1, "j": 2}, {"k": 3, "j": 4}]})
    assert _assert_declared_is_produced(ds.select(r=col("s").struct.keys())) == pa.list_(
        pa.string()
    )


def test_the_list_higher_order_operations_declare_their_element_type():
    """`transform` takes its element type from the *body*; `filter` keeps the input's."""
    ds = bt.from_pydict({"a": [[1, 2], [3]]})
    a = col("a")
    assert _assert_declared_is_produced(ds.select(r=a.list.filter(element() > 1))) == pa.list_(
        pa.int64()
    )
    assert _assert_declared_is_produced(ds.select(r=a.list.transform(element() * 2))) == pa.list_(
        pa.int64()
    )
    # The body's own type is what the list carries, so a cast inside it must be seen.
    widened = ds.select(r=a.list.transform(element().cast("float64")))
    assert _assert_declared_is_produced(widened) == pa.list_(pa.float64())
    assert _assert_declared_is_produced(ds.select(r=a.list.drop_nulls())) == pa.list_(pa.int64())


@pytest.mark.parametrize(
    ("left", "right", "op"),
    [
        (a, b, op)
        for a, b in itertools.product(
            ["d10_2", "d5_0", "d20_4", "d12_3", "i", "f"], ["d10_2", "d5_0", "d20_4", "i", "f"]
        )
        for op in ("add", "sub", "mul", "mod")
    ],
)
def test_decimal_arithmetic_declares_what_it_produces(left, right, op):
    """`decimal <op> int` and `decimal <op> double` both declared `null`.

    The engine's rule is exact and was measured rather than assumed: the integer side is
    coerced to the *decimal side's own type* and the two-decimal rule then applies, while a
    float dominates and the whole thing becomes a double.
    """
    shapes = [(10, 2), (5, 0), (20, 4), (12, 3)]
    cols = {
        f"d{p}_{s}": pa.array([D(1).quantize(D(1).scaleb(-s))], pa.decimal128(p, s))
        for p, s in shapes
    }
    cols["i"] = pa.array([3], pa.int64())
    cols["f"] = pa.array([2.5], pa.float64())
    ds = bt.from_arrow(pa.table(cols))
    expr = {
        "add": col(left) + col(right),
        "sub": col(left) - col(right),
        "mul": col(left) * col(right),
        "mod": col(left) % col(right),
    }[op]
    _assert_declared_is_produced(ds.select(r=expr))


# --- 2. the identity rewrite that retyped a column ---------------------------


def test_multiplying_a_boolean_by_one_stays_an_integer():
    """`b * 1 -> b` dropped an operation that was doing the promotion."""
    ds = bt.from_pydict({"b": [True, False, None]})
    out = ds.select(r=col("b") * lit(1))
    assert _assert_declared_is_produced(out) == pa.int64()
    assert out.to_pydict()["r"] == [1, 0, None]


def test_subtracting_zero_from_a_boolean_stays_an_integer():
    ds = bt.from_pydict({"b": [True, False, None]})
    out = ds.select(r=col("b") - lit(0))
    assert _assert_declared_is_produced(out) == pa.int64()
    assert out.to_pydict()["r"] == [1, 0, None]


@pytest.mark.parametrize("values", [[3, 4], [1.5, -0.0]])
def test_the_identity_rewrite_still_fires_for_a_numeric_column(values):
    """The guard must not cost the optimization where it was always sound.

    Checked on the plan rather than on the result, because the rewrite is invisible in the
    values by construction — that is what makes it an identity.
    """
    from batcher.kyber.rules.normalize.simplify import simplify_expressions
    from batcher.plan.expr_ir import Binary

    ds = bt.from_pydict({"x": values})
    for expr in (col("x") * lit(1), col("x") - lit(0)):
        plan = ds.select(r=expr)._plan
        simplified = simplify_expressions(plan)
        assert not any(isinstance(item.expr, Binary) for item in simplified.items), (
            "the identity was left in the plan for a numeric column"
        )


def test_the_identity_rewrite_does_not_fire_for_a_boolean_column():
    from batcher.kyber.rules.normalize.simplify import simplify_expressions
    from batcher.plan.expr_ir import Binary

    ds = bt.from_pydict({"b": [True, False]})
    plan = ds.select(r=col("b") * lit(1))._plan
    simplified = simplify_expressions(plan)
    assert any(isinstance(item.expr, Binary) for item in simplified.items), (
        "the multiply was dropped from a boolean column, which retypes it"
    )


def test_a_string_times_one_is_refused_rather_than_answered():
    """The rewrite was also turning an invalid operation into an identity.

    DuckDB rejects `s * 1` outright. Batcher answered the string back, because the identity
    fired before anything looked at the operand's type — so the query "succeeded" with a
    column the arithmetic could never have produced.
    """
    ds = bt.from_pydict({"s": ["a", "b"]})
    with pytest.raises(RuntimeError):
        ds.select(r=col("s") * lit(1)).collect()


def test_a_date_shift_by_zero_is_still_a_date_shift():
    """`date - 0` is a *real* operation — DuckDB shifts a DATE by an integer day count.

    The guard must not take this one out with the boolean: with the identity no longer
    dropped, the engine computes the shift itself and answers the same date, which is what
    `date - 1` does one day earlier.
    """
    ds = bt.from_pydict({"d": [dt.date(2024, 3, 5), dt.date(2020, 1, 1)]})
    out = ds.select(r=col("d") - lit(0))
    assert _assert_declared_is_produced(out) == pa.date32()
    assert out.to_pydict()["r"] == [dt.date(2024, 3, 5), dt.date(2020, 1, 1)]


def test_a_timestamp_minus_zero_now_refuses_the_way_duckdb_does():
    """The rewrite was answering three queries DuckDB rejects, and now none of them.

    `timestamp - <integer>` and `<non-numeric> * 1` have no meaning — DuckDB raises a binder
    error for both — but the identity fired before anything looked at the operand, so the
    engine returned the column unchanged whenever the constant happened to be the identity
    element. `t - 1` raised while `t - 0` succeeded: the same expression family answering
    two different ways depending on a literal.
    """
    ds = bt.from_pydict({"t": [dt.datetime(2024, 3, 5)]})
    with pytest.raises(RuntimeError):
        ds.select(r=col("t") - lit(0)).collect()
    dates = bt.from_pydict({"d": [dt.date(2024, 3, 5)]})
    with pytest.raises(RuntimeError):
        dates.select(r=col("d") * lit(1)).collect()


# --- 3. the window family ----------------------------------------------------


def test_a_windowed_sum_over_a_decimal_declares_the_double_it_produces():
    """The one place in the window family where the declaration disagreed with a run.

    A *windowed* sum folds in f64 — `bc_runtime::window::agg` accumulates every numeric
    fold there as a double — so a decimal input comes back as a double. The *grouped* sum
    keeps the decimal, and so does DuckDB's windowed `SUM`, which is why the declaration
    said `decimal(10,2)`: it was describing the aggregate's rule, not the window's.

    Declared as what the engine makes, because that is this function's contract. The
    representation divergence itself is recorded in `competitor_parity_census.md`; closing
    it means giving the window fold a decimal accumulator.
    """
    ds = bt.from_arrow(
        pa.table(
            {
                "g": pa.array(["a", "a", "b"]),
                "i": pa.array([1, 2, 1], pa.int64()),
                "m": pa.array(
                    [D("1.25"), D("22.50"), D("3.00")],
                    pa.decimal128(10, 2),
                ),
            }
        )
    )
    windowed = ds.select(r=col("m").sum().over(partition_by="g", order_by="i"))
    assert _assert_declared_is_produced(windowed) == pa.float64()
    assert windowed.to_pydict()["r"] == [1.25, 23.75, 3.0]
    # The grouped sum is the one that keeps the decimal — the two genuinely differ.
    grouped = ds.group_by("g").agg(r=col("m").sum())
    assert _assert_declared_is_produced(grouped) == pa.decimal128(10, 2)


@pytest.mark.parametrize("column", ["i", "f"])
def test_a_windowed_sum_over_an_integer_or_float_is_unchanged(column):
    """The guard must not retype the shapes that were always right."""
    ds = bt.from_pydict({"g": ["a", "a", "b"], "i": [1, 2, 3], "f": [1.5, 2.5, 3.5]})
    out = ds.select(r=col(column).sum().over(partition_by="g", order_by="i"))
    want = pa.int64() if column == "i" else pa.float64()
    assert _assert_declared_is_produced(out) == want
