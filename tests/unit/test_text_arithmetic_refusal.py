"""Arithmetic on a text operand is a plan-time `PlanError`, never a ``null`` schema (AP-181).

`Dataset.schema` used to report ``x: null`` for ``select(x=col("s") + 1)``, and the error
appeared only at execution -- or not at all on an empty relation, which quietly returned a
``null`` column. The type analysis now refuses the pairs the engine always refuses.

The refusal must never claim more than the engine does, so every refused pair is executed
on the engine through the raw IR node (which bypasses the control-plane check) and must fail
there too. The pairs the engine *accepts* -- a string ``//`` a number, a date minus a string
-- must stay accepted.
"""

from __future__ import annotations

import datetime as dt
import itertools
from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir.core import Binary
from batcher.plan.types.infer import arithmetic
from batcher.plan.types.infer.arithmetic import _TEXT_REFUSING_OPS

_COLUMNS = {
    "s": pa.array(["1"]),
    "ls": pa.array(["1"], pa.large_string()),
    "b": pa.array([b"1"]),
    "i": pa.array([1]),
    "f": pa.array([1.0]),
    "t": pa.array([True]),
    "dec": pa.array([Decimal("1.00")], pa.decimal128(10, 2)),
}
_TEXT = ("s", "ls", "b")
_TABLE = pa.table(_COLUMNS)


def _refused_pairs() -> list[tuple[str, str, str]]:
    names = list(_COLUMNS)
    return [
        (op, left, right)
        for op in _TEXT_REFUSING_OPS
        for left, right in itertools.product(names, names)
        if left in _TEXT or right in _TEXT
    ]


@pytest.mark.parametrize(("op", "left", "right"), _refused_pairs())
def test_every_refused_pair_is_refused_by_the_engine_too(monkeypatch, op, left, right):
    expr = Binary(op, bt.col(left), bt.col(right))
    with pytest.raises(bt.PlanError, match="arithmetic is not defined on text"):
        _ = bt.from_arrow(_TABLE).select(x=expr).schema
    # With the plan-time check switched off, the engine rejects the same pair on its own.
    monkeypatch.setattr(arithmetic, "_refuse_text_arithmetic", lambda *_: None)
    with pytest.raises(bt.ExecutionError, match=r"(?i)arithmetic|cast"):
        bt.from_arrow(_TABLE).select(x=expr).collect()


def test_the_engine_accepted_pairs_stay_accepted():
    """`//` casts a string to Float64 and a date minus a string parses it as a date."""
    ds = bt.from_pydict({"s": ["4"], "i": [2], "d": [dt.date(2024, 1, 3)], "ds": ["2024-01-01"]})
    out = ds.select(q=bt.col("s") // bt.col("i"), gap=bt.col("d") - bt.col("ds")).to_pydict()
    assert out == {"q": [2.0], "gap": [2]}


def test_the_error_names_both_types_and_the_remedy():
    with pytest.raises(bt.PlanError) as info:
        _ = bt.from_pydict({"s": ["a"]}).select(x=bt.col("s") * 2).schema
    message = str(info.value)
    assert "'*'" in message and "string" in message and "int64" in message
    assert "cast" in message and "bt.concat" in message


def test_meta_output_type_answers_without_a_dataset():
    schema = pa.schema({"a": pa.int32(), "f": pa.float32(), "s": pa.string()})
    assert (bt.col("a") + 1).meta.output_type(schema) == pa.int64()  # widened like the engine
    assert (bt.col("a") * bt.col("f")).meta.output_type(schema) == pa.float64()
    assert (bt.col("a") > 1).meta.output_type(schema) == pa.bool_()
    assert bt.col("missing").meta.output_type(schema) is None
    with pytest.raises(bt.PlanError):
        (bt.col("s") - 1).meta.output_type(schema)


def test_meta_output_type_agrees_with_the_engine():
    """The answer is the type a real run produces, over a schema with narrow columns."""
    table = pa.table({"a": pa.array([1, 2], pa.int32()), "f": pa.array([0.5, 1.5], pa.float32())})
    for expr in [bt.col("a") // 2, bt.col("a") / 2, bt.col("a") + bt.col("f"), bt.col("a").abs()]:
        ran = bt.from_arrow(table).select(x=expr).collect().schema.field("x").type
        assert expr.meta.output_type(table.schema) == ran
