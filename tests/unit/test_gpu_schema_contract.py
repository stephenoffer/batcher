"""The device tier's free oracle: the engine declares the columns, the device must produce them.

Every defect the device tier has shipped is a column-*type* defect with correct values — a DATE
coming back `timestamp[ms]`, an integer `abs` widening to double, an empty cuDF string column
converting as Arrow `null`. All three were right in CI and wrong on hardware, because the
translator's suite runs on pandas and a pandas stand-in cannot see a cuDF-only type.

`LogicalPlan.available_schema` can. It is the engine's own static type analysis — the one
`Dataset.schema` is answered from — and it runs before either backend does, on no rows, with no
device. So the contract this file pins is: whatever the device produced, its columns are the
columns the engine declared, or the query uses the CPU engine.

The cases below are the three shipped defects restated as *schemas*, which is what makes them
catchable here. A value comparison sees none of them.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.api.terminal.gpu_backend.verify import (
    declared_schema,
    enforce_schema_contract,
    schema_contract_violation,
)

pytestmark = pytest.mark.unit


def _plan(build, table: pa.Table):
    """The logical plan `build` produces over `table`."""
    return build(bt.from_arrow(table))._plan


NUMBERS = pa.table(
    {
        "g": pa.array([1, 1, 2], pa.int64()),
        "n": pa.array([1, -2, 3], pa.int32()),
        "v": pa.array([1.5, 2.5, 3.5], pa.float64()),
        "d": pa.array([dt.date(2024, 1, 2)] * 3, pa.date32()),
        "s": pa.array(["a", "b", "c"], pa.string()),
    }
)


# --- what the engine declares ---------------------------------------------------------


def test_the_engine_declares_a_plans_columns_without_running_it():
    """The whole point: no rows, no engine call, no device — just the plan."""
    declared = declared_schema(_plan(lambda ds: ds.group_by("g").agg(t=col("v").sum()), NUMBERS))
    assert declared is not None
    assert declared.names == ["g", "t"]
    assert declared.field("t").type == pa.float64()


def test_a_narrow_column_is_declared_at_the_width_the_boundary_produces():
    """`n` is `int32` in storage and `int64` out of the engine; the device must match the latter."""
    declared = declared_schema(_plan(lambda ds: ds.select("n"), NUMBERS))
    assert declared is not None
    assert declared.field("n").type == pa.int64()


def test_an_uninferable_plan_declares_nothing_and_the_result_passes():
    """`available_schema` is all-or-nothing, so "no opinion" must not read as "violation".

    A `map_batches` UDF returns whatever it returns, which no static analysis can predict. The
    contract then has nothing to hold the result to and lets it through — the position the tier
    was in before this check existed, not a stricter one.
    """
    plan = _plan(lambda ds: ds.map_batches(lambda b: b), NUMBERS)
    assert declared_schema(plan) is None
    result = pa.table({"anything": pa.array([1], pa.int64())})
    assert schema_contract_violation(result, plan) is None
    assert enforce_schema_contract(result, plan) is result


# --- the three shipped defects, as schemas --------------------------------------------


def test_a_date_returned_as_a_timestamp_is_caught():
    """Defect 1: neither dataframe library has a calendar-day type.

    pandas' `astype` happened to land back on `date32`; a device's cannot, so the same
    translation returned `timestamp[ms]` from a GPU and `date32` from CI. Same day, wrong
    column, and a fan-out cannot concatenate it with a CPU-recovered shard.
    """
    plan = _plan(lambda ds: ds.select("d"), NUMBERS)
    diverged = pa.table({"d": pa.array([0], pa.timestamp("ms"))})
    difference = schema_contract_violation(diverged, plan)
    assert difference is not None
    assert "'d'" in difference and "timestamp" in difference and "date32" in difference


def test_an_integer_abs_widened_to_double_is_caught():
    """Defect 2: `abs` is the one unary math function that keeps its input's integer type.

    Routing an integer `abs` through the float ufunc path returns `1.0` where the engine
    returns `1` — not a wrong number, a wrong column.
    """
    plan = _plan(lambda ds: ds.select(a=col("n").abs()), NUMBERS)
    assert declared_schema(plan).field("a").type == pa.int64()
    diverged = pa.table({"a": pa.array([1.0], pa.float64())})
    assert "double" in (schema_contract_violation(diverged, plan) or "")


def test_an_empty_string_column_typed_as_null_is_caught():
    """Defect 3: an **empty** cuDF string column converts to Arrow `null`.

    TPC-H q15 is empty on the benchmark data, so it returned `s_name`, `s_address` and
    `s_phone` as `null` where the engine returns `string`. Every value agreed — there were
    none. That is exactly the failure a value comparison cannot see and a schema can.
    """
    plan = _plan(lambda ds: ds.filter(col("g") > 99).select("s"), NUMBERS)
    diverged = pa.table({"s": pa.array([], pa.null())})
    assert "null" in (schema_contract_violation(diverged, plan) or "")


# --- the contract's edges -------------------------------------------------------------


def test_a_missing_or_renamed_column_is_caught_before_any_type_is_compared():
    plan = _plan(lambda ds: ds.select("g", "v"), NUMBERS)
    renamed = pa.table({"g": pa.array([1], pa.int64()), "value": pa.array([1.0], pa.float64())})
    difference = schema_contract_violation(renamed, plan)
    assert difference is not None and "columns" in difference


def test_nullability_and_metadata_are_not_part_of_the_contract():
    """Or every query would report a difference that says nothing about the answer.

    The engine's static analysis marks every inferred field nullable; a device result's flags
    come from whatever the library happened to build. Comparing them would make the check fire
    constantly and be switched off, which is worse than not having it.
    """
    plan = _plan(lambda ds: ds.select("g"), NUMBERS)
    strict = pa.schema([pa.field("g", pa.int64(), nullable=False)], metadata={b"k": b"v"})
    result = pa.table({"g": pa.array([1], pa.int64())}).cast(strict)
    assert schema_contract_violation(result, plan) is None


def test_an_agreeing_result_is_returned_unchanged_and_a_diverging_one_falls_back():
    """The two outcomes the router acts on: the device result, or `None` for the CPU engine."""
    plan = _plan(lambda ds: ds.select("g"), NUMBERS)
    agreeing = pa.table({"g": pa.array([1, 2], pa.int64())})
    assert enforce_schema_contract(agreeing, plan) is agreeing
    diverging = pa.table({"g": pa.array([1, 2], pa.int32())})
    assert enforce_schema_contract(diverging, plan) is None


# --- the translator itself, held to the contract ---------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.select("g", "v"), id="project"),
        pytest.param(lambda ds: ds.filter(col("g") > 1), id="filter"),
        pytest.param(lambda ds: ds.select(a=col("n").abs()), id="integer-abs"),
        pytest.param(lambda ds: ds.select(w=col("n") + 1), id="narrow-arithmetic"),
        pytest.param(lambda ds: ds.select("d"), id="date-passthrough"),
        pytest.param(lambda ds: ds.filter(col("g") > 99).select("s"), id="empty-string"),
        pytest.param(lambda ds: ds.group_by("g").agg(t=col("v").sum()), id="aggregate"),
        pytest.param(lambda ds: ds.group_by("g").agg(c=bt.count()), id="count"),
        pytest.param(lambda ds: ds.select("g").distinct(), id="distinct"),
        pytest.param(lambda ds: ds.sort("v", descending=True).limit(2), id="top-n"),
    ],
)
def test_the_translator_produces_the_columns_the_engine_declares(build):
    """The oracle, run for real — on pandas, so CI proves it with no GPU.

    This is the check the router applies to every device result, applied here to the host
    backend that mirrors it. A translation whose columns drift from the engine's declaration
    fails here rather than being discovered on a cluster.
    """
    pandas = pytest.importorskip("pandas", reason="the head-runnable device backend needs pandas")
    from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
    from batcher.core.gpu_plan.execute import run_chain

    plan = _plan(build, NUMBERS)
    spec = gpu_plan_ops(plan)
    assert spec is not None, "shape should be GPU-translatable"
    be = DfBackend(pandas)
    produced = be.to_arrow(run_chain(NUMBERS, spec[1], be))
    assert schema_contract_violation(produced, plan) is None
