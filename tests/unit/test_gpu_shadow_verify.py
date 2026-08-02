"""The device tier's runtime oracle: does the comparison actually see a divergence?

`core/gpu_plan/` is a second implementation of the engine's semantics — cuDF has no maintained
Rust binding, so unlike the Cranelift JIT it cannot share `bc_expr::Expr` and must translate
the same JSON IR onto a dataframe library. Its own suite runs that translator on **pandas**,
never on cuDF, because CI has no GPU, and two divergences have already shipped through that
gap. Both were *column type* bugs with correct values: a DATE returning `timestamp[ms]` on a
real device where pandas gave `date32`, and an integer `abs` widening to double.

`shadow_verify` is the answer until there is a GPU CI lane. These tests pin the part that has
to be right for it to be worth running — that the comparison catches a schema change a
row-value check would sail past, and that it does not cry wolf over floating-point
accumulation order, which legitimately differs between a device and the CPU engine.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.api.terminal.gpu_backend.verify import DeviceDivergence, compare_results

pytestmark = pytest.mark.unit


def _table(**columns) -> pa.Table:
    return pa.table(columns)


def test_identical_results_agree():
    left = _table(g=pa.array([1, 2], pa.int64()), s=pa.array([1.0, 2.0], pa.float64()))
    assert compare_results(left, left) is None


def test_row_order_alone_is_not_a_divergence():
    """A group-by fixes no output order on either backend, so order is not the contract."""
    gpu = _table(g=pa.array([2, 1], pa.int64()), s=pa.array([20.0, 10.0], pa.float64()))
    cpu = _table(g=pa.array([1, 2], pa.int64()), s=pa.array([10.0, 20.0], pa.float64()))
    assert compare_results(gpu, cpu) is None


def test_a_column_type_change_is_caught_though_every_value_matches():
    """The failure mode both shipped device bugs had: right numbers, wrong column.

    A value-only comparison passes this, and a sharded fan-out then fails to concatenate a
    device shard's `int32` with a CPU-recovered shard's `int64`.
    """
    gpu = _table(n=pa.array([1, 2, 3], pa.int32()))
    cpu = _table(n=pa.array([1, 2, 3], pa.int64()))

    difference = compare_results(gpu, cpu)

    assert difference is not None
    assert "int32" in difference and "int64" in difference


def test_a_date_returned_as_a_timestamp_is_caught():
    """The exact divergence `remember_date_alias` records: right day, wrong type."""
    import datetime

    day = datetime.date(1995, 2, 3)
    gpu = _table(d=pa.array([datetime.datetime(1995, 2, 3)], pa.timestamp("ms")))
    cpu = _table(d=pa.array([day], pa.date32()))

    difference = compare_results(gpu, cpu)

    assert difference is not None
    assert "timestamp" in difference and "date32" in difference


def test_a_renamed_or_reordered_column_is_caught():
    gpu = _table(b=pa.array([1], pa.int64()), a=pa.array([2], pa.int64()))
    cpu = _table(a=pa.array([2], pa.int64()), b=pa.array([1], pa.int64()))
    assert compare_results(gpu, cpu) is not None


def test_a_wrong_value_is_caught():
    gpu = _table(g=pa.array([1, 2], pa.int64()))
    cpu = _table(g=pa.array([1, 3], pa.int64()))

    difference = compare_results(gpu, cpu)

    assert difference is not None and "column 'g'" in difference


def test_a_missing_row_is_caught():
    gpu = _table(g=pa.array([1], pa.int64()))
    cpu = _table(g=pa.array([1, 2], pa.int64()))
    assert compare_results(gpu, cpu) is not None


def test_float_accumulation_order_is_not_a_divergence():
    """A device sums in a different order; the last bits of a large SUM legitimately differ.

    Reporting that would make the mode unusable — every aggregate would look broken — so the
    tolerance has to absorb it while staying far tighter than any real kernel error.
    """
    total = 1.0e9
    gpu = _table(s=pa.array([total], pa.float64()))
    cpu = _table(s=pa.array([total + total * 1e-13], pa.float64()))
    assert compare_results(gpu, cpu) is None


def test_a_real_numeric_error_is_still_caught_under_the_tolerance():
    """One part in 1e6 is not accumulation noise — a wrong kernel must not hide."""
    gpu = _table(s=pa.array([1.0e9], pa.float64()))
    cpu = _table(s=pa.array([1.0e9 * (1 + 1e-6)], pa.float64()))
    assert compare_results(gpu, cpu) is not None


def test_nulls_and_nan_compare_as_themselves():
    nan = float("nan")
    both_nan = _table(x=pa.array([nan], pa.float64()))
    assert compare_results(both_nan, both_nan) is None
    null_vs_nan = compare_results(_table(x=pa.array([None], pa.float64())), both_nan)
    assert null_vs_nan is not None


def test_a_divergence_is_classified_as_a_defect_not_a_decline(caplog):
    """`note_gpu_failure` must log a divergence loudly — it is never "the device declined"."""
    import logging

    from batcher.api.terminal.gpu_backend.failure import note_gpu_failure

    with caplog.at_level(logging.WARNING):
        note_gpu_failure("shadow-verify", DeviceDivergence("column 'n' is int32 vs int64"))

    assert any(record.levelno >= logging.WARNING for record in caplog.records), (
        "a device divergence was recorded quietly, which is how the two shipped ones survived"
    )
