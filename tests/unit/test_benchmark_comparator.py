"""The benchmark correctness gate (`benchmarks/harness/compare.py`) against false verdicts.

Every case here is one the gate once got wrong: a wrong answer reported as a match, or a
right one reported as a mismatch. A comparator that passes a wrong answer turns a benchmark
row into a timed claim about work nobody checked, so each case pins the verdict, not just
that the call returns.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from harness import canonical_names, results_match, type_differences  # noqa: E402
from harness.matching import MAX_GROUP_ROWS  # noqa: E402

pytestmark = pytest.mark.unit

INF = float("inf")


def _dec(value: str, scale: int) -> pa.Table:
    return pa.table({"x": pa.array([Decimal(value)], type=pa.decimal128(20, scale))})


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        # A finite answer against an infinite one: `inf <= FLOAT_RTOL * inf` used to pass it.
        pytest.param(pa.table({"x": [1.0]}), pa.table({"x": [INF]}), False, id="finite-vs-inf"),
        pytest.param(pa.table({"x": [INF]}), pa.table({"x": [1.0]}), False, id="inf-vs-finite"),
        # `inf - inf` is NaN, so identical infinities used to fail.
        pytest.param(pa.table({"x": [INF]}), pa.table({"x": [INF]}), True, id="same-inf"),
        pytest.param(pa.table({"x": [-INF]}), pa.table({"x": [-INF]}), True, id="same-neg-inf"),
        pytest.param(pa.table({"x": [-INF]}), pa.table({"x": [INF]}), False, id="opposite-inf"),
        pytest.param(
            pa.table({"x": [float("nan"), None]}),
            pa.table({"x": [None, float("nan")]}),
            True,
            id="nan-and-null",
        ),
    ],
)
def test_special_float_values(reference: pa.Table, candidate: pa.Table, expected: bool) -> None:
    assert results_match(reference, candidate)[0] is expected


def test_the_tolerance_is_symmetric() -> None:
    # 1e-9 relative of the *larger* side: with a candidate-relative tolerance these two
    # verdicts could differ at the boundary.
    a, b = pa.table({"x": [1.0e12]}), pa.table({"x": [1.0e12 + 900.0]})
    assert results_match(a, b)[0] is results_match(b, a)[0] is True
    far = pa.table({"x": [1.0e12 + 5000.0]})
    assert results_match(a, far)[0] is results_match(far, a)[0] is False


def test_case_colliding_columns_are_both_compared() -> None:
    ref = pa.table([[1], [2]], names=["x", "X"])
    assert results_match(ref, pa.table([[1], [2]], names=["x", "X"]))[0] is True
    ok, msg = results_match(ref, pa.table([[1], [999]], names=["x", "X"]))
    assert ok is False
    assert "999" in msg


def test_duplicate_names_stay_distinct() -> None:
    table = pa.table([[1], [2], [3]], names=["x", "x", "x#1"])
    names = canonical_names(table)
    assert len(set(names)) == 3


def test_decimals_compare_exactly() -> None:
    ok, _ = results_match(_dec("10000000000000000.01", 2), _dec("10000000000000000.02", 2))
    assert ok is False
    assert results_match(_dec("10000000000000000.01", 2), _dec("10000000000000000.01", 2))[0]
    # Different scales of the same value, and a decimal against an equal integer, still match.
    assert results_match(_dec("5.10", 2), _dec("5.1", 1))[0] is True
    assert results_match(_dec("5.00", 2), pa.table({"x": [5]}))[0] is True


def test_a_large_integer_against_a_float_is_compared_not_raised() -> None:
    big = pa.table({"x": pa.array([9007199254740993], type=pa.int64())})
    ok, msg = results_match(big, pa.table({"x": [9007199254740992.0]}))
    assert ok is True, msg
    ok, _ = results_match(big, pa.table({"x": [1.0e16]}))
    assert ok is False


def test_zero_column_results_keep_their_row_count() -> None:
    two, one = pa.table({"x": [1, 2]}).select([]), pa.table({"x": [1]}).select([])
    ok, msg = results_match(two, one)
    assert ok is False
    assert "row count" in msg
    assert results_match(two, pa.table({"y": [7, 8]}).select([]))[0] is True


def test_rows_are_matched_within_tolerance_not_by_sort_position() -> None:
    # Pairing by `b`, both rows agree on `a` within tolerance; sorting on the rounded
    # floats pairs them the other way round and used to fail on `b`.
    ref = pa.table({"a": [1e9, 1e9 + 0.5], "b": [1, 2]})
    assert results_match(ref, pa.table({"a": [1e9 + 0.6, 1e9], "b": [1, 2]}))[0] is True
    # The same shape with only float columns: resolved by matching inside the group.
    ref = pa.table({"a": [1e9, 1e9 + 0.5], "c": [1.0, 2.0]})
    assert results_match(ref, pa.table({"a": [1e9 + 0.6, 1e9], "c": [1.0, 2.0]}))[0] is True
    ok, _ = results_match(ref, pa.table({"a": [1e9 + 0.6, 1e9], "c": [1.0, 3.0]}))
    assert ok is False


def test_an_oversized_ambiguous_group_is_reported_not_passed() -> None:
    n = MAX_GROUP_ROWS + 1
    ref = pa.table({"a": [float(i) for i in range(n)], "c": [0.0] * n})
    bad = pa.table({"a": [float(i) for i in range(n)], "c": [0.0] * (n - 1) + [1.0]})
    ok, msg = results_match(ref, bad)
    assert ok is False
    assert "not searched" in msg


def test_type_differences_are_a_separate_opt_in_gate() -> None:
    narrow = pa.table({"x": pa.array([1], type=pa.int32())})
    wide = pa.table({"x": pa.array([1], type=pa.int64())})
    assert results_match(narrow, wide)[0] is True
    assert type_differences(narrow, wide) == ["x: int32 vs int64"]
    ok, msg = results_match(narrow, wide, strict_types=True)
    assert ok is False
    assert "type mismatch" in msg
    assert results_match(wide, wide, strict_types=True)[0] is True
