"""`Dataset.equals(ordered=False)` compares multisets without a per-row Python touch.

The comparison used to be `sorted(map(repr, table.to_pylist()))` on both sides: a Python
dict and a repr string per row, in the control plane, which is the one thing it must never
do (`CLAUDE.md` invariant #2). Sorting both relations by every column answers the same
question in compiled Arrow code and ran 33x faster on a million rows.

The sort is only sound where the ordering and Arrow's equality agree on which values are
the same value, which floating point, nested, and dictionary columns do not — floats
disagree in *both* directions (`-0.0 == 0.0` but `NaN != NaN` under Arrow equality, and the
reverse row-wise). Those keep the row-wise path, and the tests below pin that the observable
answer is identical either way.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.dataset.frame import _multiset_sortable


def _ds(table: pa.Table):
    return bt.from_arrow(table)


# --- the fast path is taken exactly where it is sound --------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arrow_type", "sortable"),
    [
        (pa.int64(), True),
        (pa.string(), True),
        (pa.bool_(), True),
        (pa.timestamp("us"), True),
        (pa.binary(), True),
        (pa.decimal128(10, 2), True),
        (pa.float64(), False),
        (pa.float32(), False),
        (pa.list_(pa.int64()), False),
        (pa.map_(pa.string(), pa.int64()), False),
        (pa.dictionary(pa.int32(), pa.string()), False),
    ],
)
def test_sortable_guard_matches_what_arrow_can_actually_do(arrow_type, sortable):
    assert _multiset_sortable(pa.schema([pa.field("c", arrow_type)])) is sortable


@pytest.mark.unit
def test_guard_rejects_a_schema_where_any_one_column_is_unsafe():
    schema = pa.schema([pa.field("ok", pa.int64()), pa.field("bad", pa.float64())])
    assert not _multiset_sortable(schema)


# --- semantics are unchanged ---------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({"x": [1, 2, 3]}, {"x": [3, 1, 2]}, True),
        ({"x": [1, 2, 3]}, {"x": [1, 2, 4]}, False),
        ({"x": [1, 1, 2]}, {"x": [1, 2, 2]}, False),  # multiset, not set
        ({"x": [1, 2]}, {"x": [1, 2, 2]}, False),  # differing row counts
        ({"x": [1, None]}, {"x": [None, 1]}, True),  # nulls are values here
        ({"x": [1, None]}, {"x": [1, 1]}, False),
        ({"x": []}, {"x": []}, True),
        ({"x": [1], "y": ["a"]}, {"x": [1], "y": ["a"]}, True),
        ({"x": [1, 2], "y": ["a", "b"]}, {"x": [2, 1], "y": ["b", "a"]}, True),
        # a column pair that must not be compared position-wise after sorting one column
        ({"x": [1, 2], "y": ["a", "b"]}, {"x": [1, 2], "y": ["b", "a"]}, False),
    ],
)
def test_unordered_equals_on_the_fast_path(left, right, expected):
    assert _ds(pa.table(left)).equals(_ds(pa.table(right))) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([0.0], [-0.0], False),  # row-wise distinguishes the sign of zero; Arrow does not
        ([math.nan], [math.nan], True),  # row-wise says NaN is NaN; Arrow equality does not
        ([1.0, 2.0], [2.0, 1.0], True),
        ([1.0], [1.5], False),
    ],
    ids=["signed-zero", "nan", "reordered", "different"],
)
def test_float_columns_keep_the_row_wise_answer(left, right, expected):
    """The float edges are why the fast path excludes them: these four answers are the ones
    that shipped, and routing them through the Arrow sort would flip the first two —
    `-0.0 == 0.0` and `NaN != NaN` under Arrow equality, the reverse row-wise."""
    lt = pa.table({"c": pa.array(left, pa.float64())})
    rt = pa.table({"c": pa.array(right, pa.float64())})
    assert _ds(lt).equals(_ds(rt)) is expected


@pytest.mark.unit
def test_nested_and_dictionary_columns_still_compare():
    nested = pa.table({"c": pa.array([[1, 2], [3]])})
    assert _ds(nested).equals(_ds(pa.table({"c": pa.array([[3], [1, 2]])})))
    assert not _ds(nested).equals(_ds(pa.table({"c": pa.array([[1], [2, 3]])})))


@pytest.mark.unit
def test_ordered_comparison_is_untouched():
    a = _ds(pa.table({"x": [1, 2, 3]}))
    b = _ds(pa.table({"x": [3, 2, 1]}))
    assert not a.equals(b, ordered=True)
    assert a.equals(b, ordered=False)
    assert a.equals(a.sort("x"), ordered=True)


@pytest.mark.unit
def test_schema_and_column_mismatches_short_circuit():
    a = _ds(pa.table({"x": [1]}))
    assert not a.equals(_ds(pa.table({"y": [1]})))  # names
    assert not a.equals(_ds(pa.table({"x": ["1"]})))  # types
    # Int8/16/32 are normalized to Int64 at the FFI boundary, so a narrow-vs-wide integer
    # pair is genuinely equal by the time `equals` sees it. Pinned so the type check above
    # is not later "fixed" into asserting a difference the engine deliberately erases.
    assert a.equals(_ds(pa.table({"x": pa.array([1], pa.int32())})))


@pytest.mark.unit
def test_equals_still_compares_results_not_plans():
    ds = _ds(pa.table({"x": [1, 2, 3]}))
    assert ds.equals(ds.filter(bt.col("x") > 0))
    assert not ds.equals(ds.filter(bt.col("x") > 1))
