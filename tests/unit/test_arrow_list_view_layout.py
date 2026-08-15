"""`plan.types.layout` respells the list-view layouts the FFI reader cannot import.

Unit-level, so it runs without the engine: the property under test is that the rebuild is
*value-preserving and structurally valid* for every container a list view can hide inside.
Both halves matter. `validate(full=True)` is the one that catches the failure this module
exists for -- pyarrow's own ``list_view -> list`` cast returns an array that *looks* right
and fails validation, and an unvalidated array corrupts whatever reads it next rather than
raising.

The value comparison uses `to_pylist`, which is per-row Python. That is fine here and only
here: these are three-row fixtures in a unit test, not a hot path.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.plan.types.layout import importable_array, importable_batch, importable_type

pytestmark = pytest.mark.unit


def _lv(values, value_type: pa.DataType | None = None) -> pa.Array:
    return pa.array(values, pa.list_view(value_type or pa.int64()))


# (name, array) — one entry per container a list view can appear under, plus the layout
# edge cases (a slice, an out-of-order view, all-null, empty).
_CASES = [
    ("top level", _lv([[1, 2], None, [3], []])),
    ("large list view", pa.array([[1, 2], None, [3]], pa.large_list_view(pa.int64()))),
    ("narrow value type", _lv([[1.5], [2.5]], pa.float32())),
    (
        "inside a struct",
        pa.array([{"a": [1]}, None, {"a": None}], pa.struct([("a", pa.list_view(pa.int64()))])),
    ),
    ("inside a list", pa.array([[[1], [2]], None, []], pa.list_(pa.list_view(pa.int64())))),
    ("inside a large list", pa.array([[[1]], None], pa.large_list(pa.list_view(pa.int64())))),
    (
        "inside a fixed-size list",
        pa.array([[[1], [2]], None, [[5], [6]]], pa.list_(pa.list_view(pa.int64()), 2)),
    ),
    (
        "as a map value",
        pa.array(
            [[("a", [1]), ("b", [2])], None, []], pa.map_(pa.string(), pa.list_view(pa.int64()))
        ),
    ),
    ("as a map key", pa.array([[([1], "x")]], pa.map_(pa.list_view(pa.int64()), pa.string()))),
    (
        "holding a struct",
        pa.array([[{"a": 1}], None], pa.list_view(pa.struct([("a", pa.int64())]))),
    ),
    ("all null", _lv([None, None])),
    ("empty", pa.array([], pa.list_view(pa.int64()))),
    ("sliced", _lv([[1], [2], [3], [4]]).slice(1, 2)),
    (
        "sliced list of views",
        pa.array([[[1], [2]], None, [[9]]], pa.list_(pa.list_view(pa.int64()))).slice(1, 2),
    ),
    (
        "sliced struct of views",
        pa.array(
            [{"a": [1]}, None, {"a": [7]}], pa.struct([("a", pa.list_view(pa.int64()))])
        ).slice(1, 2),
    ),
    (
        "sliced fixed-size list",
        pa.array([[[1], [2]], None, [[5], [6]]], pa.list_(pa.list_view(pa.int64()), 2)).slice(1, 2),
    ),
    (
        "sliced map",
        pa.array(
            [[("a", [1])], None, [("c", [3])]], pa.map_(pa.string(), pa.list_view(pa.int64()))
        ).slice(2, 1),
    ),
    (
        "nested three deep",
        pa.array(
            [{"a": [[{"b": [1]}]]}],
            pa.struct(
                [("a", pa.list_view(pa.list_(pa.struct([("b", pa.list_view(pa.int64()))]))))]
            ),
        ),
    ),
]


@pytest.mark.parametrize(("name", "arr"), _CASES, ids=[c[0] for c in _CASES])
def test_rebuild_preserves_values_and_validates(name: str, arr: pa.Array) -> None:
    target = importable_type(arr.type)
    assert target is not None, f"{name}: a list view must be recognized as unimportable"
    out = importable_array(arr, target)
    out.validate(full=True)
    assert out.to_pylist() == arr.to_pylist()
    assert importable_type(out.type) is None, f"{name}: the rebuild must be a fixed point"


def test_a_view_whose_ranges_are_out_of_order() -> None:
    """List-view ranges may overlap, repeat, and run backwards -- the layout's whole point.

    Rebuilding from `arr.offsets` would read the wrong rows here, which is why the offsets
    are derived from the per-row lengths and `list_flatten`'s row-order concatenation.
    """
    arr = pa.ListViewArray.from_arrays(
        pa.array([3, 0, 1], pa.int32()),
        pa.array([2, 2, 1], pa.int32()),
        pa.array([1, 2, 3, 4, 5], pa.int64()),
    )
    out = importable_array(arr, importable_type(arr.type))
    out.validate(full=True)
    assert out.to_pylist() == [[4, 5], [1, 2], [2]]


def test_pyarrows_own_cast_is_why_this_module_exists() -> None:
    """Pin the upstream defect the rebuild works around, so a fixed pyarrow is noticed.

    If this starts failing, pyarrow's ``list_view -> list`` cast has been fixed and
    `importable_array` could delegate to it. Until then the cast silently produces an array
    that fails validation.
    """
    broken = _lv([[1, 2], [3]]).cast(pa.list_(pa.int64()))
    with pytest.raises(pa.ArrowInvalid):
        broken.validate(full=True)


def test_a_batch_with_no_view_is_returned_unchanged() -> None:
    """The common case costs a schema walk and touches no buffer."""
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2]), pa.array(["a", "b"])], names=["n", "s"])
    assert importable_batch(batch) is batch


def test_a_batch_respells_only_the_view_column() -> None:
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2]), _lv([[1], [2]])], names=["n", "v"])
    out = importable_batch(batch)
    out.validate(full=True)
    assert out.schema.field("n").type == pa.int64()
    assert out.schema.field("v").type == pa.list_(pa.int64())
    # Buffer addresses, not object identity: `from_arrays` rewraps every column, so the
    # only way to state "the untouched column was not copied" is that it still points at
    # the same memory.
    before = [b.address for b in batch.column("n").buffers() if b is not None]
    after = [b.address for b in out.column("n").buffers() if b is not None]
    assert after == before
