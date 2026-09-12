"""The way in from NumPy and PyTorch — the shapes those libraries actually hand you.

`test_arrow_interop.py` pins the bridge *out* of Batcher. This is the bridge *in*, and it
covers the cases where a caller writes one ordinary line of NumPy or torch and the engine
either refused it or, once, answered it wrongly.

Three of these are regressions with a shared cause: the ingestion path normalized with
``np.asarray`` and then handed the result to ``pa.array``, so anything Arrow could not type
in one step surfaced as a pyarrow message quoting an internal dtype number — ``Unsupported
numpy type 20`` for NumPy's own table type, ``Got unsupported ScalarType BFloat16`` for the
dtype every LLM checkpoint is in. None of them named the column, the dtype, or a fix.

The masked-array case is the one that matters most and looks the least dramatic:
``np.asarray`` drops the `np.ma` subclass, so a masked value arrived as whatever sat under
the mask and was read as data. No error, no warning, wrong numbers.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


# --- NumPy structured arrays: NumPy's own table shape ----------------------------------
def test_a_structured_array_becomes_one_column_per_field():
    """``np.genfromtxt``/``np.rec.array`` produce this, and it is a table, not a column."""
    array = np.array([(1, 2.5), (3, 4.5)], dtype=[("id", "i8"), ("score", "f8")])
    assert bt.from_numpy(array).to_pydict() == {"id": [1, 3], "score": [2.5, 4.5]}


def test_a_recarray_becomes_one_column_per_field():
    array = np.rec.array([(1, 2.5)], dtype=[("id", "i8"), ("score", "f8")])
    assert bt.from_numpy(array).to_pydict() == {"id": [1], "score": [2.5]}


def test_a_structured_field_holding_a_vector_keeps_the_embedding_shape():
    """A sub-array field is a per-row vector, so it takes the same type a bare 2-D array does.

    The width is what this is about, and it is unchanged. The **element type** used to be
    `float64` here and is now the `f4` the caller wrote: a list of floats is a tensor, and the
    boundary stopped widening its child (`bc_py::normalize_list_element`), so an embedding
    column no longer doubles on the way in. Asserting the narrow type is what pins that from
    this side — the caller's dtype survives the trip.
    """
    array = np.zeros(2, dtype=[("id", "i8"), ("emb", "f4", (4,))])
    schema = bt.from_numpy(array).schema
    assert schema.field("emb").type == pa.list_(pa.float32(), 4)


def test_a_nested_compound_field_becomes_a_struct_column():
    """A C struct inside a record — routine in an h5py compound dataset — nests, not flattens.

    Only the outermost compound dtype names the rows, so a field that is itself compound is a
    struct *column*. It used to reach Arrow whole and fail with the same ``Unsupported numpy
    type 20`` the top-level case did.
    """
    array = np.zeros(2, dtype=[("id", "i8"), ("pos", [("x", "f8"), ("y", "f8")])])
    assert bt.from_numpy(array).to_pydict()["pos"] == [{"x": 0.0, "y": 0.0}] * 2


def test_a_nested_compound_field_nests_to_any_depth():
    array = np.zeros(1, dtype=[("outer", [("inner", [("z", "i8")])])])
    assert bt.from_numpy(array).to_pydict() == {"outer": [{"inner": {"z": 0}}]}


def test_a_sub_array_inside_a_nested_field_keeps_the_embedding_shape():
    """The rank rules apply at every level, not only the top one.

    And so does the tensor exemption: the list is inside a struct, and it is still a list of
    floats, so its child keeps the caller's `f4` exactly as the top-level case above does.
    """
    array = np.zeros(1, dtype=[("rec", [("emb", "f4", (3,))])])
    struct = bt.from_numpy(array).schema.field("rec").type
    assert struct.field("emb").type == pa.list_(pa.float32(), 3)


def test_a_masked_structured_array_keeps_the_mask_per_field():
    """The two features compose: a record array of measurements with missing readings."""
    data = np.array([(1,), (2,)], dtype=[("a", "i8")])
    array = np.ma.array(data, mask=[(True,), (False,)])
    assert bt.from_numpy(array).to_pydict() == {"a": [None, 2]}


def test_a_multidimensional_structured_array_is_refused_with_the_reshape():
    array = np.zeros((2, 2), dtype=[("id", "i8")])
    with pytest.raises(PlanError, match=r"reshape\(-1\)"):
        bt.from_numpy(array)


# --- Masked arrays: the silent-wrong-answer case ---------------------------------------
def test_a_masked_value_becomes_null_rather_than_the_fill_underneath_it():
    """The mask is the point. Reading through it returns numbers nobody put there."""
    array = np.ma.array([1, 2, 3], mask=[False, True, False])
    assert bt.from_numpy(array).to_pydict() == {"data": [1, None, 3]}


def test_a_masked_matrix_keeps_the_mask_per_element():
    array = np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [False, False]])
    assert bt.from_numpy(array).to_pydict() == {"data": [[1, None], [3, 4]]}


def test_an_unmasked_masked_array_is_not_given_a_null_bitmap():
    """`np.ma` with nothing masked is ordinary data; it must not acquire a validity buffer."""
    column = bt.from_numpy(np.ma.array([1, 2, 3], mask=False)).collect().column(0)
    assert column.null_count == 0


def test_a_masked_tensor_is_refused_rather_than_silently_filled():
    """Rank >= 3 becomes a fixed-shape-tensor column, which has nowhere to put the mask."""
    array = np.ma.array(np.zeros((2, 2, 2)), mask=np.zeros((2, 2, 2), dtype=bool))
    array[0, 0, 0] = np.ma.masked
    with pytest.raises(PlanError, match=r"filled\("):
        bt.from_numpy(array)


# --- Declines that used to be pyarrow dtype numbers ------------------------------------
def test_a_complex_array_names_the_split_that_fixes_it():
    with pytest.raises(PlanError, match="complex"):
        bt.from_numpy(np.array([1 + 2j]))


def test_a_zero_dimensional_array_says_it_has_no_row_axis():
    with pytest.raises(PlanError, match="atleast_1d"):
        bt.from_numpy(np.array(5))


# --- The {name: array} door agrees with the bare-array door ----------------------------
@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.arange(6.0).reshape(3, 2), pa.list_(pa.float64(), 2)),
        (np.arange(24).reshape(2, 3, 4), None),  # tensor column: compared by rank below
    ],
    ids=["matrix", "tensor"],
)
def test_a_multidimensional_array_gets_the_same_type_through_either_door(array, expected):
    """``bt.from_numpy(a)`` worked and ``bt.from_pydict({"x": a})`` raised, for the same array.

    An embedding table is built the second way far more often than the first, and the error
    it produced told the caller to convert the column to an ndarray, which is what it was.
    """
    bare = bt.from_numpy(array).schema.field(0).type
    named = bt.from_pydict({"x": array}).schema.field("x").type
    assert bare == named
    if expected is not None:
        assert named == expected


def test_a_structured_array_named_as_one_column_becomes_a_struct_column():
    """As a whole `Dataset` a record array is a table; named as one column it is that column."""
    array = np.array([(1, 2.5)], dtype=[("x", "i8"), ("y", "f8")])
    ds = bt.from_pydict({"rec": array, "id": [7]})
    assert ds.schema.field("rec").type == pa.struct([("x", pa.int64()), ("y", pa.float64())])
    assert ds.to_pydict()["rec"] == [{"x": 1, "y": 2.5}]


def test_an_untypable_column_is_named_even_when_arrow_raises_not_implemented():
    """`ArrowNotImplementedError` derives from `NotImplementedError`, not from `ValueError`.

    So it slipped past handlers built for the other three, and the column diagnosis, the
    tensor retry and the `PlanError` wrapping were all skipped for the dtypes Arrow has no
    column form for at all — exactly the ones a caller most needs told about.
    """
    with pytest.raises(PlanError, match="'c'"):
        bt.from_pydict({"c": np.array([1 + 2j])})


def test_a_one_dimensional_array_column_is_unchanged():
    """The plain path must not be rerouted; it converts on the first attempt as before."""
    assert bt.from_pydict({"x": np.arange(3)}).schema.field("x").type == pa.int64()


def test_a_list_of_same_shape_arrays_still_becomes_a_tensor_column():
    """The pre-existing per-row-arrays spelling keeps its type."""
    from batcher.io.formats.ml.tensor import is_tensor_column

    column = bt.from_pydict({"x": [np.zeros((2, 2)), np.ones((2, 2))]}).collect().column(0)
    assert is_tensor_column(column.chunk(0) if hasattr(column, "chunk") else column)


# --- PyTorch ---------------------------------------------------------------------------
def test_a_mapping_of_tensors_keeps_its_keys_as_column_names():
    """The exact inverse of `ml.to_torch`; it used to raise ``KeyError: 0``."""
    torch = pytest.importorskip("torch")
    ds = bt.from_torch({"x": torch.arange(3), "y": torch.ones(3)})
    assert ds.schema.names == ["x", "y"]
    assert ds.to_pydict()["x"] == [0, 1, 2]


def test_the_to_torch_loader_round_trips_back_through_from_torch():
    """What the loader yields is what the constructor takes. It was not, and nothing said so."""
    pytest.importorskip("torch")
    source = bt.from_pydict({"x": [1, 2, 3, 4]})
    batch = next(iter(source.ml.to_torch(batch_size=4)))
    assert bt.from_torch(batch).to_pydict()["x"] == [1, 2, 3, 4]


def test_a_features_and_labels_tuple_keeps_the_feature_matrix_shape():
    """``(features, labels)`` with 2-D features is the canonical PyTorch pair."""
    torch = pytest.importorskip("torch")
    ds = bt.from_torch((torch.arange(6).reshape(3, 2), torch.ones(3)))
    assert ds.schema.field("col_0").type == pa.list_(pa.int64(), 2)
    assert ds.count() == 3


def test_a_bfloat16_tensor_widens_to_float32_and_says_so():
    """bfloat16 is what nearly every LLM checkpoint carries; NumPy and Arrow have no such type."""
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="bfloat16"):
        assert bt.from_torch(tensor).to_pydict() == {"data": [1.0, 2.0]}


def test_a_float8_tensor_widens_the_same_way():
    """The quantized-inference dtypes take the same route, so the rule is on the failure."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8 dtype")
    tensor = torch.ones(2).to(torch.float8_e4m3fn)
    with pytest.warns(UserWarning, match="float8"):
        assert bt.from_torch(tensor).to_pydict() == {"data": [1.0, 1.0]}


def test_an_ordinary_float_tensor_warns_about_nothing():
    """The widening warning must be reachable only by a dtype that is actually widened."""
    torch = pytest.importorskip("torch")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bt.from_torch(torch.tensor([1.0, 2.0])).count() == 2
