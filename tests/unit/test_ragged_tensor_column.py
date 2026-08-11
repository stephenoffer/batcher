"""Variable-shape tensor columns: mixed-resolution arrays in one Arrow column.

Arrow's canonical tensor type needs one shape for a whole column, so a mixed-resolution image
batch — the ordinary result of decoding a folder of photos — could not be typed at all. It was
diagnosed accurately and left to the user to solve by resizing before the engine saw the data.
`io.formats.ml.ragged` carries it instead, as an ordinary struct of a binary buffer, a shape,
and a dtype.

Three properties carry the design and are tested here rather than assumed:

* the values survive exactly, including the **element type** — a `uint8` image must not
  arrive as `int64`, which is what a list-of-elements layout would have cost it at the FFI
  boundary that widens narrow numerics;
* nothing in the engine has to know about it, so it passes through filters, writes, and reads
  as any struct column would;
* the shapes it does *not* claim keep their existing types, because changing the schema of
  every embedding column in existing pipelines is not an acceptable price for this.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.formats.ml.ragged import (
    is_ragged_tensor_column,
    ragged_from_values,
    ragged_to_numpy,
    to_ragged_tensor_column,
)
from batcher.io.formats.ml.tensor import tensor_from_values

pytestmark = pytest.mark.unit

_IMAGES = [np.full((2, 2), 1, "uint8"), np.full((3, 4), 7, "uint8"), np.full((1, 5), 9, "uint8")]


# --- the column itself ------------------------------------------------------------------


def test_it_round_trips_shapes_values_and_dtype():
    column = to_ragged_tensor_column(_IMAGES)
    decoded = ragged_to_numpy(column)
    assert [a.shape for a in decoded] == [(2, 2), (3, 4), (1, 5)]
    assert all(a.dtype == np.uint8 for a in decoded)
    assert np.array_equal(decoded[1], _IMAGES[1])


def test_rows_may_disagree_on_rank_and_dtype():
    """Per-row shape *and* per-row dtype, so nothing about the rows has to line up."""
    column = to_ragged_tensor_column([np.zeros(3, "float32"), np.ones((2, 2, 2), "int16")])
    decoded = ragged_to_numpy(column)
    assert decoded[0].shape == (3,) and decoded[0].dtype == np.float32
    assert decoded[1].shape == (2, 2, 2) and decoded[1].dtype == np.int16


def test_a_null_row_stays_null():
    decoded = ragged_to_numpy(to_ragged_tensor_column([np.zeros((2, 2)), None]))
    assert decoded[1] is None
    assert decoded[0].shape == (2, 2)


def test_it_recognizes_itself_and_nothing_else():
    assert is_ragged_tensor_column(to_ragged_tensor_column(_IMAGES))
    assert not is_ragged_tensor_column(pa.array([1, 2, 3]))
    assert not is_ragged_tensor_column(pa.array([{"data": b"x", "other": 1}]))


def test_the_buffer_is_bytes_not_boxed_elements():
    """The layout choice that keeps a uint8 image one byte per pixel across the boundary."""
    column = to_ragged_tensor_column(_IMAGES)
    assert pa.types.is_binary(column.type.field("data").type)


# --- what it claims, and what it leaves alone -------------------------------------------


def test_mixed_shape_arrays_are_claimed():
    assert ragged_from_values(_IMAGES) is not None


def test_one_shape_is_left_to_the_fixed_shape_column():
    same = [np.zeros((2, 2), "uint8"), np.ones((2, 2), "uint8")]
    assert ragged_from_values(same) is None
    assert tensor_from_values(same) is not None


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1, 2, 3], id="numbers"),
        pytest.param(["a", "b"], id="strings"),
        pytest.param([[1, 2], [3]], id="lists"),
        pytest.param([np.zeros(4), np.ones(5)], id="1d-arrays"),
        pytest.param([], id="empty"),
    ],
)
def test_ordinary_columns_are_not_claimed(values):
    assert ragged_from_values(values) is None
    assert tensor_from_values(values) is None


def test_an_embedding_column_keeps_the_type_it_has_always_had():
    """1-D arrays already convert to a list column; claiming them would change live schemas."""
    ds = bt.from_pydict({"e": [np.zeros(4, "float32"), np.ones(4, "float32")]})
    assert pa.types.is_list(ds.schema.field("e").type)


# --- through the engine -----------------------------------------------------------------


def _decoded(ds, column="img"):
    return ds.to_numpy()[column]


def test_a_udf_returning_mixed_resolution_images_produces_a_ragged_column():
    out = bt.from_pydict({"id": [1, 2, 3]}).map_batches(
        lambda b: {"id": b.column("id"), "img": _IMAGES}, output_columns=["id", "img"]
    )
    assert is_ragged_tensor_column(out.schema.field("img").type)
    assert [a.shape for a in _decoded(out)] == [(2, 2), (3, 4), (1, 5)]


def test_a_constructor_takes_them_too():
    ds = bt.from_pydict({"img": _IMAGES})
    assert [a.shape for a in _decoded(ds)] == [(2, 2), (3, 4), (1, 5)]


def test_uniform_multidimensional_arrays_become_a_fixed_shape_column():
    """The other half of the same fix: this also used to be an un-typable list of arrays."""
    ds = bt.from_pydict({"img": [np.zeros((2, 2), "uint8"), np.ones((2, 2), "uint8")]})
    assert _decoded(ds).shape == (2, 2, 2)


def test_the_element_type_survives_the_engine_boundary():
    """A `list<uint8>` layout would arrive as `int64` — eight bytes a pixel."""
    ds = bt.from_pydict({"id": [1, 2, 3], "img": _IMAGES})
    assert all(a.dtype == np.uint8 for a in _decoded(ds))


def test_it_passes_through_a_filter_unchanged():
    ds = bt.from_pydict({"id": [1, 2, 3], "img": _IMAGES})
    kept = ds.filter(bt.col("id") > 1)
    assert [a.shape for a in _decoded(kept)] == [(3, 4), (1, 5)]


def test_it_survives_a_parquet_round_trip(tmp_path):
    ds = bt.from_pydict({"id": [1, 2, 3], "img": _IMAGES})
    path = str(tmp_path / "images.parquet")
    ds.write.parquet(path)
    back = bt.read.parquet(path)
    assert is_ragged_tensor_column(back.schema.field("img").type)
    decoded = _decoded(back)
    assert [a.shape for a in decoded] == [(2, 2), (3, 4), (1, 5)]
    assert np.array_equal(decoded[1], _IMAGES[1])


def test_the_numpy_batch_format_decodes_it_for_the_user_function():
    seen: list = []

    def note(batch):
        seen.append([a.shape for a in batch["img"]])
        return {"n": np.array([len(batch["img"])])}

    bt.from_pydict({"img": _IMAGES}).map_batches(
        note, batch_format="numpy", output_columns=["n"]
    ).collect()
    assert seen == [[(2, 2), (3, 4), (1, 5)]]


# --- across the batch_format conversions ------------------------------------------------


@pytest.mark.parametrize("fmt", ["pyarrow", "numpy", "pandas", "polars"])
def test_it_survives_every_batch_format_round_trip(fmt):
    """An identity `fn` must return the column it was handed, whatever it was handed it as.

    Each format damaged this differently: numpy hands back an object *array* the tensor path
    did not recognize as a sequence, polars widens `binary` to `large_binary` and `list` to
    `large_list`, and pandas reorders the struct's fields.
    """
    pytest.importorskip(fmt if fmt in {"pandas", "polars"} else "numpy")
    out = bt.from_pydict({"img": _IMAGES}).map_batches(lambda b: b, batch_format=fmt)
    assert is_ragged_tensor_column(out.schema.field("img").type)
    assert [a.shape for a in out.to_numpy()["img"]] == [(2, 2), (3, 4), (1, 5)]


def test_a_pandas_udf_sees_shaped_arrays_not_flat_ones():
    """`to_pandas` flattens a tensor column, so the `fn` used to receive 9-element vectors
    for a 3x3 image with the shape gone before it was called."""
    pytest.importorskip("pandas")
    seen: list = []

    def note(frame):
        seen.append(frame["t"].iloc[0].shape)
        return frame

    fixed = bt.from_pydict({"id": [1, 2]}).map_batches(
        lambda b: {"id": b.column("id"), "t": np.zeros((2, 3, 3), "uint8")},
        output_columns=["id", "t"],
    )
    out = fixed.map_batches(note, batch_format="pandas")
    assert out.to_numpy()["t"].shape == (2, 3, 3)  # and it comes back a tensor column
    assert seen == [(3, 3)]
