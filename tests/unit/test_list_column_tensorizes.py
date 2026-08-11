"""A feature matrix stored as a plain `List<T>` reaches the training loop.

`FixedSizeList<T, W>` was the only list spelling the tensor converters recognized. But that
is not the spelling most feature columns actually have: `bt.from_pydict`, a Parquet or JSON
read, and `collect_list` all produce a *variable-length* `List<T>`, whose type does not
record a width even when every row has the same one.

So an embedding column built any of those ordinary ways became an object array of per-row
lists, which is not numeric, which meant it was dropped from every yielded batch — with no
error and no warning. The training loop read a `KeyError`, or worse, trained on whatever
columns did survive.

A list column can hold a rectangle and usually does, so the widths are read off the offsets
(one vectorized subtraction over `n + 1` integers, not a pass over the data) and a uniform
column is reshaped to `(n, W)`. A genuinely ragged one, or one with nulls, still has no
rectangular form and is still dropped — nulls especially, because a null row's span is empty
and placing it would slide every later row up under the wrong label.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.interop.arrays import uniform_list_to_matrix

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch", reason="the tensor converters need torch")


def _first_batch(ds: bt.Dataset, rows: int = 2) -> dict:
    return next(iter(ds.ml.iter_torch_batches(batch_size=rows)))


def test_a_list_embedding_column_reaches_the_training_loop() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0], "e": [[1.0, 2.0], [3.0, 4.0]]})
    batch = _first_batch(ds)

    assert "e" in batch, "the feature column was dropped from the batch"
    assert tuple(batch["e"].shape) == (2, 2)
    assert batch["e"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_the_width_comes_from_the_data_not_the_type() -> None:
    ds = bt.from_pydict({"e": [[1, 2, 3], [4, 5, 6]]})
    assert tuple(_first_batch(ds)["e"].shape) == (2, 3)


def test_a_fixed_size_list_still_converts() -> None:
    """The spelling that already worked must keep working, identically."""
    values = pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0, 3.0, 4.0]), 2)
    ds = bt.from_arrow(pa.table({"x": [1.0, 2.0], "e": values}).to_batches())
    batch = _first_batch(ds)

    assert tuple(batch["e"].shape) == (2, 2)
    assert batch["e"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize(
    ("name", "column"),
    [
        ("ragged", [[1.0], [2.0, 3.0]]),  # no rectangular form exists
        ("null row", [None, [2.0, 3.0]]),  # placing it would misalign every later row
        ("strings", [["a", "b"], ["c", "d"]]),  # not numeric
    ],
)
def test_a_column_with_no_rectangular_form_is_still_dropped(name, column) -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0], "e": column})
    batch = _first_batch(ds)

    assert "e" not in batch, f"{name} column must not be reshaped"
    assert "x" in batch  # and its siblings are unaffected


def test_the_matrix_helper_reports_what_it_cannot_reshape() -> None:
    """The single implementation both the loader and the converter path share."""
    assert uniform_list_to_matrix(pa.array([[1.0, 2.0], [3.0, 4.0]])).shape == (2, 2)
    assert uniform_list_to_matrix(pa.array([[1.0], [2.0, 3.0]])) is None  # ragged
    assert uniform_list_to_matrix(pa.array([None, [2.0, 3.0]])) is None  # null row
    assert uniform_list_to_matrix(pa.array([1.0, 2.0])) is None  # not a list at all
    assert uniform_list_to_matrix(pa.array([[], []], type=pa.list_(pa.float64()))) is None


def test_a_large_list_column_converts_too() -> None:
    """`large_list` is what the HuggingFace fast tokenizers emit."""
    values = pa.array([[1, 2], [3, 4]], type=pa.large_list(pa.int64()))
    assert uniform_list_to_matrix(values).tolist() == [[1, 2], [3, 4]]


def test_a_sliced_list_column_reads_from_its_own_offset() -> None:
    """A batch is routinely a slice of a larger array; the child buffer is not rebased."""
    column = pa.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert uniform_list_to_matrix(column.slice(1, 2)).tolist() == [[3.0, 4.0], [5.0, 6.0]]


def test_the_converter_path_sees_the_same_matrix() -> None:
    """`to_numpy_batches` feeds `map_batches(batch_format="numpy")` and `ds.ml.to_torch`."""
    from batcher.interop.arrays import to_numpy_batches

    batch = pa.RecordBatch.from_pydict({"e": [[1.0, 2.0], [3.0, 4.0]]})
    arrays = next(iter(to_numpy_batches([batch])))

    assert arrays["e"].dtype == np.float64  # not an object array of per-row lists
    assert arrays["e"].shape == (2, 2)
