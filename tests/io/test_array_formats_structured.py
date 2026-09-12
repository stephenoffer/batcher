"""A compound (record) array read from a file is a table, wherever it is stored.

NumPy's structured dtype is how a C struct is stored, and three of this engine's readers
meet one: `bt.from_numpy` in memory, the `.npy` reader on disk, and the HDF5/Zarr array
readers for a compound dataset. Only the first understood it. The other two handed the
`void` array straight to Arrow and failed with ``ArrowNotImplementedError: Unsupported
numpy type 20`` — a message naming neither the fields the file held nor the fact that
those fields are the columns the caller was asking for.

That matters most for HDF5, where a compound dataset is the ordinary way instrument,
simulation and genomics files record their rows, and for `.npy`, which is what
``np.rec.array`` and ``np.genfromtxt`` save.

All three now split the fields through one function (`io.interop.structured_to_columns`),
so the same array reads as the same table whichever door it comes in by. These tests hold
the three against each other rather than against a hardcoded expectation, because agreement
is the property that was missing.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.io

#: A record array with an integer, a float and a bytes field — the ordinary shape.
_ROWS = np.array(
    [(1, 2.5, b"aa"), (3, 4.5, b"bb")],
    dtype=[("id", "i8"), ("score", "f8"), ("tag", "S2")],
)

#: A record whose field is itself a vector, which must keep the embedding convention.
_WITH_VECTOR = np.zeros(3, dtype=[("id", "i8"), ("emb", "f4", (4,))])


def _npy(tmp_path, array, name="a.npy"):
    path = tmp_path / name
    np.save(str(path), array)
    return bt.read(str(path), format="numpy")


def _hdf5(tmp_path, array, name="a.h5"):
    h5py = pytest.importorskip("h5py", reason="h5py not installed")
    path = tmp_path / name
    with h5py.File(str(path), "w") as handle:
        handle.create_dataset("rows", data=array)
    return bt.read(str(path), format="hdf5", dataset="rows")


def test_a_structured_npy_reads_as_one_column_per_field(tmp_path):
    assert _npy(tmp_path, _ROWS).to_pydict() == {
        "id": [1, 3],
        "score": [2.5, 4.5],
        "tag": [b"aa", b"bb"],
    }


def test_a_compound_hdf5_dataset_reads_as_one_column_per_field(tmp_path):
    assert _hdf5(tmp_path, _ROWS).to_pydict() == {
        "id": [1, 3],
        "score": [2.5, 4.5],
        "tag": [b"aa", b"bb"],
    }


def test_the_three_doors_agree_on_the_same_array(tmp_path):
    """Agreement is the property, so it is asserted directly rather than field by field."""
    in_memory = bt.from_numpy(_ROWS).to_pydict()
    assert _npy(tmp_path, _ROWS).to_pydict() == in_memory
    assert _hdf5(tmp_path, _ROWS).to_pydict() == in_memory


@pytest.mark.parametrize("reader", ["npy", "hdf5"])
def test_a_vector_field_keeps_the_embedding_convention(reader, tmp_path):
    """A sub-array field is a per-row vector; flattening it would lose the row boundary.

    The element stays ``float32``. This asserted ``float64`` until the boundary stopped
    widening a *list element* — an embedding column is the shape the multimodal path is
    built on, and widening its child doubles (for ``uint8``, octuples) a payload that
    usually reaches a UDF without touching a kernel at all. See
    `bc_py::normalize::normalize_list_element` and `io.source.inmemory._widen_narrow_type`.
    So the width here is the deliberate behaviour, not an accident to be widened back.
    """
    ds = _npy(tmp_path, _WITH_VECTOR) if reader == "npy" else _hdf5(tmp_path, _WITH_VECTOR)
    assert ds.schema.field("emb").type == pa.list_(pa.float32(), 4)
    # The declared schema is a promise about what `collect()` returns; a vector field is
    # exactly where the two used to be able to drift, since only one of them widened.
    assert ds.collect().schema.field("emb").type == ds.schema.field("emb").type


def test_the_npy_schema_is_answered_from_the_header_without_reading_rows(tmp_path):
    """`schema()` is answered from the `.npy` header, so it must know the compound layout too.

    It did not: the header path built a one-column ``data`` schema while the read produced
    the fields, and the disagreement surfaced as ``KeyError: 'Field "data" does not exist in
    schema'`` — an error about neither the file nor the dtype.
    """
    ds = _npy(tmp_path, _ROWS)
    assert ds.schema.names == ["id", "score", "tag"]
    assert ds.schema.names == ds.collect().schema.names


def test_a_projection_over_a_compound_dataset_selects_a_field(tmp_path):
    """The fields are real columns, so the reader's projection has to reach them."""
    assert _hdf5(tmp_path, _ROWS).select("id").to_pydict() == {"id": [1, 3]}


# --- the layouts that must not move ------------------------------------------------------
def test_a_plain_2d_hdf5_dataset_still_reads_as_positional_columns(tmp_path):
    ds = _hdf5(tmp_path, np.arange(12).reshape(6, 2))
    assert ds.schema.names == ["c0", "c1"]


def test_a_plain_1d_hdf5_dataset_still_reads_as_value(tmp_path):
    assert _hdf5(tmp_path, np.arange(4)).to_pydict() == {"value": [0, 1, 2, 3]}


def test_a_plain_2d_npy_still_reads_as_one_vector_column(tmp_path):
    ds = _npy(tmp_path, np.arange(6.0).reshape(3, 2))
    assert ds.schema.field("data").type == pa.list_(pa.float64(), 2)


def test_an_npz_member_still_keeps_its_archive_key(tmp_path):
    """Only the single-array file expands; an archive's keys are its column names."""
    path = tmp_path / "a.npz"
    np.savez(str(path), a=np.arange(3), b=np.arange(3.0))
    assert bt.read(str(path), format="numpy").to_pydict() == {
        "a": [0, 1, 2],
        "b": [0.0, 1.0, 2.0],
    }


def test_a_dtype_with_no_arrow_form_names_the_fix_rather_than_a_dtype_number(tmp_path):
    """Delegating to the shared converter brings its declines with it.

    A complex `.npy` used to fail with ``Unsupported numpy type 15``. It still fails — Arrow
    has no complex type — but now says which conversion fixes it.
    """
    path = tmp_path / "c.npy"
    np.save(str(path), np.array([1 + 2j]))
    with pytest.raises(Exception, match="complex"):
        bt.read(str(path), format="numpy").collect()
