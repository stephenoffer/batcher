"""Unifying tensor columns whose shapes disagree, which is how a ragged column survives.

A `map_batches` UDF returning arrays of differing shapes produces one *ragged* column when a
single call sees every row. Run per partition, the same UDF yields a `fixed_shape_tensor` for
any partition whose rows happen to agree -- so reconciling the partitions asks for the common
type of two fixed-shape tensors with different shapes. There is one, and it is the encoding
the single-node path already chose.

The distributed end of this is `tests/integration/test_ragged_tensor_distributed.py`; these
are the same rule without a Ray cluster, so the reconciler stays covered by the PR gate that
`CLAUDE.md` notes never runs the distributed path at all.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.io.formats.ml.ragged import (
    is_ragged_tensor_column,
    ragged_tensor_type,
    ragged_to_numpy,
    to_ragged_tensor_column,
)
from batcher.io.formats.ml.tensor import tensor_from_values
from batcher.io.schema.evolution import normalize_batch, reconcile_batches, unify_schemas


def _fixed(shape: tuple[int, ...], rows: int = 2) -> pa.Array:
    return tensor_from_values([np.full(shape, i + 1, "uint8") for i in range(rows)])


def _schema_of(array: pa.Array) -> pa.Schema:
    return pa.schema([pa.field("img", array.type)])


def test_two_fixed_tensors_of_different_shape_unify_to_ragged():
    unified = unify_schemas([_schema_of(_fixed((1, 2))), _schema_of(_fixed((2, 2)))])
    assert is_ragged_tensor_column(unified.field("img").type)


def test_two_fixed_tensors_of_the_same_shape_are_left_alone():
    """The regression guard: widening these to ragged would cost every uniform column its
    tensor type for nothing."""
    same = _fixed((2, 2))
    unified = unify_schemas([_schema_of(same), _schema_of(same)])
    assert unified.field("img").type.equals(same.type)


def test_a_fixed_tensor_unifies_with_an_already_ragged_column():
    ragged = to_ragged_tensor_column([np.zeros((1, 2), "uint8"), np.zeros((3, 2), "uint8")])
    unified = unify_schemas([_schema_of(_fixed((2, 2))), _schema_of(ragged)])
    assert unified.field("img").type.equals(ragged_tensor_type())


def test_normalize_batch_re_encodes_a_fixed_tensor_as_ragged_keeping_every_row():
    """`cast` cannot reach the ragged struct from an extension type, so this is a real
    conversion and the values have to be checked, not just the type."""
    fixed = _fixed((2, 2), rows=3)
    batch = pa.RecordBatch.from_arrays([fixed], names=["img"])
    target = pa.schema([pa.field("img", ragged_tensor_type())])
    out = normalize_batch(batch, target)
    assert is_ragged_tensor_column(out.column("img"))
    assert out.num_rows == 3
    before = list(fixed.to_numpy_ndarray())
    after = ragged_to_numpy(out.column("img"))
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_reconcile_batches_joins_partitions_that_disagree_on_shape():
    """The whole path: two 'partitions' whose UDF output shapes differ, concatenated.

    This is the operation that raised `SchemaError` on the distributed path -- and only
    there, because single-node never splits the rows into two calls in the first place.
    """
    a = pa.RecordBatch.from_arrays([_fixed((1, 2), rows=2)], names=["img"])
    b = pa.RecordBatch.from_arrays([_fixed((2, 2), rows=2)], names=["img"])
    out = reconcile_batches([a, b])
    table = pa.Table.from_batches(out)
    assert table.num_rows == 4
    shapes = [arr.shape for arr in ragged_to_numpy(table.column("img").combine_chunks())]
    assert shapes == [(1, 2), (1, 2), (2, 2), (2, 2)]


@pytest.mark.parametrize("dtype", [pa.int64(), pa.string()])
def test_a_tensor_column_still_refuses_a_genuinely_incompatible_neighbour(dtype):
    """The arm must not turn every unification into a ragged column."""
    from batcher._internal.errors import SchemaError

    with pytest.raises(SchemaError, match="incompatible types"):
        unify_schemas([_schema_of(_fixed((2, 2))), pa.schema([pa.field("img", dtype)])])
