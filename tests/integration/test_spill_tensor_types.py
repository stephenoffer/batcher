"""A tensor column that spills must still be a tensor.

`fixed_shape_tensor` is the type every Batcher tensor column carries -- a decoded image, a
sampled clip, an audio waveform, a model output, and anything `bt.from_numpy` builds. The
shape lives in the Arrow extension type rather than in the data, which is what lets a
column round-trip to a correctly-shaped training tensor, and the docs say so.

An extension type survives an in-memory run because the FFI boundary passes it through
unnormalized. A **group-key round trip** did not: a spilled or distributed whole-row
``DISTINCT`` lowers to a group-by over every column, and the key came back as plain
storage, which then picked up the boundary's ordinary narrow-type widening on the way out.
So a `fixed_shape_tensor(uint8, [2, 2, 3])` returned as `fixed_size_list<int64>[12]` --
the same numbers, eight times the width, and no longer a tensor.

Nothing downstream complains, because that is a perfectly valid column of a different
type. A model handed it sees a flat vector where it expected an image, and the only place
the difference is visible is the schema nobody prints.

Two neighbouring cases are deliberately *not* fixed here, and are pinned as they are:

* the **distributed** reducer has the same defect on its ``materialize=False`` branch,
  where `materialize_reduce_output` types the result from the IPC file it wrote rather
  than from the operator's declared schema -- one line in
  `dist/executors/partition_io/_sources.py`;
* a **group-by the user wrote** loses the extension type single-node as well, so
  restoring only the spilled side would make spill disagree with single-node in the
  other direction. See `test_a_group_by_on_a_tensor_key_is_left_alone_deliberately`.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

_ROWS = 400
_SHAPE = (2, 2, 3)
_WIDTH = _SHAPE[0] * _SHAPE[1] * _SHAPE[2]


def _tensor_table() -> pa.Table:
    """Rows of distinct `uint8` tensors, plus an ordinary key to group or sort by."""
    values = pa.array(np.arange(_ROWS * _WIDTH, dtype="uint8"))
    storage = pa.FixedSizeListArray.from_arrays(values, _WIDTH)
    tensors = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(pa.uint8(), _SHAPE), storage)
    keys = pa.array([i % 20 for i in range(_ROWS)], type=pa.int64())
    return pa.table({"k": keys, "t": tensors})


@pytest.mark.parametrize("num_partitions", [1, 4])
def test_a_spilled_distinct_keeps_the_tensor_type(num_partitions):
    """The regression. Same type and same values as the in-memory run."""
    dataset = bt.from_arrow(_tensor_table()).select("t")

    in_memory = dataset.distinct().collect()
    spilled = dataset.distinct().collect(spill=True, num_partitions=num_partitions)

    assert isinstance(in_memory.schema.field("t").type, pa.FixedShapeTensorType), (
        "the in-memory result must be a tensor for this comparison to mean anything"
    )
    assert spilled.schema.field("t").type == in_memory.schema.field("t").type
    assert tuple(spilled.schema.field("t").type.shape) == _SHAPE
    assert spilled.num_rows == in_memory.num_rows

    def rows(table):
        return sorted(tuple(row) for row in table.column("t").combine_chunks().to_pylist())

    assert rows(spilled) == rows(in_memory)


def test_a_group_by_on_a_tensor_key_is_left_alone_deliberately():
    """The neighbouring case, pinned as it is rather than fixed, because it is not the same bug.

    A `DISTINCT` keeps the extension type single-node and lost it when spilled, so the two
    disagreed and restoring the declared schema makes them agree. A group-by the *user*
    wrote loses it single-node **too** -- it comes back as the plain storage -- so restoring
    the spilled side would have made spill disagree with single-node in the opposite
    direction, which is the invariant this is supposed to protect.

    What is left is a real discrepancy of its own: the two paths return the same storage
    layout at different widths, and neither matches the schema the plan declares. That is
    engine-side, in how a group key round-trips, and is deeper than the lowering fixed here.
    This test exists so that whoever fixes it sees this expectation fail and updates both.
    """
    dataset = bt.from_arrow(_tensor_table())
    query = dataset.group_by("t").agg(n=bt.col("k").count())

    in_memory = query.collect().schema.field("t").type
    spilled = query.collect(spill=True, num_partitions=4).schema.field("t").type

    declared = query._plan.available_schema().arrow.field("t").type
    assert isinstance(declared, pa.FixedShapeTensorType), "the plan still declares a tensor"
    assert not isinstance(in_memory, pa.FixedShapeTensorType), (
        "single-node group-by now keeps the tensor type -- fix the spilled side to match"
    )
    assert not isinstance(spilled, pa.FixedShapeTensorType)


def test_a_decoded_image_keeps_its_shape_through_a_spill():
    """The shape this exists for: decode to a tensor, then dedup out-of-core.

    A near-duplicate pass over an image corpus is exactly a `DISTINCT` over a decoded
    column, and it is the shape most likely to exceed memory and therefore to spill.
    """
    import struct
    import zlib

    def png(shade: int) -> bytes:
        def chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
        row = b"\x00" + bytes((shade, 40, 200 - shade)) * 2
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(row * 2))
            + chunk(b"IEND", b"")
        )

    frames = pa.array([png(i % 200) for i in range(200)], type=pa.binary())
    dataset = bt.from_arrow(pa.table({"b": frames})).select(t=bt.col("b").image.to_tensor(2, 2))

    in_memory = dataset.distinct().collect()
    spilled = dataset.distinct().collect(spill=True, num_partitions=4)

    assert isinstance(in_memory.schema.field("t").type, pa.FixedShapeTensorType)
    assert spilled.schema.field("t").type == in_memory.schema.field("t").type
    assert tuple(spilled.schema.field("t").type.shape) == (2, 2, 3)
    assert spilled.num_rows == in_memory.num_rows
