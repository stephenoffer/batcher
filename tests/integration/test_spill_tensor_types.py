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
* a **group-by the user wrote**, and a **join carrying a tensor payload**, lost the
  extension type single-node as well -- so restoring only the spilled side would have
  made spill disagree with single-node in the other direction. Both are now restored
  once at the boundary every relational result passes through, which is what lets all
  three paths agree instead of being repaired one at a time.
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


def test_a_group_by_on_a_tensor_key_keeps_it_on_both_paths():
    """The case that first had to be left alone, now fixed at the boundary instead.

    Restoring the type per operator could only ever fix one path at a time: a group-by the
    user wrote lost the extension type single-node *as well*, so repairing just the spilled
    side would have traded one mismatch for another. Doing it once where every relational
    result passes through means single-node, spilled and distributed all get the same
    answer, and none of them can disagree with another.

    This test was originally written to assert the *unfixed* shape, so that whoever fixed
    it would see it fail. That is what happened.
    """
    dataset = bt.from_arrow(_tensor_table())
    query = dataset.group_by("t").agg(n=bt.col("k").count())

    declared = query._plan.available_schema().arrow.field("t").type
    in_memory = query.collect()
    spilled = query.collect(spill=True, num_partitions=4)

    assert isinstance(declared, pa.FixedShapeTensorType)
    assert in_memory.schema.field("t").type == declared
    assert spilled.schema.field("t").type == declared
    assert spilled.num_rows == in_memory.num_rows


def test_a_join_carrying_a_tensor_keeps_it():
    """A join is the other hash-keyed operator that dropped the label.

    Joining a decoded frame to anything -- detections to images, labels to clips -- is
    ordinary, and the payload came back as plain storage.
    """
    dataset = bt.from_arrow(_tensor_table())
    right = dataset.select(k2=bt.col("k")).distinct()
    joined = dataset.join(right, left_on="k", right_on="k2").select("t")

    declared = joined._plan.available_schema().arrow.field("t").type
    assert isinstance(declared, pa.FixedShapeTensorType)
    assert joined.collect().schema.field("t").type == declared


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
