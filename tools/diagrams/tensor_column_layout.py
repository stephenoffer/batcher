#!/usr/bin/env python3
"""Draw `tensor_column_layout.svg` — a batch of images as one Arrow buffer, against
the object column it would otherwise be.

Source of truth: `python/batcher/io/formats/ml/tensor.py` (the canonical
`arrow.fixed_shape_tensor` extension over a `FixedSizeList`, the shape in the field
metadata, `as_tensor_column` reinterpreting storage with no data copy; the `uint8`
value type and the `(224, 224, 3)` shape are that module's own examples),
`python/batcher/interop/formats.py` (what `RecordBatch.to_pandas()` hands back for a
tensor column, and why the batch is re-wrapped before a user function sees it) and
`python/batcher/interop/arrays.py` (`_column_to_numpy`, `arrays_to_torch` and its
`zero_copy` DLPack view).

The right-hand panel deliberately mirrors `arrow_memory_layout`: the load-bearing
difference is the buffer that is *absent*, because a fixed-size row needs no offsets.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 960, 560

body = [
    band(24, 20, 440, 276, "ONE PYTHON OBJECT PER ROW", "grey"),
    card(48, 72, 88, 58, "ndarray"),
    card(148, 72, 88, 58, "ndarray"),
    card(248, 72, 88, 58, "ndarray"),
    card(348, 72, 88, 58, "ndarray"),
    note(48, 154, "n separate allocations, one per row."),
    note(48, 182, "Every row is a pointer, so the batch's bytes are"),
    note(48, 200, "scattered and nothing can be handed to a kernel"),
    note(48, 218, "or across the FFI boundary whole."),
    note(48, 246, "This is what RecordBatch.to_pandas() gives for a"),
    note(48, 264, "tensor column, which is why the batch is re-wrapped"),
    note(48, 282, "before a user function ever sees it."),
    band(496, 20, 440, 276, "ONE ARROW COLUMN", "blue"),
    card(520, 72, 392, 54, "validity bitmap", "one bit per row"),
    card(520, 136, 392, 54, "values", "n x 150528 uint8, contiguous"),
    note(520, 212, "No offsets buffer: every row is the same size,"),
    note(520, 230, "so row i begins at byte i x 150528."),
    note(520, 258, "arrow.fixed_shape_tensor over a FixedSizeList, with"),
    note(520, 276, "the shape (224, 224, 3) in the field metadata."),
    label(716, 322, "read back"),
    arrow(704, 300, 704, 332),
    band(24, 336, 912, 180, "WHAT A READER GETS BACK", "grey"),
    card(48, 376, 420, 72, "to_numpy_ndarray()", "(n, 224, 224, 3), shape from the type"),
    card(496, 376, 416, 72, "arrays_to_torch(zero_copy=True)", "a DLPack view over that buffer"),
    note(
        48,
        480,
        "The shape travels with the data, so the column crosses the FFI boundary and comes back shaped:",
    ),
    note(
        48,
        498,
        "no IR tag and no two-sided contract, which is what choosing the canonical type buys.",
    ),
    note(
        480,
        546,
        "The default torch path owns a writable copy instead: a training loop mutates its batch, and the Arrow buffer is read-only.",
        anchor="middle",
    ),
]

write("tensor_column_layout", svg(W, H, "".join(body)))
print("wrote tensor_column_layout.svg")
