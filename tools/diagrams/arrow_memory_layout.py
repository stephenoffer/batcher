#!/usr/bin/env python3
"""Draw `arrow_memory_layout.svg` — a RecordBatch's buffers, and what a slice does
not copy.

Source of truth: `crates/bc-arrow/src/lib.rs` — `Morsel`, `DEFAULT_MORSEL_ROWS`,
`fixed_width`, and above all `slice_bytes`, whose doc comment records why the naive
measure is wrong: `Array::get_array_memory_size` reports the whole backing
allocation, so a relation of n morsels over one parent buffer reads as n times its
real footprint. `crates/bc-interp/src/ops/morsel.rs::emit_uniform` is where the
engine makes those slices (`b.slice(off, len)`), and `crates/bc-py/src/lib.rs`
states the C-Data-Interface boundary the last band describes.

The buffer shapes themselves are the Arrow columnar specification, which is the
engine's one columnar contract; nothing here is a Batcher-specific encoding.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 960, 642

COL1, COL2 = 56, 490
CW = 414

body = [
    band(24, 20, 912, 300, "ONE RECORDBATCH: A SCHEMA, AND BUFFERS PER COLUMN", "blue"),
    label(COL1, 62, "id : Int64"),
    card(COL1, 76, CW, 54, "validity bitmap", "one bit per row, absent when nothing is null"),
    card(COL1, 140, CW, 54, "values", "eight bytes per row, back to back"),
    note(COL1, 222, "No offsets buffer: every value is the same width,"),
    note(COL1, 240, "so row i begins at byte i x 8."),
    label(COL2, 62, "name : Utf8"),
    card(COL2, 76, CW, 54, "validity bitmap", "one bit per row"),
    card(COL2, 140, CW, 54, "offsets", "n + 1 int32s"),
    card(COL2, 204, CW, 54, "values", "every row's bytes end to end, unpadded"),
    note(COL2, 286, "Row i is values[ offsets[i] .. offsets[i+1] ]:"),
    note(COL2, 304, "a per-row length nobody has to store."),
    band(24, 336, 912, 180, "WHAT A ZERO-COPY SLICE DOES", "grey"),
    card(56, 376, 340, 72, "the parent batch", "one allocation per buffer"),
    arrow(396, 412, 520, 412),
    label(458, 400, "batch.slice(off, len)", anchor="middle"),
    card(520, 376, 384, 72, "a second RecordBatch", "an offset and a length, nothing else"),
    note(56, 476, "Copied: nothing. But get_array_memory_size still reports the parent's whole allocation, which is why"),
    note(56, 494, "every size decision in the engine measures a morsel with bc_arrow::slice_bytes instead."),
    band(24, 532, 912, 88, "WHY THE FFI BOUNDARY COSTS NOTHING", "blue"),
    note(56, 572, "Those same buffer pointers cross to pyarrow through the Arrow C Data Interface. No copy, no"),
    note(56, 590, "serialization: the morsel Rust hands back is the batch Python already holds."),
]

write("arrow_memory_layout", svg(W, H, "".join(body)))
print("wrote arrow_memory_layout.svg")
