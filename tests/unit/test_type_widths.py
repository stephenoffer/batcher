"""`plan.types.widths` — per-column byte widths derived from the Arrow type.

The cost model's byte axes (broadcast eligibility, memory, IO) are only as good as the
width they are fed. Before this, an unmeasured relation was costed at a flat
`bytes_per_row` constant regardless of schema, so a two-`int64` join key (16 B/row) and
a 20-column payload were both 64 B/row — which over-sized narrow build sides ~4x and
forfeited the broadcast join they should have had.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.plan.types import (
    DEFAULT_VARLEN_BYTES,
    column_bytes,
    projected_row_bytes,
    schema_row_bytes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pa.int64(), 8.0),
        (pa.float64(), 8.0),
        (pa.int32(), 4.0),
        (pa.int8(), 1.0),
        (pa.date32(), 4.0),
        (pa.timestamp("us"), 8.0),
        (pa.bool_(), 1.0),
    ],
)
def test_fixed_width_types_are_exact(dtype, expected):
    assert column_bytes(dtype) == expected


def test_variable_length_types_use_the_prior_plus_offsets():
    # A string's true width needs a measurement; the prior carries its offset buffer.
    assert column_bytes(pa.string()) == DEFAULT_VARLEN_BYTES + 4.0
    assert column_bytes(pa.large_string()) == DEFAULT_VARLEN_BYTES + 8.0


def test_dictionary_costs_only_its_index():
    # The values live in a shared dictionary, so a row costs its index, not the value.
    assert column_bytes(pa.dictionary(pa.int32(), pa.string())) == 4.0
    assert column_bytes(pa.dictionary(pa.int8(), pa.string())) == 1.0


def test_struct_sums_its_fields():
    dtype = pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.int32())])
    assert column_bytes(dtype) == 12.0


def test_fixed_size_list_is_exact_from_its_type():
    # The embedding column, and the one nested case needing no prior: the length is in
    # the type. Charging it a flat scalar prior (the old behavior) under-read a 768-dim
    # float32 vector as 32 B/row instead of 3 KiB — a ~96x under-estimate of the memory
    # envelope on exactly the workload that spills.
    assert column_bytes(pa.list_(pa.float32(), 768)) == 768 * 4.0
    assert column_bytes(pa.list_(pa.float64(), 1536)) == 1536 * 8.0
    # Nested exactly, too — a fixed-size list of structs.
    inner = pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.int32())])
    assert column_bytes(pa.list_(inner, 4)) == 4 * 12.0


def test_variable_length_list_scales_with_its_element_type():
    # No length in the type, so the count is a prior — but the *element* is known, so a
    # list of 8-byte values must not cost the same as a list of 1-byte ones.
    assert column_bytes(pa.list_(pa.float64())) > column_bytes(pa.list_(pa.int8()))
    # And it carries its offset buffer, wide for the large variant.
    assert column_bytes(pa.large_list(pa.int64())) - column_bytes(pa.list_(pa.int64())) == 4.0


def test_extension_type_is_sized_by_its_storage():
    # THE multimodal case. Every decoded image, audio waveform, video frame stack, and
    # model output in Batcher is the canonical `arrow.fixed_shape_tensor` extension type
    # (io/formats/ml/tensor.py, ml/decode/media.py, core/udf/call.py). None of the
    # `pa.types.is_*` predicates see through an extension label, so before this a
    # 224x224x3 uint8 image was sized at the 32 B varlen prior against a true 150,528 —
    # a 4,704x under-estimate, in the direction that makes a memory envelope too small
    # and a build side look broadcastable when replicating it would OOM every worker.
    image = pa.fixed_shape_tensor(pa.uint8(), [224, 224, 3])
    assert column_bytes(image) == 224 * 224 * 3
    # And an embedding column produced the same way is exact, not a prior.
    assert column_bytes(pa.fixed_shape_tensor(pa.float32(), [768])) == 768 * 4


def test_map_is_sized_as_a_list_of_entries():
    # A map is how every semi-structured source (JSON objects with open-ended keys,
    # Parquet MAP groups, Avro maps) lands in Arrow, and it matches none of the list
    # predicates — so it used to fall through to the flat 32 B scalar prior.
    entry = column_bytes(pa.string()) + column_bytes(pa.int64())
    assert column_bytes(pa.map_(pa.string(), pa.int64())) > entry
    # Bigger values make a bigger map: the element types are not ignored.
    assert column_bytes(pa.map_(pa.string(), pa.float64())) > column_bytes(
        pa.map_(pa.string(), pa.int8())
    )


def test_null_column_costs_nothing():
    # Arrow's `null` type is pure metadata with no value buffer. Charging it the varlen
    # prior invented 32 B/row for a column occupying none — the shape a JSON or CSV
    # source produces for a field it saw only nulls in.
    assert column_bytes(pa.null()) == 0.0


def test_run_end_encoded_is_bounded_by_one_run_per_row():
    # An honest worst case (a run of length 1), not an invented compression ratio —
    # and still far below the 32 B prior it used to hit.
    ree = pa.run_end_encoded(pa.int32(), pa.int64())
    assert column_bytes(ree) == column_bytes(pa.int32()) + column_bytes(pa.int64())
    assert column_bytes(ree) < DEFAULT_VARLEN_BYTES


def test_union_is_sized_by_its_layout():
    # A sparse union allocates every variant for every row; a dense union allocates only
    # the chosen one. Both used to fall through to the scalar prior, which under-reads a
    # sparse union by roughly its arity — the shape an Avro union takes.
    fields = [pa.field("a", pa.int64()), pa.field("b", pa.float64())]
    sparse = column_bytes(pa.sparse_union(fields))
    dense = column_bytes(pa.dense_union(fields))
    assert sparse > dense
    assert sparse >= column_bytes(pa.int64()) + column_bytes(pa.float64())


def test_view_types_carry_the_view_struct_not_an_offset():
    # A string view inlines short values in a 16-byte struct instead of an offset pair.
    assert column_bytes(pa.string_view()) == DEFAULT_VARLEN_BYTES + 16.0
    # A list view carries an offset *and* a size buffer, so two buffers wide per row.
    assert column_bytes(pa.list_view(pa.int64())) == column_bytes(pa.large_list(pa.int64()))


def test_schema_row_bytes_sums_the_columns():
    # The exact shape that decides a broadcast: a two-int64 join key is 16 B/row,
    # not the 64 B/row a flat per-row constant would have charged it.
    schema = pa.schema([pa.field("k", pa.int64()), pa.field("v", pa.int64())])
    assert schema_row_bytes(schema) == 16.0


def test_projected_row_bytes_sums_only_the_projected_columns():
    # The whole point of the projected form: a query naming one column of a wide relation
    # is sized by that column, and never by a walk of the other 104.
    schema = pa.schema(
        [pa.field("a", pa.int64()), pa.field("b", pa.float64()), pa.field("c", pa.string())]
    )
    assert projected_row_bytes(schema, ["a"]) == 8.0
    assert projected_row_bytes(schema, ["a", "b"]) == 16.0
    assert projected_row_bytes(schema, ["c"]) == column_bytes(pa.string())


def test_projected_row_bytes_with_no_projection_is_the_whole_schema():
    schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.float64())])
    assert projected_row_bytes(schema, None) == schema_row_bytes(schema)
    assert projected_row_bytes(schema, []) == schema_row_bytes(schema)


def test_projected_row_bytes_ignores_a_name_the_schema_does_not_carry():
    # The caller is sizing a read. A projection naming a column the source lacks means the
    # plan's column resolution and this schema disagree — a reason to under-estimate a byte
    # total, never to fail the query at its memory guard.
    schema = pa.schema([pa.field("a", pa.int64())])
    assert projected_row_bytes(schema, ["a", "missing"]) == 8.0


def test_projected_row_bytes_repeats_are_charged_per_reference():
    # A projection is a list of reads, so naming a column twice reads it twice.
    schema = pa.schema([pa.field("a", pa.int64())])
    assert projected_row_bytes(schema, ["a", "a"]) == 16.0
