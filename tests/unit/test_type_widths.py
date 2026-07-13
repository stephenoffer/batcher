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

from batcher.plan.types import DEFAULT_VARLEN_BYTES, column_bytes, schema_row_bytes

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


def test_schema_row_bytes_sums_the_columns():
    # The exact shape that decides a broadcast: a two-int64 join key is 16 B/row,
    # not the 64 B/row a flat per-row constant would have charged it.
    schema = pa.schema([pa.field("k", pa.int64()), pa.field("v", pa.int64())])
    assert schema_row_bytes(schema) == 16.0
