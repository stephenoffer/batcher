"""Sizing the *input* from metadata, before a byte of it is read.

The in-memory path resolves every source to Arrow batches before the engine starts, so
a query's resident cost is dominated by what it *scans*, not by what it returns — a
`GROUP BY` yielding four rows still materializes every projected column of every row.
That term is invisible to the plan estimate, which sizes operator working sets. These
pin the metadata-only figure the conductor uses to route such a query out-of-core
instead of discovering the problem as an OOM.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.api.orchestration.sizing import projected_input_bytes as _projected_input_bytes

pytestmark = pytest.mark.unit


class _Sized:
    """A source that declares its size without a scan, as parquet does from its footer."""

    def __init__(self, rows: int | None, schema: pa.Schema):
        self._rows = rows
        self._schema = schema

    def row_count(self) -> int | None:
        return self._rows

    def schema(self) -> pa.Schema:
        return self._schema


class _Unsizable:
    """A source that cannot count itself (a socket, a generator, an opaque connector)."""

    def row_count(self) -> int | None:
        return None

    def schema(self) -> pa.Schema:
        return pa.schema([pa.field("a", pa.int64())])


_WIDE = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64()), pa.field("c", pa.int64())])


def test_size_is_rows_times_the_row_width():
    assert _projected_input_bytes([_Sized(1_000, _WIDE)], {}) == 1_000 * 24


def test_a_projection_only_charges_the_columns_it_reads():
    # Pushdown is the difference between scanning three columns and one, and the estimate
    # has to see that or it routes a narrow scan out-of-core on the width of columns the
    # query never touches.
    full = _projected_input_bytes([_Sized(1_000, _WIDE)], {})
    projected = _projected_input_bytes([_Sized(1_000, _WIDE)], {0: ["a"]})
    assert projected == 1_000 * 8
    assert projected < full


def test_sizes_sum_across_sources():
    srcs = [_Sized(1_000, _WIDE), _Sized(2_000, _WIDE)]
    assert _projected_input_bytes(srcs, {}) == 3_000 * 24


def test_one_unsizable_source_makes_the_whole_figure_unknown():
    # A partial sum understates the total, and understating is the direction that OOMs —
    # so an unknown source yields `0` ("no figure"), never a small number that reads as
    # "fits". The caller treats 0 as absence of evidence, not as evidence of fitting.
    assert _projected_input_bytes([_Sized(10**9, _WIDE), _Unsizable()], {}) == 0
    assert _projected_input_bytes([_Unsizable()], {}) == 0


def test_a_source_that_raises_while_describing_itself_is_unknown_not_zero():
    class _Broken:
        def row_count(self) -> int:
            return 10

        def schema(self):
            raise RuntimeError("no schema available")

    assert _projected_input_bytes([_Broken()], {}) == 0


def test_an_embedding_column_is_charged_its_real_width():
    # The regression this pairs with: a `fixed_size_list` embedding was charged a flat
    # scalar prior, so a vector workload's input read as ~1% of its true size — the exact
    # shape whose scan does not fit. 768 float32s is 3 KiB per row, not 32 bytes.
    schema = pa.schema([pa.field("v", pa.list_(pa.float32(), 768))])
    assert _projected_input_bytes([_Sized(1_000, schema)], {}) == 1_000 * 768 * 4
