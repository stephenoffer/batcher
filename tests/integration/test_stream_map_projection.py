"""A streamed `map_batches` reads the columns the pipeline needs, not the whole source.

`collect()` has always narrowed the scan to `kyber.required_columns_per_source`, and so has
the relational branch of the streaming router. The `map_batches` branch did not: it drove
`stream_windowed` with no projection, so a UDF declaring `input_columns` over a wide table
still decoded every column of every window and discarded the rest. That is the widest read
in the engine landing on the one API that exists for inputs too large to collect.

The saving is invisible to a result comparison by construction — a projection cannot change
rows — so these tests assert on what the *source is asked for*, and separately that the rows
are unchanged.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.integration

_WIDE = 12
_ROWS = 4_000


@pytest.fixture
def wide_table(tmp_path) -> str:
    """A parquet file with `_WIDE` columns, of which a UDF will read two."""
    cols = {f"c{i}": pa.array([float(i * r) for r in range(_ROWS)]) for i in range(_WIDE)}
    path = str(tmp_path / "wide.parquet")
    pq.write_table(pa.table(cols), path, row_group_size=500)
    return path


def _sum_two(batch: pa.RecordBatch) -> pa.RecordBatch:
    import pyarrow.compute as pc

    return pa.record_batch({"s": pc.add(batch.column("c0"), batch.column("c1"))})


def test_streamed_map_reads_only_the_declared_columns(wide_table) -> None:
    """Asserted on the batch the UDF is *handed*, not on the one it returns.

    The return width is 1 whether or not the scan was narrowed, so asserting on it would
    pass before the fix as well.
    """
    seen: list[list[str]] = []

    def peek(batch: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(batch.schema.names)
        return _sum_two(batch)

    ds = bt.read.parquet(wide_table).map_batches(
        peek, input_columns=["c0", "c1"], output_columns=["s"]
    )
    list(ds.iter_batches())

    assert seen, "the UDF never ran"
    assert {tuple(sorted(names)) for names in seen} == {("c0", "c1")}


def test_streamed_map_matches_the_collected_result(wide_table) -> None:
    """The projection is a read narrowing, so the rows must be identical either way."""
    ds = bt.read.parquet(wide_table).map_batches(
        _sum_two, input_columns=["c0", "c1"], output_columns=["s"]
    )

    collected = ds.collect().column("s").to_pylist()
    streamed = [v for b in ds.iter_batches() for v in b.column("s").to_pylist()]

    assert streamed == collected
    assert len(streamed) == _ROWS


def test_an_undeclared_udf_still_sees_every_column(wide_table) -> None:
    """`input_columns=None` means the `fn` is a black box, and the only safe projection is
    none at all — narrowing it would hand the UDF a batch missing a column it reads."""
    seen: list[int] = []

    def peek(batch: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(batch.num_columns)
        return batch

    ds = bt.read.parquet(wide_table).map_batches(peek)
    list(ds.iter_batches())

    assert seen and set(seen) == {_WIDE}
