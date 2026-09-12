"""The map fan-out's byte term must size the *projected* read, not the whole relation.

`_byte_partition_count` bounds what one task holds, and a source's own byte total covers
every column whatever the query asked for. Sizing a narrow query over a wide table on the
unprojected total shards it as if it read the wide table: on TPC-H sf100 `lineitem`, one
`float64` column of sixteen, the source reports 124.8 GB against a 4.8 GB read, and the byte
term returned 465 partitions where the row term asked for 301. The term meant to be a memory
*bound* was setting the fan-out, and that pipeline ran 2,257 ms at 465 partitions against
1,758 ms at 301.

`_minus_pruned_columns` subtracts only the columns nobody reads. These pin both halves: that
it shrinks with the projection, and that it errs toward *more* partitions when a pruned
column's width is under-modelled — the direction an OOM guard has to fail in.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.dist.executors import map as mapmod

pytestmark = pytest.mark.unit


@pytest.fixture
def wide(tmp_path):
    """A wide-ish Parquet source: one narrow numeric column among several fat string ones."""
    n = 20_000
    table = pa.table(
        {
            "v": pa.array(np.arange(n, dtype="int64")),
            **{f"s{i}": pa.array(["x" * 64] * n) for i in range(6)},
        }
    )
    pq.write_table(table, tmp_path / "wide.parquet", row_group_size=2_000)
    import batcher as bt

    return bt.read.parquet(str(tmp_path / "wide.parquet"))


def test_the_byte_term_shrinks_with_the_projection(wide):
    """Reading one narrow column must not be sized like reading every fat one.

    Asserted on the bytes rather than the partition count: the count is
    `ceil(bytes / target_bytes_per_task)` and a fixture small enough to build in a unit test
    rounds to one partition either way, which would make the comparison vacuous.
    """
    narrow = wide.select("v").map_batches(lambda b: b, output_columns=["v"])
    whole = wide.map_batches(lambda b: b, output_columns=wide.schema.names)
    src = wide._sources[0]
    rows = mapmod._source_total_rows(src)
    reported = mapmod._source_total_bytes(src)

    narrow_bytes = mapmod._minus_pruned_columns(src, narrow._plan, rows, reported)
    whole_bytes = mapmod._minus_pruned_columns(src, whole._plan, rows, reported)
    assert narrow_bytes < whole_bytes / 4, (
        f"one int64 column of seven must size well below the whole relation "
        f"({narrow_bytes:.0f} vs {whole_bytes:.0f} bytes)"
    )


def test_pruning_an_under_modelled_column_asks_for_MORE_partitions_not_fewer(wide):
    """The safety direction, stated as a test.

    A variable-length column's width is a prior, and a media column's real bytes can exceed
    it by orders of magnitude. Only the *pruned* columns are modelled, so subtracting a prior
    that is too small removes too little and the estimate stays high — more tasks, smaller
    each. The retained columns are never estimated: they keep the source's authoritative
    share. So the result can never fall below the projected columns' own modelled total.
    """
    src = wide._sources[0]
    rows = mapmod._source_total_rows(src)
    reported = mapmod._source_total_bytes(src)
    assert reported is not None, "the fixture's source must report a byte total"

    narrow = wide.select("v").map_batches(lambda b: b, output_columns=["v"])
    got = mapmod._minus_pruned_columns(src, narrow._plan, rows, reported)

    from batcher.plan.types.widths import projected_row_bytes, schema_row_bytes

    schema = src.schema()
    ratio = projected_row_bytes(schema, ["v"]) / schema_row_bytes(schema)
    assert got >= reported * ratio, "never below the dimensionless scaled share"
    assert got <= reported, "pruning columns cannot make the read bigger than the relation"


def test_no_projection_leaves_the_source_total_alone(wide):
    """With nothing pruned there is nothing to subtract — the authoritative total stands."""
    src = wide._sources[0]
    rows = mapmod._source_total_rows(src)
    reported = mapmod._source_total_bytes(src)
    whole = wide.map_batches(lambda b: b, output_columns=wide.schema.names)
    assert mapmod._minus_pruned_columns(src, whole._plan, rows, reported) == pytest.approx(
        reported, rel=1e-6
    )
