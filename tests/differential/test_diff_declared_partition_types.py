"""A declared Hive partition type reads the way DuckDB's `hive_types` does (AP-432).

Discovery types a key from its values, so ``k=01``/``k=02`` read back as int64 ``1``/``2``:
the leading zeros and the type are both gone. ``partitioning=`` now takes a
`pyarrow.Schema` or a ``{column: type}`` mapping that fixes the named keys' types, which
is DuckDB's ``hive_types`` and Spark's partition schema.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pytest
from tests._harness import assert_same

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.formats.structured.parquet.dataset import ParquetDatasetSource

pytestmark = pytest.mark.differential


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Zero-padded string keys, a null key, a duplicate, and a second (date) key."""
    out = str(tmp_path_factory.mktemp("padded") / "t")
    bt.from_pydict(
        {
            "k": ["01", "02", "02", None, "10"],
            "day": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
            "v": [1, 2, 3, 4, 5],
        }
    ).write.parquet(out, partition_by=["k", "day"])
    return out


def _duck(tree, hive_types: str):
    duckdb = pytest.importorskip("duckdb")
    return duckdb.connect().sql(
        f"SELECT * FROM read_parquet('{tree}/**/*.parquet', hive_partitioning=true, "
        f"hive_types={hive_types})"
    )


@pytest.mark.parametrize(
    "partitioning",
    [{"k": pa.string()}, {"k": "string"}, pa.schema([("k", pa.string())])],
    ids=["mapping", "alias", "schema"],
)
def test_declared_string_key_keeps_leading_zeros(tree, partitioning):
    got = bt.read.parquet(tree, partitioning=partitioning).to_arrow()
    assert got.schema.field("k").type == pa.string()
    # The undeclared key is still discovered, and still promoted to a date.
    assert got.schema.field("day").type == pa.date32()
    assert_same(got, _duck(tree, "{'k': 'VARCHAR', 'day': 'DATE'}"))


def test_undeclared_key_is_still_inferred(tree):
    # The control: without the declaration the key is int64, which is the lossy default
    # the option exists to correct, and what DuckDB infers too.
    got = bt.read.parquet(tree).to_arrow()
    assert got.schema.field("k").type == pa.int64()
    assert_same(got, _duck(tree, "{'k': 'BIGINT', 'day': 'DATE'}"))


def test_declared_type_filters_on_the_string_value(tree):
    got = bt.read.parquet(tree, partitioning={"k": pa.string()}).filter(bt.col("k") == "02")
    duck = _duck(tree, "{'k': 'VARCHAR', 'day': 'DATE'}").filter("k = '02'")
    assert_same(got.to_arrow(), duck)


def test_declared_types_survive_the_split_pickle(tree):
    source = ParquetDatasetSource(tree, partitioning={"k": pa.string()})
    splits = [pickle.loads(pickle.dumps(s)) for s in source.splits()]
    rows = pa.Table.from_batches([b for s in splits for b in s.read()])
    assert sorted(v for v in rows.column("k").to_pylist() if v is not None) == [
        "01",
        "02",
        "02",
        "10",
    ]
    # Typed and untyped reads of the same tree must not share a scan-cache identity.
    untyped = ParquetDatasetSource(tree)
    assert source.identity() != untyped.identity()
    assert {s.identity() for s in source.splits()}.isdisjoint(
        s.identity() for s in untyped.splits()
    )


def test_a_type_for_a_column_that_is_not_a_key_is_refused(tree):
    with pytest.raises(FormatError, match="is not a partition key"):
        bt.read.parquet(tree, partitioning={"nope": pa.string()}).to_arrow()


def test_a_non_type_is_refused():
    with pytest.raises(FormatError, match="expected a pyarrow type"):
        ParquetDatasetSource("unused", partitioning={"k": 3})
