"""Edges of multi-file schema reconciliation, and the splits that carry it to a worker.

Three groups, each pinned against an oracle rather than a hand-written expectation:

- **Strict mode holds every file to file 0's schema, and a cast may not change a value.**
  Arrow's safe cast let ``5`` become ``true``, a timestamp lose its time of day as a
  ``date32``, ``0.1`` round to ``float32`` and a UTC timestamp become a naive one, all
  without an error. The oracle is the file's own values: a conformed read either returns
  them unchanged or raises `SchemaError` naming the file, the column and both types.
- **The evolution modes agree with DuckDB's ``union_by_name``** on the rows, and the places
  they deliberately differ (case-sensitive names, ``uint64`` beside ``int64``, a naive
  timestamp beside an aware one) are pinned so a change to them is a decision.
- **A strict multi-file source's splits carry the contract.** A split rebuilds a one-file
  reader on the worker, and a one-file source's contract is its own file, so the check never
  ran distributed and the gather reconciled whatever came back: a renamed column returned
  one row of two. Reading every split here is the distributed read in miniature.
"""

from __future__ import annotations

import pickle
from datetime import date, datetime

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import SchemaError
from batcher.io.formats.structured.parquet.source import ParquetSource
from batcher.io.schema import conform_batch, normalize_batch, unify_schemas
from batcher.io.splits import ConformedSplit, RowGroupSplit

pytestmark = pytest.mark.differential


def _write(tmp_path, tables: list[pa.Table]) -> str:
    for i, table in enumerate(tables):
        pq.write_table(table, tmp_path / f"f{i}.parquet")
    return str(tmp_path)


def _rows(table: pa.Table) -> list[str]:
    """An order-independent, type-sensitive fingerprint of a table's rows."""
    ordered = table.select(sorted(table.column_names))
    return sorted(repr(row) for row in ordered.to_pylist())


def _duck_union(directory: str) -> pa.Table:
    return duckdb.sql(
        f"select * from read_parquet('{directory}/*.parquet', union_by_name=true)"
    ).to_arrow_table()


# --- strict mode: a cast that would change a value raises -------------------------------


_LOSSY = {
    "int_to_bool": (pa.array([True]), pa.array([5])),
    "timestamp_to_date": (
        pa.array([date(1970, 1, 1)]),
        pa.array([datetime(1970, 1, 1, 1, 0)], pa.timestamp("us")),
    ),
    "double_to_float": (pa.array([1.5], pa.float32()), pa.array([0.1])),
    "utc_to_naive": (
        pa.array([1], pa.timestamp("us")),
        pa.array([1], pa.timestamp("us", "UTC")),
    ),
    "double_to_int": (pa.array([1]), pa.array([2.5])),
}


@pytest.mark.parametrize("case", sorted(_LOSSY))
def test_a_lossy_strict_conformance_raises_naming_file_column_and_types(tmp_path, case):
    first, second = _LOSSY[case]
    directory = _write(tmp_path, [pa.table({"a": first}), pa.table({"a": second})])
    with pytest.raises(SchemaError) as info:
        bt.read.parquet(directory).collect(distributed=False)
    message = str(info.value)
    assert "f1.parquet" in message
    assert "'a'" in message
    assert str(second.type) in message and str(first.type) in message


# Each case: file 0's column, file 1's column, and the values the read must return -- file
# 1's own values, unchanged, in the declared type.
_LOSSLESS = {
    "int_to_bool_zero_one": (pa.array([True]), pa.array([0, 1]), [True, False, True]),
    "midnight_timestamp_to_date": (
        pa.array([date(1970, 1, 1)]),
        pa.array([datetime(1970, 1, 2)], pa.timestamp("us")),
        [date(1970, 1, 1), date(1970, 1, 2)],
    ),
    "exact_double_to_float": (pa.array([1.5], pa.float32()), pa.array([2.5]), [1.5, 2.5]),
    "int32_to_int64": (pa.array([1]), pa.array([2], pa.int32()), [1, 2]),
    "dictionary_to_string": (pa.array(["x"]), pa.array(["y"]).dictionary_encode(), ["x", "y"]),
}


@pytest.mark.parametrize("case", sorted(_LOSSLESS))
def test_a_value_preserving_strict_conformance_still_reads(tmp_path, case):
    """The check refuses a changed value, not a differing type: these convert exactly."""
    first, second, expected = _LOSSLESS[case]
    directory = _write(tmp_path, [pa.table({"a": first}), pa.table({"a": second})])
    out = bt.read.parquet(directory).collect(distributed=False)
    assert out.column("a").to_pylist() == expected


def test_a_null_in_a_column_the_first_file_declares_non_nullable_is_a_schema_error(tmp_path):
    required = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    directory = _write(
        tmp_path,
        [
            pa.Table.from_pylist([{"a": 1}], schema=required),
            pa.table({"a": pa.array([None], pa.int64())}),
        ],
    )
    with pytest.raises(SchemaError, match="non-nullable") as info:
        bt.read.parquet(directory).collect(distributed=False)
    assert "f1.parquet" in str(info.value)


def test_conform_batch_leaves_a_conforming_batch_untouched():
    batch = pa.record_batch({"a": [1, 2]})
    assert conform_batch(batch, batch.schema, path="f0") is batch


# --- the evolution modes against DuckDB's union_by_name ----------------------------------


_DRIFT = {
    "added_column": [pa.table({"a": [1, 2]}), pa.table({"a": [3], "b": ["x"]})],
    "int_widening": [pa.table({"a": pa.array([1], pa.int32())}), pa.table({"a": [2**40]})],
    "date_into_timestamp": [
        pa.table({"a": pa.array([date(1970, 1, 1)])}),
        pa.table({"a": pa.array([datetime(1970, 1, 1, 1)], pa.timestamp("us"))}),
    ],
    "reordered": [pa.table({"a": [1], "b": ["x"]}), pa.table({"b": ["y"], "a": [2]})],
    "int_then_float": [pa.table({"a": [1]}), pa.table({"a": [2.5]})],
}


@pytest.mark.parametrize("case", sorted(_DRIFT))
def test_union_returns_duckdbs_union_by_name_rows(tmp_path, case):
    directory = _write(tmp_path, _DRIFT[case])
    ours = bt.read.parquet(directory, schema_mode="union").collect(distributed=False)
    assert _rows(ours) == _rows(_duck_union(directory))


def test_union_keeps_a_and_capital_a_apart_where_duckdb_folds_them(tmp_path):
    """Column names are case-sensitive here; DuckDB matches them case-insensitively."""
    directory = _write(tmp_path, [pa.table({"A": [1]}), pa.table({"a": [2]})])
    ours = bt.read.parquet(directory, schema_mode="union").collect(distributed=False)
    assert ours.column_names == ["A", "a"]
    assert len(_duck_union(directory).column_names) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (pa.array([1], pa.timestamp("us")), pa.array([1], pa.timestamp("us", "UTC"))),
        (pa.array([b"x"]), pa.array(["y"])),
    ],
    ids=["naive_vs_aware_timestamp", "binary_vs_string"],
)
def test_union_rejects_a_pair_with_no_lossless_common_type(tmp_path, first, second):
    directory = _write(tmp_path, [pa.table({"a": first}), pa.table({"a": second})])
    with pytest.raises(SchemaError, match="incompatible types"):
        bt.read.parquet(directory, schema_mode="union").collect(distributed=False)


def test_string_and_large_string_read_back_as_string(tmp_path):
    """The reconciled type is `large_string`, and the engine boundary reads it as `string`."""
    first, second = pa.table({"a": pa.array(["x"], pa.large_string())}), pa.table({"a": ["y"]})
    assert unify_schemas([first.schema, second.schema]).field("a").type == pa.large_string()
    directory = _write(tmp_path, [first, second])
    out = bt.read.parquet(directory, schema_mode="union").collect(distributed=False)
    assert out.schema.field("a").type == pa.string()


def test_uint64_beside_int64_names_the_file_whose_value_does_not_fit(tmp_path):
    """`uint64` and `int64` meet at `int64`, which is what the engine holds every integer as.

    DuckDB widens the pair to a 128-bit integer instead. A value above ``2**63`` has no
    `int64` form, so the read raises naming the file that holds it rather than wrapping.
    """
    directory = _write(
        tmp_path,
        [pa.table({"a": pa.array([2**63], pa.uint64())}), pa.table({"a": pa.array([-1])})],
    )
    with pytest.raises(SchemaError) as info:
        bt.read.parquet(directory, schema_mode="union").collect(distributed=False)
    message = str(info.value)
    assert "f0.parquet" in message and "uint64" in message and "int64" in message


def test_uint64_beside_int64_reads_when_every_value_fits(tmp_path):
    directory = _write(
        tmp_path,
        [pa.table({"a": pa.array([7], pa.uint64())}), pa.table({"a": pa.array([-1])})],
    )
    ours = bt.read.parquet(directory, schema_mode="union").collect(distributed=False)
    assert ours.schema.field("a").type == pa.int64()
    assert sorted(ours.column("a").to_pylist()) == sorted(
        int(v) for v in _duck_union(directory).column("a").to_pylist()
    )


def test_latest_declares_a_column_nullable_when_an_older_file_lacks_it(tmp_path):
    required = pa.schema([pa.field("a", pa.int64(), False), pa.field("b", pa.int64(), False)])
    directory = _write(
        tmp_path,
        [pa.table({"a": [1]}), pa.Table.from_pylist([{"a": 2, "b": 3}], schema=required)],
    )
    out = bt.read.parquet(directory, schema_mode="latest").collect(distributed=False)
    assert sorted(out.to_pylist(), key=lambda r: r["a"]) == [{"a": 1, "b": None}, {"a": 2, "b": 3}]


def test_latest_raises_when_an_older_value_does_not_survive_the_newer_type(tmp_path):
    directory = _write(tmp_path, [pa.table({"a": [5]}), pa.table({"a": [True]})])
    with pytest.raises(SchemaError) as info:
        bt.read.parquet(directory, schema_mode="latest").collect(distributed=False)
    assert "f0.parquet" in str(info.value)


def test_normalize_batch_names_the_file_it_was_given():
    batch = pa.record_batch({"a": pa.array([2**63], pa.uint64())})
    with pytest.raises(SchemaError, match=r"part-7\.parquet"):
        normalize_batch(batch, pa.schema([("a", pa.int64())]), path="part-7.parquet")


# --- strict splits carry the contract to the worker --------------------------------------


def _read_splits(source) -> pa.Table:
    """Every split read after a pickle round trip: the distributed read in miniature."""
    batches = []
    for split in source.splits():
        batches.extend(pickle.loads(pickle.dumps(split)).read(None))
    return pa.Table.from_batches(batches)


def test_a_renamed_column_raises_through_the_splits_as_it_does_locally(tmp_path):
    directory = _write(tmp_path, [pa.table({"A": [1]}), pa.table({"a": [2]})])
    with pytest.raises(SchemaError, match="missing column 'A'"):
        bt.read.parquet(directory).collect(distributed=False)
    with pytest.raises(SchemaError, match="missing column 'A'"):
        _read_splits(ParquetSource(directory))


def test_a_widened_file_reads_through_the_splits_at_the_declared_type(tmp_path):
    directory = _write(tmp_path, [pa.table({"a": [1]}), pa.table({"a": pa.array([2], pa.int32())})])
    source = ParquetSource(directory)
    whole = source.read()
    via_splits = _read_splits(source)
    assert via_splits.schema == source.schema() == whole[0].schema
    assert sorted(via_splits.column("a").to_pylist()) == [1, 2]


def test_a_split_whose_footer_matches_is_left_for_the_fast_reader(tmp_path):
    directory = _write(tmp_path, [pa.table({"a": [1]}), pa.table({"a": [2]})])
    splits = ParquetSource(directory)._held_to_contract(
        [RowGroupSplit(f"{directory}/f{i}.parquet", (0,), 1) for i in range(2)]
    )
    assert all(isinstance(s, RowGroupSplit) for s in splits)


def test_a_split_whose_footer_differs_is_wrapped(tmp_path):
    directory = _write(tmp_path, [pa.table({"a": [1]}), pa.table({"a": pa.array([2], pa.int32())})])
    planned = [RowGroupSplit(f"{directory}/f{i}.parquet", (0,), 1) for i in range(2)]
    splits = ParquetSource(directory)._held_to_contract(planned)
    assert isinstance(splits[0], RowGroupSplit)
    assert isinstance(splits[1], ConformedSplit)
    assert splits[1].identity() != planned[1].identity()
    assert splits[1].read(["a"])[0].schema == pa.schema([("a", pa.int64())])


def test_a_single_file_source_is_never_wrapped(tmp_path):
    directory = _write(tmp_path, [pa.table({"a": [1]})])
    assert not any(isinstance(s, ConformedSplit) for s in ParquetSource(directory).splits())
