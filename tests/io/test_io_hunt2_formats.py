"""Wave-2 IO format round-trip fidelity regressions.

Each test fails on the pre-fix behavior:

- JSON write upcast an integer column containing a null to float64 via pandas'
  ``to_pandas``, silently turning ``9007199254740993`` into ``9007199254740992.0``
  and changing the column's type on a round-trip (data corruption).
- A byte-range-split CSV read inferred each range's column types independently, so
  an early all-integer range parsed a column as ``int64`` while a later range with a
  string value parsed it as ``string`` — the ranges of one file disagreed with each
  other and with the source's advertised schema (mixed-type corruption / crash).
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.structured.csv import CSVSource

pytestmark = pytest.mark.unit


def _roundtrip_json(table: pa.Table) -> pa.Table:
    path = tempfile.mktemp(suffix=".json")
    try:
        SINKS.get("json")().write(table, path)
        return pa.Table.from_batches(SOURCES.get("json")(path).read())
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_json_int_column_with_null_preserves_type_and_value() -> None:
    # int64 with a null must NOT become float64 (pandas upcast) on a JSON round-trip.
    big = 9007199254740993  # 2**53 + 1: not representable as float64
    table = pa.table({"a": pa.array([1, None, big], type=pa.int64())})
    out = _roundtrip_json(table)
    assert pa.types.is_integer(out.schema.field("a").type), out.schema.field("a").type
    assert out.column("a").to_pylist() == [1, None, big]


def test_json_uint64_max_written_exactly() -> None:
    # The writer must not upcast a uint64 column with a null to float (which would
    # mangle 2**64 - 1). (pyarrow.json read-back of values > int64 max is a separate
    # reader limitation, so this pins the write side where the corruption was.)
    from batcher.io.formats.semistructured.json import _ndjson_bytes

    umax = 18446744073709551615  # 2**64 - 1
    table = pa.table({"u": pa.array([None, 5, umax], type=pa.uint64())})
    assert str(umax).encode() in _ndjson_bytes(table)


def test_csv_range_splits_agree_on_column_type() -> None:
    # A column that is integer-looking in the first range and string-looking later
    # must parse to ONE consistent type across every byte-range split, matching the
    # source schema — not int64 in one range and string in another.
    path = tempfile.mktemp(suffix=".csv")
    lines = ["v", *[str(i) for i in range(2000)], *[f"abc{i}" for i in range(2000)]]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        size = os.path.getsize(path)
        src = CSVSource(path)
        declared = src.schema().field("v").type
        splits = src.splits(target_size=size // 4)
        assert len(splits) > 1  # the file really did split into ranges
        types: set[str] = set()
        rows = 0
        for split in splits:
            batches = split.read()
            if not batches:
                continue
            tbl = pa.Table.from_batches(batches)
            types.add(str(tbl.schema.field("v").type))
            rows += tbl.num_rows
        assert types == {str(declared)}, types
        assert rows == 4000  # no rows dropped
    finally:
        if os.path.exists(path):
            os.remove(path)
