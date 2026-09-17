"""The shared fixture for the engine round-trip suites (`tests/integration/test_interop_*`).

One table covers every case those suites are asked to hold: nulls (row 1 is null in every
column but the key and the dictionary), an empty result (`empty`), a nested list and struct,
a timestamp in a non-UTC zone, a decimal, a dictionary-encoded column, a string of 200,000
UTF-8 bytes, and binary holding a NUL byte and an empty value.

The expected *types* are pinned here rather than derived, because a type that drifts is the
defect these suites exist to catch: a derivation would move with it. Two schemas are pinned.
`ENGINE_SCHEMA` is what Batcher produces for the table itself: the dictionary decodes and
``large_string`` narrows to ``string`` at the engine boundary. `FOREIGN_SCHEMA` is what it
produces for the same rows arriving back from Polars or Daft, which export binary and list
columns with 64-bit offsets and so arrive as ``large_binary`` and ``large_list``.
"""

from __future__ import annotations

import datetime as dt
import decimal
from typing import Any

import pyarrow as pa

import batcher as bt

_TOKYO = dt.timezone(dt.timedelta(hours=9))
_STRUCT = pa.struct([("x", pa.int64()), ("y", pa.string())])
_TS = pa.timestamp("us", tz="Asia/Tokyo")
_DEC = pa.decimal128(10, 2)

TABLE = pa.table(
    {
        "id": pa.array([0, 1, 2], pa.int64()),
        "i": pa.array([1, None, -3], pa.int64()),
        "s": pa.array(["a", None, "é" * 100_000], pa.large_string()),
        "b": pa.array([b"\x00\xff", None, b""], pa.binary()),
        "l": pa.array([[1, 2], None, []], pa.list_(pa.int64())),
        "st": pa.array([{"x": 1, "y": "a"}, None, {"x": None, "y": "b"}], _STRUCT),
        "ts": pa.array(
            [
                dt.datetime(2024, 1, 1, 9, tzinfo=_TOKYO),
                None,
                dt.datetime(2024, 6, 1, 12, tzinfo=_TOKYO),
            ],
            _TS,
        ),
        "d": pa.array([decimal.Decimal("1.25"), None, decimal.Decimal("-3.50")], _DEC),
        "dict": pa.array(["u", "v", None]).dictionary_encode(),
    }
)

#: The rows, in key order, as Python values. A dictionary cell reads as its value.
ROWS: list[dict[str, Any]] = TABLE.to_pylist()

ENGINE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("i", pa.int64()),
        ("s", pa.string()),
        ("b", pa.binary()),
        ("l", pa.list_(pa.int64())),
        ("st", _STRUCT),
        ("ts", _TS),
        ("d", _DEC),
        ("dict", pa.string()),
    ]
)

FOREIGN_SCHEMA = ENGINE_SCHEMA.set(3, pa.field("b", pa.large_binary())).set(
    4, pa.field("l", pa.large_list(pa.int64()))
)


def dataset() -> bt.Dataset:
    """The fixture as a Batcher dataset, sorted on its key so its row order is defined."""
    return bt.from_arrow(TABLE).sort("id")


def empty() -> bt.Dataset:
    """The fixture filtered to no rows: a result with a schema and nothing else."""
    return bt.from_arrow(TABLE).filter(bt.col("id") < 0)


def assert_types(schema: pa.Schema, expected: pa.Schema) -> None:
    """Every column's name and Arrow type, in order, ignoring nullability and metadata."""
    got = [(field.name, field.type) for field in schema]
    want = [(field.name, field.type) for field in expected]
    assert got == want
