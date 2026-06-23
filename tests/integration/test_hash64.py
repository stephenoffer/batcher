"""`str.hash64` — deterministic FNV-1a hashing for surrogate keys / change detection.

Not a DuckDB differential (DuckDB's ``hash()`` uses a different algorithm); instead
we lock in determinism, known FNV-1a vectors, null propagation, and the
surrogate-key composition pattern.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _fnv1a64(b: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in b:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    # Reinterpret the u64 digest as a signed i64 (Arrow Int64).
    return h - 2**64 if h >= 2**63 else h


def test_hash64_matches_fnv_and_propagates_null():
    t = pa.table({"s": ["a", "foobar", "customer-42|2024-06-23", None]})
    out = bt.from_arrow(t).select(h=col("s").str.hash64()).to_pydict()
    assert out["h"][0] == _fnv1a64(b"a")
    assert out["h"][1] == _fnv1a64(b"foobar")
    assert out["h"][2] == _fnv1a64(b"customer-42|2024-06-23")
    assert out["h"][3] is None


def test_hash64_is_partition_independent():
    """The hash of a value is identical regardless of how rows are batched —
    the property that makes it safe as a distributed surrogate key."""
    rows = [{"s": f"k{i}"} for i in range(50)]
    whole = pa.Table.from_pylist(rows)
    batched = pa.Table.from_batches(
        [pa.RecordBatch.from_pylist(rows[:13]), pa.RecordBatch.from_pylist(rows[13:])]
    )
    a = bt.from_arrow(whole).select(h=col("s").str.hash64()).to_pydict()
    b = bt.from_arrow(batched).select(h=col("s").str.hash64()).to_pydict()
    assert a["h"] == b["h"]
