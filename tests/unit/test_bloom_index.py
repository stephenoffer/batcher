"""`BloomIndex` — the data-skipping membership index, and its Rust↔Python agreement."""

from __future__ import annotations

import batcher._native as nat
import pyarrow as pa

from batcher.plan.bloom_index import BloomIndex, canonical_bytes


def _bloom(values, arrow_type=None, n=1000):
    batch = pa.record_batch({"c": pa.array(values, arrow_type)})
    return nat.build_column_bloom([batch], 0, n)


# --- canonical_bytes ----------------------------------------------------------


def test_canonical_bytes_int_and_str():
    assert canonical_bytes(5) == (5).to_bytes(8, "little", signed=True)
    assert canonical_bytes(-7) == (-7).to_bytes(8, "little", signed=True)
    assert canonical_bytes("hi") == b"hi"


def test_canonical_bytes_unindexable():
    assert canonical_bytes(1.5) is None  # float
    assert canonical_bytes(True) is None  # bool (int subclass, excluded)
    assert canonical_bytes(2**70) is None  # outside Int64


# --- from_bytes ---------------------------------------------------------------


def test_from_bytes_malformed():
    assert BloomIndex.from_bytes(b"") is None
    assert BloomIndex.from_bytes(b"\x00\x01\x02") is None
    assert BloomIndex.from_bytes(None) is None


# --- Rust build ↔ Python check agreement (the critical contract) --------------


def test_int_no_false_negatives():
    vals = [1, 5, 42, 9_700_123, -7, 1_000_000]
    idx = BloomIndex.from_bytes(_bloom(vals, pa.int64()))
    assert idx is not None
    assert all(idx.contains(v) for v in vals)  # every inserted value present


def test_int_absent_values_pruned():
    idx = BloomIndex.from_bytes(_bloom(list(range(50)), pa.int64()))
    # Values far outside the inserted set are (almost all) absent — the data-skip win.
    absent = sum(not idx.contains(v) for v in range(10_000, 10_200))
    assert absent > 190  # ~all of 200 absent (1% fp budget)


def test_string_membership():
    idx = BloomIndex.from_bytes(_bloom(["alice", "bob", "zoë", "carol"]))
    assert idx.contains("alice") and idx.contains("zoë")  # incl. non-ASCII
    assert not idx.contains("dave")


def test_narrow_int_widened_and_matched():
    # Int32 is normalized to Int64 at the FFI boundary, matching the int literal encoding.
    idx = BloomIndex.from_bytes(_bloom([3, 4], pa.int32()))
    assert idx.contains(3) and not idx.contains(999)


def test_unindexable_column_returns_none():
    assert _bloom([1.0, 2.0], pa.float64()) is None


def test_contains_unindexable_value_is_conservative():
    # A float probe against an int index can't be encoded → assume present (no prune).
    idx = BloomIndex.from_bytes(_bloom([1, 2, 3], pa.int64()))
    assert idx.contains(1.5) is True
