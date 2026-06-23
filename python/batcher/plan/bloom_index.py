"""`BloomIndex` — a data-skipping membership index over a column's values.

A per-column bloom filter lets the optimizer prune a scan for an *equality* (or
`IN`) predicate the way zone-map min/max bounds prune a *range* one — but where
min/max cannot help: a point lookup whose value lies inside `[min, max]` yet is
absent from a high-cardinality column. `col = 9700123` over a 10M-row id column is
the canonical case.

The index is built in Rust (`bc_py::build_column_bloom`, fast over Arrow) but
**checked here, in pure Python**, because the optimizer (Kyber) may not call the
native engine. That requires a *portable* hash both sides compute identically:
FNV-1a-64 over a canonical byte encoding of the value (a signed 8-byte little-endian
integer, or raw UTF-8 for text). The bit layout and the double-hashing index scheme
match `bc_sketches::BloomFilter` exactly, so a Rust-built bloom round-trips through
the same `to_bytes` wire format this reader parses.

No false negatives: a value that was inserted always tests present, so a `False`
from `contains` is definitive — the predicate cannot match and the scan is pruned.
"""

from __future__ import annotations

import struct

__all__ = ["BloomIndex", "canonical_bytes"]

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def _fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit — a tiny, portable hash computed identically in Rust and here."""
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & _MASK64
    return h


def canonical_bytes(value: object) -> bytes | None:
    """The canonical byte encoding a column bloom hashes, or None if unindexable.

    Must match `bc_py`'s builder byte-for-byte: a Python `int` → signed 8-byte
    little-endian (the engine's normalized Int64), `str` → UTF-8. Other types
    (float, bool, date) are not indexed — equality on them is rare and their byte
    encodings are fiddly to keep identical across the FFI — so the bloom is simply
    not consulted for them.
    """
    if isinstance(value, bool):
        return None  # bool is an int subclass; not indexed
    if isinstance(value, int):
        try:
            return value.to_bytes(8, "little", signed=True)
        except OverflowError:
            return None  # outside Int64 → cannot be in an Int64 column's index
    if isinstance(value, str):
        return value.encode("utf-8")
    return None


class BloomIndex:
    """A read-only view over a serialized `bc_sketches::BloomFilter` (see module doc)."""

    __slots__ = ("_words", "num_bits", "num_hashes")

    def __init__(self, num_bits: int, num_hashes: int, words: list[int]) -> None:
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self._words = words

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomIndex | None:
        """Parse the `BloomFilter::to_bytes` wire format, or None if malformed."""
        if data is None or len(data) < 12:
            return None
        num_bits = struct.unpack_from("<Q", data, 0)[0]
        num_hashes = struct.unpack_from("<I", data, 8)[0]
        body = data[12:]
        if num_bits == 0 or num_bits % 64 != 0 or len(body) != (num_bits // 64) * 8:
            return None
        words = list(struct.unpack(f"<{num_bits // 64}Q", body))
        return cls(num_bits, num_hashes, words)

    def contains(self, value: object) -> bool:
        """Whether `value` *may* be present. `False` is definitive (never inserted);
        `True` may be a false positive. An unindexable value conservatively returns
        `True` (cannot prove absence)."""
        encoded = canonical_bytes(value)
        if encoded is None:
            return True  # cannot encode → cannot prove absence → assume present
        return self.contains_hash(_fnv1a_64(encoded))

    def contains_hash(self, h: int) -> bool:
        # Double hashing (`h1 + i·h2`) over the same positions bc_sketches sets.
        h1 = h & 0xFFFFFFFF
        h2 = ((h >> 32) | 1) & _MASK64
        for i in range(self.num_hashes):
            pos = (h1 + (i * h2 & _MASK64)) % self.num_bits
            if not (self._words[pos // 64] >> (pos % 64)) & 1:
                return False
        return True
