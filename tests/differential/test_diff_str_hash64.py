"""`Expr.str.hash64` against an independent FNV-1a 64-bit reference.

DuckDB has no FNV-1a function, so the oracle here is a ten-line reference implementation
of the published algorithm (offset basis ``0xcbf29ce484222325``, prime ``0x100000001b3``,
over the UTF-8 bytes, reinterpreted as a signed Int64). The docstring promises the hash is
stable across partitions and runs because surrogate keys and SCD change detection are built
on it, so the value itself is the contract, not merely its determinism.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

_MASK = (1 << 64) - 1


def _fnv1a64(s: str | None) -> int | None:
    """The FNV-1a 64-bit hash of `s`'s UTF-8 bytes, as a signed 64-bit integer."""
    if s is None:
        return None
    h = 0xCBF29CE484222325
    for byte in s.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & _MASK
    return h - (1 << 64) if h >= 1 << 63 else h


VALUES = ["abc", "", None, "a", "aa", "héllo", "日本語", "abc", "x" * 1000]


def _hash(values: list[str | None]) -> pa.Table:
    table = pa.table({"s": pa.array(values, pa.string())})
    return bt.from_arrow(table).select(h=bt.col("s").str.hash64()).collect()


def test_hash64_matches_the_reference_on_nulls_empty_and_multibyte():
    out = _hash(VALUES)
    assert out.schema.field("h").type == pa.int64()
    assert out.column("h").to_pylist() == [_fnv1a64(v) for v in VALUES]


def test_hash64_on_empty_input_and_one_row():
    assert _hash([]).num_rows == 0
    assert _hash(["abc"]).column("h").to_pylist() == [-1792535898324117685]


def test_hash64_is_the_same_on_every_execution_path():
    """Stable across partitions: a multi-morsel input hashed collected, spilled and streamed."""
    values = [f"key-{i % 977}" if i % 13 else None for i in range(40_000)]
    ds = bt.from_arrow(pa.table({"s": values})).select(bt.col("s"), h=bt.col("s").str.hash64())
    expected = {v: _fnv1a64(v) for v in set(values)}
    tables = [ds.collect(), ds.collect(spill=True)]
    tables.append(pa.Table.from_batches(list(ds.iter_batches())))
    for out in tables:
        assert out.num_rows == len(values)
        for s, h in zip(out.column("s").to_pylist(), out.column("h").to_pylist(), strict=True):
            assert h == expected[s]
