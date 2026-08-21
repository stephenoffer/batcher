"""Differential tests for a **binary** ``ORDER BY`` against DuckDB.

``Binary``, ``LargeBinary`` and ``FixedSizeBinary`` are byte-lexicographic sort keys and
arrow orders them exactly as it orders ``Utf8`` — but until the byte-key sort landed they
matched none of the engine's fast paths. The sort fell to an unstable ``lexsort`` with a
row-index tie-break column, the parallel sample-sort declined the type outright and ran
single-threaded, and the distributed range partitioner refused it. That is the shape a
fixed-width key over a wide payload takes, which is the canonical large-sort workload.

DuckDB reads all three as ``BLOB`` and orders them by the same byte comparison, so it is
the oracle here as it is everywhere else.

The payload column ``p`` is the input row position, and Batcher's byte-key sort is stable,
so ties come back in ``p`` order. That makes DuckDB's ``ORDER BY k, p`` a *total* order
Batcher's raw output must match row for row — which is what makes these assertions
meaningful rather than a multiset comparison.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered, assert_tables_equal

# Comfortably over the 131_072-row threshold that turns on the parallel sample-sort.
PARALLEL_ROWS = 200_000

# The three spellings of a byte key that are not text, with the width `pa.binary(width)`
# needs. `None` means variable-length.
BINARY_TYPES = [
    ("binary", None),
    ("large_binary", None),
    ("fixed_size_binary", 8),
]


def _binary_type(kind: str, width: int | None) -> pa.DataType:
    """The pyarrow type for one row of [`BINARY_TYPES`]."""
    if kind == "binary":
        return pa.binary()
    if kind == "large_binary":
        return pa.large_binary()
    return pa.binary(width)


def _keys(n: int, distinct: int, width: int | None, *, nulls: bool) -> list[bytes | None]:
    """Deterministic keys that tie heavily (``n // distinct`` duplicates each).

    Values are built from a multiplicative hash so the bytes are spread rather than
    ascending, and **zero bytes are deliberately common**: a `0` byte is the one byte a
    zero-padded key pack can confuse with the end of a shorter value, so a test whose keys
    avoid it would not exercise the case the pack has to decline.
    """
    out: list[bytes | None] = []
    for i in range(n):
        if nulls and i % 97 == 0:
            out.append(None)
            continue
        value = ((i * 7919) % distinct).to_bytes(4, "big")
        # A variable-length column gets genuinely varying lengths, which is the other half of
        # what the pack has to reason about.
        out.append(value.rjust(width, b"\x00") if width else value[: 2 + (i % 3)])
    return out


def _table(n: int, distinct: int, kind: str, width: int | None, *, nulls: bool = False) -> pa.Table:
    """A `(k, p)` relation: a binary key of `kind` and the input row position."""
    return pa.table(
        {
            "k": pa.array(_keys(n, distinct, width, nulls=nulls), type=_binary_type(kind, width)),
            "p": pa.array(range(n), type=pa.int64()),
        }
    )


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
@pytest.mark.parametrize("descending", [False, True])
def test_parallel_binary_sort_matches_duckdb(duck, kind, width, descending):
    """The parallel sample-sort over a binary key, which used to be a serial comparison sort."""
    t = _table(PARALLEL_ROWS, distinct=5_000, kind=kind, width=width)
    duck.register("t", t)
    direction = "DESC" if descending else "ASC"
    out = bt.from_arrow(t).sort("k", descending=descending).collect()
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {direction}, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
@pytest.mark.parametrize("nulls_first", [False, True])
def test_binary_sort_places_nulls_like_duckdb(duck, kind, width, nulls_first):
    """Nulls group at the end the flag names, in input order, above the parallel threshold.

    The parallel path routes nulls to a dedicated end bucket rather than by comparison, so
    this is the assertion that the bucket it picks is the one the serial sort would.
    """
    t = _table(PARALLEL_ROWS, distinct=1_000, kind=kind, width=width, nulls=True)
    duck.register("t", t)
    placement = "NULLS FIRST" if nulls_first else "NULLS LAST"
    out = bt.from_arrow(t).sort("k", nulls_first=nulls_first).collect()
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k ASC {placement}, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
def test_serial_binary_sort_matches_duckdb(duck, kind, width):
    """Below the parallel threshold the serial sort answers, and must agree with it."""
    t = _table(4_096, distinct=64, kind=kind, width=width, nulls=True)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC NULLS LAST, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
def test_binary_top_n_matches_duckdb(duck, kind, width):
    """``ORDER BY <binary> LIMIT k`` takes the packed-prefix selection path, not the full sort."""
    t = _table(PARALLEL_ROWS, distinct=50_000, kind=kind, width=width, nulls=True)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").limit(100).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC NULLS LAST, p ASC LIMIT 100"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
def test_binary_multi_key_sort_matches_duckdb(duck, kind, width):
    """A binary *leading* key with a second key: the ranges must still sort by the full list."""
    t = _table(PARALLEL_ROWS, distinct=64, kind=kind, width=width)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k", "p", descending=[False, True]).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC, p DESC"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
@pytest.mark.parametrize("rows", [0, 1, 3])
def test_degenerate_binary_sorts_match_duckdb(duck, kind, width, rows):
    """Empty, one row, and a handful — the sizes every specialized path has to decline on."""
    t = _table(rows, distinct=2, kind=kind, width=width)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k", descending=True).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k DESC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
def test_a_constant_binary_key_keeps_input_order(duck, kind, width):
    """Every key equal: the sort must be the identity, which is what stability means.

    This is the shape the parallel sample-sort splits into constant ranges, and the one where
    an unstable sort's damage is invisible to a multiset comparison — so it is asserted
    against a *total* order that names the tie-break.
    """
    n = PARALLEL_ROWS
    value = (b"\x00" * (width - 1) + b"\x07") if width else b"\x07"
    t = pa.table(
        {
            "k": pa.array([value] * n, type=_binary_type(kind, width)),
            "p": pa.array(range(n), type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").collect()
    assert out.column("p").to_pylist() == list(range(n))
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize(("kind", "width"), BINARY_TYPES)
def test_binary_sort_agrees_across_execution_paths(kind, width):
    """`collect`, `collect(spill=True)` and `iter_batches` must produce the identical relation.

    The cross-product `CLAUDE.md` names, on the key type that had no stable single-key path:
    the spilling sort runs its comparison over per-run slices and the streaming one over
    per-batch slices, and if those disagree with the whole-relation sort about *ties* the
    difference is a reordering no order-independent assertion can see.
    """
    t = _table(PARALLEL_ROWS, distinct=1_000, kind=kind, width=width, nulls=True)
    ds = bt.from_arrow(t).sort("k")
    collected = ds.collect()
    assert_tables_equal(ds.collect(spill=True), collected, ordered=True)
    assert_tables_equal(
        pa.Table.from_batches(list(ds.iter_batches()), schema=collected.schema),
        collected,
        ordered=True,
    )
