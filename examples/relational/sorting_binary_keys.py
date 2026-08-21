"""Sorting by a binary key: hashes, UUIDs, and fixed-layout records.

A `binary` column is a first-class sort key and is ordered by the same byte comparison a
string is, so a hash, a checksum, a UUID, or a key someone already encoded sorts without
being decoded first. All three Arrow spellings order identically: variable-length `binary`
and `large_binary`, and fixed-width `binary(n)`.

The shape at the end is the one the Sort Benchmark family (GraySort, CloudSort) defines: a
10-byte key over a 90-byte payload. It is worth knowing because it is the extreme of a ratio
most large sorts have -- the key is narrow, the payload is wide, and what the sort really
costs is moving the payload once.

    python examples/relational/sorting_binary_keys.py
"""

from __future__ import annotations

import hashlib

import pyarrow as pa

import batcher as bt


def main() -> None:
    # A digest column: the natural key of a content-addressed table, and a type that reads
    # back as `binary` from every format that stores it.
    rows = [f"row-{i:04d}" for i in range(1_000)]
    digests = [hashlib.sha256(r.encode()).digest()[:10] for r in rows]
    table = pa.table({"digest": pa.array(digests, type=pa.binary(10)), "row": rows})

    ordered = bt.from_arrow(table).sort("digest").to_pydict()
    # Byte-lexicographic, exactly as Arrow and DuckDB order a BLOB. Asserted as an *ordered*
    # sequence: comparing a sort's output as a set is how a sort bug survives a test suite.
    assert ordered["digest"] == sorted(digests)
    # And the payload column travelled with its key rather than being re-derived.
    expected = [row for _, row in sorted(zip(digests, rows, strict=True))]
    assert ordered["row"] == expected
    print("first digest:", ordered["digest"][0].hex())

    # The three binary spellings are one ordering. Only the storage differs: `binary(n)` has
    # no offset buffer, and the engine can prove that a padded comparison of its bytes is
    # exact, which a variable-length column holding a zero byte does not allow.
    for arrow_type in (pa.binary(), pa.large_binary(), pa.binary(10)):
        column = pa.table({"k": pa.array(digests, type=arrow_type)})
        assert bt.from_arrow(column).sort("k").to_pydict()["k"] == sorted(digests)

    # Descending and null placement work as they do for any other key. Nulls sort last by
    # default in both directions; say `nulls_first=True` if you want the other end.
    with_nulls = pa.table({"k": pa.array([b"\x02", None, b"\x00\xff", b"\x01"], type=pa.binary())})
    ds = bt.from_arrow(with_nulls)
    assert ds.sort("k").to_pydict()["k"] == [b"\x00\xff", b"\x01", b"\x02", None]
    assert ds.sort("k", descending=True).to_pydict()["k"] == [b"\x02", b"\x01", b"\x00\xff", None]
    assert ds.sort("k", nulls_first=True).to_pydict()["k"] == [None, b"\x00\xff", b"\x01", b"\x02"]

    # The fixed-layout record: 10 bytes of key, 90 of payload. The engine range-partitions on
    # the key and gathers the payload exactly once, which is why the payload's width costs far
    # less than sorting it as part of the key would.
    # Both halves fixed width, which is what the benchmark's record actually is -- and the
    # cheapest shape to move, since a fixed stride needs no offset buffer to chase.
    records = pa.table(
        {
            "key": pa.array(digests, type=pa.binary(10)),
            "payload": pa.array([b"\xab" * 90] * len(digests), type=pa.binary(90)),
        }
    )
    sorted_records = bt.from_arrow(records).sort("key").collect()
    assert sorted_records.column("key").to_pylist() == sorted(digests)
    assert sorted_records.num_rows == len(digests)
    print("sorted", sorted_records.num_rows, "100-byte records")

    # The same call distributes: passing `distributed=True` to `collect` samples the leading
    # key for byte quantiles and has each worker sort its own range, so a sort keyed on bytes
    # is not capped at one machine. Not run here, because it needs a cluster.


if __name__ == "__main__":
    main()
