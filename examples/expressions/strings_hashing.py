"""Hashing and encoding a string column: keys, checksums, and safe transport.

Hashes give you a fixed-width key from arbitrary text, which is how you bucket, partition,
or pseudonymize without a lookup table. Encodings move bytes through channels that only
accept text. Neither is encryption: a hash is one-way, base64 is not secret at all.

    python examples/expressions/strings_hashing.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ids = bt.from_pydict({"email": ["ada@example.com", "bob@example.com", "ada@example.com"]})

    hashed = ids.with_columns(
        h64=col("email").str.hash64(),
        xx=col("email").str.xxhash64(),
        crc=col("email").str.crc32(),
        md5=col("email").str.md5(),
        sha1=col("email").str.sha1(),
        sha256=col("email").str.sha256(),
    ).to_pydict()

    print({k: v[:2] for k, v in hashed.items()})

    # A hash is deterministic: the same input gives the same digest every time, which is
    # what makes it usable as a stable key.
    assert hashed["sha256"][0] == hashed["sha256"][2]
    assert hashed["sha256"][0] != hashed["sha256"][1]
    assert hashed["h64"][0] == hashed["h64"][2]
    # SHA-256 is 64 hex characters.
    assert len(hashed["sha256"][0]) == 64
    assert len(hashed["md5"][0]) == 32

    # The bucketing this exists for: a stable shard from a high-cardinality key.
    bucketed = ids.select(bucket=col("email").str.hash64().mod(4)).to_pydict()
    print("buckets:", bucketed["bucket"])
    assert all(0 <= b < 4 for b in bucketed["bucket"])
    assert bucketed["bucket"][0] == bucketed["bucket"][2]

    # Encodings, which are reversible and therefore not a privacy control.
    text = bt.from_pydict({"s": ["hello", "wörld"]})
    encoded = text.with_columns(
        b64=col("s").str.base64(),
        hexed=col("s").str.hex(),
        url=col("s").str.url_encode(),
    ).to_pydict()
    print(encoded)

    # Round-trip both ways.
    back = text.select(
        from_b64=col("s").str.base64().str.from_base64(),
        from_hex=col("s").str.hex().str.unhex(),
        from_url=col("s").str.url_encode().str.url_decode(),
    ).to_pydict()
    print(back)
    assert back["from_b64"] == ["hello", "wörld"]
    assert back["from_url"] == ["hello", "wörld"]

    # Byte lengths differ from character lengths once you leave ASCII.
    lengths = text.select(
        chars=col("s").str.len_chars(), octets=col("s").str.octet_length()
    ).to_pydict()
    print(lengths)
    assert lengths["chars"] == [5, 5]
    assert lengths["octets"][1] == 6  # the umlaut costs two bytes


if __name__ == "__main__":
    main()
