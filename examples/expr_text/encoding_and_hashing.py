"""Encoding bytes and hashing values.

Hex and base64 are encodings: reversible, and about transport. Hashes are not reversible
and are about identity — a stable fingerprint you can join on or compare without carrying
the original value around.

    python examples/expr_text/encoding_and_hashing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_name").head(100)

    encoded = customer.select(
        "c_name",
        hexed=col("c_name").str.hex(),
        b64=col("c_name").str.base64(),
        url=col("c_name").str.url_encode(),
    ).with_columns(
        from_hex=col("hexed").str.unhex(),
        from_b64=col("b64").str.from_base64(),
        from_url=col("url").str.url_decode(),
    )

    result = encoded.head(2).to_pydict()
    print(result["hexed"][0][:40], "...")

    full = encoded.to_pydict()
    # Encodings round-trip exactly.
    assert full["from_hex"] == full["c_name"]
    assert full["from_b64"] == full["c_name"]
    assert full["from_url"] == full["c_name"]

    # Hashes are fixed width and collision-free on this input.
    hashes = customer.select(
        md5=col("c_name").str.md5(),
        sha1=col("c_name").str.sha1(),
        sha256=col("c_name").str.sha256(),
        crc=col("c_name").str.crc32(),
    ).to_pydict()
    assert all(len(value) == 32 for value in hashes["md5"])
    assert all(len(value) == 40 for value in hashes["sha1"])
    assert all(len(value) == 64 for value in hashes["sha256"])
    assert len(set(hashes["sha256"])) == customer.count()

    # A hash is deterministic: the same input hashes the same way twice.
    again = customer.select(sha256=col("c_name").str.sha256()).to_pydict()
    assert again["sha256"] == hashes["sha256"]


if __name__ == "__main__":
    main()
