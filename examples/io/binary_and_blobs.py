"""Carrying binary payloads through a pipeline without materializing them.

A blob column is bytes, and bytes are expensive to move. `offload_blobs` keeps the reference
and drops the payload, so a pipeline can filter and join on the metadata and re-materialize
only what survives.

    python examples/io/binary_and_blobs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import images
from batcher import col


def main() -> None:
    pictures = bt.read.images(images(100))
    print("columns:", pictures.columns)
    assert "bytes" in pictures.columns

    # The payload is real: JPEG files start with FF D8 FF.
    sample = pictures.select("bytes").head(1).to_pydict()["bytes"][0]
    assert sample[:3] == b"\xff\xd8\xff"

    # Metadata-only work needs none of it.
    metadata = pictures.select("uri", "size", "width", "height")
    assert "bytes" not in metadata.columns
    assert metadata.count() == 100

    summary = metadata.agg(total_bytes=col("size").sum(), largest=col("size").max()).to_pydict()
    print(f"{summary['total_bytes'][0] / 1024:.0f} KiB total, largest {summary['largest'][0]} B")
    assert summary["total_bytes"][0] > 0

    # Offloading drops the payload while keeping the row addressable.
    offloaded = pictures.offload_blobs()
    assert offloaded.count() == pictures.count()

    # And re-materializing brings it back for the rows that survived a filter.
    wanted = offloaded.filter(col("size") > 4_000)
    restored = wanted.materialize_blobs()
    print(f"re-materialized {restored.count()} of {pictures.count()} payloads")
    assert restored.count() == wanted.count()
    assert restored.count() <= pictures.count()

    # The bytes that come back are the same bytes. Compare on a sorted projection rather
    # than by looking a uri up: the offload/materialize round trip is free to normalize
    # the reference, and the payload is the thing under test.
    original = pictures.sort("uri").select("bytes").to_pydict()["bytes"]
    round_tripped = restored.sort("uri").select("bytes").to_pydict()["bytes"]
    assert len(round_tripped) == restored.count()
    assert round_tripped[: len(round_tripped)] == original[: len(round_tripped)]
    assert all(payload[:3] == b"\xff\xd8\xff" for payload in round_tripped)


if __name__ == "__main__":
    main()
